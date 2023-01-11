#%%
# Imports

import copy
import datetime
import glob
import itertools
import mimetypes
import os
import time
from pathlib import Path
from typing import (Dict, Generator, Iterable, Iterator, List, Optional,
                    Sequence, Set, Tuple, Union)

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import meta_utils
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from captum.attr import DeepLift, IntegratedGradients, NoiseTunnel, Saliency
from captum.attr import visualization as viz
from sklearn import metrics, model_selection, preprocessing
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

sns.set()

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
cudnn.benchmark = True
#%%
# Config
# TODO Refactor to Config file

def fish_name_fn(x): return str(x).split("/")[-2]
def asl_name_fn(x): return str(x).split("/")[-2]

experiment_name = "test_asl_starter"
ds_path = Path("/media/hdd/Datasets/asl/")
ds_name = "asl"
name_fn = asl_name_fn
image_size = 224
batch_size = 128
epoch_steps = [1,2]
enable_proxy_attention = True
change_subset_attention = 0.01
validation_split = 0.3
shuffle_dataset = True
num_gpu = 1
transfer_imagenet = False
subset_images = 5000 # for testing purposes
#%%

os.makedirs("runs/", exist_ok=True)
fname_start = f'runs/{ds_name}_{experiment_name}+{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}_subset-{subset_images}'  # unique_name
print(f"[INFO] : File name = {fname_start}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
# TODO Options for more config
data_transforms = {
    'train':  A.Compose(
        [
            A.RandomResizedCrop(image_size, image_size, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    ),
    'val': A.Compose([
        A.Resize(image_size, image_size),
        A.CenterCrop(image_size, image_size, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
        p=1.0,
    )

}
#%%
# Data part
# Get all image files
# TODO Refactor into a get_data function
# TODO Allow options for Proxy data
all_files = meta_utils.get_files(ds_path/"train")
if subset_images !=None:
    all_files = all_files[:subset_images]

# Put them in a data frame for encoding
df = pd.DataFrame.from_dict(
    {x: name_fn(x) for x in all_files}, orient="index"
).reset_index()
df.columns = ["image_id", "label"]
# Convert labels to integers
temp = preprocessing.LabelEncoder()
df["label"] = temp.fit_transform(df.label.values)

# Save label map
label_map = {i: l for i, l in enumerate(temp.classes_)}

# Kfold splits
df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
stratify = StratifiedKFold(n_splits=2)
for i, (t_idx, v_idx) in enumerate(
    stratify.split(X=df.image_id.values, y=df.label.values)
):
    df.loc[v_idx, "kfold"] = i
df.to_csv("train_folds.csv", index=False)

#%%
class ImageClassDs(Dataset):
    def __init__(
        self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None
    ):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms
        self.classes = self.df["label"]

    def __getitem__(self, index):
        im_path = self.df.iloc[index]["image_id"]
        x = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if self.transforms:
            x = self.transforms(image=x)["image"]

        y = self.df.iloc[index]["label"]
        return {
            "x": x,
            "y": y,
        }

    def __len__(self):
        return len(self.df)

#%%
train = df.loc[df["kfold"] != 1]
val = df.loc[df["kfold"] == 1]
image_datasets = {
    "train": ImageClassDs(train, ds_path, train=True, transforms=data_transforms["train"]),

    "val": ImageClassDs(val, ds_path, train=False, transforms=data_transforms["val"]),
}

dataloaders = {
    "train": torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=batch_size,
        shuffle=True, num_workers=4* num_gpu, pin_memory = True),
        
    "val": torch.utils.data.DataLoader(
        image_datasets["val"],
        batch_size=batch_size,
        shuffle=False, num_workers=4*num_gpu, pin_memory = True),
    }

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
#%%
# Model Defs
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    writer = SummaryWriter(log_dir=fname_start, comment = fname_start)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    pbar = tqdm(range(num_epochs), total=num_epochs)
    for epoch in pbar:
        # print(f'Epoch {epoch}/{num_epochs - 1}')
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            optimizer.zero_grad(set_to_none=True)
            scaler = torch.cuda.amp.GradScaler()
            for inps in tqdm(dataloaders[phase], total=len(dataloaders[phase]), leave = False):
                inputs = inps['x'].to(device, non_blocking=True)
                labels = inps['y'].to(device, non_blocking=True)

                # zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast():
                        if phase == 'train':
                            outputs = model(inputs)
                        else:
                            with torch.no_grad():
                                outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # TODO 
                        # TODO Save images pipeline
                        # TODO Config to choose what to save

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        # optimizer.step()

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix({
                    'Phase': "running",
                    'Loss': running_loss/dataset_sizes[phase],
                    # 'Acc' : running_corrects.double() / dataset_sizes[phase],
                })

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            pbar.set_postfix({
                'Phase': phase,
                'Loss': epoch_loss,
                'Acc': epoch_acc
            })
            if phase == 'train':
                writer.add_scalar('Loss/Train', epoch_loss, epoch)
                writer.add_scalar('Acc/Train', epoch_acc, epoch)
            if phase == 'val':
                writer.add_scalar('Loss/Val', epoch_loss, epoch)
                writer.add_scalar('Acc/Val', epoch_acc, epoch)


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            # TODO Save best model

        # print()

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#%%
num_classes = len(label_map.keys())

# TODO add more networks, transformers etc
if transfer_imagenet == True:
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model_ft.parameters():
        param.requires_grad = False
else:
    model_ft = models.resnet18(weights=None)
    for param in model_ft.parameters():
        param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-4)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#%%
#TODO Proxy loop function, custom schedule
model_ft = train_model(model_ft, criterion, optimizer_ft,
                       exp_lr_scheduler, num_epochs=10)

#%%