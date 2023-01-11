#%%
# Imports

import copy
import datetime
import glob
import itertools
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import (
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import albumentations as A
import cv2
import matplotlib.pyplot as plt
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

import proxyattention

sns.set()

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
cudnn.benchmark = True
#%%
# Config
# TODO Refactor this into a JSON with options
experiment_params = {
    "experiment_name": "test_asl_starter",
    "ds_path": Path("/media/hdd/Datasets/asl/"),
    "ds_name": "asl",
    "name_fn": proxyattention.data_utils.asl_name_fn,
    "image_size": 224,
    "batch_size": 128,
    "epoch_steps": [1, 2],
    "enable_proxy_attention": True,
    "change_subset_attention": 0.01,
    "validation_split": 0.3,
    "shuffle_dataset": True,
    "num_gpu": 1,
    "transfer_imagenet": False,
    "subset_images": 5000,

}
config = proxyattention.configuration.Experiment(params=experiment_params)

# Make dirs
logging.info("Directories made/checked")
os.makedirs("runs/", exist_ok=True)
fname_start = f'runs/{config.ds_name}_{config.experiment_name}+{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}_subset-{config.subset_images}'  # unique_name

config.fname_start = fname_start

logging.basicConfig(filename=fname_start, encoding="utf-8", level=logging.DEBUG)
logging.info(f"[INFO] : File name = {fname_start}")
print(f"[INFO] : File name = {fname_start}")

config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
# Data part
train, val, label_map = proxyattention.data_utils.create_folds(config)
image_datasets, dataloaders, dataset_sizes = proxyattention.data_utils.create_dls(
    train, val, config
)
class_names = image_datasets["train"].classes
#%%
num_classes = len(label_map.keys())

# TODO add more networks, transformers etc
if config.transfer_imagenet == True:
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model_ft.parameters():
        param.requires_grad = False
else:
    model_ft = models.resnet18(weights=None)
    for param in model_ft.parameters():
        param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(config.device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-4)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#%%
# TODO Proxy loop function, custom schedule
# TODO Proxy attention tabular support
model_ft = proxyattention.training.train_model(
    model_ft,
    config.criterion,
    config.optimizer_ft(model_ft.parameters(), lr=config.lr),
    config.exp_lr_scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=1,
    config=config,
)
#%%
