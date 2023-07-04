# %%
# Imports

import copy
from datetime import datetime
import logging
import os
import time
from pathlib import Path

import matplotlib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

matplotlib.use("Agg")  # no UI backend
import pickle

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.image import (
    deprocess_image,
    preprocess_image,
    show_cam_on_image,
)
from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from torchvision import transforms
from tqdm import tqdm
from operator import attrgetter

import time
import gc
import copy
import argparse as ap
import ast
from operator import itemgetter
# from pytorch_memlab import LineProfiler, profile
# import torchsnooper
# import pytorch_lightning as pl
# from lightning.pytorch import LightningModule, Trainer, seed_everything, LightningDataModule
# from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
# from lightning.pytorch.callbacks.progress import TQDMProgressBar
# from lightning.pytorch.loggers import CSVLogger# torch.autograd.set_detect_anomaly(True)
# from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from itertools import combinations
import numpy as np
import itertools
# from lightning.pytorch.loggers import TensorBoardLogger

# from multiprocessing import Pool
# from torch.multiprocessing import Pool
import torch.multiprocessing as mp

from .meta_utils import *
# from pympler import tracker

# sns.set()

cudnn.benchmark = True
logging.basicConfig(level=logging.ERROR)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#%%
import sys
# %%
set_batch_size_dict = {
    "vgg16": 16,
    "vit_base_patch16_224": 16,
    "resnet18": 128,
    "resnet50": 32,
    "efficientnet_b0" : 64,
}


dict_decide_change = {
    "mean": torch.mean,
    "max": torch.max,
    "min": torch.min,
    "halfmax": lambda x: torch.max(x) / 2,
}


# TODO Smoothing maybe?
dict_gradient_method = {
    "gradcam": GradCAM,
    "gradcamplusplus": GradCAMPlusPlus,
    "eigencam": EigenCAM,
}

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)

tfm = transforms.ToPILImage()


def reset_params(model):
    for param in model.parameters():
        param.requires_grad = False


# TODO find some models to use from the repo
def choose_network(config):
    if config["model"] == "vision_transformer":
        config["model"] = "vit_small_patch32_224"
    # Define the number of classes
    model = timm.create_model(
        config["model"],
        pretrained=config["transfer_imagenet"],
        num_classes=config["num_classes"],
    ).to(config["device"], non_blocking = True)
    model.train()
    return model


#%%


def proxy_one_batch(config, input_wrong, cam):
    grads = cam(input_tensor=input_wrong.to(config["device"]), targets=None)
    grads = torch.Tensor(grads).to(config["device"]).unsqueeze(1).expand(-1, 3, -1, -1)
    normalized_inps = inv_normalize(input_wrong)
    
    if config["pixel_replacement_method"] != "blended":
        output =  torch.where(
            grads > config["proxy_threshold"],
            dict_decide_change[config["pixel_replacement_method"]](grads),
            normalized_inps,
        )
    else:
        output= torch.where(
            grads > config["proxy_threshold"],
            (1 - config["proxy_image_weight"] * grads) * normalized_inps,
            normalized_inps,
        )
    del grads
    return output

def proxy_callback(config, input_wrong_full, label_wrong_full, cam):
    writer = config["writer"]
    logging.info("Performing Proxy step")

    # TODO Save Classwise fraction
    chosen_inds = int(np.ceil(config["change_subset_attention"] * len(label_wrong_full)))
    # TODO some sort of decay?
    # TODO Remove min and batchify

    writer.add_scalar(
        "Number_Chosen", chosen_inds, config["global_run_count"]
    )

    input_wrong_full = input_wrong_full[:chosen_inds]
    label_wrong_full = label_wrong_full[:chosen_inds]

    processed_labels = []
    processed_thresholds = []
    logging.info("[INFO] Started proxy batches")

    for i in tqdm(range(0, len(input_wrong_full), config["batch_size"]), desc="Running proxy"):
        try:
            input_wrong = input_wrong_full[i:i+config["batch_size"]]
            label_wrong = label_wrong_full[i:i+config["batch_size"]]

            try:
                input_wrong = torch.squeeze(torch.stack(input_wrong, dim=1))
                label_wrong = torch.squeeze(torch.stack(label_wrong, dim=1))
            except:
                input_wrong = torch.squeeze(input_wrong)
                label_wrong = torch.squeeze(label_wrong)
            
            if i == 0:
                writer.add_images(
                    "original_images",
                    inv_normalize(input_wrong),
                    # input_wrong,
                    config["global_run_count"],
                )
            start_time = time.time()
            thresholded_ims = proxy_one_batch(config, input_wrong.to(config["device"]), cam)
            end_time = time.time()
            processed_thresholds.extend(thresholded_ims.detach().cpu())
            processed_labels.extend(label_wrong)
            # Log how long it took to run the proxy step
            writer.add_scalar(
                "Time_Proxy", end_time - start_time, config["global_run_count"]
            )

            logging.info("[INFO] Ran proxy step")
            if i == 0:
                writer.add_images(
                    "converted_proxy",
                    thresholded_ims,
                    config["global_run_count"],
                )

            logging.info("[INFO] Saving the images")
        except ValueError:
            pass
    
    try:
        processed_thresholds = torch.stack(processed_thresholds, dim = 0).detach()
        batch_size = processed_thresholds.size(0)


        for ind in tqdm(range(batch_size), total=batch_size, desc="Saving images"):
            label = config["label_map"][processed_labels[ind].item()]
            save_name = (
                config["ds_path"] / label / f"proxy-{ind}-{config['global_run_count']}.jpeg"
            )
            tfm(processed_thresholds[ind, :, :, :]).save(save_name)
    except RuntimeError:
        pass

def plot_confusion_matrix(config, model, dataloaders):
    writer = config["writer"]
    logging.info("Plotting confusion matrix")
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, inps in tqdm(
            enumerate(dataloaders["val"]), total=len(dataloaders["val"]), leave=False
        ):
            inputs = inps["x"].to(config["device"], non_blocking=True)
            labels = inps["y"].to(config["device"], non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    # save to tensorboard
    fig = plt.figure()
    plt.imshow(cm)
    plt.show()
    writer.add_figure("Confusion Matrix", fig, config["global_run_count"])

    classreport = classification_report(y_true, y_pred, target_names=config["label_map"].values())
    writer.add_text("Classification Report", classreport, config["global_run_count"])

def one_epoch(config, pbar, model, optimizer, dataloaders, target_layers, scheduler = None):
    writer = config["writer"]
    config["global_run_count"] += 1

    input_wrong = []
    label_wrong = []

    criterion = nn.CrossEntropyLoss()

    for phase in ["train", "val"]:
        # logging.info(f"[INFO] Phase = {phase}")
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        running_presicion = 0
        running_recall = 0

        scaler = torch.cuda.amp.GradScaler()
        for i, inps in tqdm(
                enumerate(dataloaders[phase]), total=len(dataloaders[phase]), leave=False
            ):

            inputs = inps["x"].to(config["device"], non_blocking=True)
            labels = inps["y"].to(config["device"], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # if "vit" not in config["model"]: 
                # Enable fp16 for all models except ViT
            with torch.set_grad_enabled(phase == "train"):
                with torch.cuda.amp.autocast():
                    if phase == "train":
                        outputs = model(inputs)

                    else:
                        with torch.no_grad():
                            outputs = model(inputs)
                    _, preds = torch.max(outputs.data.detach(), 1)
                    loss = criterion(outputs, labels)
                    # precision, recall = precision_score(labels, outputs.data.detach(), average="macro"), recall_score(labels, outputs.data.detach(), average="macro")
                
                running_loss += loss.item()
                running_corrects += (preds == labels).sum().item()
                # running_presicion += precision
                # running_recall += recall

                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            
            if config["proxy_step"] == True and phase == "train":
                wrong_indices = (labels != preds).nonzero()
                input_wrong.extend(inputs[wrong_indices].detach().cpu())
                label_wrong.extend(labels[wrong_indices].detach().cpu())
            
                
        if config["proxy_step"] == True and phase == 'train':
            cam = dict_gradient_method[config["gradient_method"]](
                model=model, target_layers=target_layers, use_cuda=True
            )
            proxy_callback(config, input_wrong, label_wrong, cam)
            writer.add_scalar("proxy_step", True, config["global_run_count"])
        else:
            # pass
            writer.add_scalar("proxy_step", False, config["global_run_count"])
        
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = 100. * running_corrects/len(dataloaders[phase].dataset)
        # epoch_precision = running_presicion / len(dataloaders[phase].dataset)
        # epoch_recall = running_recall / len(dataloaders[phase].dataset)
        pbar.set_postfix(
                {
                    "Phase": "running",
                    "Loss": epoch_loss
                }
            )
        if phase == "train":
            writer.add_scalar(
                "Loss/Train", epoch_loss, config["global_run_count"]
            )
            writer.add_scalar(
                "Acc/Train", epoch_acc, config["global_run_count"]
            )
            # writer.add_scalar(
            #     "Precision/Train", epoch_precision, config["global_run_count"]
            # )
            # writer.add_scalar(
            #     "Recall/Train", epoch_recall, config["global_run_count"]
            # )

        if phase == "val":
            writer.add_scalar(
                "Loss/Val", epoch_loss, config["global_run_count"]
            )
            writer.add_scalar(
                "Acc/Val", epoch_acc, config["global_run_count"]
            )

            # writer.add_scalar(
            #     "Precision/Val", epoch_precision, config["global_run_count"]
            # )
            # writer.add_scalar(
            #     "Recall/Val", epoch_recall, config["global_run_count"]
            # )
            # save confusion matrix
            # writer.add_figure(
            #     "Confusion matrix",
            #     plot_confusion_matrix(
            #         config = config, model = model, dataloaders = dataloaders
            #     ),
            #     config["global_run_count"],
            # )

            plot_confusion_matrix(
                config = config, model = model, dataloaders = dataloaders
            )

            save_path = Path(config["fname_start"]) / "checkpoint"
            config["save_path"] = save_path
            if config["global_run_count"] % config["log_every"] == 0:
                torch.save(
                    {
                        # "config": config,
                        "epoch": config["global_run_count"],
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": epoch_loss,
                    },
                    save_path,
                )

            config["final_acc"] = epoch_acc
        writer.add_scalar(
            "global_run_count", config["global_run_count"], config["global_run_count"]
        )



# %%
# TODO Better transfer learning params. more trainable layers

# @profile
def setup_train_round(
    config,model = None, num_epochs=1, load_check=None
):

    # logger = TensorBoardLogger(config["fname_start"], name=config["fname_start"])
    config["writer"] = SummaryWriter(
        log_dir=config["fname_start"], comment=config["fname_start"]
    )

    train, val = create_folds(config)
    image_datasets, dataloaders, dataset_sizes = create_dls(train, val, config)
    # class_names = image_datasets["train"].classes
    config["num_classes"] = len(config["label_map"].keys())
    config["dataset_sizes"] = dataset_sizes

    config["criterion"] = nn.CrossEntropyLoss()

    model = choose_network(config)

    # model = torch.compile(model, mode= "reduce-overhead")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    if load_check == True:
        chk = torch.load(config["save_path"], map_location = config["device"])
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
   
    target_layers = find_target_layer(config, model)

    pbar = tqdm(
        range(config["global_run_count"], config["global_run_count"] + num_epochs),
        total=num_epochs,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer, 2e-3, epochs=num_epochs, steps_per_epoch=len(dataloaders["train"])
    )

    for _ in pbar:
        one_epoch(config, pbar, model, optimizer,dataloaders, target_layers, scheduler)
    for key, value in config.items():
        config["writer"].add_text(key, str(value))

    config["writer"].close()

    print("GPU freed")

def train_proxy_steps( config):

    fname_start = f'{config["main_run_dir"]}{config["experiment_name"]}_{datetime.now().strftime("%d%m%Y_%H%M%S")}'
    config["fname_start"] = fname_start


    config["ds_path"] = config["dataset_info"][config["ds_name"]]["path"]
    config["name_fn"] = config["dataset_info"][config["ds_name"]]["name_fn"]

    config["batch_size"] = set_batch_size_dict[config["model"]]

    clear_proxy_images(config=config)
    config["global_run_count"] = 0
   

    for i, step in enumerate(config["proxy_steps"]):
        load_check = i > 0
        if step == "p":
            config["load_proxy_data"] = True
            config["proxy_step"] = True
            setup_train_round(
                config=config,
                # model = model,
                num_epochs=1,
                load_check=load_check,
            )
        else:
            config["load_proxy_data"] = False
            config["proxy_step"] = False
            setup_train_round(
                config=config,
                # model=model,
                num_epochs=step,
                load_check=load_check,
            )

        if config["clear_every_step"] == True:
            clear_proxy_images(config=config)  # Clean directory

    if config["clear_every_step"] == False:
        clear_proxy_images(config=config)  # Clean directory
    return config["final_acc"]

def train_single_round(config):
    return train_proxy_steps(config)

