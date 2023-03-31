# %%
# Imports

import copy
from datetime import datetime
import logging
import os
import time
from pathlib import Path

import matplotlib

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

from .data_utils import clear_proxy_images, create_dls, create_folds, get_parent_name
from .meta_utils import get_files, save_pickle, read_pickle
import time
import gc
import copy
import argparse as ap
import ast
from operator import itemgetter
from collections import defaultdict

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

# from pympler import tracker

# sns.set()

cudnn.benchmark = True
logging.basicConfig(level=logging.ERROR)
# %%
import sys

# %%
set_batch_size_dict = {
    "vgg16": 16,
    "vit_base_patch16_224": 32,
    "resnet18": 32,
    "resnet50": 32,
    "efficientnet_b0": 32,
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
    ).to(config["device"], non_blocking=True)
    model.train()
    return model


# %%


def proxy_one_batch(config, input_wrong, cam):
    # Compute gradients using CAM
    grads = cam(input_tensor=input_wrong.to(config["device"]), targets=None)
    # Convert to Tensor and expand its dimensions
    grads = (
        torch.as_tensor(grads, device=config["device"])
        .unsqueeze(1)
        .expand(-1, 3, -1, -1)
    )
    # Normalize the input
    normalized_inps = inv_normalize(input_wrong)

    # Choose pixel replacement method based on 'config'
    if config["pixel_replacement_method"] != "blended":
        output = torch.where(
            grads > config["proxy_threshold"],
            dict_decide_change[config["pixel_replacement_method"]](grads),
            normalized_inps,
        )
    else:
        output = torch.where(
            grads > config["proxy_threshold"],
            (1 - config["proxy_image_weight"] * grads) * normalized_inps,
            normalized_inps,
        )

    return output


def proxy_callback(config, input_wrong_full, label_wrong_full, cam):
    # Set up logger and writer
    logger = logging.getLogger(__name__)
    writer = config["writer"]

    logger.info("Performing Proxy step")
    chosen_inds = int(
        np.ceil(config["change_subset_attention"] * len(label_wrong_full))
    )
    writer.add_scalar("Number_Chosen", chosen_inds, config["global_run_count"])

    # Process only a subset of inputs and labels specified by 'chosen_inds'
    input_wrong_full = input_wrong_full[:chosen_inds]
    label_wrong_full = label_wrong_full[:chosen_inds]

    processed_labels = []
    processed_thresholds = []
    batch_size = config["batch_size"]
    logger.info("[INFO] Started proxy batches")

    for i in tqdm(range(0, len(input_wrong_full), batch_size), desc="Running proxy"):
        try:
            input_wrong = input_wrong_full[i : i + batch_size]
            label_wrong = label_wrong_full[i : i + batch_size]

            # Stack tensors if needed
            if isinstance(input_wrong[0], torch.Tensor):
                input_wrong = torch.stack(input_wrong)
            if isinstance(label_wrong[0], torch.Tensor):
                label_wrong = torch.stack(label_wrong)

            # Run proxy_one_batch() to get thresholded images
            if i == 0:
                writer.add_images(
                    "original_images",
                    inv_normalize(input_wrong),
                    config["global_run_count"],
                )
            thresholded_ims = proxy_one_batch(config, input_wrong, cam)
            processed_thresholds.append(thresholded_ims)
            processed_labels.append(label_wrong)

            logger.info("[INFO] Ran proxy step")
            if i == 0:
                writer.add_images(
                    "converted_proxy",
                    thresholded_ims,
                    config["global_run_count"],
                )

            logger.info("[INFO] Saving the images")
        except ValueError:
            pass

    # Concatenate all thresholded images and corresponding labels
    processed_thresholds = torch.cat(processed_thresholds, dim=0).detach().cpu()
    processed_labels = torch.cat(processed_labels, dim=0)

    # Save each thresholded image to disk
    label_map = config["label_map"]
    ds_path = config["ds_path"]
    for ind in tqdm(
        range(len(processed_labels)), total=len(processed_labels), desc="Saving images"
    ):
        label = label_map[processed_labels[ind].item()]
        save_name = ds_path / label / f"proxy-{ind}-{config['global_run_count']}.jpeg"
        tfm(processed_thresholds[ind]).save(save_name)


# def proxy_callback(config, input_wrong_full, label_wrong_full, cam):
#     writer = config["writer"]
#     logging.info("Performing Proxy step")

#     # TODO Save Classwise fraction
#     chosen_inds = int(np.ceil(config["change_subset_attention"] * len(label_wrong_full)))
#     writer.add_scalar(
#         "Number_Chosen", chosen_inds, config["global_run_count"]
#     )

#     input_wrong_full = input_wrong_full[:chosen_inds]
#     label_wrong_full = label_wrong_full[:chosen_inds]

#     processed_labels = []
#     processed_thresholds = []
#     logging.info("[INFO] Started proxy batches")

#     for i in tqdm(range(0, len(input_wrong_full), config["batch_size"]), desc="Running proxy"):
#         try:
#             input_wrong = input_wrong_full[i:i+config["batch_size"]]
#             label_wrong = label_wrong_full[i:i+config["batch_size"]]

#             try:
#                 input_wrong = torch.squeeze(torch.stack(input_wrong, dim=1))
#                 label_wrong = torch.squeeze(torch.stack(label_wrong, dim=1))
#             except:
#                 input_wrong = torch.squeeze(input_wrong)
#                 label_wrong = torch.squeeze(label_wrong)

#             if i == 0:
#                 writer.add_images(
#                     "original_images",
#                     inv_normalize(input_wrong),
#                     # input_wrong,
#                     config["global_run_count"],
#                 )


#             # TODO run over all the batches
#             thresholded_ims = proxy_one_batch(config, input_wrong, cam)
#             processed_thresholds.extend(thresholded_ims)
#             processed_labels.extend(label_wrong)


#             logging.info("[INFO] Ran proxy step")
#             if i == 0:
#                 writer.add_images(
#                     "converted_proxy",
#                     thresholded_ims,
#                     config["global_run_count"],
#                 )

#             logging.info("[INFO] Saving the images")
#         except ValueError:
#             pass
#     processed_thresholds = torch.stack(processed_thresholds, dim = 0).detach().cpu()
#     batch_size = processed_thresholds.size(0)


#     for ind in tqdm(range(batch_size), total=batch_size, desc="Saving images"):
#         label = config["label_map"][processed_labels[ind].item()]
#         save_name = (
#             config["ds_path"] / label / f"proxy-{ind}-{config['global_run_count']}.jpeg"
#         )
#         tfm(processed_thresholds[ind, :, :, :]).save(save_name)


def one_epoch(
    config, pbar, model, optimizer, dataloaders, target_layers, scheduler=None
):
    writer = config["writer"]
    # mem = tracker.SummaryTracker()
    config["global_run_count"] += 1
    # scheduler = config["scheduler"]

    # input_wrong = torch.Tensor().to(config["device"], non_blocking = True)
    # label_wrong = torch.Tensor().to(config["device"], non_blocking = True)

    input_wrong = []
    label_wrong = []

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(dataloaders["train"]))

    criterion = nn.CrossEntropyLoss()

    for phase in ["train", "val"]:
        # logging.info(f"[INFO] Phase = {phase}")
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0

        scaler = torch.cuda.amp.GradScaler()
        for i, inps in tqdm(
            enumerate(dataloaders[phase]), total=len(dataloaders[phase]), leave=False
        ):
            inputs = inps["x"].to(config["device"], non_blocking=True)
            labels = inps["y"].to(config["device"], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(phase == "train"):
                with torch.cuda.amp.autocast():
                    if phase == "train":
                        outputs = model(inputs)

                    else:
                        with torch.no_grad():
                            outputs = model(inputs)
                    _, preds = torch.max(outputs.data.detach(), 1)
                    loss = criterion(outputs, labels)
                    class_correct = defaultdict(int)
                    class_total = defaultdict(int)

                    with torch.set_grad_enabled(phase == "train"):
                        with torch.cuda.amp.autocast():
                            if phase == "train":
                                outputs = model(inputs)
                            else:
                                with torch.no_grad():
                                    outputs = model(inputs)

                            _, preds = torch.max(outputs.data.detach(), 1)

                            # Loop over the predictions and update the number of correct and total samples for each class
                            for i in range(len(labels)):
                                label = labels[i].item()
                                class_correct[label] += preds[i] == label
                                class_total[label] += 1

                            # Compute the accuracy for each class and the average accuracy across all classes
                            accuracies = {}
                            avg_accuracy = 0.0
                            for label in class_total:
                                accuracy = (
                                    float(class_correct[label]) / class_total[label]
                                )
                                accuracies[label] = accuracy
                                avg_accuracy += accuracy
                            avg_accuracy /= len(class_total)

                running_loss += loss.item()
                running_corrects += (preds == labels).sum().item()

                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            if config["proxy_step"] == True and phase == "train":
                wrong_indices = (labels != preds).nonzero()
                input_wrong.extend(inputs[wrong_indices])
                label_wrong.extend(labels[wrong_indices])

        if config["proxy_step"] == True and phase == "train":
            cam = dict_gradient_method[config["gradient_method"]](
                model=model, target_layers=target_layers, use_cuda=True
            )
            proxy_callback(config, input_wrong, label_wrong, cam)
            writer.add_scalar("proxy_step", True, config["global_run_count"])
        else:
            # pass
            writer.add_scalar("proxy_step", False, config["global_run_count"])

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = 100.0 * running_corrects / len(dataloaders[phase].dataset)
        pbar.set_postfix({"Phase": "running", "Loss": epoch_loss})
        # if phase == "train":
        #     writer.add_scalar(
        #         "Loss/Train", epoch_loss, config["global_run_count"]
        #     )
        #     writer.add_scalar(
        #         "Acc/Train", epoch_acc, config["global_run_count"]
        #     )
        #     writer.add_scalar(
        #         "Classwise/Train", avg_accuracy, config["global_run_count"]
        #     )
        writer.add_scalar(
            f"Loss/{phase.capitalize()}", epoch_loss, config["global_run_count"]
        )
        writer.add_scalar(
            f"Acc/{phase.capitalize()}", epoch_acc, config["global_run_count"]
        )
        writer.add_scalar(
            f"ClasswiseAcc/{phase.capitalize()}",
            avg_accuracy,
            config["global_run_count"],
        )

        if phase == "val":
            # writer.add_scalar(
            #     "Loss/Val", epoch_loss, config["global_run_count"]
            # )
            # writer.add_scalar(
            #     "Acc/Val", epoch_acc, config["global_run_count"]
            # )
            # writer.add_scalar(
            #     "Classwise/Val", avg_accuracy, config["global_run_count"]
            # )

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

            # trial.report(epoch_acc, config["global_run_count"])
            config["final_acc"] = epoch_acc
        writer.add_scalar(
            "global_run_count", config["global_run_count"], config["global_run_count"]
        )


def find_target_layer(config, model):
    if config["model"] == "resnet18":
        return [model.layer4[-1].conv2]
    elif config["model"] == "resnet50":
        return [model.layer4[-1].conv2]
    elif config["model"] == "efficientnet_b0":
        return [model.conv_head]
    elif config["model"] == "FasterRCNN":
        return model.backbone
    elif config["model"] == "vgg16" or config["model"] == "densenet161":
        return [model.features[-3]]
    elif config["model"] == "mnasnet1_0":
        return model.layers[-1]
    elif config["model"] == "vit_base_patch16_224":
        return [model.norm]
    else:
        raise ValueError("Unsupported model type!")


# %%
# TODO Better transfer learning params. more trainable layers


# @profile
def setup_train_round(config, model=None, num_epochs=1, load_check=None):
    config["writer"] = SummaryWriter(
        log_dir=config["fname_start"], comment=config["fname_start"]
    )

    train, val = create_folds(config)
    image_datasets, dataloaders, dataset_sizes = create_dls(train, val, config)
    class_names = image_datasets["train"].classes
    config.update(
        {
            "num_classes": len(config["label_map"]),
            "dataset_sizes": create_dls(*create_folds(config), config),
            "criterion": nn.CrossEntropyLoss(),
        }
    )

    model = choose_network(config)

    # model = torch.compile(model, mode= "reduce-overhead")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    if load_check == True:
        chk = torch.load(config["save_path"], map_location=config["device"])
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
        one_epoch(config, pbar, model, optimizer, dataloaders, target_layers, scheduler)
    for key, value in config.items():
        config["writer"].add_text(key, str(value))

    config["writer"].close()

    # Clean up after training
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU freed")


def train_proxy_steps(config):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()

    fname_start = f'{config["main_run_dir"]}{config["experiment_name"]}_{datetime.now().strftime("%d%m%Y_%H%M%S")}'
    config.update(
        {
            "fname_start": fname_start,
            "ds_path": config["dataset_info"][config["ds_name"]]["path"],
            "name_fn": config["dataset_info"][config["ds_name"]]["name_fn"],
            "batch_size": set_batch_size_dict[config["model"]],
            "global_run_count": 0,
        }
    )

    clear_proxy_images(config=config)

    for i, step in enumerate(config["proxy_steps"]):
        config.update({"proxy_step": True}) if step == "p" else config.update(
            {"proxy_step": False}
        )
        setup_train_round(
            config=config, num_epochs=1 if step == "p" else step, load_check=i > 0
        )

        if config["clear_every_step"]:
            clear_proxy_images(config=config)

    if not config["clear_every_step"]:
        clear_proxy_images(config=config)

    return config["final_acc"]


def train_single_round(config):
    return train_proxy_steps(config)
