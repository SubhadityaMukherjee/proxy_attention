# %%
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
from typing import (Dict, Generator, Iterable, Iterator, List, Optional,
                    Sequence, Set, Tuple, Union)

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
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
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import proxyattention
sns.set()

os.environ["TORCH_HOME"] = "/mnt/e/Datasets/"
# %%
# Config
# TODO Refactor this into a JSON with options
experiment_params = {
    "experiment_name": "test_asl_starter",
    "ds_path": Path("/mnt/e/Datasets/asl/asl_alphabet_train/asl_alphabet_train"),
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
    "proxy_steps": 2,
    "proxy_threshold": 0.008,
    "pixel_replacement_method": "mean",
    "model": "resnet18",
    "proxy_steps": [2, "p", 2],
}
# config = proxyattention.configuration.Experiment(params=experiment_params)

config = {
    "experiment_name": "test_asl_starter",
    "ds_path": Path("/mnt/e/Datasets/asl/asl_alphabet_train/asl_alphabet_train"),
    "ds_name": "asl",
    "name_fn": proxyattention.data_utils.asl_name_fn,
    "image_size": 224,
    "batch_size": 128,
    "epoch_steps": [1, 2],
    "enable_proxy_attention": True,
    "change_subset_attention": tune.loguniform(0.1,0.8),
    "validation_split": 0.3,
    "shuffle_dataset": tune.choice([True, False]),
    "num_gpu": 1,
    "transfer_imagenet": False,
    "subset_images": 5000,
    "proxy_threshold": tune.loguniform(0.008, 0.01),
    "pixel_replacement_method": tune.choice(["mean", "max", "min", "black", "white"]),
    "model": "resnet18",
    # "proxy_steps": tune.choice([[1, "p", 1], [3, "p", 1], [1, 1], [3,1]]),
    "proxy_steps": tune.choice([[1, "p", 1],[1, 1]]),
}   

# Make dirs
logging.info("Directories made/checked")
os.makedirs("runs/", exist_ok=True)
# TODO unique_name
fname_start = f'runs/{config["ds_name"]}_{config["experiment_name"]}+{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}_subset-{config["subset_images"]}'

config["fname_start"]= fname_start

# TODO logging
logging.basicConfig(filename=fname_start, encoding="utf-8", level=logging.DEBUG)
logging.info(f"[INFO] : File name = {fname_start}")
print(f"[INFO] : File name = {fname_start}")

config["device"]= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%


# %%
# TODO Proxy loop function, custom schedule
# TODO Proxy attention tabular support


def decide_pixel_replacement(original_image, method="mean"):
    if method == "mean":
        return original_image.mean()
    elif method == "max":
        return original_image.max()
    elif method == "min":
        return original_image.min()
    elif method == "black":
        return 0.0
    elif method == "white":
        return 255.0


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=25,
    proxy_step=False,
    config=None,
):
    writer = SummaryWriter(log_dir=config["fname_start"], comment=config["fname_start"])
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    pbar = tqdm(range(num_epochs), total=num_epochs)
    for epoch in pbar:
        # print(f'Epoch {epoch}/{num_epochs - 1}')
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            optimizer.zero_grad(set_to_none=True)
            scaler = torch.cuda.amp.GradScaler()
            for inps in tqdm(
                dataloaders[phase], total=len(dataloaders[phase]), leave=False
            ):
                inputs = inps["x"].to(config["device"], non_blocking=True)
                labels = inps["y"].to(config["device"], non_blocking=True)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    with torch.cuda.amp.autocast():
                        if phase == "train":
                            outputs = model(inputs)
                        else:
                            with torch.no_grad():
                                outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # TODO Save images pipeline
                        # TODO Config to choose what to save
                        # TODO Proxy getting called randomly??
                        if proxy_step == True and phase == "train":
                            print("[INFO] : Proxy")
                            logging.info("Proxy")
                            wrong_indices = (labels != preds).nonzero()
                            saliency = Saliency(model)
                            # TODO Other methods
                            grads = saliency.attribute(inputs, labels)[wrong_indices]
                            grads = np.transpose(
                                grads.squeeze().cpu().detach().numpy(), (0, 2, 3, 1)
                            )

                            # TODO Save Classwise fraction
                            frac_choose = 0.25
                            chosen_inds = range(int(np.ceil(frac_choose * len(labels))))

                            original_images = [
                                inputs[ind].permute(1, 2, 0).cpu().detach()
                                for ind in chosen_inds
                            ]

                            for ind in tqdm(chosen_inds, total=len(chosen_inds)):
                                # TODO Split these into individual comprehensions for speed
                                original_images[ind][
                                    grads[ind].mean(axis=2) > config["proxy_threshold"]] = decide_pixel_replacement(
                                    original_image=original_images[ind],
                                    method=config["pixel_replacement_method"],
                                )

                            for ind in tqdm(chosen_inds, total=len(chosen_inds)):
                                plt.imshow(np.uint8(original_images[ind]))
                                plt.axis("off")
                                plt.gca().set_axis_off()
                                plt.margins(x=0)
                                plt.autoscale(False)
                                label = config["label_map"][labels[ind].item()]
                                save_name = (
                                    config["ds_path"]/ label / f"proxy-{ind}-{epoch}.png"
                                )
                                plt.savefig(
                                    save_name, bbox_inches="tight", pad_inches=0
                                )

                    # backward + optimize only if in training phase
                    if phase == "train":
                        scaler.scale(loss).backward()
                        # optimizer.step()

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix(
                    {
                        "Phase": "running",
                        "Loss": running_loss / dataset_sizes[phase],
                        # 'Acc' : running_corrects.double() / dataset_sizes[phase],
                    }
                )

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            pbar.set_postfix({"Phase": phase, "Loss": epoch_loss, "Acc": epoch_acc})

            # TODO Add more loss functions
            # TODO Classwise accuracy
            if phase == "train":
                writer.add_scalar("Loss/Train", epoch_loss, epoch)
                writer.add_scalar("Acc/Train", epoch_acc, epoch)
            if phase == "val":
                writer.add_scalar("Loss/Val", epoch_loss, epoch)
                writer.add_scalar("Acc/Val", epoch_acc, epoch)
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = Path(config["fname_start"])/"checkpoint"
                    torch.save((model.state_dict(), optimizer.state_dict()), path)

                tune.report(loss= epoch_loss, accuracy=epoch_acc)


            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            
            # TODO Save best model

        # print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    # TODO Change returns to normal
    return model


# %%


def setup_train_round(config, proxy_step=False, num_epochs=1):
    # Data part
    train, val = proxyattention.data_utils.create_folds(config)
    image_datasets, dataloaders, dataset_sizes = proxyattention.data_utils.create_dls(
        train, val, config
    )
    class_names = image_datasets["train"].classes
    config["num_classes"]= len(config["label_map"].keys())

    model_ft = proxyattention.training.choose_network(config)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    trained_model = train_model(
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=num_epochs,
        config=config,
        proxy_step=proxy_step,
    )


def train_proxy_steps(config):
    assert torch.cuda.is_available()
    for step in config["proxy_steps"]:
        if step == "p":
            setup_train_round(config=config, proxy_step=True, num_epochs=1)
        else:
            setup_train_round(config=config, proxy_step=False, num_epochs=step)

def hyperparam_tune(config):
    ray.init(num_gpus=1, num_cpus=12)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=30,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        train_proxy_steps,
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end= True,
        max_failures=100,
        num_samples=50,
        resources_per_trial={
            "gpu":1,
            "cpu":4,
        }
        )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    print(result)
    

# %%
# train_proxy_steps(config=config)
hyperparam_tune(config=config)
