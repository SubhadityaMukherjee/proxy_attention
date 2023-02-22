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
from captum.attr import DeepLift, IntegratedGradients, NoiseTunnel, Saliency, GuidedGradCam
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

from .data_utils import create_folds, create_dls, clear_proxy_images
from .meta_utils import *
import pickle

sns.set()

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
cudnn.benchmark = True

# %%


def reset_params(model):
    for param in model.parameters():
        param.requires_grad = False


def choose_network(config):
    # vit_tiny_patch16_224.augreg_in21k_ft_in1k
    if config["model"]== "vision_transformer":
        config["model"]= "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    # Define the number of classes
    model = timm.create_model(
        config["model"], pretrained=config["transfer_imagenet"], num_classes=config["num_classes"]).to(config["device"])
    model.train()
    return model


# %%
# TODO Proxy attention tabular support

def decide_pixel_replacement(original_image, method="mean"):
    if method == "mean":
        val =  original_image.mean()
    elif method == "max":
        val =  original_image.max()
    elif method == "min":
        val =  original_image.min()
    elif method == "black":
        val =  0.0
    elif method == "white":
        val = 255.0
    elif method == "half":
        val = 127.5
   
    return val

def permute_and_detach(obj):
    return obj.permute(1, 2, 0).cpu().detach()


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
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.485/0.229, -0.485/0.229],
        std=[1/0.229, 1/0.229, 1/0.229]
    )
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    pbar = tqdm(range(num_epochs), total=num_epochs)
    for epoch in pbar:
        config["global_run_count"] += 1
        
        # Each epoch has a training and validation phase
        input_wrong = []
        label_wrong = []
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
                        if proxy_step == True and phase == "train":
                            print("[INFO] : Proxy")
                            logging.info("Proxy")
                            wrong_indices = (labels != preds).nonzero()
                            # input_wrong = input_wrong.stack(inputs[wrong_indices])
                            input_wrong.extend(inputs[wrong_indices])
                            label_wrong.extend(labels[wrong_indices])

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

            if proxy_step == True:
                print("Performing Proxy step")
                # TODO Save Classwise fraction
                frac_choose = 0.25
                chosen_inds = int(np.ceil(frac_choose * len(label_wrong)))
                #TODO some sort of decay?
                #TODO Conver to batches to run over more
                chosen_inds = min(50, chosen_inds)

                writer.add_scalar("Number_Chosen", chosen_inds, config["global_run_count"])
                print(f"{chosen_inds} images chosen to run proxy on")

                print(len(input_wrong) , len(label_wrong))
                input_wrong = input_wrong[:chosen_inds]
                try:
                    input_wrong = torch.squeeze(torch.stack(input_wrong, dim=0))
                except:
                    input_wrong = torch.squeeze(input_wrong)
                
                writer.add_images('original_images', inv_normalize(input_wrong), config["global_run_count"])
                label_wrong = label_wrong[:chosen_inds]
                try:
                    label_wrong = torch.squeeze(torch.stack(label_wrong, dim=1))
                except:
                    label_wrong = torch.squeeze(label_wrong)
                
                if config["gradient_method"] == "saliency":
                    saliency = Saliency(model)
                elif config["gradient_method"] == "guidedgradcam":
                    if config["model"] == "resnet18":
                        chosen_layer = model.layer3[-1].conv2
                    saliency = GuidedGradCam(model, chosen_layer)

                # TODO Other methods
                print(input_wrong.size() , label_wrong.size())
                grads = saliency.attribute(
                    input_wrong, label_wrong
                )
                grads = np.transpose(
                    grads.squeeze().cpu().detach().numpy(), (0, 2, 3, 1)
                )

                print("Calculating permutes and sending to CPU")
                # TODO replace these with direct array operations?
                # original_images = [
                # permute_and_detach(ind)
                # for ind in tqdm(input_wrong, total=len(input_wrong))
                # ]
                print(input_wrong.size())
                original_images = [
                    ind.permute(1, 2, 0).cpu().detach() for ind in input_wrong
                ]

                print("Calculating pixel replacement method")
                # pixel_replacement = [
                #     decide_pixel_replacement(x, config["pixel_replacement_method"])
                #     for x in tqdm(original_images, total=len(original_images))
                # ]

                print("Calculating gradient thresholds")
                # grad_thresholds = [
                #     calc_grad_threshold(grad) for grad in tqdm(grads, total=len(grads))
                # ]

                for ind in tqdm(range(len(label_wrong)), total=len(label_wrong)):
                    # TODO Split these into individual comprehensions for speed
                    # TODO Check if % of image is gone or not
                    original_images[ind][
                        grads[ind] > config["proxy_threshold"]
                    ] = decide_pixel_replacement(
                        original_image=original_images[ind],
                        method=config["pixel_replacement_method"],
                    )

                # TODO : Dont save this everytime I guess??
                orig2 = torch.Tensor(np.stack(original_images)).permute(0, 3,1,2)

                orig2 = inv_normalize(orig2)
                writer.add_images('converted_proxy', orig2, config["global_run_count"],  dataformats='NCHW')

                print("Saving the images")
                cm = plt.get_cmap("viridis")

                with open("/mnt/e/CODE/Github/improving_robotics_datasets/src/proxyattention/pickler.pkl", "wb") as f:
                    pickle.dump((model, saliency, grads, input_wrong, label_wrong, original_images), f)

                # TODO Prune least important weights/filters? Pruning by explaining
                for ind in tqdm(range(len(label_wrong)), total=len(label_wrong)):
                    print(ind, original_images[ind])
                    plt.imshow(np.uint8(original_images[ind]))
                    plt.axis("off")
                    plt.gca().set_axis_off()
                    plt.margins(x=0)
                    plt.autoscale(False)
                    label = config["label_map"][label_wrong[ind].item()]
                    save_name = config["ds_path"] / label / f"proxy-{ind}-{config['global_run_count']}.png"
                    plt.savefig(save_name, bbox_inches="tight", pad_inches=0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            pbar.set_postfix({"Phase": phase, "Loss": epoch_loss, "Acc": epoch_acc})

            # TODO Add more loss functions
            # TODO Classwise accuracy
            if proxy_step == True:
                writer.add_scalar("proxy_step", True, config["global_run_count"])
            else:
                writer.add_scalar("proxy_step", False, config["global_run_count"])

            if phase == "train":
                writer.add_scalar("Loss/Train", epoch_loss, config["global_run_count"])
                writer.add_scalar("Acc/Train", epoch_acc, config["global_run_count"])
            if phase == "val":
                writer.add_scalar("Loss/Val", epoch_loss, config["global_run_count"])
                writer.add_scalar("Acc/Val", epoch_acc, config["global_run_count"])
                with tune.checkpoint_dir(config["global_run_count"]) as checkpoint_dir:
                    save_path = Path(config["fname_start"] + str(config["global_run_count"])) / "checkpoint"
                    torch.save((model.state_dict(), optimizer.state_dict()), save_path)

                tune.report(loss=epoch_loss, accuracy=epoch_acc)

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# %%
def setup_train_round(config, proxy_step=False, num_epochs=1):
    # Data part
    train, val = create_folds(config)
    image_datasets, dataloaders, dataset_sizes = create_dls(
        train, val, config
    )
    class_names = image_datasets["train"].classes
    config["num_classes"] = len(config["label_map"].keys())

    model_ft = choose_network(config)
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
    # config["global_run_count"] = 0
    for step in config["proxy_steps"]:
        if step == "p":
            setup_train_round(config=config, proxy_step=True, num_epochs=1)
        else:
            setup_train_round(config=config, proxy_step=False, num_epochs=step)
        
        clear_proxy_images(config=config) # Clean directory
        # config["global_run_count"] += 1

def hyperparam_tune(config):
    ray.init(num_gpus=1, num_cpus=12)
    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=30, grace_period=1, reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        train_proxy_steps,
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
        max_failures=100,
        num_samples=50,
        resources_per_trial={
            "gpu": 1,
            "cpu": 8,
        },
        local_dir=config["fname_start"],
    )

    df_res = result.get_dataframe()
    df_res.to_csv(Path(config["fname_start"]+str(config["global_run_count"])) / "result_log.csv")
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
        )
    )

    print(result)

