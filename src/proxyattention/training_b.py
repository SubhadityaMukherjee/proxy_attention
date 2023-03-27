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
from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter

from torchvision import transforms
from tqdm import tqdm

from .data_utils import clear_proxy_images, create_dls, create_folds, get_parent_name
from .meta_utils import get_files, save_pickle, read_pickle
import time

# import gc
import copy

# from pytorch_memlab import LineProfiler, profile

import numpy as np

import torch.multiprocessing as mp

cudnn.benchmark = True
logging.basicConfig(level=logging.ERROR)
# %%
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

    grads = cam(input_tensor=input_wrong.detach().to(config["device"]), targets=None)
    # grads = cam(input_tensor=normalized_inps.detach().to(config["device"]), targets=None)
    grads = torch.Tensor(grads).to(config["device"]).unsqueeze(1).expand(-1, 3, -1, -1)

    normalized_inps = inv_normalize(input_wrong.float())
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


def proxy_callback(config, input_wrong_full, label_wrong_full, model):
    writer = config["writer"]
    logging.info("Performing Proxy step")

    # TODO Save Classwise fraction
    # chosen_inds = int(
    #     np.ceil(config["change_subset_attention"] * len(label_wrong_full))
    # )
    # chosen_inds = 100
    chosen_inds = len(label_wrong_full)
    chosen_inds = 50 if chosen_inds > 50 else chosen_inds
    # TODO some sort of decay?
    # TODO Remove min and batchify

    writer.add_scalar("Number_Chosen", chosen_inds, config["global_run_count"])
    print(f"{chosen_inds} images chosen to run proxy on")

    input_wrong_full = input_wrong_full[:chosen_inds]
    label_wrong_full = label_wrong_full[:chosen_inds]

    processed_labels = []
    processed_thresholds = []
    logging.info("[INFO] Started proxy batches")
    # model.eval()
    target_layers = find_target_layer(config, model.float())
    cam = dict_gradient_method[config["gradient_method"]](
                model=model.float(), target_layers=target_layers, use_cuda=True, 
            )

    # with torch.cuda.amp.autocast(dtype=torch.float32):
    for i in tqdm(
        range(0, len(input_wrong_full), config["batch_size"]), desc="Running proxy"
    ):

        
        input_wrong = input_wrong_full[i : i + config["batch_size"]]
        label_wrong = label_wrong_full[i : i + config["batch_size"]]

        # try:
        input_wrong = torch.squeeze(torch.stack(input_wrong, dim=0))
        label_wrong = torch.squeeze(torch.stack(label_wrong, dim=0))
        # except:
        #     input_wrong = torch.squeeze(input_wrong)
        #     label_wrong = torch.squeeze(label_wrong)

        if i == 0:
            writer.add_images(
                "original_images",
                inv_normalize(input_wrong),
                # input_wrong,
                config["global_run_count"],
            )

        thresholded_ims = proxy_one_batch(config, input_wrong, cam)
        processed_thresholds.extend(thresholded_ims)
        processed_labels.extend(label_wrong)

        logging.info("[INFO] Ran proxy step")
        if i == 0:
            writer.add_images(
                "converted_proxy",
                thresholded_ims,
                config["global_run_count"],
            )

        logging.info("[INFO] Saving the images")
    # model.train()
    processed_thresholds = torch.stack(processed_thresholds, dim=0).detach().cpu()
    batch_size = processed_thresholds.size(0)

    for ind in tqdm(range(batch_size), total=batch_size, desc="Saving images"):
        label = config["label_map"][processed_labels[ind].item()]
        save_name = (
            config["ds_path"] / label / f"proxy-{ind}-{config['global_run_count']}.jpeg"
        )
        tfm(processed_thresholds[ind, :, :, :]).save(save_name)



def one_epoch(config, pbar, model, optimizer, dataloaders, scheduler):
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

        scaler = torch.cuda.amp.GradScaler(enabled=False)
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

                    optimizer.zero_grad(set_to_none=True)
                    running_loss += loss.item()
                    running_corrects += (preds == labels).sum().item()

                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                if config["proxy_step"] == True and phase == "train":
                    wrong_indices = (labels != preds).nonzero()
                    input_wrong.extend(inputs[wrong_indices])
                    label_wrong.extend(labels[wrong_indices])
        with torch.set_grad_enabled(phase == "train"):
            with torch.cuda.amp.autocast():
                if config["proxy_step"] == True and phase == "train":
                    # proxy_callback(config, input_wrong, label_wrong, model)
                    writer.add_scalar("proxy_step", True, config["global_run_count"])
                    logging.info("Performing Proxy step")

                    # TODO Save Classwise fraction
                    # chosen_inds = int(
                    #     np.ceil(config["change_subset_attention"] * len(label_wrong_full))
                    # )
                    # chosen_inds = 100
                    chosen_inds = len(label_wrong)
                    chosen_inds = 50 if chosen_inds > 50 else chosen_inds
                    # TODO some sort of decay?
                    # TODO Remove min and batchify

                    writer.add_scalar("Number_Chosen", chosen_inds, config["global_run_count"])
                    print(f"{chosen_inds} images chosen to run proxy on")

                    input_wrong = input_wrong[:chosen_inds]
                    label_wrong = label_wrong[:chosen_inds]

                    processed_labels = []
                    processed_thresholds = []
                    logging.info("[INFO] Started proxy batches")
                    # model.eval()
                    target_layers = find_target_layer(config, model.float())
                    cam = dict_gradient_method[config["gradient_method"]](
                                model=model.float(), target_layers=target_layers, use_cuda=True, 
                            )

                    # with torch.cuda.amp.autocast(dtype=torch.float32):
                    for i in tqdm(
                        range(0, len(input_wrong), config["batch_size"]), desc="Running proxy"
                    ):

                        input_wrong = input_wrong[i : i + config["batch_size"]]
                        label_wrong = input_wrong[i : i + config["batch_size"]]

                        # try:
                        input_wrong = torch.squeeze(torch.stack(input_wrong, dim=1))
                        label_wrong = torch.squeeze(torch.stack(label_wrong, dim=1))

                        if i == 0:
                            writer.add_images(
                                "original_images",
                                inv_normalize(input_wrong),
                                # input_wrong,
                                config["global_run_count"],
                            )
                        grads = cam(input_tensor=input_wrong.detach().to(config["device"]), targets=None)
                        grads = torch.Tensor(grads).to(config["device"]).unsqueeze(1).expand(-1, 3, -1, -1)

                        normalized_inps = inv_normalize(input_wrong.float())
                        if config["pixel_replacement_method"] != "blended":
                            thresholded_ims = torch.where(
                                grads > config["proxy_threshold"],
                                dict_decide_change[config["pixel_replacement_method"]](grads),
                                normalized_inps,
                            )
                        else:
                            thresholded_ims = torch.where(
                                grads > config["proxy_threshold"],
                                (1 - config["proxy_image_weight"] * grads) * normalized_inps,
                                normalized_inps,
                            )


                        # thresholded_ims = proxy_one_batch(config, input_wrong, cam)
                        processed_thresholds.extend(thresholded_ims)
                        processed_labels.extend(label_wrong)

                        logging.info("[INFO] Ran proxy step")
                        if i == 0:
                            writer.add_images(
                                "converted_proxy",
                                thresholded_ims,
                                config["global_run_count"],
                            )

                        logging.info("[INFO] Saving the images")
                    # model.train()
                    processed_thresholds = torch.stack(processed_thresholds, dim=0).detach().cpu()
                    batch_size = processed_thresholds.size(0)

                    for ind in tqdm(range(batch_size), total=batch_size, desc="Saving images"):
                        label = config["label_map"][processed_labels[ind].item()]
                        save_name = (
                            config["ds_path"] / label / f"proxy-{ind}-{config['global_run_count']}.jpeg"
                        )
                        tfm(processed_thresholds[ind, :, :, :]).save(save_name)

    
                else:
                    writer.add_scalar("proxy_step", False, config["global_run_count"])

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = 100.0 * running_corrects / len(dataloaders[phase].dataset)
                pbar.set_postfix({"Phase": "running", "Loss": epoch_loss})
                if phase == "train":
                    writer.add_scalar("Loss/Train", epoch_loss, config["global_run_count"])
                    writer.add_scalar("Acc/Train", epoch_acc, config["global_run_count"])
                if phase == "val":
                    writer.add_scalar("Loss/Val", epoch_loss, config["global_run_count"])
                    writer.add_scalar("Acc/Val", epoch_acc, config["global_run_count"])

                    save_path = Path(config["fname_start"]) / "checkpoint"
                    config["save_path"] = save_path
                    if config["global_run_count"] % config["log_every"] == 0:
                        torch.save(
                            {
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

def one_epoch_old(config, pbar, model, optimizer, dataloaders, scheduler):
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

        scaler = torch.cuda.amp.GradScaler(enabled=False)
        for i, inps in tqdm(
            enumerate(dataloaders[phase]), total=len(dataloaders[phase]), leave=False
        ):
            inputs = inps["x"].to(config["device"], non_blocking=True)
            labels = inps["y"].to(config["device"], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(phase == "train"):
                with torch.cuda.amp.autocast(enabled=False):
                    if phase == "train":
                        outputs = model(inputs)

                    else:
                        with torch.no_grad():
                            outputs = model(inputs)
                    _, preds = torch.max(outputs.data.detach(), 1)

                    loss = criterion(outputs, labels)

                    optimizer.zero_grad(set_to_none=True)
                    running_loss += loss.item()
                    running_corrects += (preds == labels).sum().item()

            if phase == "train":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                if config["proxy_step"] == True and phase == "train":
                    wrong_indices = (labels != preds).nonzero()
                    input_wrong.extend(inputs[wrong_indices])
                    label_wrong.extend(labels[wrong_indices])
        with torch.set_grad_enabled(phase == "train"):
            with torch.cuda.amp.autocast():
                if config["proxy_step"] == True and phase == "train":
                    proxy_callback(config, input_wrong, label_wrong, model)
                    writer.add_scalar("proxy_step", True, config["global_run_count"])
                else:
                    writer.add_scalar("proxy_step", False, config["global_run_count"])

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = 100.0 * running_corrects / len(dataloaders[phase].dataset)
                pbar.set_postfix({"Phase": "running", "Loss": epoch_loss})
                if phase == "train":
                    writer.add_scalar("Loss/Train", epoch_loss, config["global_run_count"])
                    writer.add_scalar("Acc/Train", epoch_acc, config["global_run_count"])
                if phase == "val":
                    writer.add_scalar("Loss/Val", epoch_loss, config["global_run_count"])
                    writer.add_scalar("Acc/Val", epoch_acc, config["global_run_count"])

                    save_path = Path(config["fname_start"]) / "checkpoint"
                    config["save_path"] = save_path
                    if config["global_run_count"] % config["log_every"] == 0:
                        torch.save(
                            {
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


def find_target_layer(config, model):
    if config["model"] == "resnet18":
        return [model.layer4[-1].conv2]
    elif config["model"] == "resnet50":
        return [model.layer4[-1].conv2]
    elif config["model"] == "efficientnet_b0":
        return [model.conv_head]
    elif config["model"] == "FasterRCNN":
        return [model.backbone]
    elif config["model"] == "vgg16" or config["model"] == "densenet161":
        return [model.features[-3]]
    elif config["model"] == "mnasnet1_0":
        return [model.layers[-1]]
    elif config["model"] == "vit_base_patch16_224":
        return [model.norm]
    else:
        raise ValueError("Unsupported model type!")


# %%
def setup_train_round(config, model=None, num_epochs=1, load_check=None):
    config["writer"] = SummaryWriter(
        log_dir=config["fname_start"], comment=config["fname_start"]
    )

    train, val = create_folds(config)
    image_datasets, dataloaders, dataset_sizes = create_dls(train, val, config)
    config["num_classes"] = len(config["label_map"].keys())
    config["dataset_sizes"] = dataset_sizes

    config["criterion"] = nn.CrossEntropyLoss()

    model = choose_network(config)

    # model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Save config info to tensorboard

    if load_check == True:
        chk = torch.load(config["save_path"])
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])

    # target_layers = find_target_layer(config, model)
    # config["target_layers"] = target_layers

    pbar = tqdm(
        range(config["global_run_count"], config["global_run_count"] + num_epochs),
        total=num_epochs,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer, 2e-3, epochs=num_epochs, steps_per_epoch=len(dataloaders["train"])
    )

    for _ in pbar:
        one_epoch(config, pbar, model, optimizer, dataloaders, scheduler)

    for key, value in config.items():
        config["writer"].add_text(key, str(value))

    print("GPU freed")


def train_proxy_steps(config):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()

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
            config["proxy_step"] = True
            setup_train_round(
                config=config,
                # model = model,
                num_epochs=1,
                load_check=load_check,
            )
        else:
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
