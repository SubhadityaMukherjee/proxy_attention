# %%
# Imports

import copy
import datetime
import logging
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no UI backend
import pickle

import numpy as np
import ray
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
import optuna
from optuna.storages import RetryFailedTrialCallback
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from operator import attrgetter
from fastai.vision.all import *
from fastai.callback.all import GradientAccumulation, MixedPrecision

from .data_utils import clear_proxy_images, create_dls, create_folds, get_parent_name
from .meta_utils import get_files, save_pickle, read_pickle
import time

# sns.set()

cudnn.benchmark = True

# %%
set_batch_size_dict = {
    "vgg16": 32,
    "vit_base_patch16_224": 32,
    "resnet18": 64,
    "resnet50": 32,
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
    # model = timm.create_model(
    #     config["model"],
    #     pretrained=config["transfer_imagenet"],
    #     num_classes=config["num_classes"],
    # ).to(config["device"])
    # model.train()
    # return model


#%%


def proxy_one_batch(config, input_wrong):
    grads = config["cam"](input_tensor=input_wrong, targets=None)
    grads = torch.Tensor(grads).to("cuda").unsqueeze(1).expand(-1, 3, -1, -1)
    normalized_inps = inv_normalize(input_wrong)
    if config["pixel_replacement_method"] != "blended":
        return torch.where(
            grads > config["proxy_threshold"],
            dict_decide_change[config["pixel_replacement_method"]](grads),
            normalized_inps,
        )
    else:
        return torch.where(
            grads > config["proxy_threshold"],
            (1 - config["proxy_image_weight"] * grads) * normalized_inps,
            normalized_inps,
        )


def proxy_callback(config, input_wrong, label_wrong):
    logging.info("Performing Proxy step")

    # TODO Save Classwise fraction
    chosen_inds = int(np.ceil(config["change_subset_attention"] * len(label_wrong)))
    # TODO some sort of decay?
    # TODO Remove min and batchify
    chosen_inds = min(config["batch_size"], chosen_inds)

    config["writer"].add_scalar(
        "Number_Chosen", chosen_inds, config["global_run_count"]
    )
    logging.info(f"{chosen_inds} images chosen to run proxy on")

    input_wrong = input_wrong[:chosen_inds]
    label_wrong = label_wrong[:chosen_inds]

    try:
        input_wrong = torch.squeeze(torch.stack(input_wrong, dim=1))
        label_wrong = torch.squeeze(torch.stack(label_wrong, dim=1))
    except:
        input_wrong = torch.squeeze(input_wrong)
        label_wrong = torch.squeeze(label_wrong)

    config["writer"].add_images(
        "original_images",
        inv_normalize(input_wrong),
        # input_wrong,
        config["global_run_count"],
    )

    # save_pickle((cam, input_wrong, config,tfm))

    # TODO run over all the batches
    thresholded_ims = proxy_one_batch(config, input_wrong)

    logging.info("[INFO] Ran proxy step")
    config["writer"].add_images(
        "converted_proxy",
        thresholded_ims,
        config["global_run_count"],
    )

    logging.info("[INFO] Saving the images")

    for ind in tqdm(range(len(label_wrong)), total=len(label_wrong)):
        label = config["label_map"][label_wrong[ind].item()]
        save_name = (
            config["ds_path"] / label / f"proxy-{ind}-{config['global_run_count']}.png"
        )
        tfm(thresholded_ims[ind]).save(save_name)


# class CancelFitException(Exception):
#     pass


# class CancelBatchException(Exception):
#     pass


# class CancelEpochException(Exception):
#     pass


# class Callback:
#     order = 0


# def run_cbs(cbs, method_nm, learn=None):
#     for cb in sorted(cbs, key=attrgetter("order")):
#         method = getattr(cb, method_nm, None)
#         if method is not None:
#             method(learn)


class ProxyCallback(Callback):
    order = 3

    def before_fit(self):
        self.input_wrong = []
        self.label_wrong = []

    def after_epoch(self):

        if self.model.training:
            wrong_indices = (self.batch["y"] != self.preds).nonzero()
            self.input_wrong.extend(self.batch["x"][wrong_indices])
            self.label_wrong.extend(self.batch["y"][wrong_indices])

            proxy_callback(self.config, self.input_wrong, self.label_wrong)
            if self.config["proxy_step"] == True:
                self.config["writer"].add_scalar(
                    "proxy_step", True, self.config["global_run_count"]
                )
            else:
                self.config["writer"].add_scalar(
                    "proxy_step", False, self.config["global_run_count"]
                )


# class TensorBoardWrite(Callback):
#     order = 4

#     def after_fit(self):

#         if self.model.training:
#             self.config["writer"].add_scalar(
#                 "Loss/Train", self.epoch_loss, self.config["global_run_count"]
#             )
#             self.config["writer"].add_scalar(
#                 "Acc/Train", self.epoch_acc, self.config["global_run_count"]
#             )

#         if self.model.train == False:
#             self.config["writer"].add_scalar(
#                 "Loss/Val", self.epoch_loss, self.config["global_run_count"]
#             )
#             self.config["writer"].add_scalar(
#                 "Acc/Val", self.epoch_acc, self.config["global_run_count"]
#             )

#             save_path = Path(self.config["fname_start"]) / "checkpoint"
#             self.config["save_path"] = save_path
#             torch.save(
#                 {
#                     "config": self.config,
#                     "epoch": self.config["global_run_count"],
#                     "model_state_dict": self.model.state_dict(),
#                     "optimizer_state_dict": self.opt.state_dict(),
#                     "loss": self.epoch_loss,
#                 },
#                 save_path,
#             )

#             self.trial.report(self.epoch_acc, self.config["global_run_count"])
#             self.config["final_acc"] = self.epoch_acc


# class GradScalerCallback(Callback):
#     order = 1

#     def before_fit(self):
#         self.gradscalar = torch.cuda.amp.GradScaler()
#         self.gradient_scaling = True


# class Metrics(Callback):
#     def __init__(self):
#         self.losses,self.val_losses,self.lrs,self.moms,self.metrics,self.nb_batches = [],[],[],[],[],[]
#     def before_batch(self):
#         if self.model.train:
#             self.losses.append(self.loss)
# order = 3

# def before_fit(self):
#     self.running_loss = 0.0
#     self.running_corrects = 0

# # def after_batch(self):

# def after_epoch(self):
#     self.running_loss += self.loss.item() * self.batch["x"].size(0)
#     self.running_corrects += torch.sum(self.preds == self.batch["y"])
#     if self.model.training:
#         phase = "train"
#     else:
#         phase = "val"
#     self.epoch_loss = self.running_loss / self.config["dataset_sizes"][phase]
#     self.epoch_acc = (
#         self.running_corrects.double() / self.config["dataset_sizes"][phase]
#     )


# class DeviceCB(Callback):
#     def __init__(self):
#         self.device = self.config["device"]
#     def before_fit(self, learn):
#         if hasattr(learn.model, 'to'): learn.model.to(self.device)
#     def before_batch(self, learn): learn.batch = to_device(learn.batch, device=self.device)


# class Learner:
#     def __init__(self, config, trial, model, dls, cbs, opt_func):
#         self.config = config
#         self.model = model
#         self.dls = dls
#         self.config = config
#         self.loss_func = self.config["criterion"]
#         self.cbs = cbs
#         self.opt_func = opt_func
#         self.lr = self.config["lr"]
#         self.device = self.config["device"]

#     def one_batch(self):
#         self.batch["x"] = self.batch["x"].to(self.config["device"], non_blocking=True)
#         self.batch["y"] = self.batch["y"].to(self.config["device"], non_blocking=True)
#         self.preds = self.model(self.batch["x"])
#         self.loss = self.loss_func(self.preds, self.batch["y"])

#         if self.model.training:
#             if self.gradient_scaling == True:
#                 self.gradscalar.scale(self.loss).backward()
#                 self.gradscalar.step(self.opt)
#                 self.gradscalar.update()
#             else:
#                 self.loss.backward()
#                 self.opt.step()
#             self.opt.zero_grad()

#     def one_epoch(self, train):
#         self.train = train
#         self.model.train(train)
#         self.dl = self.dls["train"] if train else self.dls["val"]
#         try:
#             self.callback("before_epoch")
#             self.opt.zero_grad(set_to_none=True)
#             for self.iter, self.batch in enumerate(self.dl):
#                 try:
#                     self.callback("before_batch")
#                     self.one_batch()
#                     self.callback("after_batch")
#                 except CancelBatchException:
#                     pass
#             self.callback("after_epoch")
#         except CancelEpochException:
#             pass

#     def fit(self, n_epochs):
#         self.n_epochs = n_epochs
#         self.epochs = n_epochs
#         self.opt = self.opt_func(self.model.parameters(), self.lr)
#         try:
#             self.callback("before_fit")
#             for self.epoch in self.epochs:
#                 self.one_epoch(True)
#                 self.one_epoch(False)
#                 self.config["global_run_count"] += 1
#             self.callback("after_fit")
#         except CancelFitException:
#             pass

#     def callback(self, method_nm):
#         run_cbs(self.cbs, method_nm, self)


# %%
def one_epoch(config, trial, pbar, model, optimizer, dataloaders):
    config["global_run_count"] += 1

    input_wrong = []
    label_wrong = []

    for phase in ["train", "val"]:
        # logging.info(f"[INFO] Phase = {phase}")
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
                    loss = config["criterion"](outputs, labels)

                    if config["proxy_step"] == True and phase == "train":
                        # logging.info("[INFO] : Proxy")
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
                    "Loss": running_loss / config["dataset_sizes"][phase],
                    # 'Acc' : running_corrects.double() / dataset_sizes[phase],
                }
            )

        if phase == "train":
            config["scheduler"].step()

        if config["proxy_step"] == True:
            proxy_callback(config, input_wrong, label_wrong)
            config["writer"].add_scalar("proxy_step", True, config["global_run_count"])
        else:
            config["writer"].add_scalar("proxy_step", False, config["global_run_count"])

        epoch_loss = running_loss / config["dataset_sizes"][phase]
        epoch_acc = running_corrects.double() / config["dataset_sizes"][phase]

        pbar.set_postfix({"Phase": phase, "Loss": epoch_loss, "Acc": epoch_acc})

        # TODO Add more loss functions
        # TODO Classwise accuracy

        if phase == "train":
            config["writer"].add_scalar(
                "Loss/Train", epoch_loss, config["global_run_count"]
            )
            config["writer"].add_scalar(
                "Acc/Train", epoch_acc, config["global_run_count"]
            )
        if phase == "val":
            config["writer"].add_scalar(
                "Loss/Val", epoch_loss, config["global_run_count"]
            )
            config["writer"].add_scalar(
                "Acc/Val", epoch_acc, config["global_run_count"]
            )

            save_path = Path(config["fname_start"]) / "checkpoint"
            config["save_path"] = save_path
            torch.save(
                {
                    "config": config,
                    "epoch": config["global_run_count"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                save_path,
            )

            trial.report(epoch_acc, config["global_run_count"])
            config["final_acc"] = epoch_acc

        # deep copy the model
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            # best_model_wts = copy.deepcopy(model.state_dict())


def find_target_layer(config, model):
    if config["model"] == "resnet18":
        return [model.layer4[-1].conv2]
    elif config["model"] == "resnet50":
        return [model.layer4[-1].conv2]
    elif config["model"] == "FasterRCNN":
        return model.backbone
    elif config["model"] == "vgg16" or config["model"] == "densenet161":
        return [model.features[-3]]
    elif config["model"] == "mnasnet1_0":
        return model.layers[-1]
        # elif config["model"] == "vit_small_patch32_224":
        return [model.norm]
    elif config["model"] == "vit_base_patch16_224":
        return [model.norm]
        # target_layers = model.layers[-1].blocks[-1].norm1
    else:
        raise ValueError("Unsupported model type!")


def train_model(
    trial,
    model,
    optimizer,
    dataloaders,
    num_epochs=25,
    config=None,
    load_check=False,
    callbacks=None,
):
    config["writer"] = SummaryWriter(
        log_dir=config["fname_start"], comment=config["fname_start"]
    )

    # Save config info to tensorboard
    for key, value in config.items():
        config["writer"].add_text(key, str(value))
    since = time.time()

    if load_check == True:
        chk = torch.load(config["save_path"])
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    target_layers = find_target_layer(config, model)
    config["cam"] = dict_gradient_method[config["gradient_method"]](
        model=model, target_layers=target_layers, use_cuda=True
    )

    # pbar = tqdm(
    #     range(config["global_run_count"], config["global_run_count"] + num_epochs),
    #     total=num_epochs,
    # )
    # for _ in pbar: one_epoch(config,trial, pbar, model, optimizer,dataloaders)
    # learn = Learner(
    #     config=config,
    #     trial=trial,
    #     model=model,
    #     dls=dataloaders,
    #     cbs=callbacks,
    #     opt_func=optimizer,
    # )

    # for _ in pbar:
    #     learn.fit(pbar)
    # learn = Lea
    learn = vision_learner(
        dls=dataloaders,
        arch=config["model"],
        n_out=config["num_classes"],
        metrics=error_rate,
    )
    learn.fit(
        start_epoch=config["global_run_count"],
        n_epoch=config["global_run_count"] + num_epochs,
    )

    time_elapsed = time.time() - since
    logging.info(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    logging.info(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    return learn.model


# %%
# TODO Better transfer learning params. more trainable layers
def setup_train_round(
    trial, config, proxy_step=False, num_epochs=1, load_check=None, callbacks=None
):
    # Data part
    train, val = create_folds(config)
    image_datasets, dataloaders, dataset_sizes = create_dls(train, val, config)
    class_names = image_datasets["train"].classes
    config["num_classes"] = len(config["label_map"].keys())
    config["dataset_sizes"] = dataset_sizes

    model_ft = choose_network(config)
    config["criterion"] = nn.CrossEntropyLoss()

    # Check this as well
    if torch.cuda.device_count() > 1:
        print("Multi GPU : ", torch.cuda.device_count(), "GPUs")
        model_ft = nn.DataParallel(model_ft)

    # Observe that all parameters are being optimized
    # TODO Fix this for tranasfer learning . Reduce rate
    # if config["transfer_imagenet"] == True:
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-5)
    # else:
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-4)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)
    config["lr"] = 1e-3
    optimizer_ft = optim.Adam

    callbacks = [MixedPrecision]

    dls = ImageDataLoaders.from_name_func(
        config["ds_path"],
        get_image_files(config["ds_path"]),
        valid_pct=0.2,
        label_func=parent_label,
        item_tfms=Resize(224),
    )

    learn = vision_learner(
        dls,
        config["model"],
        metrics=error_rate,
        n_out=config["num_classes"],
        pretrained=config["transfer_imagenet"],
        cbs=callbacks
    )
    learn.fine_tune(1)


def train_proxy_steps(trial, config):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    # tune.utils.wait_for_gpu(target_util = 0.03)
    # time.sleep(60)
    config["change_subset_attention"] = trial.suggest_float(
        "change_subset_attention", 0.1, 1.0
    )
    config["proxy_image_weight"] = trial.suggest_float("proxy_image_weight", 0.1, 1.0)
    config["proxy_threshold"] = trial.suggest_float("proxy_threshold", 0.1, 1.0)
    config["model"] = trial.suggest_categorical(
        "model", ["resnet18", "vgg16", "resnet50", "vit_base_patch16_224"]
    )
    config["gradient_method"] = trial.suggest_categorical(
        "gradient_method", ["gradcamplusplus", "gradcam", "eigencam"]
    )
    config["ds_name"] = trial.suggest_categorical(
        "ds_name", ["asl", "imagenette", "caltech256"]
    )

    fname_start = f'{config["main_run_dir"]}{config["experiment_name"]}_{datetime.now().strftime("%d%m%Y_%H:%M:%S")}'
    config["fname_start"] = fname_start

    # fname_start = f'{config["main_run_dir"]}{config["ds_name"]}_{config["experiment_name"]}+{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}_ps-{str(config["proxy_steps"])}_gradient-{str(config["gradient_method"])}_px-{str(config["pixel_replacement_method"])}-subs-{str(config["change_subset_attention"])}_pt-{str(config["proxy_threshold"])}_cs-{str(config["clear_every_step"])}'

    config["ds_path"] = config["dataset_info"][config["ds_name"]]["path"]
    config["name_fn"] = config["dataset_info"][config["ds_name"]]["name_fn"]

    config["batch_size"] = set_batch_size_dict[config["model"]]

    clear_proxy_images(config=config)
    # config["fname_start"] = fname_start
    config["global_run_count"] = 0

    for i, step in enumerate(config["proxy_steps"]):
        load_check = i > 0
        if step == "p":
            # config["load_proxy_data"] = True
            setup_train_round(
                trial,
                config=config,
                proxy_step=True,
                num_epochs=1,
                load_check=load_check,
            )
        else:
            # config["load_proxy_data"] = False
            setup_train_round(
                trial,
                config=config,
                proxy_step=False,
                num_epochs=step,
                load_check=load_check,
            )

        if config["clear_every_step"] == True:
            clear_proxy_images(config=config)  # Clean directory

    if config["clear_every_step"] == False:
        clear_proxy_images(config=config)  # Clean directory

    return config["final_acc"]


def train_single_round(trial, config):
    return train_proxy_steps(trial, config)
