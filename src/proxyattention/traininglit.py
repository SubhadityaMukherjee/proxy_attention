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

from .data_utils import clear_proxy_images, DataModule
from .meta_utils import save_pickle, read_pickle, get_files
import time

# import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import preprocessing
import random
import albumentations as A
import cv2
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torchmetrics.functional import accuracy
import timm

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
    elif config["model"] == "vit_base_patch16_224":
        return [model.norm]
    else:
        raise ValueError("Unsupported model type!")


inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)
dict_decide_change = {
    "mean": torch.mean,
    "max": torch.max,
    "min": torch.min,
    "halfmax": lambda x: torch.max(x) / 2,
}

def get_last_checkpoint(checkpoint_folder):
    "https://github.com/Lightning-AI/lightning/issues/4176"
    if os.path.exists(checkpoint_folder):
        past_experiments = sorted(
            Path(checkpoint_folder).iterdir(), key=os.path.getmtime
        )

        for experiment in past_experiments[::-1]:
            experiment_folder = os.path.join(experiment, "checkpoints")
            if os.path.exists(experiment_folder):
                checkpoints = os.listdir(experiment_folder)

                if len(checkpoints):
                    checkpoints.sort()
                    path = os.path.join(experiment_folder, checkpoints[-1])
                    return path

    return None


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.choose_network()
        self.loss_func = F.cross_entropy
        # self.save_hyperparameters()

    
    def choose_network(self):
        return timm.create_model(
            self.config["model"],
            pretrained=self.config["transfer_imagenet"],
            num_classes=self.config["num_classes"],
        )
    
    def training_step(self, batch, batch_idx):
        self.config["global_run_count"] += 1
        x, y = batch["x"], batch["y"]
        logits = self.model(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def evaluate(self, batch, stage=None):
        x, y = batch["x"], batch["y"]
        logits = self.model(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.config["num_classes"]
        )
        self.config["final_acc"] = acc
        if stage:
            self.log(f"ptl/{stage}_loss", loss, prog_bar=True)
            self.log(f"ptl/{stage}_acc", acc, prog_bar=True)

            self.logger.experiment.add_scalar(f"Loss/{stage}",
                                            loss,
                                            self.config["global_run_count"])

            self.logger.experiment.add_scalar(f"Accuracy/{stage}",
                                            acc,
                                            self.config["global_run_count"])

def setup_train_round(config, proxy_step = False, num_epochs = 1, chk = None):
    dm = DataModule(config=config)
    dm.setup()
    model = Model(config=config)
    os.makedirs(config["fname_start"], exist_ok=True)
    callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
            ModelCheckpoint(
                save_last=True,
                monitor="step",
                mode="max",
                dirpath=config["fname_start"],
                filename="full_train",
            ),
            StochasticWeightAveraging(swa_lrs=1e-2),
        ]
    trainer = Trainer(
        max_epochs=num_epochs,
        default_root_dir=config["fname_start"],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None, 
        logger=TensorBoardLogger("tb_logs", name=config["fname_start"]),
        log_every_n_steps=1,
        accumulate_grad_batches=4,  # number of batches to accumulate gradients over
        precision=16,  # enables mixed precision training with 16-bit floating point precision
        resume_from_checkpoint=chk,
        callbacks = callbacks
        )

    model.validation_step_end = None
    model.validation_epoch_end = None
    if config["chk"] is not None:
        trainer.fit(
            model, dm, dm, ckpt_path=config["chk"]
        )
        trainer.test(model, dm, ckpt_path=config["chk"])
    else:
        trainer.fit(model, dm)
        trainer.test(model, dm)


def proxy_one_batch(config, input_wrong):
    grads = config["cam"](input_tensor=input_wrong, targets=None)
    grads = torch.Tensor(grads).to(config["device"]).unsqueeze(1).expand(-1, 3, -1, -1)
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


def proxy_callback(config, all_wrong_input, all_wrong_input_label):
    logging.info("Performing Proxy step")

    # TODO Save Classwise fraction
    chosen_inds = int(
        np.ceil(config["change_subset_attention"] * len(all_wrong_input_label))
    )
    # TODO some sort of decay?
    # TODO Remove min and batchify
    # chosen_inds = min(config["batch_size"], chosen_inds)
    if chosen_inds > 2:

        config["writer"].add_scalar(
            "Number_Chosen", chosen_inds, config["global_run_count"]
        )
        logging.info(f"{chosen_inds} images chosen to run proxy on")

        input_wrong = all_wrong_input[:chosen_inds]
        label_wrong = all_wrong_input_label[:chosen_inds]

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

        # logging.info("[INFO] Ran proxy step")
        config["writer"].add_images(
            "converted_proxy",
            thresholded_ims,
            config["global_run_count"],
        )

        # logging.info("[INFO] Saving the images")
        tfm = transforms.ToPILImage()

        for ind in tqdm(range(len(input_wrong)), total=len(input_wrong)):
            label = config["vocab"][label_wrong[ind].item()]
            save_name = (
                config["ds_path"]
                / label
                / f"proxy-{ind}-{config['global_run_count']}.png"
            )
            tfm(thresholded_ims[ind]).save(save_name)


# class ProxyCallback(Callback):
#     def before_train(self):
#         self.input_wrong = []
#         self.label_wrong = []

#     def after_train(self):
#         if self.training == True:
#             wrong_indices = (
#                 self.learn.yb[0] != torch.max(self.learn.pred, 1)[1]
#             ).nonzero()

#             self.input_wrong.extend(self.learn.xb[0][wrong_indices])
#             self.label_wrong.extend(self.learn.yb[0][wrong_indices])
#             self.learn.config["writer"] = self.learn.tensor_board.writer

#             proxy_callback(self.learn.config, self.input_wrong, self.label_wrong)


# # %%
# def setup_train_round_back(config, proxy_step=False, num_epochs=1):

#     dls = create_dls(config=config)
#     # TODO Write your own dahm TensorBoardCallback AGAIN

#     callbacks = [
#         MixedPrecision,
#         ProgressCallback,
#         TensorBoardCallback(log_dir=config["fname_start"], trace_model=False),
#     ]
#     if proxy_step == True:
#         callbacks.append(ProxyCallback())

#     learn = vision_learner(
#         dls,
#         config["model"],
#         metrics=[accuracy,error_rate],
#         pretrained=config["transfer_imagenet"],
#         cbs=callbacks,
#     )

#     target_layers = find_target_layer(config, learn)
#     config["cam"] = dict_gradient_method[config["gradient_method"]](
#         model=learn.model, target_layers=target_layers, use_cuda=True
#     )
#     learn.config = config
#     # try:
#     #     learn.load(Path(config["fname_start"])/"saved_model")
#     #     # save_model(config["fname_start"]/"saved_model", learn)
#     # except:
#     #     pass
#     learn.train_iter = config["global_run_count"]
#     try:
#         learn.load(Path(config["fname_start"])/f"saved_net_{config['global_run_count']}")
#         learn.fit_one_cycle(
#             n_epoch=num_epochs,
#             # start_epoch=config["global_run_count"],
#             cbs=[
#                 SaveModelCallback(
#                     every_epoch = True, monitor="accuracy", fname= Path(config["fname_start"])/"saved_net"
#                 )
#             ],
#         )

#     except:
#         learn.fit_one_cycle(
#             n_epoch=num_epochs,
#             cbs=[
#                 SaveModelCallback(
#                     every_epoch = True, monitor="accuracy",fname= Path(config["fname_start"])/"saved_net"
#                 )
#             ],
#         )
#     # learn.save(Path(config["fname_start"])/"saved_model")

#     # TODO Export, add cam to saved object
#     # learn.export()

#     config["global_run_count"] += num_epochs
#     config["final_acc"] = learn.recorder.metrics[0].value.item()


def train_proxy_steps(trial, config):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()

    fname_start = f'{config["main_run_dir"]}{config["experiment_name"]}_{datetime.now().strftime("%d%m%Y_%H:%M:%S")}'

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
    config["clear_every_step"] = trial.suggest_categorical(
        "clear_every_step", [True, False]
    )

    config["fname_start"] = fname_start

    config["ds_path"] = config["dataset_info"][config["ds_name"]]["path"]
    config["name_fn"] = config["dataset_info"][config["ds_name"]]["name_fn"]

    config["batch_size"] = set_batch_size_dict[config["model"]]

    config["criterion"] = nn.CrossEntropyLoss()

    config["global_run_count"] = 0
    config["lr"] = 1e-3

    clear_proxy_images(config=config)
    for i, step in enumerate(config["proxy_steps"]):
        chk = get_last_checkpoint(config["fname_start"]) if i > 0 else None
        config["chk"] = chk
        if step == "p":
            # config["load_proxy_data"] = True
            setup_train_round(
                config=config,
                proxy_step=True,
                num_epochs=1 + i,chk=chk
            )
        else:
            # config["load_proxy_data"] = False
            setup_train_round(
                config=config,
                proxy_step=False,
                num_epochs=step+ i , chk = chk,
            )

        if config["clear_every_step"] == True:
            clear_proxy_images(config=config)  # Clean directory

    if config["clear_every_step"] == False:
        clear_proxy_images(config=config)  # Clean directory

    return config["final_acc"]


def train_single_round(trial, config):
    return train_proxy_steps(trial, config)
