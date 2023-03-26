import argparse
import os
from pprint import pprint

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
import proxyattention
import logging
from datetime import datetime
from itertools import combinations
import numpy as np
import itertools


def get_optimizer(parameters) -> torch.optim.Optimizer:
    return torch.optim.Adam(parameters, lr=0.001, weight_decay=0.0001)

def get_lr_scheduler_config(optimizer: torch.optim.Optimizer) -> dict:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001
    )
    lr_scheduler_config = {
        'scheduler': scheduler,
        'monitor': 'val/loss',
        'interval': 'epoch',
        'frequency': 1,
    }
    return lr_scheduler_config

class ImageTransform:
    def __init__(self, is_train: bool, img_size: int | tuple =224):
        if is_train:
            self.transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)

class DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 8,
        subset : int|bool = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = ImageFolder(
            root=os.path.join(root_dir),
            transform=ImageTransform(is_train=True, img_size=self.img_size),
        )
        self.train_dataset, self.val_dataset = self.train_val_dataset(self.train_dataset, subset = subset)
        # self.val_dataset = ImageFolder(
        #     root=os.path.join(root_dir, 'val'),
        #     transform=ImageTransform(is_train=False, img_size=self.img_size),
        # )
        self.classes = self.train_dataset.classes
        self.class_to_idx = self.train_dataset.class_to_idx
    
    def train_val_dataset(self, dataset, val_split=0.25, subset= None) -> Subset:

        indices = np.arange(len(dataset))
        if subset != None:
            indices= indices[:subset]
        train_idx, val_idx = train_test_split(indices, test_size=val_split)
        return Subset(dataset, train_idx).dataset, Subset(dataset, val_idx).dataset

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        return dataloader

class Model(LightningModule):
    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = False,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=num_classes
        )
        self.model = torch.compile(self.model)
        self.train_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task = 'multiclass', num_classes = num_classes)
        self.val_loss = nn.CrossEntropyLoss()
        self.val_acc = Accuracy(task = 'multiclass', num_classes = num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        _, pred = out.max(1)

        loss = self.train_loss(out, target)
        acc = self.train_acc(pred, target)
        self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        _, pred = out.max(1)

        loss = self.val_loss(out, target)
        acc = self.val_acc(pred, target)
        self.log_dict({'val/loss': loss, 'val/acc': acc})

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters())
        lr_scheduler_config = get_lr_scheduler_config(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

def get_basic_callbacks(checkpoint_interval: int = 1) -> list:
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(
        filename='epoch{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=-1,
        every_n_epochs=checkpoint_interval,
    )
    return [ckpt_callback, lr_callback]

def get_gpu_settings(
    gpu_ids: list[int], n_gpu: int
) -> tuple[str, int | list[int] | None, str | None]:
    """Get gpu settings for pytorch-lightning trainer:
    https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags
    Args:
        gpu_ids (list[int])
        n_gpu (int)
    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    """
    if not torch.cuda.is_available():
        return "cpu", None, None

    if gpu_ids is not None:
        devices = gpu_ids
        strategy = "ddp" if len(gpu_ids) > 1 else "auto"
    elif n_gpu is not None:
        # int
        devices = n_gpu
        strategy = "ddp" if n_gpu > 1 else "auto"
    else:
        devices = 1
        strategy = "auto"

    return "gpu", devices, strategy

def get_trainer(config) -> Trainer:
    callbacks = get_basic_callbacks(checkpoint_interval=config["log_every"])
    accelerator, devices, strategy = get_gpu_settings([0], 1)
    trainer = Trainer(
        max_epochs=config["num_epoch"],
        callbacks=callbacks,
        default_root_dir=config["fname_start"],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=True,
        deterministic=True,
        precision = 16,
    )
    return trainer

if __name__ == '__main__':
    config = {
        # "experiment_name": "proxy_run",
        # "experiment_name": "baseline_run",
        "experiment_name": "ignore",
        "image_size": 224,
        "batch_size": 32,
        "enable_proxy_attention": True,
        "transfer_imagenet": True,
        "subset_images": 9000,
        "pixel_replacement_method": "blended",
        # "proxy_steps": [10, "p",9],
        # "proxy_steps": [3],
        "proxy_steps": [20],
        # "proxy_steps": [4],
        "load_proxy_data": False,
        "proxy_step": False,
        "log_every": 2

    }

    search_space = {
        "change_subset_attention": [0.8],
        # "model": ["resnet18", "vgg16", "resnet50", "vit_base_patch16_224"],
        # "model": ["resnet18","vgg16", "resnet50", "vit_base_patch16_224"],
        "model": ["vgg16","efficientnet_b0","resnet18","resnet50", "vit_base_patch16_224"],
        # "model" : ["vgg16"],
        "proxy_image_weight": [0.1],
        "proxy_threshold": [0.85],
        "gradient_method": ["gradcamplusplus"],
        # "ds_name": ["asl", "imagenette", "caltech256"],
        # "ds_name": ["asl", "imagenette"],
        "ds_name": ["cifar100", "dogs", "caltech101", "asl", "imagenette"],
        # "ds_name": ["imagenette"],
        "clear_every_step": [True],
    }

    main_run_dir = "/run/media/eragon/HDD/CODE/Github/improving_robotics_datasets/src/runs/"
    main_ds_dir = "/run/media/eragon/HDD/Datasets/"
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.environ["TORCH_HOME"] = main_ds_dir
    dataset_info = {
        "asl": {
            "path": Path(f"{main_ds_dir}asl/asl_alphabet_train/asl_alphabet_train"),
            "name_fn": proxyattention.data_utils.get_parent_name,
            "num_classes" : 29
        },
        "imagenette": {
            "path": Path(f"{main_ds_dir}/imagenette2-320/train"),
            "name_fn": proxyattention.data_utils.get_parent_name,

            "num_classes" : 10
        },
        "caltech256": {
            "path": Path(f"{main_ds_dir}/caltech256/train"),
            "name_fn": proxyattention.data_utils.get_parent_name,
            "num_classes" : 256
        },
        "tinyimagenet": {
            "path": Path(f"{main_ds_dir}/tiny-imagenet-200/train"),
            "name_fn": proxyattention.data_utils.get_parent_name,
            "num_classes" : 200
        },
        "cifar100": {
            "path": Path(f"{main_ds_dir}/CIFAR-100/train"),
            "name_fn": proxyattention.data_utils.get_parent_name,
            "num_classes" : 100
        },
        "dogs": {
            "path": Path(f"{main_ds_dir}/dogs/images/Images"),
            "name_fn": proxyattention.data_utils.get_parent_name,
            "num_classes" : 120
        },
        "caltech101": {
            "path": Path(f"{main_ds_dir}/caltech-101"),
            "name_fn": proxyattention.data_utils.get_parent_name,
            "num_classes" : 101
        }


    }


    logging.info("Directories made/checked")
    os.makedirs(main_run_dir, exist_ok=True)

    config["dataset_info"] = dataset_info
    config["main_run_dir"] = main_run_dir

    fname_start = f'{config["main_run_dir"]}{config["experiment_name"]}_{datetime.now().strftime("%d%m%Y_%H%M%S")}'
    config["fname_start"] = fname_start



    search_space_values = list(search_space.values())
    combinations = list(itertools.product(*search_space_values))
    for combination in combinations:
        params = dict(zip(search_space.keys(), combination))
        config = {**config, ** params}
        config["ds_path"] = config["dataset_info"][config["ds_name"]]["path"]
        config["name_fn"] = config["dataset_info"][config["ds_name"]]["name_fn"]

        data = DataModule(
            root_dir=config["ds_path"],
            img_size=config["image_size"],
            batch_size=32,
            num_workers=4,
            subset = config["subset_images"]
        )
        model = Model(
            model_name=config["model"], pretrained=config["transfer_imagenet"], num_classes=len(data.classes)
        )
        # model = torch.compile(model)
        config["num_epoch"] = 5
        trainer = get_trainer(config)

        print('Training classes:')
        pprint(data.class_to_idx)
        trainer.fit(model, data)
        break
