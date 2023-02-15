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
import random

from .meta_utils import *

sns.set()

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
cudnn.benchmark = True

# %%


def fish_name_fn(x):
    return str(x).split("/")[-2]


def asl_name_fn(x):
    return str(x).split("/")[-2]


# %%
# TODO Create dataset loader for tabular
# %%
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


# %%
def clear_proxy_images(config):
    all_files = get_files(config["ds_path"])
    _ = [Path.unlink(x) for x in all_files if "proxy" in str(x)]
    print("[INFO] Cleared all existing proxy images")

def create_folds(config):
    # TODO Allow options for Proxy data
    all_files = get_files(config["ds_path"])
    random.shuffle(all_files)
    if config["subset_images"]!= None:
        all_files = all_files[: config["subset_images"]]
    if config["load_proxy_data"] == False:
        all_files = [x for x in all_files if "proxy" not in str(x)]

    # Put them in a data frame for encoding
    df = pd.DataFrame.from_dict(
        {x: config["name_fn"](x) for x in all_files}, orient="index"
    ).reset_index()
    # print(df.head(5))
    df.columns = ["image_id", "label"]
    # Convert labels to integers
    temp = preprocessing.LabelEncoder()
    df["label"] = temp.fit_transform(df.label.values)

    # Save label map
    label_map = {i: l for i, l in enumerate(temp.classes_)}
    rev_label_map = {l: i for i, l in enumerate(temp.classes_)}

    config["label_map"]= label_map
    config["rev_label_map"]= rev_label_map

    # Kfold splits
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    stratify = StratifiedKFold(n_splits=2)
    for i, (t_idx, v_idx) in enumerate(
        stratify.split(X=df.image_id.values, y=df.label.values)
    ):
        df.loc[v_idx, "kfold"] = i
    df.to_csv("train_folds.csv", index=False)
    logging.info("Train folds saved")

    train = df.loc[df["kfold"] != 1]
    val = df.loc[df["kfold"] == 1]

    # TODO Check if logging works
    logging.info("Train and val data created")

    return train, val


# %%
def create_dls(train, val, config):
    # TODO Options for more config
    data_transforms = {
        "train": A.Compose(
            [
                A.RandomResizedCrop(config["image_size"], config["image_size"], p=1.0),
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
        "val": A.Compose(
            [
                A.Resize(config["image_size"], config["image_size"]),
                A.CenterCrop(config["image_size"], config["image_size"], p=1.0),
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
    }

    #TODO :FFCCV
    image_datasets = {
        "train": ImageClassDs(
            train, config["ds_path"], train=True, transforms=data_transforms["train"]
        ),
        "val": ImageClassDs(
            val, config["ds_path"], train=False, transforms=data_transforms["val"]
        ),
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            batch_size=config["batch_size"],
            shuffle=True,
            # num_workers=4 * config["num_gpu"],
            # pin_memory=True,
            num_workers = 8,
        ),
        "val": torch.utils.data.DataLoader(
            image_datasets["val"],
            batch_size=config["batch_size"],
            shuffle=False,
            # num_workers=4 * config["num_gpu"],
            # pin_memory=True,
            num_workers = 8,
        ),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    return image_datasets, dataloaders, dataset_sizes