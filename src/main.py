# %%
# Imports
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
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
from sklearn import metrics, model_selection, preprocessing
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm
from functools import partial
import proxyattention
from PIL import Image
from torchvision.utils import save_image
import subprocess

import logging

logging.basicConfig(level=logging.DEBUG)
print("Done imports")

# sns.set()


config = {
    "experiment_name": "baseline_run",
    "image_size": 224,
    "batch_size": 32,
    "enable_proxy_attention": True,
    "transfer_imagenet": True,
    "subset_images": 9000,
    "pixel_replacement_method": "blended",
    "load_proxy_data": False,
    "proxy_step": False,
    "log_every": 2

}

# Proxy search space
# search_space = {
#     "change_subset_attention" : [0.8, 0.5, 0.2],
#     # "model": ["resnet18", "vgg16", "resnet50", "vit_base_patch16_224"],
#     # "model": ["resnet18", "vgg16", "resnet50"],
#     "model": ["resnet18"],
#     "proxy_image_weight" : [0.1, 0.2, 0.4, 0.8, 0.95],
#     "proxy_threshold": [0.85],
#     "gradient_method" : ["gradcamplusplus"],
#     "ds_name" : ["asl", "imagenette", "caltech256"],
#     "clear_every_step": [True, False],
# }

# No proxy search space
search_space = {
    "change_subset_attention": [0.8],
    # "model": ["resnet18", "vgg16", "resnet50", "vit_base_patch16_224"],
    "model": ["resnet18","vgg16", "resnet50", "vit_base_patch16_224"],
    # "model" : ["vgg16"],
    "proxy_image_weight": [0.1],
    "proxy_threshold": [0.85],
    "gradient_method": ["gradcamplusplus"],
    # "ds_name": ["asl", "imagenette", "caltech256"],
    "ds_name": ["asl", "imagenette"],
    # "ds_name": ["imagenette"],
    "clear_every_step": [True],
    "proxy_steps": [20],
}


def get_approx_trial_count(search_space):
    total = 1
    for key in search_space.keys():
        total *= len(search_space[key])
    return total


logging.info(f"[INFO]: Approx trial count = {get_approx_trial_count(search_space)}")

computer_choice = "pc"
# pc, cluster

# Make dirs
if computer_choice == "pc":
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
}


logging.info("Directories made/checked")
os.makedirs(main_run_dir, exist_ok=True)

config["dataset_info"] = dataset_info
config["main_run_dir"] = main_run_dir


if __name__ == "__main__":
    resume_broken = False

    if resume_broken == True:
        i, combinations = proxyattention.meta_utils.read_pickle("combination_train.pkl")[0]
        combinations = combinations[i::]

    else:
        search_space_values = list(search_space.values())
        combinations = list(itertools.product(*search_space_values))

    for i,combination in tqdm(enumerate(combinations), total = len(combinations), desc="All training"):
        proxyattention.meta_utils.save_pickle((i, combinations), fname = "combination_train.pkl")
        params = dict(zip(search_space.keys(), combination))
        config = {**config, ** params}
        proxyattention.meta_utils.save_pickle(config, fname= f"current_config.pkl")
        subprocess.run(["python", "./proxyattention/training.py"])

# %%
