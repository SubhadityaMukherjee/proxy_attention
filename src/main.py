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
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import proxyattention
from PIL import Image
from torchvision.utils import save_image

import logging

logging.basicConfig(level=logging.DEBUG)
print("Done imports")

# sns.set()


# %%
# Config

# config = {
#     "experiment_name": "asl_prune_params_subset",
#     # "ds_path": Path("/mnt/e/Datasets/asl/asl_alphabet_train/asl_alphabet_train"),
#     "ds_name": "asl",
#     # "name_fn": proxyattention.data_utils.asl_name_fn,
#     "image_size": 224,
#     "batch_size": 32,
#     "epoch_steps": [1, 2],
#     "enable_proxy_attention": True,
#     # "change_subset_attention": tune.loguniform(0.1, 0.8),
#     "change_subset_attention": tune.grid_search([0.3, 0.5, 0.8]),
#     # "shuffle_dataset": tune.choice([True, False]),
#     "num_gpu": 1,
#     "num_cpu": 10,
#     # "transfer_imagenet": False,
#     "transfer_imagenet": tune.grid_search([True, False]),
#     "subset_images": 8000,
#     "proxy_threshold": tune.loguniform(0.8, .95),
#     # "pixel_replacement_method": tune.grid_search(["mean", "max", "min", "halfmax"]),
#     # "pixel_replacement_method": tune.grid_search(["blended", "max", "min"]),
#     "pixel_replacement_method": tune.grid_search(["max"]),
#     "model": tune.grid_search(["resnet18"]),
#     # "model": tune.grid_search(["vit_base_patch8_224", "resnet18"]),
#     # "proxy_steps": tune.choice([[1, "p", 1], [3, "p", 1], [1, 1], [3,1]]),
#     # "proxy_steps": tune.choice([["p", 1],[1, 1], ["p",1], [1, "p",1], [1,1,1]]),
#     # "proxy_steps": tune.choice([[10, "p",10, "p", 30, "p", 20], [70], [70, "p"], [30, "p", 40], [30, "p", 40, "p"], [10, "p",10, "p", 30, "p", 20, "p"], ["p", 70]]),
#     "proxy_steps": tune.grid_search([[5, "p", 5]]),
#     # "proxy_steps": tune.grid_search([[2, "p", 5]]),
#     # "proxy_steps": tune.grid_search([[10, "p", 10], [21]]),
#     # "proxy_steps": ["p"],
#     "load_proxy_data": False,
#     # "global_run_count" : 0,
#     # "gradient_method": tune.grid_search(["gradcam", "gradcamplusplus", "eigencam"]),
#     "gradient_method": "gradcamplusplus",
#     # "aug_smooth" : tune.choice([True, False]),
#     # "eigen_smooth" : tune.choice([True, False]),
#     # "clear_every_step": tune.grid_search([True, False]),
#     "clear_every_step": True,
# }


config = {
    "experiment_name": "threshold_test",
    # "ds_name": tune.grid_search(['asl', 'imagenette']),
    "ds_name": tune.grid_search(['imagenette']),
    "image_size": 224,
    "batch_size": 16,
    "enable_proxy_attention": True,
    "change_subset_attention": tune.grid_search([0.8]),
    # "change_subset_attention": tune.grid_search([0.3, 0.5, 0.8]),
    "num_gpu": 1,
    "num_cpu": 10,
    "transfer_imagenet": tune.grid_search([False]),
    "subset_images": 8000,
    # "proxy_threshold": 0.85,
    # "proxy_threshold": tune.loguniform(0.8, .95),
    "proxy_threshold": tune.grid_search([np.random.random() for _ in range(50)]),
    # "proxy_image_weight" : tune.loguniform(0.1, 0.8),
    # "proxy_image_weight" : tune.loguniform(0.1, 0.8),
    # "proxy_image_weight" : 0.4,
    "proxy_image_weight" : tune.grid_search([0.1, 0.2, 0.4, 0.8, 0.95]),
    "pixel_replacement_method": tune.grid_search(["blended"]), 
    "model": tune.grid_search(["resnet18"]),
    "proxy_steps": tune.grid_search([[10,"p",10], [21]]),
    # "proxy_steps": tune.grid_search([[10,"p",10]]),
    # "proxy_steps": tune.grid_search([[1, "p"]]),
    "load_proxy_data": False,
    "gradient_method": "gradcamplusplus",
    # "gradient_method": tune.grid_search(["gradcam", "eigencam"]),
    # "clear_every_step": False,
    "clear_every_step": tune.grid_search([True]),
}

# Test config
# config = {
#     "experiment_name": "testing_things",
#     "ds_name": "asl",
#     "image_size": 224,
#     "batch_size": 32,
#     "enable_proxy_attention": True,
#     "change_subset_attention": 0.8,
#     "num_gpu": 1,
#     "num_cpu": 10,
#     "transfer_imagenet": False,
#     "subset_images": 8000,
#     "proxy_threshold": 0.8,
#     "pixel_replacement_method": "max",
#     "model": "resnet18",
#     "proxy_steps": tune.grid_search([[1, "p", 1]]),
#     "gradient_method": "gradcamplusplus",
#     "clear_every_step": True,
#     "load_proxy_data": False,
# }

computer_choice = "pc"
# pc, mac, cluster

# Make dirs
if computer_choice == "pc":
    main_run_dir = "/mnt/e/CODE/Github/improving_robotics_datasets/src/runs/"
    main_ds_dir = "/mnt/e/Datasets/"
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif computer_choice == "mac":
    main_run_dir = "/Users/eragon/Documents/CODE/Github/improving_robotics_datasets/src/runs/"
    main_ds_dir = "/Users/eragon/Documents/CODE/Datasets/"
    config["device"] = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

os.environ["TORCH_HOME"] = main_ds_dir
dataset_info = {
    "asl": {"path": Path(f"{main_ds_dir}asl/asl_alphabet_train/asl_alphabet_train") , "name_fn": proxyattention.data_utils.get_parent_name},
    "imagenette": {"path": Path(f"{main_ds_dir}/imagenette2-320/train") , "name_fn": proxyattention.data_utils.get_parent_name},
}


logging.info("Directories made/checked")
os.makedirs(main_run_dir, exist_ok=True)
fname_start = f'{main_run_dir}{config["ds_name"]}_{config["experiment_name"]}+{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}_ps-{str(config["proxy_steps"])}'

config["fname_start"] = fname_start
config["dataset_info"] = dataset_info
config["main_run_dir"] = main_run_dir

logging.basicConfig(filename=fname_start, encoding="utf-8", level=logging.DEBUG)
logging.info(f"[INFO] : File name = {fname_start}")
print(f"[INFO] : File name = {fname_start}")
#%%

proxyattention.training.hyperparam_tune(config=config)

# %%
