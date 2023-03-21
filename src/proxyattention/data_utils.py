# %%
# Imports

import logging
import os
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from albumentations.pytorch import ToTensorV2
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import random

from fastai.vision.all import get_image_files, ImageDataLoaders, Resize

cudnn.benchmark = True

# TODO Create dataset loader for tabular
def get_parent_name(x):
    return str(x).split("/")[-2]


# %%
def clear_proxy_images(config):
    all_files = get_image_files(config["ds_path"])
    _ = [Path.unlink(x) for x in all_files if "proxy" in str(x)]
    print("[INFO] Cleared all existing proxy images")


def create_dls(config):
    all_files = get_image_files(config["ds_path"])
    random.shuffle(all_files)
    if config["subset_images"] is not None:
        all_files = all_files[: config["subset_images"]]
    if config["load_proxy_data"] is False:
        all_files = [x for x in all_files if "proxy" not in str(x)]
    dls = ImageDataLoaders.from_lists(
        path=config["ds_path"],
        fnames=all_files,
        labels=[get_parent_name(x) for x in all_files],
        item_tfms=Resize(224),
    )

    config["num_classes"] = dls.c
    config["vocab"] = dls.vocab
    return dls
