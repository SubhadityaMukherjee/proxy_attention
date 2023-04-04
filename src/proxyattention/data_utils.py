# %%
# Imports

import logging
import os
from pathlib import Path

# import albumentations as A
import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

# from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from PIL import Image
from torchvision.io import read_image
import random
from tqdm import tqdm
from typing import Dict

from .meta_utils import get_files

cudnn.benchmark = True


# %%
def get_parent_name(x):
    return str(x).split("/")[-2]


# %%
# TODO Create dataset loader for tabular
# %%
class ImageClassDs(Dataset):
    def __init__(
        self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None
    ) -> None:
        self.df = df
        self.x, self.y = self.df["image_id"].values, self.df["label"].values
        self.imfolder = imfolder
        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        im_path = self.x[index]
        x = Image.open(im_path)
        if len(x.getbands()) != 3:
            x = x.convert("RGB")

        if self.transforms:
            x = self.transforms(x)

        y = self.y[index]
        return {
            "x": x,
            "y": y,
        }

    def __len__(self) -> int:
        return len(self.df)

#Paired Dataset for Image Segmentation
class ImageSegmentationDs(Dataset):
    def __init__(
        self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None
    ) -> None:
        self.df = df
        self.x, self.y = self.df["image_id"].values, self.df["image_id_2"].values
        self.imfolder = imfolder
        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x = Image.open(self.x[index])
        y = Image.open(self.y[index])
        if len(x.getbands()) != 3:
            x = x.convert("RGB")
            y = y.convert("RGB")

        if self.transforms:
            x = self.transforms(x)
            y = self.transforms(y)

        # y = self.y[index]
        return {
            "x": x,
            "y": y,
        }

    def __len__(self) -> int:
        return len(self.df)

# %%
def clear_proxy_images(config: Dict[str, str]) -> None:
    all_files = get_files(config["ds_path"])
    _ = [Path.unlink(x) for x in all_files if "proxy" in str(x)]
    print("[INFO] Cleared all existing proxy images")


def create_folds(config):
    all_files = get_files(config["ds_path"])
    random.shuffle(all_files)
    if config["subset_images"] is not None:
        all_files = all_files[: config["subset_images"]]
    if config["load_proxy_data"] is False:
        all_files = [x for x in all_files if "proxy" not in str(x)]
    # all_files = [x for x in tqdm(all_files, total = len(all_files), desc = "Verifying image files") if Image.verify(x)]

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

    config["label_map"] = label_map
    config["rev_label_map"] = rev_label_map

    # Kfold splits
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    stratify = StratifiedKFold(n_splits=2, shuffle=True)
    for i, (t_idx, v_idx) in enumerate(
        stratify.split(X=df.image_id.values, y=df.label.values)
    ):
        df.loc[v_idx, "kfold"] = i
    df.to_csv("train_folds.csv", index=False)
    logging.info("Train folds saved")

    train = df.loc[df["kfold"] != 1]
    val = df.loc[df["kfold"] == 1]

    logging.info("Train and val data created")

    return train, val


# %%
def create_dls(train, val, config):
    # TODO Compare with other augmentation techniques
    # data_transforms = {
    #     "train": A.Compose(
    #         [

    #             A.Resize(config["image_size"], config["image_size"]),
    #             # A.RandomResizedCrop(config["image_size"], config["image_size"], p=1.0),
    #             A.Normalize(
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225],
    #                 max_pixel_value=255.0,
    #                 p=1.0,
    #             ),
    #             ToTensorV2(p=1.0),
    #         ],
    #         p=1.0,
    #     ),
    #     "val": A.Compose(
    #         [
    #             A.Resize(config["image_size"], config["image_size"]),
    #             # A.CenterCrop(config["image_size"], config["image_size"], p=1.0),
    #             A.Normalize(
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225],
    #                 max_pixel_value=255.0,
    #                 p=1.0,
    #             ),
    #             ToTensorV2(p=1.0),
    #         ],
    #         p=1.0,
    #     ),
    # }
    data_transforms_train = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            # transforms.ColorJitter(hue=0.05, saturation=0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, interpolation=Image.BILINEAR),
            transforms.ToTensor(),  # use ToTensor() last to get everything between 0 & 1
        ]
    )

    data_transforms_val = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),  # use ToTensor() last to get everything between 0 & 1
        ]
    )
    # data_transforms_train = torch.jit.script(data_transforms_train)
    # data_transforms_val = torch.jit.script(data_transforms_val)

    image_datasets = {
        "train": ImageClassDs(
            train, config["ds_path"], train=True, transforms=data_transforms_train
        ),
        "val": ImageClassDs(
            val, config["ds_path"], train=False, transforms=data_transforms_val
        ),
    }
    num_work = 4
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=num_work,
            pin_memory=True,
        ),
        "val": torch.utils.data.DataLoader(
            image_datasets["val"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=num_work,
            pin_memory=True,
        ),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    return image_datasets, dataloaders, dataset_sizes


# %%
def batchify(dataset, idxs):
    "Return a list of items for the supplied dataset and idxs"
    tss = [dataset[i][0] for i in idxs]
    ys = [dataset[i][1] for i in idxs]
    return (tss, ys)


def itemize(batch):
    # take a batch and create a list of items. Each item represent a tuple of (tseries, y)
    tss, ys = batch
    b = [(ts, y) for ts, y in zip(tss, ys)]
    return b


def get_list_items(dataset, idxs):
    "Return a list of items for the supplied dataset and idxs"
    list = [dataset[i] for i in idxs]
    return list


def get_batch(dataset, idxs):
    "Return a batch based on list of items from dataset at idxs"
    # list_items = [(image2tensor(PILImage.create(dataset[i][0])), dataset[i][1]) for i in idxs]
    # tdl = TfmdDL(list_items, bs=2, num_workers=0)
    # tdl.to(default_device())
    # return tdl.one_batch()
    pass
