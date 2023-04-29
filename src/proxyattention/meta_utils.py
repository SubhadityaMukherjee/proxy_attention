import mimetypes
from pathlib import Path
from typing import Generator, Iterable
import pickle
import seaborn as sns
import clipboard
import pandas as pd

from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    ScoreCAM,
    EigenGradCAM
)
from pytorch_grad_cam.utils.image import (
    deprocess_image,
    preprocess_image,
    show_cam_on_image,
)
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from typing import Dict
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import logging
import timm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sns.set()
import os

# from fast.ai source code
image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)

def get_parent_name(x):
    return str(x).split("/")[-2]

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

tfm = transforms.ToPILImage()
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
def clear_proxy_images(config: Dict[str, str]):
    all_files = get_files(config["ds_path"])
    try:
        _ = [Path.unlink(x) for x in tqdm(all_files, total = len(all_files)) if "proxy" in str(x)]
    except:
        pass
    print("[INFO] Cleared all existing proxy images")

def get_name_without_proxy(save_name):
    save_name = str(save_name) # convert Path object to string for manipulation
    start_index = save_name.find("proxy") # find index of "proxy" substring
    end_index = save_name.find(".jpeg") # find index of ".jpeg" substring
    return save_name[:start_index] + save_name[end_index+5:] # concatenate everything before "label" and after ".jpeg"


def create_folds(config):
    all_files = get_files(config["ds_path"])

    # print(all_files[:10])
    if config["load_proxy_data"] is False:
        all_files = [x for x in all_files if "proxy" not in str(x)]
    else:
        # Remove original images that have been replaced by proxy images to maintain 1:1 ratio for the sake of a fair comparison
        proxy_files = [x for x in all_files if "proxy" in str(x)]
        replaced_files = [get_name_without_proxy(x) for x in proxy_files]
        all_files = [x for x in all_files if str(x) not in replaced_files]
    random.shuffle(all_files)
    if config["subset_images"] is not None:
        all_files = all_files[: config["subset_images"]]
   

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
    num_work = 8
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=num_work,
            # pin_memory=True,
            pin_memory=False,
        ),
        "val": torch.utils.data.DataLoader(
            image_datasets["val"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=num_work,
            pin_memory=False,
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

def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    # Rank 0 tensors in PyTorch are not really iterable
    return isinstance(o, (Iterable, Generator)) and getattr(o, "ndim", 1)


def is_coll(o):
    "Test whether `o` is a collection (i.e. has a usable `len`)"
    # Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(o, "__len__") and getattr(o, "ndim", 1)


def is_array(x):
    "`True` if `x` supports `__array__` or `iloc`"
    return hasattr(x, "__array__") or hasattr(x, "iloc")


def listify(o=None, *rest, use_list=False, match=None):
    "Convert `o` to a `list`"
    if rest:
        o = (o,) + rest
    if use_list:
        res = list(o)
    elif o is None:
        res = []
    elif isinstance(o, list):
        res = o
    elif isinstance(o, str) or is_array(o):
        res = [o]
    elif is_iter(o):
        res = list(o)
    else:
        res = [o]
    if match is not None:
        if is_coll(match):
            match = len(match)
        if len(res) == 1:
            res = res * match
        else:
            assert len(res) == match, "Match length mismatch"
    return res


def setify(o):
    "Turn any list like-object into a set."
    return o if isinstance(o, set) else set(listify(o))


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders = []
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path, followlinks=followlinks)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return res


def save_pickle(*args, fname="pickler.pkl"):
    with open(fname, "wb") as f:
        pickle.dump(args, f)


def read_pickle(fname="pickler.pkl"):
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    return obj

def check_proxy(string): return "p" in str(string)
def calc_stats(values):
    return f"min: {values.min()} \nmax: {values.max()} \navg: {values.mean()}"
def convert_float(df, cols, totype= float):
    for col in cols:
        df[col] = df[col].astype(totype)

def fix_tensorboard_names(df)->pd.DataFrame:
    """Fixes the names of the columns in the tensorboard csv file."""
    df = df[df["global_run_count"].isna() == False]

    convert_float(df, ["global_run_count"], int)
    df = df.fillna(0)
    # Fix naming
    df = df.rename(columns={"Acc/Val":"accuracy", "proxy_steps":"step_schedule"})
    # Fix types
    convert_float(df, ["change_subset_attention", "proxy_threshold", "accuracy"], float)
    convert_float(df, ["transfer_imagenet"], bool)

    df["has_proxy"] = df["step_schedule"].apply(check_proxy)
    return df


def return_grouped_results(df, group_cols ,filter = None, index_cols = (["ds_name", ("accuracy")]), print_latex = False):

    if filter != None:
        df = df.reset_index()
        for key in filter.keys():
            df = df[df[key] == filter[key]]
    final_df = pd.DataFrame(df.groupby(group_cols, as_index=True).mean(numeric_only = True)["accuracy"]).sort_values(index_cols, ascending=False)
    if print_latex == True:
        clipboard.copy(final_df.to_latex())

    return final_df

def show_cam_on_image(image, mask, weight = 0.6):
    mask,current_image, colormap = deprocess_image(tfm(mask)), image, cv2.COLORMAP_JET
    current_image = current_image.permute(1, 2, 0).cpu().numpy()
    # image = np.asarray(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = (1 - weight) * heatmap + weight * current_image
    cam = cam / np.max(cam)
    im = np.uint8(255 * cam)
    return Image.fromarray(im)

def plot_images_grad(image, grads, title,weight = 0.6, figsize=(10,10)):

    cams = [show_cam_on_image(image[i], grads[i], weight) for i in range(len(image))]
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i in range(rows):
        for j in range(cols):
            img_index = i*cols + j
            if img_index < len(cams):
                axes[i][j].imshow(cams[img_index])
                axes[i][j].set_title(title[img_index])
            axes[i][j].axis('off')
    # plt.show()
    return plt

def plot_grid(image_list,title_list = None, rows = 4, cols = 4, figsize=(10,10)):

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows > 1:
        for i in range(rows):
            for j in range(cols):
                img_index = i*cols + j
                if img_index < len(image_list):
                    axes[i][j].imshow(tfm(image_list[img_index]))
                    if title_list != None:
                        axes[i][j].set_title(title_list[img_index])
                axes[i][j].axis('off')
    else:
        for j in range(cols):
            img_index = j
            axes[j].imshow(tfm(image_list[img_index]))
            if title_list != None:
                axes[j].set_title(title_list[img_index])
            axes[j].axis('off')

def find_target_layer(config, model):
    if config["model"] == "resnet18":
        return [model.layer4[-1].conv2]
    elif config["model"] == "resnet50":
        return [model.layer4[-1].conv2]
    elif config["model"] == "efficientnet_b0":
        return [model.conv_head]
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


def get_single_cam(compare, method = GradCAMPlusPlus):
    target_layer = find_target_layer(config={"model": compare[0]}, model = compare[-1])

    return method(
        model=compare[-1].cpu(), target_layers=target_layer, use_cuda=False
    )

def iter_dl_and_plot(dataloader, cam_1, cam_2):
    image, _ = next(iter(dataloader))
    #cam1 has proxy while cam2 does not
    grads_1 = cam_1(input_tensor=image, targets=None)
    grads_2 = cam_2(input_tensor=image, targets=None)

    plot_images_grad(image, grads_1, title="proxy")
    plot_images_grad(image, grads_2, title="noproxy")

def get_row_from_index(read_agg_res, index_check):
    temp_df = read_agg_res[read_agg_res["count"] == index_check].reset_index()
    model_name = temp_df["model"].values[0]
    save_path = (
        temp_df["save_path"]
        .values[0]
        .replace("improving_robotics_datasets", "proxy_attention")
    )
    num_classes = temp_df["num_classes"].values[0]
    ds_name = temp_df["ds_name"].values[0]

    model = timm.create_model(
    model_name=model_name,
    pretrained=True,
    num_classes=int(num_classes),
    )

    sd = model.state_dict()

    model = timm.create_model(
        model_name=model_name,
        pretrained=True,
        num_classes=int(num_classes),
    ).to("cuda")

    model.load_state_dict(sd)
    model.eval()

    return model_name, save_path, num_classes, ds_name, model