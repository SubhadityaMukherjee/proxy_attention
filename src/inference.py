# %%
import copy
from datetime import datetime
import logging
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")  # no UI backend
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
from pytorch_grad_cam.utils.image import (
    deprocess_image,
    preprocess_image,
    show_cam_on_image,
)
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from operator import attrgetter

from proxyattention.data_utils import (
    clear_proxy_images,
    create_dls,
    create_folds,
    get_parent_name,
)
from proxyattention.meta_utils import get_files, save_pickle, read_pickle, return_grouped_results, fix_tensorboard_names
from proxyattention.training import choose_network, dict_gradient_method, inv_normalize, find_target_layer

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
import gc
import copy
import argparse as ap
import ast
import pandas as pd
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst


import cv2
import matplotlib.pyplot as plt

# sns.set()

cudnn.benchmark = True

computer_choice = "pc"
config = {}
# pc, cluster

# Make dirs
if computer_choice == "linux":
    main_run_dir = (
        "/run/media/eragon/HDD/CODE/Github/improving_robotics_datasets/src/runs/"
    )
    main_ds_dir = "/run/media/eragon/HDD/Datasets/"
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

elif computer_choice == "pc":
    main_run_dir = Path(
        "/mnt/d/CODE/thesis_runs/proper_runs/"
    )
    main_ds_dir = Path("/mnt/d/Datasets/")
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ds_val_paths = {
    "asl": {
        "path" : f"{main_ds_dir}/asl/asl_alphabet_test/",
        "name_func": lambda x: x.split("/")[-1].split("_")[0],
    },
    "cifar100": {
        "path" : f"{main_ds_dir}/CIFAR-100/test",
        "name_func": get_parent_name,},
    "imagenette": {
        "path" : f"{main_ds_dir}/imagenette2-320/val",
        "name_func": get_parent_name,
    },
    "caltech256": {
        "path" : f"{main_ds_dir}/caltech256/valid",
        "name_func": get_parent_name,
    },
    # "dogs": {
    #     "path": "/run/media/eragon/HDD/Datasets/dogs/test",
    #     "name_func": get_parent_name,
    # }, #todo
    # "plantdisease": {
    #     "path": "/run/media/eragon/HDD/Datasets/plantdisease/test",
    #     "name_func": get_parent_name,
    # }, #todo
}

os.environ["TORCH_HOME"] = str(main_ds_dir)

# %%
read_agg_res = read_pickle("./results/aggregated_runs.csv")[0]
read_agg_res = fix_tensorboard_names(read_agg_res)
# %%
read_agg_res.head()
#%%
return_grouped_results(read_agg_res, ["index","ds_name", "model", "global_run_count", "final_acc", "save_path","has_proxy", "step_schedule"], filter={"ds_name": "cifar100", "model":"resnet18"})

#%%
index_check = ["/mnt/d/CODE/thesis_runs/proper_runs/proxy_run_28032023_092528/events.out.tfevents.1679988566.eragon", "/mnt/d/CODE/thesis_runs/proper_runs/baseline_run_26032023_124800/events.out.tfevents.1679827680.eragon"]
#%%
def get_row_from_index(read_agg_res, index_check):
    temp_df = read_agg_res[read_agg_res["index"] == index_check]
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
#%%
def get_single_cam(compare):
    target_layer = find_target_layer(config={"model": compare[0]}, model = compare[-1])

    return GradCAMPlusPlus(
        model=compare[-1].cpu(), target_layers=target_layer, use_cuda=False
    )

#%%
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

tfm = transforms.ToPILImage()
#%%
compare_1, compare_2 = get_row_from_index(read_agg_res, index_check[0]), get_row_from_index(read_agg_res, index_check[1])
ds_name = compare_1[-2]
#%%
cam_1, cam_2 = get_single_cam(compare_1), get_single_cam(compare_2)
#%%
ds_val_path = ds_val_paths[ds_name]["path"]
ds_val_name_func = ds_val_paths[ds_name]["name_func"]
# Create an ImageFolder dataset using the images_folder_path
dataset = ImageFolder(ds_val_path, transform=transform)

# Create a DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
dataloader

#%%
image, _ = next(iter(dataloader))
# image = image.to("cuda")
#%%
#cam1 has proxy while cam2 does not
grads_1 = cam_1(input_tensor=image, targets=None)
grads_2 = cam_2(input_tensor=image, targets=None)
#%%
def plot_images(images):
    # Create a grid of images with a maximum of 16 images per row
    num_images = len(images)
    num_rows = min(4, int(np.ceil(num_images / 4)))
    num_cols = min(4, int(np.ceil(num_images / num_rows)))
    
    # Create the figure and axes objects
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    
    # Flatten the axes array if it has more than one dimension
    if len(axes.shape) > 1:
        axes = axes.flatten()
    
    # Plot each image on its corresponding subplot
    for i, ax in enumerate(axes[:num_images]):
        ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    
    # Show the plot
    plt.savefig("test.png")
#%%
plot_images(image)
#%%


def show_cam_on_image(image, mask):
    mask,current_image, colormap = deprocess_image(tfm(mask)), image, cv2.COLORMAP_JET
    current_image = current_image.permute(1, 2, 0).cpu().numpy()
    # image = np.asarray(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = (1 - .5) * heatmap + .5 * current_image
    cam = cam / np.max(cam)
    im = np.uint8(255 * cam)
    return Image.fromarray(im)

#%%
cams = [show_cam_on_image(image[i], grads_1[i]) for i in range(len(image))]
#%%
rows = 4
cols = 4

fig, axes = plt.subplots(rows, cols, figsize=(10,10))

for i in range(rows):
    for j in range(cols):
        img_index = i*cols + j
        if img_index < len(cams):
            axes[i][j].imshow(cams[img_index])
        axes[i][j].axis('off')
plt.show()
#%%