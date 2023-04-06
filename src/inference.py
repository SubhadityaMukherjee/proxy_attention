# %%
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
from proxyattention.meta_utils import get_files, save_pickle, read_pickle
from proxyattention.training import choose_network, dict_gradient_method, inv_normalize

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

import matplotlib.pyplot as plt

# sns.set()

cudnn.benchmark = True
# %%
read_agg_res = read_pickle("./results/aggregated_runs.csv")[0]
read_agg_res = read_agg_res[read_agg_res["global_run_count"].isna() == False]
read_agg_res["global_run_count"] = read_agg_res["global_run_count"].astype(int)

# %%
read_agg_res.head()
# %%
test_row = read_agg_res.iloc[3:4]
test_row
# %%
model_name = test_row["model"].values[0]
save_path = (
    test_row["save_path"]
    .values[0]
    .replace("improving_robotics_datasets", "proxy_attention")
)
num_classes = test_row["num_classes"].values[0]
ds_name = test_row["ds_name"].values[0]
# %%
ds_val_paths = {
    "asl": {
        "path": "/run/media/eragon/HDD/Datasets/asl/asl_alphabet_test/",
        "name_func": lambda x: x.split("/")[-1].split("_")[0],
    },
    "cifar100": {
        "path": "/run/media/eragon/HDD/Datasets/CIFAR-100/test",
        "name_func": get_parent_name,},
    "imagenette": {
        "path": "/run/media/eragon/HDD/Datasets/imagenette2-320/val",
        "name_func": get_parent_name,
    },
    "caltech256": {
        "path": "/run/media/eragon/HDD/Datasets/caltech256/valid",
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
# %%
test_row["ds_name"].values[0]
# %%

# %%
model = timm.create_model(
    model_name=model_name,
    pretrained=True,
    num_classes=int(num_classes),
    # checkpoint_path=save_path,
)

sd = model.state_dict()

model = timm.create_model(
    model_name=model_name,
    pretrained=True,
    num_classes=int(num_classes),
    # checkpoint_path=save_path,
).to("cuda")

model.load_state_dict(sd)

model.eval()
# %%


def find_target_layer(modelname, model):
    if modelname == "resnet18":
        return [model.layer4[-1].conv2]
    elif modelname == "resnet50":
        return [model.layer4[-1].conv2]
    elif modelname == "efficientnet_b0":
        return [model.conv_head]
    elif modelname == "FasterRCNN":
        return model.backbone
    elif modelname == "vgg16" or modelname == "densenet161":
        return [model.features[-3]]
    elif modelname == "mnasnet1_0":
        return model.layers[-1]
    elif modelname == "vit_small_patch32_224":
        return [model.norm]
    elif modelname == "vit_base_patch16_224":
        return [model.norm]
    else:
        raise ValueError("Unsupported model type!")


# %%

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

gradcam = GradCAMPlusPlus(
    model=model, target_layers=find_target_layer(model_name, model), use_cuda=True
)
tfm = transforms.ToPILImage()

#%%
# Define the path to the folder containing the images you want to visualize
try:
    ds_val_path = ds_val_paths[ds_name]["path"]
    ds_val_name_func = ds_val_paths[ds_name]["name_func"]
    # Create an ImageFolder dataset using the images_folder_path
    dataset = ImageFolder(ds_val_path, transform=transform)

    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataloader

    # Loop through each image in the dataloader
    for image, _ in dataloader:
        image = image.to("cuda")
        grads = gradcam(input_tensor=image, targets=None)
        print(grads.max(), grads.min(), grads.mean(), grads.std())

        # print(grads.shape, image.shape)
        # grads = torch.Tensor(grads).unsqueeze(1).expand(-1, 3, -1, -1)

        # image = inv_normalize(image)
        
#         cam_metric = ROADMostRelevantFirst(percentile=75)
#         scores, perturbation_visualizations = cam_metric(image, 
#   grads, targets= None, model= model, return_visualization=True, )
        # normalized_inps = inv_normalize(image)
        # Image.show(normalized_inps.squeeze())
        # Image.fromarray(normalized_inps.).show()
        # tfm(grads.squeeze()).save("./results/test.png")
        # print(grads.shape)
        # image = np.array(tfm(image[0]))
        # image = np.transpose(image, (2,0,1))
        # grads = np.array(tfm(grads[0]))
        # grads = np.expand_dims(grads, axis=0)
        # # convert 0 axis to 3 channels
        # grads = np.repeat(grads, 3, axis=0)

        # final_overlay = image + (1- 0.5) * grads
        # final_overlay = np.transpose(final_overlay, (1,2,0))
        # print(final_overlay)
        # Image.fromarray(final_overlay).show()
        # np.add()
        # print(tfm(grads[0]))

        # cam_image = show_cam_on_image(image[0].detach().cpu().numpy(), grads)


        # Show the results
        # cam_image.show()
        
        # # Get the maximum values along the channel axis
        # max_vals = torch.max(grads, dim=0)[0]
        
        # # Flatten the tensor and convert to a list
        # max_vals_list = max_vals.view(-1).tolist()
        
        # # Sort the list in descending order
        # max_vals_list.sort(reverse=True)
        
        # # Plot a bar graph of the top 10 values
        # plt.bar(range(10), max_vals_list[:10])
        
        # # Label the x-axis and y-axis
        # plt.xlabel("Top 10 Values")
        # plt.ylabel("Value")
        
        # Display the plot
        # plt.show()
        # plt.savefig("./results/test.png")

        break

except KeyError:
    pass



# %%
