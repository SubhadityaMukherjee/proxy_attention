#%%
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

from .data_utils import clear_proxy_images, create_dls, create_folds, get_parent_name
from .meta_utils import get_files, save_pickle, read_pickle
from .training import choose_network, dict_gradient_method, inv_normalize
import time
import gc
import copy
import argparse as ap
import ast

# sns.set()

cudnn.benchmark = True

#%%
train, val = create_folds(config)
image_datasets, dataloaders, dataset_sizes = create_dls(train, val, config)
class_names = image_datasets["train"].classes
config["num_classes"] = len(config["label_map"].keys())

#%%

model = choose_network(config)
model.eval()  # Set model to evaluate mode
chk = torch.load(config["save_path"])
model.load_state_dict(chk["model_state_dict"])

#%%
with torch.no_grad():
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)


#%%
for i, inps in tqdm(
        enumerate(dataloaders[phase]), total=len(dataloaders[phase]), leave=False
    ):
        inputs = inps["x"].to(config["device"], non_blocking=True)
        labels = inps["y"].to(config["device"], non_blocking=True)

target_layers = find_target_layer(config, model)
config["cam"] = dict_gradient_method[config["gradient_method"]](
        model=model, target_layers=target_layers, use_cuda=True
)

try:
    input_wrong = torch.squeeze(torch.stack(input_wrong, dim=1))
    label_wrong = torch.squeeze(torch.stack(label_wrong, dim=1))
except:
    input_wrong = torch.squeeze(input_wrong)
    label_wrong = torch.squeeze(label_wrong)


grayscale_cams = config["cam"](input_tensor=input_wrong, targets=None)
grads = torch.Tensor(grayscale_cams, device= config["device"]).unsqueeze(1).expand(-1, 3, -1, -1).detach()
normalized_inps = inv_normalize(input_wrong)

from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst
cam_metric = ROADMostRelevantFirst(percentile=75)
scores, perturbation_visualizations = cam_metric(input_tensor, 
  grayscale_cams, targets, model, return_visualization=True)

# You can also average accross different percentiles, and combine
# (LeastRelevantFirst - MostRelevantFirst) / 2
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirstAverage,ROADLeastRelevantFirstAverage,ROADCombined
cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
scores = cam_metric(input_tensor, grayscale_cams, targets, model)