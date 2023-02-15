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
import timm
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
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import pickle

sns.set()

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
cudnn.benchmark = True

# %%


def reset_params(model):
    for param in model.parameters():
        param.requires_grad = False


def choose_network(config):
    # vit_tiny_patch16_224.augreg_in21k_ft_in1k
    if config["model"]== "vision_transformer":
        config["model"]= "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    # Define the number of classes
    model = timm.create_model(
        config["model"], pretrained=config["transfer_imagenet"], num_classes=config["num_classes"]).to(config["device"])
    model.train()
    return model


# %%
# TODO Proxy attention tabular support

def decide_pixel_replacement(original_image, method="mean"):
    if method == "mean":
        return original_image.mean()
    elif method == "max":
        return original_image.max()
    elif method == "min":
        return original_image.min()
    elif method == "black":
        return 0.0
    elif method == "white":
        return 255.0


# def calc_grad_threshold(obj):
#     return obj.mean(axis=2) > config["proxy_threshold"].sample()


def permute_and_detach(obj):
    return obj.permute(1, 2, 0).cpu().detach()


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=25,
    proxy_step=False,
    config=None,
):
    writer = SummaryWriter(log_dir=config["fname_start"]+str(config["global_run_count"]), comment=config["fname_start"])
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    pbar = tqdm(range(num_epochs), total=num_epochs)
    for epoch in pbar:
        # print(f'Epoch {epoch}/{num_epochs - 1}')
        # print('-' * 10)

        # Each epoch has a training and validation phase
        input_wrong = []
        label_wrong = []
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            optimizer.zero_grad(set_to_none=True)
            scaler = torch.cuda.amp.GradScaler()
            for inps in tqdm(
                dataloaders[phase], total=len(dataloaders[phase]), leave=False
            ):
                inputs = inps["x"].to(config["device"], non_blocking=True)
                labels = inps["y"].to(config["device"], non_blocking=True)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    with torch.cuda.amp.autocast():
                        if phase == "train":
                            outputs = model(inputs)
                        else:
                            with torch.no_grad():
                                outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if proxy_step == True and phase == "train":
                            print("[INFO] : Proxy")
                            logging.info("Proxy")
                            wrong_indices = (labels != preds).nonzero()
                            # input_wrong = input_wrong.stack(inputs[wrong_indices])
                            input_wrong.extend(inputs[wrong_indices])
                            label_wrong.extend(labels[wrong_indices])

                    # backward + optimize only if in training phase
                    if phase == "train":
                        scaler.scale(loss).backward()
                        # optimizer.step()

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix(
                    {
                        "Phase": "running",
                        "Loss": running_loss / dataset_sizes[phase],
                        # 'Acc' : running_corrects.double() / dataset_sizes[phase],
                    }
                )

            if phase == "train":
                scheduler.step()

            if phase == "train" and proxy_step == True:
                print("Performing Proxy step")
                # TODO Save Classwise fraction
                frac_choose = 0.25
                chosen_inds = int(np.ceil(frac_choose * len(label_wrong)))
                #TODO some sort of decay?
                #TODO Conver to batches to run over more
                chosen_inds = min(50, chosen_inds)

                writer.add_scalar("Number_Chosen", chosen_inds, epoch)
                print(f"{chosen_inds} images chosen to run proxy on")

                print(len(input_wrong) , len(label_wrong))
                input_wrong = input_wrong[:chosen_inds]
                input_wrong = torch.squeeze(torch.stack(input_wrong, dim=0))
                label_wrong = label_wrong[:chosen_inds]
                label_wrong = torch.squeeze(torch.stack(label_wrong, dim=1))
                # label_wrong = label_wrong.expand(-1, 2)
                # print(len(label_wrong), label_wrong[0].size(), torch.cat(input_wrong,axis = 0).size())

                saliency = Saliency(model)
                # TODO Other methods
                # print(torch.cat(tuple(input_wrong)).shape)
                # print(torch.cat(tuple(label_wrong)).shape)
                # print(torch.cat(input_wrong, dim = 1).shape)
                print(input_wrong.size() , label_wrong.size())
                grads = saliency.attribute(
                    input_wrong, label_wrong
                )
                grads = np.transpose(
                    grads.squeeze().cpu().detach().numpy(), (0, 2, 3, 1)
                )

                print("Calculating permutes and sending to CPU")
                # TODO replace these with direct array operations?
                # original_images = [
                # permute_and_detach(ind)
                # for ind in tqdm(input_wrong, total=len(input_wrong))
                # ]
                print(input_wrong.size())
                original_images = [
                    ind.permute(1, 2, 0).cpu().detach() for ind in input_wrong
                ]

                print("Calculating pixel replacement method")
                # pixel_replacement = [
                #     decide_pixel_replacement(x, config["pixel_replacement_method"])
                #     for x in tqdm(original_images, total=len(original_images))
                # ]

                print("Calculating gradient thresholds")
                # grad_thresholds = [
                #     calc_grad_threshold(grad) for grad in tqdm(grads, total=len(grads))
                # ]

                for ind in tqdm(range(len(label_wrong)), total=len(label_wrong)):
                    # original_images[ind][grad_thresholds[ind]] = pixel_replacement[ind]
                    # TODO Split these into individual comprehensions for speed
                    # TODO Check if % of image is gone or not
                    original_images[ind][
                        grads[ind].mean(axis=2) > config["proxy_threshold"]
                    ] = decide_pixel_replacement(
                        original_image=original_images[ind],
                        method=config["pixel_replacement_method"],
                    )
                original_images = [np.uint8(x) for x in original_images]

                with open("testorig.pkl", "wb+") as f:
                    pickle.dump(original_images, f)
                # original_images_tensor = torch.from_numpy(original_images)
                # img_grid = torchvision.utils.make_grid(original_images_tensor[:max(16, len(original_images))])
                # writer.add_image('batch_vis', img_grid)

                print("Saving the images")
                cm = plt.get_cmap("viridis")

                # TODO Fix image colors xD
                for ind in tqdm(range(len(label_wrong)), total=len(label_wrong)):
                    plt.imshow(original_images[ind])
                    plt.axis("off")
                    plt.gca().set_axis_off()
                    plt.margins(x=0)
                    plt.autoscale(False)
                    label = config["label_map"][label_wrong[ind].item()]
                    save_name = config["ds_path"] / label / f"proxy-{ind}-{epoch}.png"

                    # data = cm(np.uint8(original_images[ind])*255)
                    # # Image.fromarray(data).save(save_name)
                    # save_image(original_images[ind], save_name)
                    # Image.fromarray(cm(((original_images[ind]) * 255).astype(np.uint8))).save(save_name)
                    plt.savefig(save_name, bbox_inches="tight", pad_inches=0)
                    # TODO Copy a batch to the results

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            pbar.set_postfix({"Phase": phase, "Loss": epoch_loss, "Acc": epoch_acc})

            # TODO Add more loss functions
            # TODO Classwise accuracy
            if proxy_step == True:
                writer.add_scalar("proxy_step", True)
            else:
                writer.add_scalar("proxy_step", False)

            if phase == "train":
                writer.add_scalar("Loss/Train", epoch_loss, epoch)
                writer.add_scalar("Acc/Train", epoch_acc, epoch)
            if phase == "val":
                writer.add_scalar("Loss/Val", epoch_loss, epoch)
                writer.add_scalar("Acc/Val", epoch_acc, epoch)
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    save_path = Path(config["fname_start"] + str(config["global_run_count"])) / "checkpoint"
                    torch.save((model.state_dict(), optimizer.state_dict()), save_path)

                tune.report(loss=epoch_loss, accuracy=epoch_acc)

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# %%
def setup_train_round(config, proxy_step=False, num_epochs=1):
    # Data part
    #TODO Configure data for proxy attention
    train, val = create_folds(config)
    image_datasets, dataloaders, dataset_sizes = create_dls(
        train, val, config
    )
    class_names = image_datasets["train"].classes
    config["num_classes"] = len(config["label_map"].keys())

    model_ft = choose_network(config)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    trained_model = train_model(
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=num_epochs,
        config=config,
        proxy_step=proxy_step,
    )


def train_proxy_steps(config):
    assert torch.cuda.is_available()
    config["global_run_count"] = 0
    for step in config["proxy_steps"]:
        if step == "p":
            setup_train_round(config=config, proxy_step=True, num_epochs=1)
            config["load_proxy_data"] = True
        else:
            setup_train_round(config=config, proxy_step=False, num_epochs=step)
            config["load_proxy_data"] = False
        config["global_run_count"] += 1

def tune_func(config):
    # tune.utils.wait_for_gpu(target_util = .1)
    train_proxy_steps(config=config)

def hyperparam_tune(config):
    ray.init(num_gpus=1, num_cpus=12)
    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=30, grace_period=1, reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        tune_func,
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
        max_failures=100,
        num_samples=50,
        resources_per_trial={
            "gpu": 1,
            "cpu": 8,
        },
        local_dir=config["fname_start"],
    )

    df_res = result.get_dataframe()
    df_res.to_csv(Path(config["fname_start"]+str(config["global_run_count"])) / "result_log.csv")
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
        )
    )

    print(result)

#%%
with open("/mnt/e/CODE/Github/improving_robotics_datasets/pure_pytorch_src/runs/asl_test_asl_starter+15022023_13:56:59_subset-8000/train_proxy_steps_2023-02-15_13-57-04/train_proxy_steps_3ed6f_00000_0_proxy_steps=p_1_2023-02-15_13-57-04/testorig.pkl", "rb+") as f:
    orig = pickle.load(f)
    saliency, grads, input_wrong, label_wrong, original_images = orig
#%%
grads_test = saliency.attribute(
                    input_wrong, label_wrong
                )
#%%
grads_test.shape
#%%
grads_test.max(), grads_test.min(), grads_test.mean()
#%%
grads_test[2].mean(axis=0).shape
#%%
part_orig = original_images.copy()
#%%
for ind in tqdm(range(len(label_wrong)), total=len(label_wrong)):
    # original_images[ind][grad_thresholds[ind]] = pixel_replacement[ind]
    # TODO Split these into individual comprehensions for speed
    # TODO Check if % of image is gone or not
    part_orig[ind][
        grads[ind] > 0.008
    ] = 255.0

#%%
inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.485/0.229, -0.485/0.229],
        std=[1/0.229, 1/0.229, 1/0.229]
    )

orig2 = torch.Tensor(np.stack(part_orig)).permute(0, 3,1,2)
orig2.shape
#%%

inv_normalize(orig2)
#%%
orig2[1] = inv_normalize(orig2[1])
#%%
orig2[1].shape
#%%
# orig2 = np.uint8(orig2[1])
# orig2[1].min(), orig2[0].max()
#%%
# plt.imshow(orig2[1].transpose(1,2))
# plt.axis("off")
# plt.gca().set_axis_off()
# plt.margins(x=0)
# plt.autoscale(False)
#%%
def show(imgs):
    import torchvision.transforms.functional as F
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        # img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# %%
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
#%%
# np.stack(orig).shape
#%%
# resize the images to (3, 224, 224)
# orig2 = np.stack(orig)  # reshape to (50, 3, 224, 224)
# print(orig2.shape)
# orig_grid = make_grid(orig2)  # create a grid of images with 10 columns
# print(orig_grid.shape)
# create a summary writer to log the images
writer = SummaryWriter()

# log the grid of images to TensorBoard
writer.add_images('orig_images', orig2, global_step=0, dataformats='NCHW')

# close the summary writer
writer.close()
# %%
