# %%
# Imports

import copy
import datetime
import logging
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no UI backend
import pickle

import numpy as np
import ray
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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from .data_utils import clear_proxy_images, create_dls, create_folds
from .meta_utils import get_files, save_pickle, read_pickle

# sns.set()

os.environ["TORCH_HOME"] = "/mnt/e/Datasets/"
cudnn.benchmark = True

# %%
#TODO Add a "weighted method"

dict_decide_change = {
    "mean": torch.mean,
    "max": torch.max,
    "min": torch.min,
    "halfmax": lambda x: torch.max(x) / 2,
}



#TODO Smoothing maybe?
dict_gradient_method = {
        "gradcam" : GradCAM,
        "gradcamplusplus" : GradCAMPlusPlus,
        "eigencam" : EigenCAM,
    }    

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def reset_params(model):
    for param in model.parameters():
        param.requires_grad = False

#TODO find some models to use from the repo
def choose_network(config):
    # vit_tiny_patch16_224.augreg_in21k_ft_in1k
    if config["model"] == "vision_transformer":
        config["model"] = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    # Define the number of classes
    model = timm.create_model(
        config["model"],
        pretrained=config["transfer_imagenet"],
        num_classes=config["num_classes"],
    ).to(config["device"])
    model.train()
    return model
#%%

def perform_proxy_step(cam, input_wrong, config):
    grads = cam(input_tensor=input_wrong, targets=None)
    grads = (
        torch.Tensor(grads).to("cuda").unsqueeze(1).expand(-1, 3, -1, -1)
    )
    normalized_inps = inv_normalize(input_wrong)
    if config["pixel_replacement_method"] != "blended":
        return torch.where(
            grads > config["proxy_threshold"],
            dict_decide_change[config["pixel_replacement_method"]](grads),
            normalized_inps,
        )
    else:
        #TODO Fix this
        return torch.where(
            grads > config["proxy_threshold"],
            (1- 0.2 * grads) * normalized_inps,
            normalized_inps,
        )
        # return torch.clamp(torch.where(
        #     grads > config["proxy_threshold"],
        #     grads * normalized_inps,
        #     normalized_inps,
        # ), max = 255.0, min = 0.0)



# %%
# TODO Proxy attention tabular support


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
    writer = SummaryWriter(log_dir=config["fname_start"], comment=config["fname_start"])

    # Save config info to tensorboard
    for key, value in config.items():
        writer.add_text(key, str(value))
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    pbar = tqdm(range(config["global_run_count"], config["global_run_count"]+num_epochs), total=num_epochs)

    if config["model"] == "resnet18":
        target_layers = [model.layer4[-1].conv2]
    elif config["model"] == "resnet50":
        target_layers = [model.layer4[-1].conv2]
    elif config["model"] == "FasterRCNN":
        target_layers = model.backbone
    elif config["model"] == "VGG" or config["model"] == "densenet161":
        target_layers = model.features[-1]
    elif config["model"] == "mnasnet1_0":
        target_layers = model.layers[-1]
    elif config["model"] == "ViT":
        target_layers = model.blocks[-1].norm1
    elif config["model"] == "SwinT":
        target_layers = model.layers[-1].blocks[-1].norm1
    else:
        raise ValueError("Unsupported model type!")



    cam = dict_gradient_method[config["gradient_method"]](model=model, target_layers=target_layers, use_cuda=True)

    tfm = transforms.ToPILImage()
    for epoch in pbar:
        config["global_run_count"] += 1

        # Each epoch has a training and validation phase
        input_wrong = []
        label_wrong = []
        for phase in ["train", "val"]:
            logging.info(f"[INFO] Phase = {phase}")
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
                            logging.info("[INFO] : Proxy")
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

            if proxy_step == True and phase == "train":
                logging.info("Performing Proxy step")
                print("Performing Proxy step")
                # TODO Save Classwise fraction
                chosen_inds = int(np.ceil(config["change_subset_attention"] * len(label_wrong)))
                # TODO some sort of decay?
                # TODO Remove min and batchify
                chosen_inds = min(config["batch_size"], chosen_inds)

                writer.add_scalar(
                    "Number_Chosen", chosen_inds, config["global_run_count"]
                )
                logging.info(f"{chosen_inds} images chosen to run proxy on")

                input_wrong = input_wrong[:chosen_inds]
                label_wrong = label_wrong[:chosen_inds]

                try:
                    input_wrong = torch.squeeze(torch.stack(input_wrong, dim=1))
                    label_wrong = torch.squeeze(torch.stack(label_wrong, dim=1))
                except:
                    input_wrong = torch.squeeze(input_wrong)
                    label_wrong = torch.squeeze(label_wrong)

                writer.add_images(
                    "original_images",
                    inv_normalize(input_wrong),
                    # input_wrong,
                    config["global_run_count"],
                )

                # save_pickle((cam, input_wrong, config,tfm))
                
                # TODO run over all the batches
                thresholded_ims= perform_proxy_step(cam, input_wrong, config)

                logging.info("[INFO] Ran proxy step")
                writer.add_images(
                    "converted_proxy",
                    thresholded_ims,
                    config["global_run_count"],
                )

                logging.info("[INFO] Saving the images")

                for ind in tqdm(range(len(label_wrong)), total=len(label_wrong)):
                    label = config["label_map"][label_wrong[ind].item()]
                    save_name = (
                        config["ds_path"]
                        / label
                        / f"proxy-{ind}-{config['global_run_count']}.png"
                    )
                    tfm(thresholded_ims[ind]).save(save_name)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            pbar.set_postfix({"Phase": phase, "Loss": epoch_loss, "Acc": epoch_acc})

            # TODO Add more loss functions
            # TODO Classwise accuracy
            if proxy_step == True:
                writer.add_scalar("proxy_step", True, config["global_run_count"])
            else:
                writer.add_scalar("proxy_step", False, config["global_run_count"])

            if phase == "train":
                writer.add_scalar("Loss/Train", epoch_loss, config["global_run_count"])
                writer.add_scalar("Acc/Train", epoch_acc, config["global_run_count"])
            if phase == "val":
                writer.add_scalar("Loss/Val", epoch_loss, config["global_run_count"])
                writer.add_scalar("Acc/Val", epoch_acc, config["global_run_count"])
                with tune.checkpoint_dir(config["global_run_count"]) as checkpoint_dir:
                    save_path = Path(config["fname_start"]) / "checkpoint"
                    torch.save((model.state_dict(), optimizer.state_dict()), save_path)

                tune.report(loss=epoch_loss, accuracy=epoch_acc)

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logging.info(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# %%
#TODO Better transfer learning params. more trainable layers
def setup_train_round(config, proxy_step=False, num_epochs=1):
    # Data part
    train, val = create_folds(config)
    image_datasets, dataloaders, dataset_sizes = create_dls(train, val, config)
    class_names = image_datasets["train"].classes
    config["num_classes"] = len(config["label_map"].keys())

    model_ft = choose_network(config)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #TODO Fix this for tranasfer learning . Reduce rate
    # if config["transfer_imagenet"] == True:
        # optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-5)
    # else:
        # optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-4)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)

    # Decay LR 
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, verbose = True)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, len(dataloaders["train"]))
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

    fname_start = f'/mnt/e/CODE/Github/improving_robotics_datasets/src/runs/{config["ds_name"]}_{config["experiment_name"]}+{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}_ps-{str(config["proxy_steps"])}_gradient-{str(config["gradient_method"])}_px-{str(config["pixel_replacement_method"])}-subs-{str(config["change_subset_attention"])}_pt-{str(config["proxy_threshold"])}_cs-{str(config["clear_every_step"])}'

    config["fname_start"] = fname_start
    config["global_run_count"] = 0
    

    for step in config["proxy_steps"]:
        if step == "p":
            # config["load_proxy_data"] = True
            setup_train_round(config=config, proxy_step=True, num_epochs=1)
        else:
            # config["load_proxy_data"] = False
            setup_train_round(config=config, proxy_step=False, num_epochs=step)

        if config["clear_every_step"] == True:
            clear_proxy_images(config=config)  # Clean directory

    if config["clear_every_step"] == False:
        clear_proxy_images(config=config)  # Clean directory


def hyperparam_tune(config):
    ray.init()
    scheduler = ASHAScheduler(
        max_t=30,
        grace_period=10,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_proxy_steps),
            resources={"cpu": config["num_cpu"], "gpu": config["num_gpu"],},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            # search_alg=OptunaSearch(),
            max_concurrent_trials=5,
        ),
        run_config=ray.air.RunConfig(progress_reporter=reporter),
        param_space=config,
    )
    result = tuner.fit()


    df_res = result.get_dataframe()
    df_res.to_csv(Path(config["fname_start"] + "result_log.csv"))
    # best_trial = result.get_best_result("loss", "min", "last")
