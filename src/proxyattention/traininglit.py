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
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR

# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F


from .data_utils import clear_proxy_images, create_dls, create_folds
from .meta_utils import get_files, save_pickle, read_pickle


# sns.set()

os.environ["TORCH_HOME"] = "/mnt/e/Datasets/"
cudnn.benchmark = True
#%%

#%%
class LitModel(LightningModule):

    # TODO Proxy attention tabular support
    def __init__(self, config, proxy_step, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.config = config

        self.dict_decide_change = {
            "mean": torch.mean,
            "max": torch.max,
            "min": torch.min,
            "halfmax": lambda x: torch.max(x) / 2,
        }

        self.dict_gradient_method = {
            "gradcam": GradCAM,
            "gradcamplusplus": GradCAMPlusPlus,
            "eigencam": EigenCAM,
        }

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

        self.model = self.choose_network()
        self.tfm = transforms.ToPILImage()

        self.input_wrong = []
        self.label_wrong = []
        self.proxy_step = proxy_step

    def select_target_layer(self):
        if self.config["model"] == "resnet18":
            target_layers = [self.model.layer4[-1].conv2]
        elif self.config["model"] == "resnet50":
            target_layers = [self.model.layer4[-1].conv2]
        elif self.config["model"] == "FasterRCNN":
            target_layers = self.model.backbone
        elif self.config["model"] == "VGG" or self.config["model"] == "densenet161":
            target_layers = self.model.features[-1]
        elif self.config["model"] == "mnasnet1_0":
            target_layers = self.model.layers[-1]
        elif self.config["model"] == "ViT":
            target_layers = self.model.blocks[-1].norm1
        elif self.config["model"] == "SwinT":
            target_layers = self.model.layers[-1].blocks[-1].norm1
        else:
            raise ValueError("Unsupported model type!")
        return target_layers

    def choose_network(self):
        # TODO find some models to use from the repo
        # vit_tiny_patch16_224.augreg_in21k_ft_in1k
        if self.config["model"] == "vision_transformer":
            self.config["model"] = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
        # Define the number of classes
        return timm.create_model(
            self.config["model"],
            pretrained=self.config["transfer_imagenet"],
            num_classes=self.config["num_classes"],
        )

    def perform_proxy_step(self, cam , input_wrong):
        
        grads = cam(input_tensor=input_wrong, targets=None)
        grads = torch.Tensor(grads).to("cuda").unsqueeze(1).expand(-1, 3, -1, -1)
        normalized_inps = self.inv_normalize(input_wrong)
        if self.config["pixel_replacement_method"] != "blended":
            return torch.where(
                grads > self.config["proxy_threshold"],
                self.dict_decide_change[self.config["pixel_replacement_method"]](grads),
                normalized_inps,
            )
        else:
            # TODO Fix this
            return torch.where(
                grads > self.config["proxy_threshold"],
                (1 - 0.2 * grads) * normalized_inps,
                normalized_inps,
            )

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.config["global_run_count"] += 1
        if self.proxy_step == True:
            _, preds = torch.max(logits, 1)
            wrong_indices = (y != preds).nonzero()
            self.input_wrong.extend(x[wrong_indices])
            self.label_wrong.extend(y[wrong_indices])

            logging.info("Performing Proxy step")
            print("Performing Proxy step")

            # TODO Save Classwise fraction
            chosen_inds = int(
                np.ceil(self.config["change_subset_attention"] * len(self.label_wrong))
            )
            # TODO some sort of decay?
            # TODO Remove min and batchify
            chosen_inds = min(self.config["batch_size"], chosen_inds)

            self.log("ptl/Number_chosen", chosen_inds)
            logging.info(f"{chosen_inds} images chosen to run proxy on")

            self.input_wrong = self.input_wrong[:chosen_inds]
            self.label_wrong = self.label_wrong[:chosen_inds]

            try:
                self.input_wrong = torch.squeeze(torch.stack(self.input_wrong, dim=1))
                self.label_wrong = torch.squeeze(torch.stack(self.label_wrong, dim=1))
            except:
                self.input_wrong = torch.squeeze(self.input_wrong)
                self.label_wrong = torch.squeeze(self.label_wrong)

            # TODO fix this
            tb_logger = None
            self.logger.experiment.add_image(
                "original_images",
                self.inv_normalize(self.input_wrong),
                self.global_step,
                dataformats="NCHW",
            )
            self.target_layers = self.select_target_layer()
            cam = self.dict_gradient_method[self.config["gradient_method"]](
            model=self.model, target_layers=self.target_layers, use_cuda = True
            )

            self.thresholded_ims = self.perform_proxy_step(cam, self.input_wrong)

            # TODO fix this
            self.logger.experiment.add_image(
                "converted_images",
                self.thresholded_ims,
                self.global_step,
                dataformats="NCHW",
            )

            for ind in tqdm(range(len(self.label_wrong)), total=len(self.label_wrong)):
                label = self.config["label_map"][self.label_wrong[ind].item()]
                save_name = (
                    self.config["ds_path"]
                    / label
                    / f"proxy-{ind}-{self.config['global_run_count']}.png"
                )
                self.tfm(self.thresholded_ims[ind]).save(save_name)

        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.config["batch_size"]
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def evaluate(self, batch, stage=None):
        x, y = batch["x"], batch["y"]
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.config["num_classes"]
        )

        if stage:
            self.log(f"ptl/{stage}_loss", loss, prog_bar=True)
            self.log(f"ptl/{stage}_acc", acc, prog_bar=True)


#%%
def get_last_checkpoint(checkpoint_folder):
    "https://github.com/Lightning-AI/lightning/issues/4176"
    if os.path.exists(checkpoint_folder):
        past_experiments = sorted(
            Path(checkpoint_folder).iterdir(), key=os.path.getmtime
        )

        for experiment in past_experiments[::-1]:
            experiment_folder = os.path.join(experiment, "checkpoints")
            if os.path.exists(experiment_folder):
                checkpoints = os.listdir(experiment_folder)

                if len(checkpoints):
                    checkpoints.sort()
                    path = os.path.join(experiment_folder, checkpoints[-1])
                    return path

    return None


# %%
def setup_train_round(config, proxy_step=False, num_epochs=1, chk=None):

    # TODO Fix global run count
    train, val = create_folds(config)
    image_datasets, dataloaders, dataset_sizes = create_dls(train, val, config)
    class_names = image_datasets["train"].classes
    config["num_classes"] = len(config["label_map"].keys())

    model = LitModel(config, proxy_step, lr=0.05)
    os.makedirs(config["fname_start"], exist_ok=True)

    trainer = Trainer(
        max_epochs=num_epochs,
        default_root_dir=config["checkpoint_path"],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=TensorBoardLogger("tb_logs", name=config["fname_start"]),
        log_every_n_steps=2,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
            TuneReportCallback(
                {"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_acc"},
                on="validation_end",
            ),
            ModelCheckpoint(
                save_last=True,
                monitor="step",
                mode="max",
                dirpath=config["checkpoint_path"],
                filename="full_train",
            ),
            StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        accumulate_grad_batches=4,  # number of batches to accumulate gradients over
        precision=16,  # enables mixed precision training with 16-bit floating point precision
        resume_from_checkpoint=chk,
    )
    model.validation_step_end = None
    model.validation_epoch_end = None

    # try:
    #     chks_list = [
    #         x
    #         for x in os.listdir(
    #             Path(config["checkpoint_path"]) / "version_0/checkpoints"
    #         )
    #         if "ckpt" in x
    #     ]
    #     trainer.fit(
    #         model,
    #         dataloaders["train"],
    #         dataloaders["val"],
    #         ckpt_path=config["checkpoint_path"] / chks_list[-1],
    #     )
    #     print("Restored checkpoint")
    # except:
    if config["chk"] is not None:
        trainer.fit(
            model, dataloaders["train"], dataloaders["val"], ckpt_path=config["chk"]
        )
        trainer.validate(model, dataloaders["val"], ckpt_path=config["chk"])
    else:
        trainer.fit(model, dataloaders["train"], dataloaders["val"])
        trainer.validate(model, dataloaders["val"])


# %%
def train_proxy_steps(config):
    assert torch.cuda.is_available()

    fname_start = f'/mnt/e/CODE/Github/improving_robotics_datasets/src/runs/{config["ds_name"]}_{config["experiment_name"]}+{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}_ps-{str(config["proxy_steps"])}_gradient-{str(config["gradient_method"])}_px-{str(config["pixel_replacement_method"])}-subs-{str(config["change_subset_attention"])}_pt-{str(config["proxy_threshold"])}_cs-{str(config["clear_every_step"])}'

    config["fname_start"] = fname_start
    config["global_run_count"] = 0

    config["checkpoint_path"] = fname_start

    for i, step in enumerate(config["proxy_steps"]):
        chk = get_last_checkpoint(config["checkpoint_path"]) if i > 0 else None
        config["chk"] = chk
        if step == "p":
            # config["load_proxy_data"] = True
            setup_train_round(config=config, proxy_step=True, num_epochs=1 + i, chk=chk)
        else:
            # config["load_proxy_data"] = False
            setup_train_round(
                config=config, proxy_step=False, num_epochs=step + i, chk=chk
            )

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

    reporter = CLIReporter(metric_columns=["ptl/val_acc", "training_iteration"])

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_proxy_steps),
            resources={
                "cpu": config["num_cpu"],
                "gpu": config["num_gpu"],
            },
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
