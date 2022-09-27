#%% In[2]:
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import config_context
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from IPython.display import display
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from pytorch_lightning.loggers import TensorBoardLogger

sn.set()

#%%
## Augmentation

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            RandomHorizontalFlip(p=0.75),
            RandomChannelShuffle(p=0.75),
            RandomThinPlateSpline(p=0.75),
        )

        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out

class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.0



#%%
## Pipeline

class LogParameters(pl.Callback):
    # weight and biases to tensorbard
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        self.d_parameters = {}
        for n,p in pl_module.named_parameters():
            self.d_parameters[n] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking: # WARN: sanity_check is turned on by default
            lp = []
            for n,p in pl_module.named_parameters():
                trainer.logger[0].experiment.add_histogram(n, p.data, trainer.current_epoch)
                # TODO add histogram to wandb too
                self.d_parameters[n].append(p.ravel().cpu().numpy())
                lp.append(p.ravel().cpu().numpy())
            p = np.concatenate(lp)
            trainer.logger[0].experiment.add_histogram('Parameters', p, trainer.current_epoch)
            # TODO add histogram to wandb too

class ModelPipeLine(LightningModule):
    def __init__(self, chosen_config):
        super().__init__()
        self.chosen_config = chosen_config
        self.config = self.create_config()
        self.model = self.config["model"]
        self.preprocess = Preprocess()  # per sample transforms
        self.transform = DataAugmentation()  # per batch augmentation_kornia
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y_hat, y):
        return self.config["loss"](y_hat, y)

    def show_batch(self, win_size=(10, 10)):
        def _to_vis(data):
            return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

        # get a batch from the training set: try with `val_datlaoader` :)
        imgs, labels = next(iter(self.train_dataloader()))
        imgs_aug = self.transform(imgs)  # apply transforms
        # use matplotlib to visualize
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs))
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs_aug))

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training:
            x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.train_accuracy.update(y_hat, y)
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", self.train_accuracy, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.val_accuracy.update(y_hat, y)
        self.log("valid_loss", loss, prog_bar=False)
        self.log("valid_acc", self.val_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config["optimizer"](self.model.parameters(), lr=1e-4)
        scheduler = self.config["scheduler"](optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]

    def prepare_data(self):
        CIFAR10(os.getcwd(), train=True, download=True, transform=self.preprocess)
        CIFAR10(os.getcwd(), train=False, download=True, transform=self.preprocess)

    def train_dataloader(self):
        dataset = CIFAR10(os.getcwd(), train=True, download=True, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=self.config["train_bs"], num_workers=12)
        return loader

    def val_dataloader(self):
        dataset = CIFAR10(os.getcwd(), train=False, download=True, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=self.config["val_bs"], num_workers=12)
        return loader
    
    def create_config(self):
        dict_configurations = {
            "resnet18_base_test": {
                "model" : torchvision.models.resnet18(pretrained=True),
                "loss" : F.cross_entropy,
                "optimizer" : torch.optim.AdamW,
                "scheduler" : torch.optim.lr_scheduler.CosineAnnealingLR,
                "train_bs" : 256,
                "val_bs" : 256,
            },
        }

        return dict_configurations[self.chosen_config]

#%%
## Training
config_chooser = "resnet18_base_test"
model = ModelPipeLine(config_chooser)
# model.show_batch(win_size=(5, 5))
#%%

trainer = Trainer(
    callbacks=[TQDMProgressBar(refresh_rate=20), ModelCheckpoint("../logs/models")],
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  
    max_epochs=4,
    logger=[CSVLogger(save_dir="../logs/"),TensorBoardLogger("../tb_logs", name=config_chooser, log_graph=True)],
    precision=16,
)

#%%
trainer.fit(model)

# #%%

# metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
# del metrics["step"]
# metrics.set_index("epoch", inplace=True)
# display(metrics.dropna(axis=1, how="all").head())
# sn.relplot(data=metrics, kind="line")