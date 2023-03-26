import torch
import typing as t
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights


from time import sleep
import timm
import os

# main_ds_dir = "/media/eragon/HDD/Datasets/"
main_ds_dir = "/run/media/eragon/HDD/Datasets/"
os.environ["TORCH_HOME"] = main_ds_dir
model_chosen = "efficientnet_b0"
# model_chosen = "vgg16"
model = timm.create_model(model_chosen, pretrained = True, num_classes = 256)
print(model)
print(model.conv_head)
# print(model.layer4[-1].conv2)


def get_batch_size(
    model: nn.Module,
    device: torch.device,
    input_shape: t.Tuple[int, int, int],
    output_shape: t.Tuple[int],
    dataset_size: int,
    max_batch_size: int = None,
    num_iterations: int = 5,
) -> int:
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:

            scaler = torch.cuda.amp.GradScaler()
            for _ in tqdm(range(num_iterations), total = num_iterations):

                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(*(batch_size, *output_shape), device=device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = F.mse_loss(targets, outputs)
                # loss.backward()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            batch_size *= 2
        except RuntimeError:
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
    return batch_size


# print(get_batch_size(model, "cuda:0", input_shape = (3, 224, 224), output_shape = (256,), dataset_size = 9000))
