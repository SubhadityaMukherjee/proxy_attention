#%%
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
# import proxyattention 
#%%
# Set the directory path for the dataset

main_ds_dir = "/home/eragon/Documents/Datasets/"
dataset_info = {
    "asl": {
        "path": Path(f"{main_ds_dir}asl/asl_alphabet_train/asl_alphabet_train"),
        "num_classes": 29,
    },
    "imagenette": {
        "path": Path(f"{main_ds_dir}/imagenette2-320/train"),
        "num_classes": 10,
    },
    "tinyimagenet": {
        "path": Path(f"{main_ds_dir}/tiny-imagenet-200/train"),
        "num_classes": 200,
    },
    "cifar100": {
        "path": Path(f"{main_ds_dir}/CIFAR-100/train"),
        "num_classes": 100,
    },
    "dogs": {
        "path": Path(f"{main_ds_dir}/dogs/images"),
        "num_classes": 120,
    },
    "caltech101": {
        "path": Path(f"{main_ds_dir}/caltech-101"),
        "num_classes": 101,
    },
    "plantdisease": {
        "path": Path(
            f"{main_ds_dir}/plantdisease/Plant_leave_diseases_dataset_without_augmentation"
        ),
        "num_classes": 39,
    },
}
#%%
for name in tqdm(dataset_info.keys(), total = len(dataset_info.keys())):
# for name in ["plantdisease"]:
    ds_name = name
    print(ds_name)
    # break

    # ds_name = "dogs"
    data_dir = dataset_info[ds_name]["path"]

    # Create a dataset object using ImageFolder
    dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        # transform=torchvision.transforms.ToTensor()
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
    )

    # Create a data loader to load the dataset in batches
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # Define the class names
    class_names = dataset.classes

    # Define the ImageNet mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))

    # Denormalize the images
    images = images * std.reshape(1, 3, 1, 1) + mean.reshape(1, 3, 1, 1)

    # Plot the images with labels
    fig, axs = plt.subplots(nrows=4, ncols=8, figsize=(16,8))
    axs = axs.flatten()

    for i in range(len(axs)):
        # Convert the image tensor to a numpy array
        img = images[i].numpy().transpose(1,2,0)

        # Plot the image and label
        axs[i].imshow(img)
        axs[i].axis('off')
        if ds_name == "asl":
            axs[i].set_title(class_names[labels[i]])
            # pass
        if ds_name == "plantdisease":
            axs[i].set_title(class_names[labels[i]][:10])
        else:
            axs[i].set_title(class_names[labels[i]].split("-")[0])

    # plt.show()
    plt.savefig(f"/home/eragon/Documents/Github/proxy_attention/writing/thesis/images/{ds_name}.pdf", transparent = True)

# %%
