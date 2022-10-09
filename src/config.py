from fastai.vision.all import *
import timm

# proxy_attention is a placeholder name for the proposed augmentation method

ds_config = {
    "fish_test_proxy": {
        "ds_path": "/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/",
        "ds_name": "fish",
        "name_fn": lambda x: str(x).split("/")[-2],
        "image_size": 224,
        "network": resnet18,
        "epoch_steps": [1, 1, 1],  # n0 epochs -> augment -> n1 epochs ...
        "enable_default_augments": True,
        "enable_proxy_attention": True,
        "change_subset_attention": 0.3,  # What % of data should be augmented with proxy attention
        "save_model_every_n_epoch": 3,
    },
    "fish_test_no_proxy": {
        "ds_path": "/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/",
        "ds_name": "fish",
        "name_fn": lambda x: str(x).split("/")[-2],
        "image_size": 224,
        "network": resnet18,
        "epoch_steps": [1, 1, 1],  # n0 epochs -> augment -> n1 epochs ...
        "enable_default_augments": True,
        "enable_proxy_attention": False,
        "change_subset_attention": 0.3,  # What % of data should be augmented with proxy attention
        "save_model_every_n_epoch": 3,
    }

}
