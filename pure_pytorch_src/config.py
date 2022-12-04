import timm

# proxy_attention is a placeholder name for the proposed augmentation method

def fish_name_fn(x): return str(x).split("/")[-2]

ds_config = {
    "fish_test_proxy": {
        "ds_path": "/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/",
        "ds_name": "fish",
        "name_fn": fish_name_fn,
        "image_size": 224,
        "batch_size": 128,
        "network": resnet18,
        "pretrained": False,
        "epoch_steps": [1, 2],  # n0 epochs -> augment -> n1 epochs ...
        "enable_default_augments": False,
        "enable_proxy_attention": True,
        "change_subset_attention": 0.01,  # What % of data should be augmented with proxy attention
        "save_model_every_n_epoch": 3,
    },
    "fish_test_no_proxy": {
        "ds_path": "/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/",
        "ds_name": "fish",
        "name_fn": fish_name_fn,
        "image_size": 224,
        "batch_size": 128,
        "network": resnet18,
        "pretrained": False,
        "epoch_steps": [1, 2],  # n0 epochs -> augment -> n1 epochs ...
        "enable_default_augments": False,
        "enable_proxy_attention": False,
        "change_subset_attention": 0.1,  # What % of data should be augmented with proxy attention
        "save_model_every_n_epoch": 3,
    },
    "fish_test_proxy_small": {
        "ds_path": "/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/",
        "ds_name": "fish",
        "name_fn": fish_name_fn,
        "image_size": 224,
        "batch_size": 128,
        "network": resnet18,
        "pretrained": False,
        "epoch_steps": [1, 2],  # n0 epochs -> augment -> n1 epochs ...
        "enable_default_augments": False,
        "enable_proxy_attention": True,
        "change_subset_attention": 0.1,  # What % of data should be augmented with proxy attention
        "save_model_every_n_epoch": 3,
    },
    "fish_test_proxy_no_train": {
        "ds_path": "/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/",
        "ds_name": "fish",
        "name_fn": fish_name_fn,
        "image_size": 224,
        "batch_size": 128,
        "network": resnet18,
        "pretrained": False,
        "epoch_steps": [0, 1],  # n0 epochs -> augment -> n1 epochs ...
        "enable_default_augments": False,
        "enable_proxy_attention": True,
        "change_subset_attention": 0.1,  # What % of data should be augmented with proxy attention
        "save_model_every_n_epoch": 3,
    },
    "asl_test_proxy_train": {
        "ds_path": "/media/hdd/Datasets/asl/asl_alphabet_train",
        "ds_name": "asl",
        "name_fn": fish_name_fn,
        "image_size": 64,
        "batch_size": 256,
        "network": resnet18,
        "pretrained": False,
        "epoch_steps": [1, 1],  # n0 epochs -> augment -> n1 epochs ...
        "enable_default_augments": False,
        "enable_proxy_attention": True,
        "change_subset_attention": 0.1,  # What % of data should be augmented with proxy attention
        "save_model_every_n_epoch": 3,
    },
    "asl_test_no_proxy_train": {
        "ds_path": "/media/hdd/Datasets/asl/asl_alphabet_train",
        "ds_name": "asl",
        "name_fn": fish_name_fn,
        "image_size": 64,
        "batch_size": 256,
        "network": resnet18,
        "pretrained": False,
        "epoch_steps": [1, 1],  # n0 epochs -> augment -> n1 epochs ...
        "enable_default_augments": False,
        "enable_proxy_attention": False,
        "change_subset_attention": 0.1,  # What % of data should be augmented with proxy attention
        "save_model_every_n_epoch": 3,
    },
    "asl_test_proxy_train_50": {
        "ds_path": "/media/hdd/Datasets/asl/asl_alphabet_train",
        "ds_name": "asl",
        "name_fn": fish_name_fn,
        "image_size": 64,
        "batch_size": 256,
        "network": resnet18,
        "pretrained": False,
        "epoch_steps": [1, 1],  # n0 epochs -> augment -> n1 epochs ...
        "enable_default_augments": False,
        "enable_proxy_attention": True,
        "change_subset_attention": .5,  # What % of data should be augmented with proxy attention
        "save_model_every_n_epoch": 3,
    },


}
