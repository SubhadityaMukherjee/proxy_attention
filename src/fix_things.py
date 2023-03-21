
import itertools

search_space = {
    "change_subset_attention": [0.8],
    # "model": ["resnet18", "vgg16", "resnet50", "vit_base_patch16_224"],
    "model": ["resnet18", "vgg16", "resnet50"],
    "proxy_image_weight": [0.1],
    "proxy_threshold": [0.85],
    "gradient_method": ["gradcamplusplus"],
    "ds_name": ["asl", "imagenette", "caltech256"],
    # "ds_name": ["caltech256"],
    "clear_every_step": [True],
}

search_space = {
    "change_subset_attention" : [0.8, 0.5, 0.2],
    # "model": ["resnet18", "vgg16", "resnet50", "vit_base_patch16_224"],
    # "model": ["resnet18", "vgg16", "resnet50"],
    "model": ["resnet18"],
    "proxy_image_weight" : [0.1, 0.2, 0.4, 0.8, 0.95],
    "proxy_threshold": [0.85],
    "gradient_method" : ["gradcamplusplus"],
    "ds_name" : ["asl", "imagenette", "caltech256"],
    "clear_every_step": [True, False],
}


# Get all combinations of the values from the search_space dictionary
search_space_values = list(search_space.values())
combinations = list(itertools.product(*search_space_values))
config = {}
for combination in combinations:
    params = dict(zip(search_space.keys(), combination))
    