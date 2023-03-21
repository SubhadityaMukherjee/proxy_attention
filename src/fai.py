#%%
from fastai.vision.all import *
from fastai.callback.all import GradientAccumulation, MixedPrecision, ProgressCallback
from fastai.callback.tensorboard import TensorBoardCallback
import logging
import proxyattention
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

from torchvision import transforms
from tqdm import tqdm
#%%
config = {
    "experiment_name" : "baseline_run",
    "image_size" : 224,
    "batch_size" : 32,
    "enable_proxy_attention" : True,
    "transfer_imagenet" : True,
    "subset_images" : 200,
    "pixel_replacement_method" : "blended",
    "proxy_steps" : [1, "p"],
    "clear_every_step" : True,
    "load_proxy_data" : False,
    "proxy_step" : True,

    # Switch to grid
    "change_subset_attention" : 0.8,
    # "model": ["resnet18", "vgg16", "resnet50", "vit_base_patch16_224"],
    "model": "resnet18",
    "proxy_image_weight" : 0.1,
    "proxy_threshold": 0.85,
    "gradient_method" : "gradcamplusplus",
    "ds_name" : "imagenette"
}
#%%

computer_choice = "pc"
# pc, cluster

# Make dirs
if computer_choice == "pc":
    main_run_dir = "/mnt/e/CODE/Github/improving_robotics_datasets/src/runs/"
    main_ds_dir = "/mnt/e/Datasets/"
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["TORCH_HOME"] = main_ds_dir
dataset_info = {
    "asl": {"path": Path(f"{main_ds_dir}asl/asl_alphabet_train/asl_alphabet_train") , "name_fn": parent_label},
    "imagenette": {"path": Path(f"{main_ds_dir}/imagenette2-320/train") , "name_fn": parent_label},
    "caltech256": {"path": Path(f"{main_ds_dir}/caltech256") , "name_fn": parent_label},
}

logging.info("Directories made/checked")
os.makedirs(main_run_dir, exist_ok=True)

config["dataset_info"] = dataset_info
config["main_run_dir"] = main_run_dir


config["ds_path"] = dataset_info[config["ds_name"]]["path"]
all_files = get_image_files(config["ds_path"])
random.shuffle(all_files)
if config["subset_images"] is not None:
    all_files = all_files[: config["subset_images"]]
if config["load_proxy_data"] is False:
    all_files = [x for x in all_files if "proxy" not in str(x)]

#%%
parent_label(get_image_files(config["ds_path"])[0])

# %%
def find_target_layer(config, learn):
    model = learn.model[0].model
    if config["model"] == "resnet18":
        return [model.layer4[-1].conv2]
    elif config["model"] == "resnet50":
        return [model.layer4[-1].conv2]
    elif config["model"] == "FasterRCNN":
        return model.backbone
    elif config["model"] == "vgg16" or config["model"] == "densenet161":
        return [model.features[-3]]
    elif config["model"] == "mnasnet1_0":
        return model.layers[-1]
        # elif config["model"] == "vit_small_patch32_224":
        return [model.norm]
    elif config["model"] == "vit_base_patch16_224":
        return [model.norm]
        # target_layers = model.layers[-1].blocks[-1].norm1
    else:
        raise ValueError("Unsupported model type!")

dict_gradient_method = {
    "gradcam": GradCAM,
    "gradcamplusplus": GradCAMPlusPlus,
    "eigencam": EigenCAM,
}



inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)
dict_decide_change = {
    "mean": torch.mean,
    "max": torch.max,
    "min": torch.min,
    "halfmax": lambda x: torch.max(x) / 2,
}

#%%
def proxy_one_batch(config, input_wrong):
    grads = config["cam"](input_tensor=input_wrong, targets=None)
    grads = torch.Tensor(grads).to("cuda").unsqueeze(1).expand(-1, 3, -1, -1)
    normalized_inps = inv_normalize(input_wrong)
    if config["pixel_replacement_method"] != "blended":
        return torch.where(
            grads > config["proxy_threshold"],
            dict_decide_change[config["pixel_replacement_method"]](grads),
            normalized_inps,
        )
    else:
        return torch.where(
            grads > config["proxy_threshold"],
            (1 - config["proxy_image_weight"] * grads) * normalized_inps,
            normalized_inps,
        )
#%%
dls = ImageDataLoaders.from_lists(
path = config["ds_path"],
    fnames = all_files,
    labels= [proxyattention.get_parent_name(x) for x in all_files],
    item_tfms=Resize(224),

)
#%%
dls.vocab
#%%
dls.c
#%%
def proxy_callback(config, all_wrong_input, all_wrong_input_label):
    logging.info("Performing Proxy step")

    # TODO Save Classwise fraction
    chosen_inds = int(np.ceil(config["change_subset_attention"] * len(all_wrong_input_label)))
    # TODO some sort of decay?
    # TODO Remove min and batchify
    # chosen_inds = min(config["batch_size"], chosen_inds)

    config["writer"].add_scalar(
        "Number_Chosen", chosen_inds, config["global_run_count"]
    )
    logging.info(f"{chosen_inds} images chosen to run proxy on")

    input_wrong = all_wrong_input[:chosen_inds]
    label_wrong = all_wrong_input_label[:chosen_inds]

    try:
        input_wrong = torch.squeeze(torch.stack(input_wrong, dim=1))
        label_wrong = torch.squeeze(torch.stack(label_wrong, dim=1))
    except:
        input_wrong = torch.squeeze(input_wrong)
        label_wrong = torch.squeeze(label_wrong)

    config["writer"].add_images(
        "original_images",
        inv_normalize(input_wrong),
        # input_wrong,
        config["global_run_count"],
    )

    # save_pickle((cam, input_wrong, config,tfm))

    # TODO run over all the batches
    thresholded_ims = proxy_one_batch(config, input_wrong)

    # logging.info("[INFO] Ran proxy step")
    config["writer"].add_images(
        "converted_proxy",
        thresholded_ims,
        config["global_run_count"],
    )

    # logging.info("[INFO] Saving the images")
    tfm = transforms.ToPILImage()

    for ind in tqdm(range(len(input_wrong)), total=len(input_wrong)):
        label = dls.vocab[label_wrong[ind].item()]
        save_name = (
            config["ds_path"] / label / f"proxy-{ind}-{config['global_run_count']}.png"
        )
        tfm(thresholded_ims[ind]).save(save_name)


class ProxyCallback(Callback):

    def before_batch(self):
        config["writer"] = self.learn.tensor_board.writer
        self.input_wrong = []
        self.label_wrong = []

    def after_batch(self):
        if self.training == True:
            wrong_indices = (self.learn.yb[0] != torch.max(self.learn.pred, 1)[1]).nonzero()

            self.input_wrong.extend(self.learn.xb[0][wrong_indices])
            self.label_wrong.extend(self.learn.yb[0][wrong_indices])

            proxy_callback(config, self.input_wrong, self.label_wrong)
            if config["proxy_step"] == True:
                config["writer"].add_scalar(
                    "proxy_step", True, config["global_run_count"]
                )
            else:
                config["writer"].add_scalar(
                    "proxy_step", False, config["global_run_count"]
                )

#%%
callbacks = [MixedPrecision, ProgressCallback, TensorBoardCallback(trace_model=False), ProxyCallback]
# callbacks = [MixedPrecision, ProgressCallback, TensorBoardCallback(trace_model=False)]
learn = vision_learner(
    dls,
    config["model"],
    metrics=error_rate,
    pretrained=config["transfer_imagenet"],
    cbs=callbacks,
)
#%%
target_layers = find_target_layer(config, learn)
config["cam"] = dict_gradient_method[config["gradient_method"]](
        model=learn.model, target_layers=target_layers, use_cuda=True
    )

#%%
config["global_run_count"] = 0
num_epochs = 3
learn.fit(num_epochs)
config["global_run_count"] += num_epochs
# %%
