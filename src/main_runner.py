# %%
# Imports
from config import ds_config
from utils import *
from fastai.vision.all import *
from fastai.callback.tensorboard import TensorBoardCallback

# from fastai.callback.tracker import
from fastai.vision.widgets import *
import os
import matplotlib.pyplot as plt
from IPython.display import Image
import argparse as ap
import datetime
from tqdm import tqdm

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
os.environ["FASTAI_HOME"] = "/media/hdd/Datasets/"


# Monkey patch batch prediction (fastai does not have this by default)
Learner.predict_batch = predict_batch
# %%
ags = ap.ArgumentParser("Additional Arguments for CLI")
ags.add_argument("--config", help="Name of config from dictionary", default="fish_test_proxy")
ags.add_argument("--name", help="Name of the experiment", required=True)
args = ags.parse_args()
ds_meta = ds_config[args.config]  # get info about dataset from the config file

# %%
# Training Part 1
path = Path(ds_meta["ds_path"])
fname_start = f'{ds_meta["ds_name"]}_{args.name}_{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}'  # unique_name
print(f"[INFO] : File name = {fname_start}")

# Check if directories all present
create_if_not_exists(f"tb_runs/{fname_start}")
create_if_not_exists(f"csv_logs/{fname_start}")
# TODO : Add reset folder
#%%
batch_tfms = aug_transforms() if ds_meta["enable_default_augments"] == True else None
fields = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=ds_meta["name_fn"],
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=RandomResizedCrop(ds_meta["image_size"], min_scale=0.5),
    batch_tfms=batch_tfms,
)
# Metrics
metrics = [accuracy, error_rate]
# Callbacks
cbs = [
    TensorBoardCallback(
        log_dir=f"tb_runs/{fname_start}", projector=True, trace_model=False
    ),
    CSVLogger(fname=f"csv_logs/{fname_start}.csv"),
]

#%%
# MAIN LOOP
training_rounds = ds_meta["epoch_steps"]
print(f"[LOG] Training rounds scheme : {training_rounds}")
total_epochs = sum(training_rounds)  # total no of epochs

# Convert [1,2,3] -> [1, 2, 'aug', 3, 'aug']
training_rounds = [
    training_rounds.insert(2 * x, "aug") for x in range(1, len(training_rounds))
]
total_epochs_passed = 0

for i, step in tqdm(enumerate(training_rounds), total=len(training_rounds)):
    if i == 0:
        # Initialize everything with new or old data
        dls = fields.dataloaders(path)
        learn = vision_learner(
            dls, ds_meta["network"], cbs=cbs, metrics=metrics
        ).to_fp16()
        fname_training = f'{ds_meta["ds_name"]}_{args.name}_{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}'  # unique_name
        learn.fine_tune(step)
        learn.save("temp_model")  # saving so can be reloaded
        # Since training is in batches, keep track of total no of epochs trained for
        total_epochs_passed += step

    if step != "aug" and i > 0:
        # Initialize everything with new or old data
        dls = fields.dataloaders(path)
        learn = vision_learner(
            dls, ds_meta["network"], cbs=cbs, metrics=metrics
        ).to_fp16()

        learn.load("temp_model")  # load model since augment has been done already
        # Continue training
        fname_training = f'{ds_meta["ds_name"]}_{args.name}_{datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")}'  # unique_name
        learn.fine_tune(step)
        learn.save("temp_model")
        # Since training is in batches, keep track of total no of epochs trained for
        total_epochs_passed += step
    else:
        # PROXY ATTENTION LOOP

        # Get the classes
        vocab_dict = {
            learn.dls.vocab[x]: x for x in range(len(learn.dls.vocab))
        }  # Get class names
        # Get images, shuffle, pick a subset
        items = get_image_files(ds_meta["ds_path"])
        items = items.shuffle()
        subset = int(ds_meta["change_subset_attention"] * len(items))
        items = items[:subset]
        # Get preds from the network for all the chosen images with "num_workers" threads
        bspred = learn.predict_batch(items, num_workers=10)
        # Get all the class names for the subset of images and convert them into the One hot encoded version that the network knows already
        item_names = list(
            map(lambda x: vocab_dict[x], list(map(ds_meta["name_fn"], items)))
        )

        # Get the index of all the images that the network predicted wrong
        # TODO : Check for confidence
        index_wrongs = [
            x for x in range(subset) if bspred[2][x] != TensorBase(item_names)[x]
        ]
        print(
            f"[INFO] : Pct wrong for step {total_epochs_passed} = {1-len(index_wrongs)/len(bspred[2])}"
        )
        if ds_meta["enable_proxy_attention"] == True:
            print("[INFO]: Running Proxy Attention")

            # RUN PROXY ATTENTION
            for im in tqdm(index_wrongs, total=len(index_wrongs)):
                img = PILImage.create(items[im])

                (x,) = first(dls.test_dl([img]))

                x_dec = TensorImage(dls.train.decode((x,))[0][0])

                # Get attention maps

                # -----
                # Grad CAM Attention map
                cls = 1
                try:
                    with HookBwd(learn.model[-2][4][-1]) as hookg:  # for other layers
                        with Hook(learn.model[-2][4][-1]) as hook:
                            output = learn.model.eval()(x.cuda())
                            act = hook.stored
                        output[0, cls].backward()
                        grad = hookg.stored
                    w = grad[0].mean(dim=[1, 2], keepdim=True)
                    cam_map = (w * act[0]).sum(0)
                    # print(x.shape,x_dec.shape, w.shape, cam_map.shape)

                except Exception as e:
                    print(e)
                # -----

                test_cam_map = cam_map.detach().cpu()
                # Resize cam map so it's the same size as the image, as the output is much smaller
                t_resized = F.resize(
                    torch.unsqueeze(test_cam_map, 0), ds_meta["image_size"]
                )
                t_resized = torch.cat([t_resized, t_resized, t_resized], dim=0)

                # IMPORTANT : Change the pixels that are of higher intensity to 0 because they did not help the network get the right answer
                x_dec[t_resized >= 0.004] = 0.0

                x_dec = torch.einsum("ijk->jki", x_dec)
                plt.imshow(x_dec)
                plt.axis("off")
                plt.savefig(rename_for_aug(items[im]))

    # Save model every n epochs
    if total_epochs_passed % ds_meta["save_model_every_n_epoch"] == 0:
        learn.save(fname_training)
