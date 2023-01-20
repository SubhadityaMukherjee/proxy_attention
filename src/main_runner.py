# %%
# Imports
import argparse as ap
import datetime
import gc
import os

import matplotlib.pyplot as plt
import torchvision.transforms.functional as transformF
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.vision.all import *
# from fastai.callback.tracker import
from fastai.vision.widgets import *
from IPython.display import Image
from tqdm import tqdm

from config import ds_config
from utils import *  # import utils at the end after fastai because the Hook function is a monkey-patch

# from torch.multiprocessing import set_start_method
# set_start_method('forkserver')

# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass


os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
os.environ["FASTAI_HOME"] = "/media/hdd/Datasets/"

# set_start_method('spawn')
# Monkey patch batch prediction (fastai does not have this by default)
Learner.predict_batch = predict_batch
# %%
ags = ap.ArgumentParser("Additional Arguments for CLI")
ags.add_argument(
    "--config", help="Name of config from dictionary", default="fish_test_proxy"
)
ags.add_argument("--name", help="Name of the experiment", required=True)
args = ags.parse_args()
ds_meta = ds_config[args.config]  # get info about dataset from the config file

if __name__ == "__main__":

    # %%
    # Training Part 1
    path = Path(ds_meta["ds_path"])
    fname_start = f'{ds_meta["ds_name"]}_{args.name}_{datetime.now().strftime("%d%m%Y_%H:%M:%S")}'  # unique_name
    print(f"[INFO] : File name = {fname_start}")

    # Check if directories all present
    create_if_not_exists(f"tb_runs/{fname_start}")
    create_if_not_exists(f"csv_logs/{fname_start}")
    # Remove previous files

    all_files = get_image_files(path)
    [Path.unlink(file) for file in all_files if "augmented_" in file.name]
    print("[INFO] : Removed Old augmented files")

    # TODO : Add reset folder
    #%%
    batch_tfms = (
        aug_transforms() if ds_meta["enable_default_augments"] == True else None
    )
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

    #%%
    # MAIN LOOP
    training_rounds = ds_meta["epoch_steps"]
    total_epochs = sum(training_rounds)  # total no of epochs

    # Convert [1,2,3] -> [1,'aug', 2, 'aug', 3, 'aug']
    [training_rounds.insert(2 * x, "aug") for x in range(len(training_rounds))]
    training_rounds.pop(0)
    print(f"[LOG] Training rounds scheme : {training_rounds}")

    # Start the loop
    total_epochs_passed = 0

    for i, step in tqdm(enumerate(training_rounds), total=len(training_rounds)):
        if i == 0:
            # Initialize everything with new or old data
            dls = fields.dataloaders(path, bs=ds_meta["batch_size"])
            cbs = [
                TensorBoardCallback(
                    log_dir=f"tb_runs/{fname_start}", projector=False, trace_model=False
                ),
                CSVLogger(fname=f"csv_logs/{fname_start}.csv"),
            ]

            learn = vision_learner(
                dls,
                ds_meta["network"],
                cbs=cbs,
                metrics=metrics,
                pretrained=ds_meta["pretrained"],
            ).to_fp16()
            fname_training = f'{ds_meta["ds_name"]}_{args.name}_{datetime.now().strftime("%d%m%Y_%H:%M:%S")}'  # unique_name
            if step > 0:
                learn.fine_tune(step)
            learn.save("temp_model")  # saving so can be reloaded
            clear_learner(learn, dls)
            print("[LOG] : Cleared learner")
            # Since training is in batches, keep track of total no of epochs trained for
            total_epochs_passed += step

        if step != "aug" and i > 0:
            # Initialize everything with new or old data
            dls = fields.dataloaders(path, bs=ds_meta["batch_size"])
            cbs = [
                TensorBoardCallback(
                    log_dir=f"tb_runs/{fname_start}", projector=False, trace_model=False
                ),
                CSVLogger(fname=f"csv_logs/{fname_start}.csv"),
            ]

            learn = vision_learner(
                dls,
                ds_meta["network"],
                cbs=cbs,
                metrics=metrics,
                pretrained=ds_meta["pretrained"],
            ).to_fp16()

            learn.load("temp_model")  # load model since augment has been done already
            # Continue training
            fname_training = f'{ds_meta["ds_name"]}_{args.name}_{datetime.now().strftime("%d%m%Y_%H:%M:%S")}'  # unique_name
            learn.fine_tune(step)
            learn.save("temp_model")
            clear_learner(learn, dls)

            print("[LOG] : Cleared learner")
            # Since training is in batches, keep track of total no of epochs trained for
            total_epochs_passed += step
        if step == "aug":

            if ds_meta["enable_proxy_attention"] == True:
                print("[INFO]: Running Proxy Attention")

                # PROXY ATTENTION LOOP

                dls = fields.dataloaders(path, bs=ds_meta["batch_size"])
                cbs = [
                    TensorBoardCallback(
                        log_dir=f"tb_runs/{fname_start}",
                        projector=False,
                        trace_model=False,
                    ),
                    CSVLogger(fname=f"csv_logs/{fname_start}.csv"),
                ]

                learn = vision_learner(
                    dls,
                    ds_meta["network"],
                    cbs=cbs,
                    metrics=metrics,
                    pretrained=ds_meta["pretrained"],
                ).to_fp16()

                learn.load(
                    "temp_model"
                )  # load model since augment has been done already
                learn.to("cpu")

                class Hook:
                    def __init__(self, m):
                        self.hook = m.register_forward_hook(self.hook_func)

                    def hook_func(self, m, i, o):
                        self.stored = o.detach().clone()

                    # Automatically register the hook when entering it
                    def __enter__(self, *args):
                        return self

                    # Automatically remove the hook when exiting it
                    def __exit__(self, *args):
                        self.hook.remove()

                class HookBwd:
                    def __init__(self, m):
                        self.hook = m.register_backward_hook(self.hook_func)

                    def hook_func(self, m, gi, go):
                        self.stored = go[0].detach().clone()

                    def __enter__(self, *args):
                        return self

                    def __exit__(self, *args):
                        self.hook.remove()

                # dls.to('cpu')

                # Get the classes
                print("[INFO] : Starting Attention Loop")
                vocab_dict = {
                    learn.dls.vocab[x]: x for x in range(len(learn.dls.vocab))
                }  # Get class names
                # Get images, shuffle, pick a subset
                items = get_image_files_exclude_augment(ds_meta["ds_path"])
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
                    x
                    for x in range(subset)
                    if bspred[2][x] != TensorBase(item_names)[x]
                ]
                print(
                    f"[INFO] : Pct wrong for step {total_epochs_passed} = {len(index_wrongs)/len(bspred[2])}"
                )

                # RUN PROXY ATTENTION
                print(f"[INFO] : Creating maps")
                # def create_im(im):
                #     img = PILImage.create(items[im])

                #     (x,) = first(dls.test_dl([img]))

                #     x_dec = TensorImage(dls.train.decode((x,))[0][0])

                #     # Get attention maps

                #     # -----
                #     # Grad CAM Attention map
                #     cls = 1
                #     try:
                #         with HookBwd(learn.model[-2][4][-1]) as hookg:  # for other layers
                #             with Hook(learn.model[-2][4][-1]) as hook:
                #                 output = learn.model.eval()(x)
                #                 act = hook.stored
                #             output[0, cls].backward()
                #             grad = hookg.stored
                #         w = grad[0].mean(dim=[1, 2], keepdim=True)
                #         cam_map = (w * act[0]).sum(0)
                #         # print(x.shape,x_dec.shape, w.shape, cam_map.shape)

                #     except Exception as e:
                #         print(e)
                #     # -----

                #     # test_cam_map = cam_map.detach().cpu()
                #     # Resize cam map so it's the same size as the image, as the output is much smaller
                #     t_resized = transformF.resize(
                #         torch.unsqueeze(cam_map, 0), ds_meta["image_size"]
                #     )
                #     t_resized = (
                #         torch.cat([t_resized, t_resized, t_resized], dim=0).detach().cpu()
                #     )

                #     # IMPORTANT : Change the pixels that are of higher intensity to 0 because they did not help the network get the right answer
                #     x_dec[t_resized >= 0.009] = 0.0

                #     x_dec = torch.einsum("ijk->jki", x_dec)
                #     plt.imshow(x_dec)
                #     plt.axis("off")
                #     ax=plt.gca()
                #     ax.get_xaxis().set_visible(False)
                #     plt.box(False)
                #     plt.savefig(rename_for_aug(items[im]), transparent = True, bbox_inches='tight',pad_inches = 0)

                # parallel(create_im, index_wrongs, progress=True, n_workers=8)

                # for im in tqdm(index_wrongs, total = len(index_wrongs)):
                #     img = PILImage.create(items[im])

                #     (x,) = first(dls.test_dl([img]))

                #     x_dec = TensorImage(dls.train.decode((x,))[0][0])

                #     # Get attention maps

                #     # -----
                #     # Grad CAM Attention map
                #     cls = 1
                #     try:
                #         with HookBwd(learn.model[-2][4][-1]) as hookg:  # for other layers
                #             with Hook(learn.model[-2][4][-1]) as hook:
                #                 output = learn.model.eval()(x.cuda())
                #                 act = hook.stored
                #             output[0, cls].backward()
                #             grad = hookg.stored
                #         w = grad[0].mean(dim=[1, 2], keepdim=True)
                #         cam_map = (w * act[0]).sum(0)
                #         # print(x.shape,x_dec.shape, w.shape, cam_map.shape)

                #     except Exception as e:
                #         print(e)
                #     # -----

                #     # test_cam_map = cam_map.detach().cpu()
                #     # Resize cam map so it's the same size as the image, as the output is much smaller
                #     t_resized = transformF.resize(
                #         torch.unsqueeze(cam_map, 0), ds_meta["image_size"]
                #     )
                #     t_resized = (
                #         torch.cat([t_resized, t_resized, t_resized], dim=0).detach().cpu()
                #     )

                #     # IMPORTANT : Change the pixels that are of higher intensity to 0 because they did not help the network get the right answer
                #     x_dec[t_resized >= 0.008] = 0.0

                #     x_dec = torch.einsum("ijk->jki", x_dec)
                #     plt.imshow(x_dec)
                #     plt.axis("off")
                #     ax=plt.gca()
                #     ax.get_xaxis().set_visible(False)
                #     plt.box(False)
                #     plt.savefig(rename_for_aug(items[im]), transparent = True, bbox_inches='tight',pad_inches = 0)

                ims = [
                    PILImage.create(items[x])
                    for x in tqdm(index_wrongs, total=len(index_wrongs))
                ]
                im_names = [items[x] for x in index_wrongs]
                test_ds_new_ims = dls.test_dl(ims, shuffle=False)
                decoded = [
                    dls.train.decode(x)[0].float()
                    for x in tqdm(test_ds_new_ims, total=len(test_ds_new_ims))
                ]
                eval_model = learn.model.eval()
                # eval_model_results = [eval_model(first(x).float().cuda()) for x in test_ds_new_ims]
                # eval_model_results = learn.predict_batch(ims)[0]
                # print(eval_model_results)
                # Hook
                cls = 1
                cam_maps = []

                print("Creating map")
                for im in tqdm(decoded, total=len(decoded)):
                    with HookBwd(learn.model[-2][4][-1]) as hookg:  # for other layers
                        with Hook(learn.model[-2][4][-1]) as hook:
                            output = eval_model(im.cuda())
                            # output = learn.predict(im)
                            act = hook.stored
                        output[0, cls].backward()
                        grad = hookg.stored
                    w = grad[0].mean(dim=[1, 2], keepdim=True)
                    cam_map = (w * act[0]).sum(0)
                    cam_maps.append(cam_map)

                print("Resizing")
                t_resized = [
                    transformF.resize(
                        torch.unsqueeze(cam_map, 0), ds_meta["image_size"]
                    )
                    for cam_map in cam_maps
                ]
                t_resized = [
                    torch.cat([x, x, x], dim=0).detach().cpu() for x in t_resized
                ]
                decoded_tensor_images = [
                    dls.train.decode(x)[0][0].float()
                    for x in tqdm(test_ds_new_ims, total=len(test_ds_new_ims))
                ]

                for ind in tqdm(range(len(t_resized))):
                    decoded_tensor_images[ind][t_resized[ind] >= 0.009] = 0.0
                    decoded_tensor_images[ind] = torch.einsum(
                        "ijk->jki", decoded_tensor_images[ind]
                    )

                for ind in tqdm(range(len(decoded_tensor_images))):
                    plt.imshow(decoded_tensor_images[ind])
                    plt.axis("off")
                    ax = plt.gca()
                    ax.get_xaxis().set_visible(False)
                    plt.box(False)
                    plt.savefig(
                        rename_for_aug(im_names[ind]),
                        transparent=True,
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                clear_learner(learn, dls)
                del bspred
                del items
                # del t_resized
                gc.collect()

        # Save model every n epochs
        if total_epochs_passed % ds_meta["save_model_every_n_epoch"] == 0:
            learn.save(fname_training)
