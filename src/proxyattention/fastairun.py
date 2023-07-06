# %%
import timm
import os
os.environ["TORCH_HOME"] = "/home/eragon/Documents/Datasets/"

# %%
from fastai.vision.all import *
from pathlib import Path

# %%
path = Path("/home/eragon/Documents/Datasets/places256/train/")

# %%
path.ls()

# %%
files = get_image_files(path)[:20000]
files

# %%
files[0].parent.name

# %%
def label_func(f): return Path(f).parent.name

# %%
dls = ImageDataLoaders.from_lists(path, fnames = files, labels = files.map(label_func), item_tfms=Resize(224), bs = 16)

# %%
dls.show_batch()

# %%
dls.c

# %%
learn = vision_learner(dls, arch ="vit_base_patch16_224" , metrics=accuracy, pretrained=True, cbs = [MixedPrecision])

# %%
learn.fine_tune(40)


