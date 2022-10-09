#%%
import timm
from fastai.vision.all import *
from fastai.vision.widgets import *
import os
import matplotlib.pyplot as plt
from IPython.display import Image

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
os.environ["FASTAI_HOME"] = "/media/hdd/Datasets/"
#%%
def predict_batch(self, item, rm_type_tfms=None, with_input=False, num_workers=10):
    """
    Monkey Patch fastai for batch predictions
    """
    dl = self.dls.test_dl(item, rm_type_tfms=rm_type_tfms, num_workers=num_workers)
    ret = self.get_preds(dl=dl, with_decoded=True)
    return ret


#%%
def create_if_not_exists(fpath):
    """
    Create folder if its not there already
    """
    if not Path.exists(Path(fpath)):
        Path.mkdir(Path(fpath), parents=True)


#%%
# GRAD Cam Hooks
class Hook:
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)

    def hook_func(self, m, i, o):
        self.stored = o.detach().clone()

    def __enter__(self, *args):
        return self

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


#%%
def rename_for_aug(fpath):
    """
    Add the word 'augmented_' before the file path so it's easy to identify them from the modified dataset
    """
    return fpath.parent / Path("augmented_" + fpath.name)
