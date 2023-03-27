#%%
from proxyattention.meta_utils import save_pickle, read_pickle
import torch
#%%

test_pick, grads2 = read_pickle("pickler.pkl")[0]
#%%
print(test_pick.type())
#%%
grads2.max(), grads2.min()
#%%
test_pick.max(), test_pick.min()
#%%
test_pick.shape, grads2.shape

#%%
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
import timm

from lightning.fabric import Fabric
# %%
fabric = Fabric(precision="16-mixed")
fabric.launch()
model = timm.create_model(
        "resnet18",
        pretrained=True,
        num_classes=10,
    )
#%%
model = fabric.setup(model)
model = torch.compile(model)
#%%
def sum_model_weights(model):
    weight_sum = 0.0
    for param in model.parameters():
        weight_sum += torch.sum(param.data)
    return weight_sum

#%%
sum_model_weights(model)
#%%
model.module
#%%
cam = GradCAMPlusPlus(model.module, [model.layer4[-1].conv2], True)
# %%
grads = cam(test_pick, targets=None)
#%%
grads.min(), grads.max()
#%%
grads
#%%
grads2
#%%
grads = torch.Tensor(grads).to(fabric.device).unsqueeze(1).expand(-1, 3, -1, -1)
#%%

grads.min(), grads.max()
#%%
torch.where(grads > 0.1)