from scipy._lib.six import reraise
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
from einops import rearrange
from einops.layers.torch import Rearrange

class EmbdClassifier(nn.Module): 
    '''lr=1e-3, batch_size=64, AdamW, weight_decay=default'''
    def __init__(self, cl_plan,  num_classes=2,  n_embed=256, pre_embed = None, **_):
        super().__init__()
        if pre_embed is not None:
            self.embd = nn.Sequential(
                nn.Embedding.from_pretrained(pre_embed, scale_grad_by_freq=True, freeze=False),
                Rearrange('b h w c -> b c h w'),
                nn.Conv2d(pre_embed.shape[1], cl_plan[0], 1,1,0, bias=False), 
            )
        else:
            self.embd = nn.Sequential(
                nn.Embedding(n_embed, cl_plan[0], scale_grad_by_freq=True),
                Rearrange('b h w c -> b c h w'),
            )
        sequential = []
        for i in range(1, len(cl_plan)):
            sequential+=[
                nn.Conv2d(cl_plan[i-1], cl_plan[i], 3,1,1, bias=True), 
                nn.InstanceNorm2d(cl_plan[i]),
                nn.ReLU(True),
                nn.Conv2d(cl_plan[i], cl_plan[i], 3,1,1, bias=True), 
                nn.InstanceNorm2d(cl_plan[i]),
                nn.ReLU(True),
                nn.MaxPool2d(2,2)
            ]
        sequential += [
            nn.Flatten(1),
            nn.Linear( cl_plan[-1]*4*4,  cl_plan[-1]*4*4, bias=True),
            nn.ReLU(True),
            nn.Linear( cl_plan[-1]*4*4, num_classes, bias=True),
        ]
        self.main = nn.Sequential(*sequential)

    def forward(self, x):
        y = self.embd(x)
        y = self.main(y)
        return y.squeeze(-1).squeeze(-1)

