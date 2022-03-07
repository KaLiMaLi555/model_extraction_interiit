# List of imports
from PIL import Image
from typing import Type, Any, Callable, Union, List, Optional

## PyTorch
import torch
from torch import Tensor
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
## PyTorch lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
# Torchvision
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pickle
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
from functools import partial
from timm.models.vision_transformer import _cfg
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from convit.models import VisionTransformer


####
#         pretrained=args.pretrained    ----> False 
#         num_classes=args.nb_classes   ----> num classes
#         drop_rate=args.drop           ----> Dropout rate (default: 0.)
#         drop_path_rate=args.drop_path ----> Drop path rate (default: 0.1)
#         drop_block_rate=args.drop_block ---> Drop block rate (default: None)
#         local_up_to_layer=args.local_up_to_layer --> number of GPSA layers (default 10)
#         locality_strength=args.locality_strength --> Determines how focused each head is around its attention center (default: 1.0)
#         embed_dim = args.embed_dim               --> embedding dimension per head (default: 48)
####

def convit_small(pretrained=False, **kwargs):
    num_heads = 9
    kwargs['embed_dim'] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/convit/convit_small.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model



class ConViTRNN(nn.Module):

    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, num_classes=400):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ConViTRNN, self).__init__()

        self.encoder = convit_small(pretrained=False,)
        self.decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=h_RNN_layers, h_RNN=h_RNN, h_FC_dim=h_FC_dim, drop_p=drop_p, num_classes=num_classes)
        
    def forward(self, x_3d):
        return self.decoder(self.encoder(x_3d))
