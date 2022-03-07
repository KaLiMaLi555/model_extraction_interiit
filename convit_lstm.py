## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


# Torchvision
from functools import partial
from timm.models.vision_transformer import _cfg
from convit.models import VisionTransformer


class DecoderRNN(nn.Module):
    def __init__(
        self,
        CNN_embed_dim=300,
        h_RNN_layers=3,
        h_RNN=256,
        h_FC_dim=128,
        drop_p=0.3,
        num_classes=400,
    ):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


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
    kwargs["embed_dim"] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/convit/convit_small.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint)
    return model


class ConViTRNN(nn.Module):
    def __init__(
        self,
        fc_hidden1=512,
        fc_hidden2=512,
        drop_p=0.3,
        CNN_embed_dim=300,
        h_RNN_layers=3,
        h_RNN=256,
        h_FC_dim=128,
        num_classes=400,
    ):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ConViTRNN, self).__init__()

        self.encoder = convit_small(
            pretrained=False,
        )
        self.decoder = DecoderRNN(
            CNN_embed_dim=CNN_embed_dim,
            h_RNN_layers=h_RNN_layers,
            h_RNN=h_RNN,
            h_FC_dim=h_FC_dim,
            drop_p=drop_p,
            num_classes=num_classes,
        )

    def forward(self, x_3d):
        return self.decoder(self.encoder(x_3d))
