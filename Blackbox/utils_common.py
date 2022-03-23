import random

import numpy as np
import torch
import torchvision.transforms.functional as TF


def set_seed(seed):
    """Function to set seeds across libraries to ensure determinism."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def swin_transform(vid):
    """Function to apply preprocessing transforms expected by Swin-T model."""
    # N, C, L, S, S
    vid_shape = vid.shape
    N, C, L, S, _ = vid.shape
    vid_swin = vid.reshape(
        (-1, vid_shape[2], vid_shape[1], *vid_shape[3:]))  # N, L, C, S, S
    vid_swin_batch = []
    for v in list(vid_swin):
        v = torch.stack(
            [
                TF.normalize(frame * 255, [123.675, 116.28, 103.53],
                             [58.395, 57.12, 57.375]).permute(1, 2, 0)
                for frame in v
            ]
        )
        v = v.reshape((-1, 1, L, S, S, C))  # 1, 1, L, S, S, C
        v = v.permute(0, 1, 5, 2, 3, 4)  # 1, 1, C, L, S, S
        v = v.reshape((-1, C, L, S, S))  # 1, C, L, S, S
        vid_swin_batch.append(v)
    vid_swin = torch.stack(vid_swin_batch)  # N, 1, C, L, S, S
    return vid_swin
