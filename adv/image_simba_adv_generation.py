# -*- coding: utf-8 -*-
# !rm -rf /content/model_extraction_interiit
# !git clone -b swint https://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@github.com/KaLiMaLi555/model_extraction_interiit

# !ls model_extraction_interiit

# !pip install -q -r /content/model_extraction_interiit/conv3d/requirements.txt
# !pip install -q -r /content/model_extraction_interiit/Video-Swin-Transformer/requirements.txt
# !pip install -q cleverhans

# # Commented out IPython magic to ensure Python compatibility.
# # %cd model_extraction_interiit/Video-Swin-Transformer/
# !pip install -v -e . --user

# !wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth

# !pip install -q mmcv==1.3.1

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack

# Import for Swin-T
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
from easydict import EasyDict
import torch
import numpy as np
import torchvision.transforms as trans
import math
from scipy.fftpack import dct, idct
import torchvision.transforms.functional as TF

import torch
import torch.nn.functional as F


import cv2
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
import pandas as pd
import pickle
import torch


def swin_transform(fake):  # N, C, L, S, S
    fake_shape = fake.shape
    fake_teacher = fake.reshape(
        (-1, fake_shape[2], fake_shape[1], *fake_shape[3:])
    )  # N, L, C, S, S
    fake_teacher_batch = []
    for vid in list(fake_teacher):
        vid = torch.stack(
            [
                TF.normalize(
                    frame * 255, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
                ).permute(1, 2, 0)
                for frame in vid
            ]
        )
        vid = vid.reshape((-1, 1, 16, 224, 224, 3))  # 1, 1, L, S, S, C
        vid = vid.permute(0, 1, 5, 2, 3, 4)  # 1, 1, C, L, S, S
        vid = vid.reshape((-1, 3, 16, 224, 224))  # 1, C, L, S, S
        fake_teacher_batch.append(vid)
    fake_teacher = torch.stack(fake_teacher_batch)  # N, 1, C, L, S, S
    return fake_teacher


# applies the normalization transformations
def apply_normalization(vid, dataset):
    if dataset == "k400":
        return swin_transform(vid)
    else:
        pass


# get most likely predictions and probabilities for a set of inputs
def get_preds(
    model, inputs, dataset_name, correct_class=None, batch_size=25, return_cpu=True
):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size) : upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        if correct_class is None:
            prob, pred = output.max(1)
        else:
            prob, pred = output[:, correct_class], torch.autograd.Variable(
                torch.ones(output.size()) * correct_class
            )
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


# get least likely predictions and probabilities for a set of inputs
def get_least_likely(model, inputs, dataset_name, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size) : upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        prob, pred = output.min(1)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


# defines a diagonal order
# order is fixed across diagonals but are randomized across channels and within the diagonal
# e.g.
# [1, 2, 5]
# [3, 4, 8]
# [6, 7, 9]
def diagonal_order(image_size, channels):
    x = torch.arange(0, image_size).cumsum(0)
    order = torch.zeros(image_size, image_size)
    for i in range(image_size):
        order[i, : (image_size - i)] = i + x[i:]
    for i in range(1, image_size):
        reverse = order[image_size - i - 1].index_select(
            0, torch.LongTensor([i for i in range(i - 1, -1, -1)])
        )
        order[i, (image_size - i) :] = image_size * image_size - 1 - reverse
    if channels > 1:
        order_2d = order
        order = torch.zeros(channels, image_size, image_size)
        for i in range(channels):
            order[i, :, :] = 3 * order_2d + i
    return order.view(1, -1).squeeze().long().sort()[1]


# defines a block order, starting with top-left (initial_size x initial_size) submatrix
# expanding by stride rows and columns whenever exhausted
# randomized within the block and across channels
# e.g. (initial_size=2, stride=1)
# [1, 3, 6]
# [2, 4, 9]
# [5, 7, 8]
def block_order(image_size, channels, initial_size=1, stride=1):
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(
        channels, initial_size, initial_size
    )
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, : (i + stride), i : (i + stride)] = perm[:num_first].view(
            channels, -1, stride
        )
        order[:, i : (i + stride), :i] = perm[num_first:].view(channels, stride, -1)
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]


# zeros all elements outside of the top-left (block_size * ratio) submatrix for every block
def block_zero(x, block_size=8, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = torch.zeros(x.size(0), x.size(1), block_size, block_size)
    mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            z[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ] = (
                x[
                    :,
                    :,
                    (i * block_size) : ((i + 1) * block_size),
                    (j * block_size) : ((j + 1) * block_size),
                ]
                * mask
            )
    return z


# applies DCT to each block of size block_size
def block_dct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ].numpy()
            submat_dct = dct(dct(submat, axis=2, norm="ortho"), axis=3, norm="ortho")
            if masked:
                submat_dct = submat_dct * mask
            submat_dct = torch.from_numpy(submat_dct)
            z[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ] = submat_dct
    return z


# applies IDCT to each block of size block_size
def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, : int(block_size * ratio[i]), : int(block_size * ratio[i])] = 1
    else:
        mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ].numpy()
            if masked:
                submat = submat * mask
            z[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ] = torch.from_numpy(
                idct(idct(submat, axis=3, norm="ortho"), axis=2, norm="ortho")
            )
    if linf_bound > 0:
        return z.clamp(-linf_bound, linf_bound)
    else:
        return z


class SimBA:
    def __init__(self, model, dataset, image_size):
        self.model = model
        self.dataset = dataset
        self.image_size = image_size
        # self.model.eval()

    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.reshape(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z

    def normalize(self, x):
        return apply_normalization(x, self.dataset)

    def get_probs(self, x, y):
        # output = self.model(self.normalize(x.cuda())).cpu()
        output = self.model(x.cuda()).cpu()
        probs = torch.index_select(output, 1, y)
        return torch.diag(probs)

    def get_preds(self, x):
        # output = self.model(self.normalize(x.cuda())).cpu()
        output = self.model(x.cuda()).cpu()
        _, preds = output.data.max(1)
        return preds

    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, num_iters=10000, epsilon=0.2, targeted=False):
        # print(x.shape)
        # print(x.view(1,-1).shape)
        # x=x.clamp(0,1)
        n_dims = x.reshape(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0)
        # print(n_dims)
        # print(x.shape)
        last_prob = self.get_probs(x, y)
        for i in range(num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (
                    right_prob < last_prob
                ):  # commenting this line made things work
                    # if right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 1000 == 0:
                print(last_prob)
        return x.squeeze()

    # # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # # (for targeted attack) <labels_batch>
    def simba_batch(
        self,
        images_batch,
        labels_batch,
        max_iters,
        freq_dims,
        stride,
        epsilon,
        linf_bound=0.0,
        order="rand",
        targeted=False,
        pixel_attack=False,
        log_every=1,
    ):
        batch_size = images_batch.size(0)
        image_size = images_batch.size(-1)
        print(self.image_size, image_size)
        assert self.image_size == image_size
        # sample a random ordering for coordinates independently per batch element
        if order == "rand":
            indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
        elif order == "diag":
            indices = diagonal_order(image_size, 3)[:max_iters]
        elif order == "strided":
            indices = block_order(image_size, 3, initial_size=freq_dims, stride=stride)[
                :max_iters
            ]
        else:
            indices = block_order(image_size, 3)[:max_iters]
        if order == "rand":
            expand_dims = freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters)
        succs = torch.zeros(batch_size, max_iters)
        queries = torch.zeros(batch_size, max_iters)
        l2_norms = torch.zeros(batch_size, max_iters)
        linf_norms = torch.zeros(batch_size, max_iters)
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: block_idct(
                z, block_size=image_size, linf_bound=linf_bound
            )
        remaining_indices = torch.arange(0, batch_size).long()
        for k in range(max_iters):
            dim = indices[k]
            expanded = (
                images_batch[remaining_indices]
                + trans(self.expand_vector(x[remaining_indices], expand_dims))
            ).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims))
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded)
            preds[remaining_indices] = preds_next
            if targeted:
                remaining = preds.ne(labels_batch)
            else:
                remaining = preds.eq(labels_batch)
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(
                    0, 1
                )
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            remaining_indices = torch.arange(0, batch_size)[remaining].long()
            if k > 0:
                succs[:, k - 1] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            diff[:, dim] = epsilon
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            adv = (
                images_batch[remaining_indices]
                + trans(self.expand_vector(left_vec, expand_dims))
            ).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices])
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (
                images_batch[remaining_indices]
                + trans(self.expand_vector(right_vec, expand_dims))
            ).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if targeted:
                right_improved = right_probs.gt(
                    torch.max(prev_probs[remaining_indices], left_probs)
                )
            else:
                right_improved = right_probs.lt(
                    torch.min(prev_probs[remaining_indices], left_probs)
                )
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            if (k + 1) % log_every == 0 or k == max_iters - 1:
                print(
                    "Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f"
                    % (
                        k + 1,
                        queries.sum(1).mean(),
                        probs[:, k].mean(),
                        remaining.float().mean(),
                    )
                )
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(
            0, 1
        )
        preds = self.get_preds(expanded)
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, max_iters - 1] = ~remaining
        return expanded, probs, succs, queries, l2_norms, linf_norms


def model_wrapper(inp):
    with torch.no_grad():
        return torch.tensor(teacher(inp, return_loss=False))


def model_wrapper_image(inp):  # b,3,224,224 assumes input with 0...1
    b_t = []
    for inp_i in inp:
        lim = [inp_i * 255 for _ in range(16)]
        b_t.append(torch.stack(lim, dim=2))  # b,3,16,224,224
        
    with torch.no_grad():
        return torch.tensor(
            teacher(swin_transform(torch.stack(b_t, dim=0).cuda()), return_loss=False)
        )


def get_random_vid(idx):
    colors = [
        (
            int(np.random.random() * 255),
            int(np.random.random() * 255),
            int(np.random.random() * 255),
        )
        for _ in range(9)
    ]

    pos = [
        [
            [int(np.random.random() * 224), int(np.random.random() * 224)],
            [int(np.random.random() * 224), int(np.random.random() * 224)],
        ]
        for _ in range(9)
    ]
    bgc = (
        int(np.random.random() * 255),
        int(np.random.random() * 255),
        int(np.random.random() * 255),
    )
    imgs = []
    for _ in range(16):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.rectangle(img, (0, 0), (224, 224), bgc, -1)
        for i in range(9):
            img = cv2.rectangle(
                img,
                tuple(pos[i][0]),
                (pos[i][0][0] + pos[i][1][0], pos[i][0][1] + pos[i][1][0]),
                colors[i],
                -1,
            )
            shiftX = int(np.random.random() * int(np.random.random() * 224)) - int(
                np.random.random() * 224
            )
            shiftY = int(np.random.random() * int(np.random.random() * 224)) - int(
                np.random.random() * 224
            )
            pos[i][0][0] = (
                pos[i][0][0]
                + shiftX
                + int(np.random.random() * int(np.random.random() * 224))
                - int(np.random.random() * 224)
            )
            pos[i][1][0] = (
                pos[i][1][0]
                + int(np.random.random() * int(np.random.random() * 224))
                - int(np.random.random() * 224)
            )
            pos[i][0][1] = (
                pos[i][0][1]
                + shiftY
                + int(np.random.random() * int(np.random.random() * 224))
                - int(np.random.random() * 224)
            )
            pos[i][1][1] = (
                pos[i][1][1]
                + int(np.random.random() * int(np.random.random() * 224))
                - int(np.random.random() * 224)
            )
            # pos[i][0][0]%=224
            # pos[i][1][0]%=224
            # pos[i][0][1]%=224
            # pos[i][1][1]%=224
        imgs.append(img)
    vid = torch.tensor(np.array(imgs), dtype=torch.float)
    vid = vid.permute(3, 0, 1, 2)
    if idx % 2:
        vid /= 255
    return vid


def get_random_img(idx):
    colors = [
        (
            int(np.random.random() * 255),
            int(np.random.random() * 255),
            int(np.random.random() * 255),
        )
        for _ in range(9)
    ]

    pos = [
        [
            [int(np.random.random() * 224), int(np.random.random() * 224)],
            [int(np.random.random() * 224), int(np.random.random() * 224)],
        ]
        for _ in range(9)
    ]
    bgc = (
        int(np.random.random() * 255),
        int(np.random.random() * 255),
        int(np.random.random() * 255),
    )
    # imgs = []
    # for _ in range(16):
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img = cv2.rectangle(img, (0, 0), (224, 224), bgc, -1)
    for i in range(9):
        img = cv2.rectangle(
            img,
            tuple(pos[i][0]),
            (pos[i][0][0] + pos[i][1][0], pos[i][0][1] + pos[i][1][0]),
            colors[i],
            -1,
        )
        shiftX = int(np.random.random() * int(np.random.random() * 224)) - int(
            np.random.random() * 224
        )
        shiftY = int(np.random.random() * int(np.random.random() * 224)) - int(
            np.random.random() * 224
        )
        pos[i][0][0] = (
            pos[i][0][0]
            + shiftX
            + int(np.random.random() * int(np.random.random() * 224))
            - int(np.random.random() * 224)
        )
        pos[i][1][0] = (
            pos[i][1][0]
            + int(np.random.random() * int(np.random.random() * 224))
            - int(np.random.random() * 224)
        )
        pos[i][0][1] = (
            pos[i][0][1]
            + shiftY
            + int(np.random.random() * int(np.random.random() * 224))
            - int(np.random.random() * 224)
        )
        pos[i][1][1] = (
            pos[i][1][1]
            + int(np.random.random() * int(np.random.random() * 224))
            - int(np.random.random() * 224)
        )
        # pos[i][0][0]%=224
        # pos[i][1][0]%=224
        # pos[i][0][1]%=224
        # pos[i][1][1]%=224
    img = torch.tensor(np.array(img), dtype=torch.float)
    img = img.permute(2, 0, 1)
    img /= 255
    return img


def get_noise_dataset_image(idx, batch_size=16):
    frames_list = []
    labels_list = []
    for i in range(int(batch_size)):
        id = (idx + i) % num_instances
        frame = torch.tensor(pickle.load(open(videos[id], "rb"))).to(torch.float32)
        frame1 = frame[0]
        # frame2=frame[1]
        frame1_ = frame1[:, 0, :, :]
        # frame2_ = frame2[:,0,:,:]
        frame1_ /= 255
        # frame2__/=255
        label = torch.tensor(labels[id])
        # label_2 = torch.tensor(labels[id])
        frames_list.append(frame1_)
        labels_list.append(label)

    frames_list = torch.stack(frames_list)
    labels_list = torch.stack(labels_list)
    return frames_list, labels_list


def save_gen_image(
    image_batch, label_batch, dir_path="/content/drive/MyDrive/Adv_gen_from_noise/"
):
    counter_gen_images = len(os.listdir(dir_path)) - 1
    for i in range(image_batch.shape[0]):
        image = image_batch[i]
        filename = os.path.join(
            dir_path, str(label_batch[i].item()) + "_" + str(counter_gen_images)
        )
        np.save(filename, image)
        counter_gen_images += 1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = "/content/model_extraction_interiit/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    checkpoint = "/content/model_extraction_interiit/Video-Swin-Transformer/swin_tiny_patch244_window877_kinetics400_1k.pth"
    cfg = Config.fromfile(config)
    teacher = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(teacher, checkpoint, map_location=device)


    teacher.to(device)
    teacher.eval()

    attacker = SimBA(model_wrapper_image, "k400", 224)

    video_dir_path = "/content/drive/MyDrive/Noise_Gen_Expt_Rohit_k400/"
    csv = pd.read_csv("/content/drive/MyDrive/Noise_Expt_Rohit_k400.csv")

    videos = csv["FileNames"]
    videos = [video_dir_path + x + ".pkl" for x in videos]
    labels = csv["Labels"]
    num_instances = len(csv)

    while True:
        batched_imgs, batched_labels = get_noise_dataset_image(np.random.randint(0, 540))
        batched_targets = batched_labels[torch.randperm(batched_labels.shape[0])]
        expanded, probs, succs, queries, l2_norms, linf_norms = attacker.simba_batch(
            batched_imgs, batched_labels, 3000, 224, 7, 0.2, log_every=100, targeted=True
        )
        final_labels = attacker.get_preds(expanded)
        print(final_labels)
        save_gen_image(expanded, final_labels)

    print(expanded.shape)
    print(probs.shape)
    print(succs.shape)
    print(queries.shape)
    print(l2_norms.shape)
    print(linf_norms.shape)
    print(probs[:, -1])
    print(succs[:, -1])
    print(queries[:, -1])
    print(l2_norms[:, -1])

    print(final_labels.shape)
    print(final_labels)
    print(batched_labels)


    # img_test = get_random_img(0)
    # img_test = img_test.unsqueeze(0)
    # test_p_y, test_y = model_wrapper_image(img_test).max(1)
    # print(test_p_y, test_y)
    # img_test = img_test.squeeze(0)
    # img_untargeted = attacker.simba_single(img_test, test_y, num_iters=3000)
    # img_untargeted = img_untargeted.unsqueeze(0)
    # out_test_p_y, out_test_y = model_wrapper_image(img_untargeted).max(1)
    # print(out_test_p_y, out_test_y)

    # img_untargeted = img_untargeted.squeeze(0)
    # img_untargeted = attacker.simba_single(img_untargeted, out_test_y, num_iters=3000)
    # img_untargeted = img_untargeted.unsqueeze(0)
    # out_test_p_y, out_test_y = model_wrapper_image(img_untargeted).max(1)
    # print(out_test_p_y, out_test_y)

    # img_untargeted = img_untargeted.squeeze(0)
    # img_untargeted = attacker.simba_single(
    #     img_untargeted, torch.tensor(5), num_iters=7000, targeted=True
    # )
    # img_untargeted = img_untargeted.unsqueeze(0)
    # out_test_p_y, out_test_y = model_wrapper_image(img_untargeted).max(1)
    # print(out_test_p_y, out_test_y)

    # adv_dict = {}


    # def run_attack(vid, vid_y, base):
    #     print(f"generating {vid_y} from {base}")
    #     vid_adv = attacker.simba_single(
    #         vid.cpu(), vid_y.cpu(), epsilon=0.2, num_iters=10000, targeted=True
    #     )
    #     vid_adv = vid_adv.unsqueeze(0)
    #     p_adv, y_adv = model_wrapper_image(vid_adv).max(1)
    #     y = int(y_adv.cpu().numpy())
    #     y_adv = y_adv.cuda()
    #     if y in adv_dict:
    #         adv_dict[y].append(vid_adv.cpu().detach().numpy())
    #     else:
    #         adv_dict[y] = [vid_adv.cpu().detach().numpy()]
    #     vid_adv = vid_adv.cuda()
    #     print(f"generated {y}")
    #     return vid_adv, y_adv, p_adv


    # def generate_from_given_video(vid_base, vid_base_y):
    #     y = int(vid_base_y.cpu().numpy())
    #     vid_base_y = vid_base_y.cuda()
    #     print(f"started for {vid_base.shape} and {y}")
    #     vid_base = vid_base.squeeze(0)
    #     if y in adv_dict:
    #         adv_dict[y].append(vid_base.cpu().detach().numpy())
    #     else:
    #         adv_dict[y] = [vid_base.cpu().detach().numpy()]
    #     vid_base = vid_base.cuda()
    #     for i in range(10):
    #         run_attack(vid_base, torch.tensor(int(np.random.rand() * 399)), vid_base_y)
    #     vid_base.cpu().detach()
    #     vid_base_y.cpu().detach()


    # for img_base, img_base_y in zip(batched_imgs, img_y):
    #     generate_from_given_video(img_base, img_base_y)

    # # from concurrent.futures import ProcessPoolExecutor
    # # with ProcessPoolExecutor() as executor:
    # #     executor.map(generate_from_given_video,zip(batched_vids,vid_y))

    # for x in adv_dict:
    #     print(x, len(adv_dict[x]))
