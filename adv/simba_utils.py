import numpy as np
import torch
import torchvision.transforms.functional as TF
import math
from scipy.fftpack import dct, idct



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
