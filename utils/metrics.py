"""
    Add a single unified script for calculating all metrics, and making all plots 
"""
import torch
from torchmetrics.functional import accuracy
import torch.nn.functional as F
import numpy as np


def topk_accuracy(preds, target, k=5):
    """
    Calculate top-k accuracy, give preds for all classes!!
    https://torchmetrics.readthedocs.io/en/latest/references/functional.html#accuracy-func
    >>> target = torch.tensor([0, 1, 2])
    >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
    tensor(0.6667)
    """
    return accuracy(preds, target, top_k=k)


def bce_logits(logits, target):
    """
    Calculate binary cross entropy loss for logits
    """
    return F.binary_cross_entropy_with_logits(logits, F.softmax(target, dim=1))


def KLDiv(logits, y):
    """
    Calculate KLDiv loss
    """
    return F.kl_div(torch.log(logits), y, reduction="batchmean")


def neg_sampling_softmax(logits, window):
    """
    implements neg sampling
    logits -- (batch, num_classes)
    """
    summed_tensor = F.avg_pool1d(logits, kernel_size=window, stride=1) * window
    r1 = torch.randint(2 * window, logits.shape[1] - window, (1,))
    t1 = torch.zeros(*logits.shape).cuda() + summed_tensor[:, r1]
    t1[:, r1 : r1 + window] = summed_tensor[:, torch.randint(0, window, (1,))]
    ret = logits / (logits + t1)
    # del t1 ########## should I do this would this affect backprop
    return ret


def train_step_log(logger, loss, accuracy_top1, accuracy_top5):
    """
    Log training metrics, given the logger object, use this to maintain uniformity in logging
    """
    logger.log(
        "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
    )
    logger.log(
        "train_acc_top1",
        accuracy_top1,
        on_step=True,
        on_epoch=True,
        logger=True,
        prog_bar=True,
    )
    logger.log(
        "train_acc_top5",
        accuracy_top5,
        on_step=True,
        on_epoch=True,
        logger=True,
        prog_bar=True,
    )


def validation_step_log(logger, loss, accuracy_top1, accuracy_top5):
    """
    Log validation metrics, given the logger object, use this to maintain uniformity in logging
    """
    logger.log(
        "val_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
    )
    logger.log(
        "val_acc_top1",
        accuracy_top1,
        on_step=True,
        on_epoch=True,
        logger=True,
        prog_bar=True,
    )
    logger.log(
        "val_acc_top5",
        accuracy_top5,
        on_step=True,
        on_epoch=True,
        logger=True,
        prog_bar=True,
    )


def test_step_log(logger, loss, accuracy_top1, accuracy_top5):
    """
    Log test metrics, given the logger object, use this to maintain uniformity in logging
    """
    logger.log(
        "test_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
    )
    logger.log(
        "test_acc_top1",
        accuracy_top1,
        on_step=True,
        on_epoch=True,
        logger=True,
        prog_bar=True,
    )
    logger.log(
        "test_acc_top5",
        accuracy_top5,
        on_step=True,
        on_epoch=True,
        logger=True,
        prog_bar=True,
    )


def neg_sampling_test():
    logits = torch.ones(4, 20).cuda()
    logits = F.softmax(logits, dim=1)
    print(logits)
    logits_neg = neg_sampling_softmax(logits, 5)
    print(logits_neg)
