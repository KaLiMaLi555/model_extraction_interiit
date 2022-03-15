"""
    Add a single unified script for calculating all metrics, and making all plots
"""
import torch
from torchmetrics.functional import accuracy
import torch.nn.functional as F
import numpy as np
import random as rn
# import pytorch_lightning as pl
# import tensorflow as tf

SEED_ = 42


def seed_all(SEED=SEED_):
    rn.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # pl.seed_everything(SEED)
    # tf.random.set_seed(SEED)


seed_all(SEED_)


def topk_accuracy(preds, target, k=5):
    """
    Calculate top-k accuracy, give preds for all classes!!
    https://torchmetrics.readthedocs.io/en/latest/references/functional.html#accuracy-func
    >>> target = torch.tensor([0, 1, 2])
    >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
    tensor(0.6667)
    """
    return accuracy(preds, torch.argmax(target, dim=1), top_k=k)


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


def train_step_log(logger, loss, accuracy):
    """
    Log training metrics, given the logger object, use this to maintain uniformity in logging
    """
    logger(
        "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
    )
    logger(
        "train_accuracy",
        accuracy,
        on_step=True,
        on_epoch=True,
        logger=True,
        prog_bar=False,
    )


def validation_step_log(logger, loss, accuracy):
    """
    Log validation metrics, given the logger object, use this to maintain uniformity in logging
    """
    logger(
        "val_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
    )
    logger(
        "val_accuracy",
        accuracy,
        on_step=True,
        on_epoch=True,
        logger=True,
        prog_bar=False,
    )


def test_step_log(logger, loss, accuracy):
    """
    Log test metrics, given the logger object, use this to maintain uniformity in logging
    """
    logger(
        "test_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=False
    )
    logger(
        "test_accuracy",
        accuracy,
        on_step=True,
        on_epoch=True,
        logger=True,
        prog_bar=True,
    )
