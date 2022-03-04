'''
    Add a single unified script for calculating all metrics, and making all plots 
'''
import torch
from torchmetrics.functional import accuracy
import torch.nn.functional as F

def topk_accuracy(preds, target, k=5):
    ''' 
        Calculate top-k accuracy, give preds for all classes!! 
        https://torchmetrics.readthedocs.io/en/latest/references/functional.html#accuracy-func 
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        tensor(0.6667)
    '''
    return accuracy(preds,target,top_k=k)

def bce_logits(logits, target):
    '''
        Calculate binary cross entropy loss for logits
    '''
    return F.binary_cross_entropy_with_logits(logits, F.softmax(target, dim=1))

def train_step_log(logger,loss,accuracy):
    '''
        Log training metrics, given the logger object, use this to maintain uniformity in logging
    '''
    logger.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
    logger.log('train_accuracy', accuracy, on_step=True, on_epoch=True, logger=True, prog_bar=True)

def validation_step_log(logger,loss,accuracy):
    '''
        Log validation metrics, given the logger object, use this to maintain uniformity in logging
    '''
    logger.log('validation_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
    logger.log('validation_accuracy', accuracy, on_step=True, on_epoch=True, logger=True, prog_bar=True)


