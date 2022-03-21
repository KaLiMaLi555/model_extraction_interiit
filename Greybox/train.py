import torch
import time
import wandb
import torch.nn as nn
from Datasets.datasets import VideoLogitDataset, VideoLabelDataset
from models.MARS.model import generate_model
from utils.mars_utils import *

from torch.autograd import Variable


def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, scheduler=None):
    best_val_acc = 0
    for epoch in range(1, epochs + 1):

        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            data_time.update(time.time() - end_time)
            inputs = torch.permute(inputs, (0, 1, 4, 2, 3))
            print(inputs.shape)
            targets = targets.to(torch.float32).argmax(dim=1)
            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, F.one_hot(targets, num_classes=400))

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies))
            wandb.log({
                'Training loss': losses.avg,
                'Training Accuracy': accuracies.avg, }, step=epoch)


    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = torch.permute(inputs, (0, 1, 4, 2, 3))
            targets = targets.to(torch.float32)
            targets = torch.nn.functional.softmax(targets)
            # pdb.set_trace()
            data_time.update(time.time() - end_time)
            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print(loss.item())
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Val_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(val_dataloader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies))
            wandb.log({
                'Validation loss': losses.avg,
                'Validation Accuracy': accuracies.avg, }, step=epoch)
    if (accuracies.avg > best_val_acc):
        best_val_acc = accuracies.avg
        torch.save(model.module.state_dict(),
                   '/content/drive/MyDrive/checkpoints/K400+UCF30k_SwinT_logit_best_model.pth')
        # wandb.save('/content/drive/MyDrive/checkpoints/MARS_K600/Exp2/best_model.pth')
