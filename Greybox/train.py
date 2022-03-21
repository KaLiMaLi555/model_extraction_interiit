from re import L
import time
from pygame import init
from vidaug import augmentors as va

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.autograd import Variable
from utils.wandb import init_wandb

from Datasets.datasets import VideoLabelDataset, VideoLogitDataset
from torch.utils.data import Dataset, DataLoader
from models.MARS.model import generate_model
from utils.mars_utils import *

def train(model, train_loader, val_loader, optimizer, criterion, epoch, scheduler=None):
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, (inputs, targets) in enumerate(train_loader):
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

        print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
            epoch,
            i + 1,
            len(train_loader),
            loss=losses,
            acc=accuracies))

    return accuracies.avg

def val(model, val_dataloader, criterion, epoch):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_dataloader):
            inputs = torch.permute(inputs, (0, 1, 4, 2, 3))
            targets = targets.to(torch.float32)
            targets = torch.nn.functional.softmax(targets)
            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print(loss.item())
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            print('Val_Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(val_dataloader),
                loss=losses,
                acc=accuracies))

    return accuracies.avg

def main():

    print("Creating Model")
    model, parameters = generate_model(cfg.pretrain_path_ucf, cfg.n_finetune_classes)

    init_wandb(model, parameters)

    print("Creating Dataloaders")
    if cfg.train_mode == 'logit':
        finetune_dataset = VideoLogitDataset(cfg.train_video_path, cfg.train_video_name_path, cfg.train_logit_path)
        val_dataset = VideoLabelDataset(cfg.val_video_path, cfg.val_class_file, cfg.val_label_file, cfg.n_finetune_classes)
    elif cfg.train_mode == 'label':
        finetune_dataset = VideoLabelDataset(cfg.train_video_path, cfg.train_video_name_path, cfg.train_label_path)
        val_dataset = VideoLabelDataset(cfg.val_video_path, cfg.val_class_file, cfg.val_label_file, cfg.n_finetune_classes)
    else:
        raise ValueError('Unknown train mode: {}'.format(cfg.train_mode))

    train_data = finetune_dataset
    val_data = val_dataset    

    train_dataloader = DataLoader(train_data, batch_size = cfg.batch_size, shuffle=True, num_workers = cfg.n_workers, pin_memory = True)
    val_dataloader   = DataLoader(val_data, batch_size = cfg.batch_size, shuffle=True, num_workers = cfg.n_workers, pin_memory = True)

    print("Creating Training Parameters")
    optimizer = optim.SGD(
        parameters,
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        dampening=cfg.dampening,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov)

    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=cfg.lr_patience)

    best_val_acc = 0
    for epoch in range(1, cfg.n_epochs + 1):
        train_acc = train(model, train_dataloader, optimizer, criterion, epoch, device)
        val_acc = val(model, val_dataloader, criterion, epoch, device)
        scheduler.step(epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.module.state_dict(), cfg.save_path)

if __name__ == '__main__':
    main()