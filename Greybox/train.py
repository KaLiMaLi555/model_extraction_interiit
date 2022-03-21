import time
from re import L

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from pygame import init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from vidaug import augmentors as va
from options.train_options import *

from Datasets.datasets import VideoLabelDataset, VideoLogitDataset
from models.MARS.model import generate_model
from utils.mars_utils import *
from utils.wandb import init_wandb

"""
    Function to train the model

    Parameters:
        model (torch.nn.Module): The model to train
        train_loader (torch.utils.data.DataLoader): The training data
        val_loader (torch.utils.data.DataLoader): The validation data
        optimizer (torch.optim.Optimizer): The optimizer to use
        criterion (torch.nn.modules.loss): The loss function to use
        epochs (int): Current epoch
        scheduler (torch.optim.lr_scheduler): The scheduler to use

    Returns:
        float: The average validation accuracy
"""
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

"""
    Function to evaluate the model on the validation set

    Parameters:
        model (torch.nn.Module): The model to evaluate
        val_dataloader (torch.utils.data.DataLoader): The dataloader for the validation set
        criterion (torch.nn.CrossEntropyLoss): The loss function
        epoch (int): The current epoch

    Returns:
        float: The average validation loss
"""
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

"""
    Main function to train the model
"""
def main():

    opt = TrainOptions()
    args = opt.initialize()
    opt.print_options(args)
    cfg = args

    print("Creating Model")
    model, parameters = generate_model(cfg.pretrain_path_ucf, cfg.n_finetune_classes)

    init_wandb(model, parameters)

    print("Creating Dataloaders")
    finetune_dataset = VideoLogitDataset(cfg.train_video_path, cfg.train_video_name_path, cfg.train_logit_path)
    val_dataset = VideoLabelDataset(cfg.val_video_path, cfg.val_class_file, cfg.val_label_file, cfg.n_finetune_classes)

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
        train_acc = train(model, train_dataloader, optimizer, criterion, epoch, scheduler)
        val_acc = val(model, val_dataloader, criterion, epoch)
        scheduler.step(epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.module.state_dict(), cfg.save_path)

if __name__ == '__main__':
    main()