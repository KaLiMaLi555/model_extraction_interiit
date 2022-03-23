import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from Datasets.datasets import VideoLabelDataset, VideoLogitDatasetMovinet, VideoLogitDatasetSwinT
from models.MARS.model import generate_model
# from vidaug import augmentors as va
from options.train_options import *
from utils.augment import va_augment
from utils.mars_utils import *

"""
    Function to train the model

    Parameters:
        model (torch.nn.Module): The model to train
        train_loader (torch.utils.data.DataLoader): The training data
        optimizer (torch.optim.Optimizer): The optimizer to use
        criterion (torch.nn.modules.loss): The loss function to use
        epochs (int): Current epoch
        scheduler (torch.optim.lr_scheduler): The scheduler to use

    Returns:
        float: The average validation accuracy
"""


def train(cfg, model, train_loader, optimizer, criterion, epoch, scheduler=None):
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = torch.permute(inputs, (0, 1, 4, 2, 3))
        targets = targets.to(torch.float32).argmax(dim=1)
        targets = targets.cuda(non_blocking=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, F.one_hot(targets, num_classes=cfg.num_classes))

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
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
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            print('Val_Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
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
    cfg = opt.initialize()
    opt.print_options(cfg)
    cfg = cfg["experiment"]

    print("Creating Model")
    model, parameters = generate_model(cfg.num_classes)

    if cfg.train_mode == "finetune":
        model.module.load_state_dict(torch.load(cfg.pretrained_ckpt_path))

    print("Model Created")

    print("Creating Dataloaders")
    if cfg.augmentations:
        augs_list = cfg.augmentations
        va_aug = va_augment(augs_list)
        if cfg.dataset == "k400":
            finetune_dataset = VideoLogitDatasetMovinet(cfg.train_vid_dir, cfg.train_logits_file, size=(224, 224), va_augments=va_aug)
        elif cfg.dataset == "k600":
            finetune_dataset = VideoLogitDatasetSwinT(cfg.train_vid_dir, cfg.train_vid_names_file, cfg.train_logits_file, size=(224, 224), va_augments=va_aug)    
    else:
        if cfg.dataset == "k400":
            finetune_dataset = VideoLogitDatasetMovinet(cfg.train_vid_dir, cfg.train_logits_file, size=(224, 224))
        elif cfg.dataset == "k600":
            finetune_dataset = VideoLogitDatasetSwinT(cfg.train_vid_dir, cfg.train_vid_names_file, cfg.train_logits_file, size=(224, 224))
    val_dataset = VideoLabelDataset(cfg.val_vid_dir, cfg.val_classes_file, cfg.val_labels_file, cfg.num_classes, size=(224, 224))

    train_data = finetune_dataset
    val_data = val_dataset

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                pin_memory=True)

    print("Creating Training Parameters")
    optimizer = optim.SGD(
        parameters,
        lr=cfg.lr,
        momentum=cfg.momentum,
        dampening=cfg.dampening,
        weight_decay=cfg.weight_decay)

    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=cfg.lr_patience)

    best_val_acc = 0
    for epoch in range(1, cfg.epochs + 1):
        train_acc = train(cfg, model, train_dataloader, optimizer, criterion, epoch, scheduler)
        val_acc = val(model, val_dataloader, criterion, epoch)
        scheduler.step(epoch)
        save_path = os.path.join(cfg.save_path, "model_epoch_" + str(epoch) + ".pth")
        torch.save(model.state_dict(), save_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_save_path = os.path.join(cfg.ckpts_dir, "best_model.pth")
            torch.save(model.state_dict(), best_save_path)

    if cfg.mode == "pretrain":
        torch.save(model.state_dict(), os.path.join(cfg.pretrained_ckpt_path))


if __name__ == '__main__':
    main()
