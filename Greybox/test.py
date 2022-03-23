import time
from re import L

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from Datasets.datasets import VideoLabelDataset, VideoLogitDataset
from models.MARS.model import generate_model
from options.test_options import TestOptions
from utils.mars_utils import *

"""
    Function to test the model

    Parameters:
        model (torch.nn.Module): The model to test
        test_dataloader (torch.utils.data.DataLoader): The test data
        criterion (torch.nn.modules.loss): The loss function to use

    Returns:
        float: The average validation accuracy
"""
def test(model, test_dataloader, criterion):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dataloader):
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

            print('Test_Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                1,
                i + 1,
                len(test_dataloader),
                loss=losses,
                acc=accuracies))

    return accuracies.avg

"""
    Main function to run the program
"""
def main():
    opt = TestOptions()
    cfg = opt.initialize()
    cfg = cfg["test"]
    opt.print_options(cfg)

    print("Creating Model")
    model, parameters = generate_model(cfg.num_classes)
    model.module.load_state_dict(torch.load(cfg.restore_from))


    print("Creating Dataloaders")
    test_dataset = VideoLabelDataset(cfg.videos_dir, cfg.classes_file, cfg.labels_file, cfg.num_classes)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print("Dataloaders created")

    criterion = nn.CrossEntropyLoss().cuda()
    test(model, test_dataloader, criterion)

if __name__ == '__main__':
    main()
