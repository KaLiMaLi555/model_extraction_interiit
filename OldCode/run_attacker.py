# List of imports
from PIL import Image
from typing import Type, Any, Callable, Union, List, Optional

## PyTorch
import torch
from torch import Tensor
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
## PyTorch lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import torchmetrics
# Torchvision
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pickle
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
from utils import metrics
from env import seed_all

import argparse

from dataloader import VideoLogitDataset, VideoLogitDatasetFromDisk
from cnn_lstm import ResCNNRNN

class WrapperModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):

        super().__init__()        
        self.model = model
        self.learning_rate = learning_rate
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = F.softmax(self.forward(x), dim=1)
        loss = metrics.KLDiv(probs, y)
        true_preds = torch.argmax(y, dim=1)
        accuracy_top1 = metrics.topk_accuracy(probs, true_preds, k=1)
        accuracy_top5 = metrics.topk_accuracy(probs, true_preds, k=5)
        metrics.train_step_log(self, loss, accuracy_top1, accuracy_top5)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = F.softmax(self.forward(x), dim=1)
        loss = metrics.KLDiv(probs, y)
        true_preds = torch.argmax(y, dim=1)
        accuracy_top1 = metrics.topk_accuracy(probs, true_preds, k=1)
        accuracy_top5 = metrics.topk_accuracy(probs, true_preds, k=5)
        metrics.validation_step_log(self, loss, accuracy_top1, accuracy_top5)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        probs = F.softmax(self.forward(x), dim=1)
        loss = metrics.KLDiv(probs, y)
        true_preds = torch.argmax(y, dim=1)
        accuracy_top1 = metrics.topk_accuracy(probs, true_preds, k=1)
        accuracy_top5 = metrics.topk_accuracy(probs, true_preds, k=5)
        metrics.test_step_log(self, loss, accuracy_top1, accuracy_top5)
        # self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for training')
    parser.add_argument('--train_logits_file', type=str)
    parser.add_argument('--val_logits_file', type=str)
    parser.add_argument('--test_logits_file', type=str)
    parser.add_argument('--train_input_dir', type=str)
    parser.add_argument('--val_input_dir', type=str)
    parser.add_argument('--test_input_dir', type=str)
    parser.add_argument('--attacker_model_name', type=str)
    parser.add_argument('--victim_model_name', type=str)
    parser.add_argument('--epochs', type=int)
    # parser.add_argument('--resnet_lstm_trainable_layers', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--load_train_from_disk', action='store_true')
    parser.add_argument('--load_val_from_disk', action='store_true')
    parser.add_argument('--load_test_from_disk', action='store_true')
    parser.add_argument('--wandb_name', type=str, default=None)

    args = parser.parse_args()

    train_input_dir = args.train_input_dir
    val_input_dir = args.val_input_dir
    test_input_dir = args.test_input_dir
    train_logits_file = args.train_logits_file
    val_logits_file = args.val_logits_file
    test_logits_file = args.test_logits_file
    attacker_model_name = args.attacker_model_name
    victim_model_name = args.victim_model_name
    learning_rate = args.learning_rate

    seed_all()

    if args.load_train_from_disk:
        train_video_data = VideoLogitDatasetFromDisk(train_input_dir, train_logits_file)
    else:
        train_video_data = VideoLogitDataset(train_input_dir, train_logits_file)

    train_size = int(len(train_video_data)*0.9)
    if val_input_dir:
        train_data = train_video_data
        if args.load_val_from_disk:
            val_data = VideoLogitDatasetFromDisk(val_input_dir, val_logits_file)
        else:
            val_data = VideoLogitDataset(val_input_dir, val_logits_file)
    else:
        train_data, val_data = data.random_split(train_video_data, [train_size, len(train_video_data) - train_size])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, drop_last=False, num_workers=2)

    if args.load_test_from_disk:
        test_video_data = VideoLogitDatasetFromDisk(test_input_dir, test_logits_file)
    else:
        test_video_data = VideoLogitDataset(test_input_dir, test_logits_file)

    test_loader = DataLoader(test_video_data, batch_size=32, shuffle=False, drop_last=False, num_workers=2)
    
    if victim_model_name == 'swin-t':
        num_classes = 400
    elif victim_model_name == 'movinet':
        num_classes = 600
    else:
        print("unknown victim name")
        exit(-1)
    

    wandb_logger = WandbLogger(project="model_extraction", log_model="all", name=args.wandb_name)
    wandb_logger.log_hyperparams({
        "train_logits_file": train_logits_file,
        "val_logits_file": val_logits_file,
        "test_logits_file": test_logits_file,
        "train_input_dir": train_input_dir,
        "val_input_dir": val_input_dir,
        "test_input_dir": test_input_dir,
        "attacker_model_name": attacker_model_name,
        "victim_model_name": victim_model_name,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
    })
    if attacker_model_name == 'resnet-lstm':
        model_internal = ResCNNRNN(num_classes=num_classes)
    else:
        print("Unknown attacker name")
        exit(-1)
    model = WrapperModel(model_internal, learning_rate=learning_rate)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    trainer = pl.Trainer(max_epochs=args.epochs,
                progress_bar_refresh_rate=20, 
                gpus=1, logger=wandb_logger, callbacks=[checkpoint_callback])
    
    seed_all()
    trainer.fit(model, train_loader, val_loader)
    # test best val model
    trainer.test(dataloaders=test_loader)
