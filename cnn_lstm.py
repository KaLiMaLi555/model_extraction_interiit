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

import argparse

# TODO: The class is implemented now for random, 
# Do if we have both the actual images and labels
class VideoLogitDataset(Dataset):

    def __init__(self, video_dir_path, logits_file):

        self.video_dir_path = video_dir_path
        self.instances = []     # Tensor of image frames
        self.logits = pickle.load(open(logits_file, 'rb'))

        self.videos = os.listdir(self.video_dir_path)
        self.get_frames()

        self.instances = torch.stack(self.instances)
        self.num_instances = len(self.instances)

    def get_frames(self):
        
        for video in tqdm(self.videos, position = 0, leave = True):
            
            image_frames = []
            video_dir = os.path.join(self.video_dir_path, video)
            images = os.listdir(video_dir)
            
            for image_name in images:
                image = Image.open(os.path.join(video_dir, image_name))
                image = np.array(image, dtype = np.float32)
                image_frames.append(torch.tensor(image))

            self.instances.append(torch.stack(image_frames))

    def __getitem__(self, idx):
        return self.instances[idx], self.logits[idx]

    def __len__(self):
        return len(self.instances)

class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=400):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


# 2D CNN encoder using ResNet-152 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        for module in modules[:-1]:
          for param in module.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            x = x_3d[:, t, :, :, :]
            x = x.reshape((-1, x.shape[3], x.shape[1], x.shape[2]))
            x = self.resnet(x)  # ResNet
            x = x.view(x.size(0), -1)             # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq

class ResCNNRNN(nn.Module):

    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, num_classes=400):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNRNN, self).__init__()

        self.encoder = ResCNNEncoder(fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2, drop_p=drop_p, CNN_embed_dim=CNN_embed_dim)
        self.decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=h_RNN_layers, h_RNN=h_RNN, h_FC_dim=h_FC_dim, drop_p=drop_p, num_classes=num_classes)
        
    def forward(self, x_3d):
        return self.decoder(self.encoder(x_3d))

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
        loss = F.kl_div(torch.log(probs), y, reduction="batchmean")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = F.softmax(self.forward(x), dim=1)
        loss = F.kl_div(torch.log(probs), y, reduction="batchmean")
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        probs = F.softmax(self.forward(x), dim=1)
        loss = F.kl_div(torch.log(probs), y, reduction="batchmean")
        preds = torch.argmax(probs, dim=1)
        true_preds = torch.argmax(y, dim=1)
        accuracy = torch.sum(torch.where(preds == true_preds, 1, 0))/preds.shape[0]
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for training')
    parser.add_argument('--train_logits_file', type=str)
    parser.add_argument('--test_logits_file', type=str)
    parser.add_argument('--train_input_dir', type=str)
    parser.add_argument('--test_input_dir', type=str)
    parser.add_argument('--attacker_model_name', type=str)
    parser.add_argument('--victim_model_name', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resnet_lstm_trainable_layers', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()

    train_input_dir = args.train_input_dir
    test_input_dir = args.test_input_dir
    train_logits_file = args.train_logits_file
    test_logits_file = args.test_logits_file
    attacker_model_name = args.attacker_model_name
    victim_model_name = args.victim_model_name
    learning_rate = args.learning_rate

    train_video_data = VideoLogitDataset(train_input_dir, train_logits_file)
    train_size = int(len(train_video_data)*0.9)
    train_data, val_data = data.random_split(train_video_data, [train_size, len(video_data) - train_size])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, drop_last=False, num_workers=2)

    test_video_data = VideoLogitDataset(test_input_dir, test_logits_file)
    test_loader = DataLoader(test_video_data, batch_size=32, shuffle=False, drop_last=False, num_workers=2)
    
    if victim_model_name == 'swin-t':
        num_classes = 400
    elif victim_model_name == 'movinet':
        num_classes = 600
    else:
        print("unknown victim name")
        exit(-1)
    

    wandb_logger = WandbLogger(project="model_extraction", log_model="all")
    wandb_logger.log_hyperparams({
        "train_logits_file": train_logits_file,
        "test_logits_file": test_logits_file,
        "train_input_dir": train_input_dir,
        "test_input_dir": test_input_dir,
        "attacker_model_name": attacker_model_name,
        "victim_model_name": victim_model_name,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
    })
    if attacker_model_name == 'resnet-lstm':
        model_internal = ResCNNRNN(num_classes=num_classes)
        wandb_logger.log_hyperparams({
            "resnet_lstm_trainable_layers": args.resnet_lstm_trainable_layers
        })
    else:
        print("Unknown attacker name")
        exit(-1)
    model = WrapperModel(model_internal, learning_rate=learning_rate)
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    trainer = pl.Trainer(max_epochs=args.epochs,
                progress_bar_refresh_rate=20, 
                gpus=1, logger=wandb_logger, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)
    # test best val model
    trainer.test(dataloaders=test_loader)
