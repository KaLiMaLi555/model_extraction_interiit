# List of imports
from PIL import Image
from typing import Type, Any, Callable, Union, List, Optional
from utils.config import process_config
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


## PyTorch
import torch
from torch import Tensor
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
## PyTorch lightning
import pytorch_lightning as pl
import torchmetrics
# Torchvision
import torchvision
from torchvision import transforms
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import pickle
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os

import argparse

config = process_config("config/config1.json")

parser = argparse.ArgumentParser(description='Overwrite Config')

parser.add_argument('--input_dir', type=str, default=config.input_dir)
parser.add_argument('--logits_file', type=str, default=config.logits_file)
parser.add_argument('--save', type=bool, default=config.save)

parser.add_argument('--wandb_api_key', type=str)
parser.add_argument('--wandb', type=bool, default=config.wandb)
parser.add_argument('--wandb_project', type=str, default=config.wandb_project)
parser.add_argument('--wandb_name', type=str, default=config.wandb_name)
parser.add_argument('--wandb_id', type=str, default=config.wandb_id)
parser.add_argument('--resume', type=int, default=config.resume)

parser.add_argument('--epochs', type=int, default=config.epochs)
parser.add_argument('--train_batch_size', type=int, default=config.train_batch_size)
parser.add_argument('--val_batch_size', type=int, default=config.val_batch_size)
parser.add_argument('--lr', type=float, default=config.lr)
parser.add_argument('--wt_decay', type=float, default=config.wt_decay)
parser.add_argument('--checkpoint_path', type=str, default=config.checkpoint_path)

args = parser.parse_args()




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
        k=0
        for video in tqdm(self.videos, position = 0, leave = True):
          if k<60:
            
            image_frames = []
            video_dir = os.path.join(self.video_dir_path, video)
            images = os.listdir(video_dir)
            
            for image_name in images:
                image = Image.open(os.path.join(video_dir, image_name))
                image = np.array(image, dtype = np.float32)
                image_frames.append(torch.tensor(image))

            self.instances.append(torch.stack(image_frames))
            k=k+1

    def __getitem__(self, idx):
      #print(self.instances[idx].shape)
      a = self.instances[idx]
      a = a.swapaxes(0,3)
      #print(a.shape)
      return a, self.logits[idx]
      #return self.instances[idx], self.logits[idx]

    def __len__(self):
        return len(self.instances)


class C3D(nn.Module):
    ''' The C3D network '''

    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv6a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv6b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(25088, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 25088)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        
        logits = self.fc8(x)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class WrapperModel(pl.LightningModule):
    def __init__(self, model, learning_rate=args.lr):
        super().__init__()        
        self.model = model
        self.learning_rate = learning_rate
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        if args.wandb:
            self.save_hyperparameters()
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = F.softmax(self.forward(x), dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, F.softmax(y, dim=1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = F.softmax(self.forward(x), dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, F.softmax(y, dim=1))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":    

    # WandB Setup
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if args.wandb and args.resume:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name = args.wandb_name,
            id=args.wandb_id,
            log_model='all',  # log all new checkpoints during training
            resume='allow'
        )

    else:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name = args.wandb_name,
            log_model='all', # log all new checkpoints during training
            resume=None
        ) 

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k =2,                 
        mode='min',                    
        every_n_epochs=1,             
        save_on_train_epoch_end=True,
        save_last = True    
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    video_data = VideoLogitDataset(args.input_dir, args.logits_file)
    train_size = int(len(video_data)*0.9)
    train_data, val_data = data.random_split(video_data, [train_size, len(video_data) - train_size])
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=2)
    model_internal = C3D(num_classes=train_data[0][1].shape[0])
    model = WrapperModel(model_internal)
    
    if config.wandb_watch:
        wandb_logger.watch(
            model,
            log='all',
            log_freq=config.wandb_watch_freq,
            log_graph=True
        )  # logs histogram of gradients and parameters

    if args.resume:
        run = wandb.init(project=args.wandb_project, resume=args.resume)
        artifact = run.use_artifact(args.checkpoint_path, type='model')
        artifact_dir = artifact.download()  #should change these lines so that user can specify path (now just for testing)
        model =C3D.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", num_classes=train_data[0][1].shape[0] )
        resume_path = "artifacts/" + artifact_path + "/model.ckpt"
        trainer = pl.Trainer(max_epochs=epochs,
            progress_bar_refresh_rate=1, 
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor],
            resume_from_checkpoint=args.checkpoint_path,
            gpus=1)


    trainer = pl.Trainer(max_epochs=epochs,
                progress_bar_refresh_rate=1, 
                log_every_n_steps=1,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, lr_monitor],
                gpus=1)

    trainer.fit(model, train_loader, val_loader)
    print("Run complete")
    wandb.finish()
