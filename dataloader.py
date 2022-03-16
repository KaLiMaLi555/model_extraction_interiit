# Import the required libraries
import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from env import set_seed
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import pickle
<<<<<<< HEAD
=======
import mmcv
import matplotlib.pyplot as plt
>>>>>>> 0cd37a07819bef52d24a2e9da56f4cb713b7be54

# TODO: The class is implemented now for random, 
# Do if we have both the actual images and labels
class CustomResizeTransform:

    def __init__(self):
        pass

    def custom_resize_transform(self, vid, size=224):
      x = vid
      video_transform = []
      for i, image in enumerate(list(x)):
          image = TF.resize(image, size+32)
          image = TF.center_crop(image, (size, size))
          image = image.permute(1, 2, 0)
          image = torch.tensor(mmcv.imnormalize(image.numpy(), np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375]), False))
          video_transform.append(image)
      return torch.stack(video_transform)

    def __call__(self, vid):
        vid = self.custom_resize_transform(vid) # Resize
        return vid

class VideoDataset(Dataset):

    def __init__(self, video_dir_path):

        self.video_dir_path = video_dir_path
        self.instances = []     # Tensor of image frames
        # self.logits = pickle.load(open(logits_file, 'rb'))
        self.transform = CustomResizeTransform()

        self.videos = sorted([x for x in os.listdir(self.video_dir_path) if os.path.isdir(os.path.join(self.video_dir_path, x))])
        self.get_frames()
        print(len(self.videos), "videos")

        self.instances = torch.stack(self.instances)
        self.num_instances = len(self.instances)

    def get_frames(self):

        for video in tqdm(self.videos, position = 0, leave = True):

            image_frames = []
            video_dir = os.path.join(self.video_dir_path, video)
            images = os.listdir(video_dir)

            for image_name in images:
                image = Image.open(os.path.join(video_dir, image_name))
                image = np.array(image, dtype = np.uint8)
                image_frames.append(torch.tensor(image))

            if self.transform:
                image_frames = self.transform(
                  torch.stack(image_frames).permute(0, 3, 1, 2)
                  ).permute(0, 2, 3, 1)
            self.instances.append(image_frames)

    def __getitem__(self, idx):
        return self.instances[idx], self.logits[idx]

    def __len__(self):
        return len(self.instances)

class VideoDatasetFromDisk(Dataset):

    def __init__(self, video_dir_path):

        self.video_dir_path = video_dir_path
        self.transform = CustomResizeTransform()
        # self.logits = pickle.load(open(logits_file, 'rb'))

        self.videos = sorted([x for x in os.listdir(self.video_dir_path) if os.path.isdir(os.path.join(self.video_dir_path, x))])
        print(len(self.videos), "videos")
        with open("test_files.txt", "w+") as f:
          f.writelines([f"{x}\n" for x in self.videos])
        self.num_instances = len(self.videos)

    def get_frames(self, video_path):
        images = os.listdir(video_path)
        image_frames = []

        for image_name in images:
            image = Image.open(os.path.join(video_path, image_name))
            image = np.array(image, dtype = np.uint8)
            image_frames.append(torch.tensor(image))
        if len(image_frames) > 0:
            return torch.stack(image_frames)
        else:
            return torch.zeros((16, 224, 224, 3))

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir_path, self.videos[idx])
        vid = self.get_frames(video_path)
        if self.transform:
            vid = vid.permute(0, 3, 1, 2)
            vid = self.transform(vid)
        pad_shape = 16 - vid.shape[0]
        vid = torch.cat((vid, torch.zeros(pad_shape, vid.shape[1], vid.shape[2], vid.shape[3])))
        # print(vid.shape, vid)
        return vid

    def __len__(self):
        return self.num_instances

class Dataloader():

    def __init__(self, video_dir_path, num_classes = 5, is_random = True):

        self.video_dir_path = video_dir_path
        self.instances = []     # Tensor of image frames
        self.labels = []        # Class labels
        self.num_classes = num_classes

        self.is_random = is_random

        self.videos = [x for x in os.listdir(self.video_dir_path) if os.path.isdir(os.path.join(self.video_dir_path, x))]
        print(self.videos)
        self.get_frames()

        self.instances = torch.stack(self.instances)
        self.num_instances = len(self.instances)

        self.image_idx_mapping = {}
        self.idx_image_mapping = {}
        
        if self.is_random:
            self.create_labels()
        else:
            self.get_labels()

    def get_frames(self):
        
        i = 0
        for video in tqdm(self.videos, position = 0, leave = True):
            
            self.image_idx_mapping[video] = i
            self.idx_image_mapping[i] = video
            i += 1
            image_frames = []
            video_dir = os.path.join(self.video_dir_path, video)
            images = os.listdir(video_dir)
            
            for image_name in images:
                image = Image.open(os.path.join(video_dir, image_name))
                image = np.array(image, dtype = np.float32)
                image_frames.append(torch.tensor(image))

            self.instances.append(torch.stack(image_frames))

    def create_labels(self):

        self.labels = np.zeros(self.num_instances)
        permute = np.arange(0, 1000)
        random.shuffle(permute)

        num_image_per_class = self.num_instances / self.num_classes
        c = -1
        for i in range(self.num_instances):
            index = permute[i]
            if i % num_image_per_class == 0:
                c+=1
            self.labels[index] = c

        self.labels = torch.tensor(self.labels)