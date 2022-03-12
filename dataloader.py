# Import the required libraries
import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
# from env import set_seed
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import pickle
import mmcv
import matplotlib.pyplot as plt

# TODO: The class is implemented now for random, 
# Do if we have both the actual images and labels
class SwinTransform:

    def __init__(self, n_clips=2, clip_len=16):
        self.n_clips = n_clips
        self.clip_len = clip_len
        self.total_frames = self.n_clips*self.clip_len

    def transform(self, vid):
        x = list(vid)
        n = self.total_frames//len(x)
        x = x*n + x[:(self.total_frames % len(x))]
        assert len(x) == self.total_frames, "clip length mismatch"
        video_transform = []
        for image in x:
            im = im.permute(2, 0, 1)
            im = TF.resize(im, 224)
            im = TF.center_crop(im, (224, 224))
            im = im.float()
            im = TF.normalize(im, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
            im = im.permute(1, 2, 0)
            video_transform.append(image)
        vid = torch.stack(video_transform)
        vid = vid.reshape((-1, self.n_clips, self.clip_len, 224, 224, 3))
        vid = vid.permute(0, 1, 5, 2, 3, 4)
        vid = vid.reshape((-1, 3, self.clip_len, 224, 224))
        return vid

    def __call__(self, vid):
        vid = self.transform(vid) # Resize
        return vid

class MovinetTransform:

    def __init__(self):
        pass

    def transform(self, vid):
        x = list(vid)
        video_transform = []
        for image in x:
            im = im.permute(2, 0, 1)
            im = TF.resize(im, 224)
            im = TF.center_crop(im, (224, 224))
            im = im.float()
            im = im/255.0
            im = im.permute(1, 2, 0)
            video_transform.append(image)
        vid = torch.stack(video_transform)
        return vid

    def __call__(self, vid):
        vid = self.transform(vid) # Resize
        return vid

class VideoDataset(Dataset):

    def __init__(self, video_dir_path, video_names_file, transforms, logits_file=None):

        self.video_dir_path = video_dir_path
        self.instances = []     # Tensor of image frames
        if logits_file is not None:
            self.logits = pickle.load(open(logits_file, 'rb'))
        else:
            self.logits = None
        self.transform = transforms
        
        with open(video_names_file) as f:
            self.videos = [os.path.join(video_dir_path, x[:-1]) for x in f.readlines()]
        self.get_frames()

        self.instances = torch.stack(self.instances)
        self.num_instances = len(self.instances)

    def get_frames(self):

        for video in tqdm(self.videos, position = 0, leave = True):

            image_frames = []
            images = os.listdir(video)

            for image_name in images:
                image = Image.open(os.path.join(video, image_name))
                image = np.array(image, dtype = np.uint8)
                image_frames.append(torch.tensor(image))

            vid = self.transform(image_frames)
            self.instances.append(vid)

    def __getitem__(self, idx):
        if self.logits is not None:
            return self.instances[idx], self.logits[idx]
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

class VideoDatasetFromDisk(Dataset):

    def __init__(self, video_dir_path, video_names_file, transforms, logits_file=None):

        self.video_dir_path = video_dir_path
        self.transform = transforms
        if logits_file is not None:
            self.logits = pickle.load(open(logits_file, 'rb'))
        else:
            self.logits = None

        with open(video_names_file) as f:
            self.videos = [os.path.join(video_dir_path, x[:-1]) for x in f.readlines()]
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
        vid = self.get_frames(self.videos[idx])
        # print(vid.shape, vid)
        if self.logits is not None:
            return self.transform(vid), self.logits[idx]  
        return self.transform(vid)

    def __len__(self):
        return self.num_instances

# class Dataloader():

#     def __init__(self, video_dir_path, num_classes = 5, is_random = True):

#         self.video_dir_path = video_dir_path
#         self.instances = []     # Tensor of image frames
#         self.labels = []        # Class labels
#         self.num_classes = num_classes

#         self.is_random = is_random

#         self.videos = [x for x in os.listdir(self.video_dir_path) if os.path.isdir(os.path.join(self.video_dir_path, x))]
#         print(self.videos)
#         self.get_frames()

#         self.instances = torch.stack(self.instances)
#         self.num_instances = len(self.instances)

#         self.image_idx_mapping = {}
#         self.idx_image_mapping = {}
        
#         if self.is_random:
#             self.create_labels()
#         else:
#             self.get_labels()

#     def get_frames(self):
        
#         i = 0
#         for video in tqdm(self.videos, position = 0, leave = True):
            
#             self.image_idx_mapping[video] = i
#             self.idx_image_mapping[i] = video
#             i += 1
#             image_frames = []
#             video_dir = os.path.join(self.video_dir_path, video)
#             images = os.listdir(video_dir)
            
#             for image_name in images:
#                 image = Image.open(os.path.join(video_dir, image_name))
#                 image = np.array(image, dtype = np.float32)
#                 image_frames.append(torch.tensor(image))

#             self.instances.append(torch.stack(image_frames))

#     def create_labels(self):

#         self.labels = np.zeros(self.num_instances)
#         permute = np.arange(0, 1000)
#         random.shuffle(permute)

#         num_image_per_class = self.num_instances / self.num_classes
#         c = -1
#         for i in range(self.num_instances):
#             index = permute[i]
#             if i % num_image_per_class == 0:
#                 c+=1
#             self.labels[index] = c

#         self.labels = torch.tensor(self.labels)


# class VideoDataset(Dataset):

#     def __init__(self, video_dir_path, video_names_file, transforms):

#         self.video_dir_path = video_dir_path
#         self.instances = []     # Tensor of image frames
#         # self.logits = pickle.load(open(logits_file, 'rb'))
#         self.transform = transforms
        
#         with open(video_names_file) as f:
#             self.videos = [os.path.join(video_dir_path, x) for x in f.readlines()]
#         self.get_frames()

#         self.instances = torch.stack(self.instances)
#         self.num_instances = len(self.instances)

#     def get_frames(self):

#         for video in tqdm(self.videos, position = 0, leave = True):

#             image_frames = []
#             images = os.listdir(video)

#             for image_name in images:
#                 image = Image.open(os.path.join(video, image_name))
#                 image = np.array(image, dtype = np.uint8)
#                 image_frames.append(torch.tensor(image))

#             vid = self.transform(image_frames)
#             self.instances.append(vid)

#     def __getitem__(self, idx):
#         return self.instances[idx]

#     def __len__(self):
#         return len(self.instances)

# class VideoDatasetFromDisk(Dataset):

#     def __init__(self, video_dir_path, video_names_file, transforms):

#         self.video_dir_path = video_dir_path
#         self.transform = transforms
#         # self.logits = pickle.load(open(logits_file, 'rb'))

#         with open(video_names_file) as f:
#             self.videos = [os.path.join(video_dir_path, x) for x in f.readlines()]
#         self.num_instances = len(self.videos)

#     def get_frames(self, video_path):
#         images = os.listdir(video_path)
#         image_frames = []

#         for image_name in images:
#             image = Image.open(os.path.join(video_path, image_name))
#             image = np.array(image, dtype = np.uint8)
#             image_frames.append(torch.tensor(image))
#         if len(image_frames) > 0:
#             return torch.stack(image_frames)
#         else:
#             return torch.zeros((16, 224, 224, 3))

#     def __getitem__(self, idx):
#         vid = self.get_frames(self.videos[idx])
#         # print(vid.shape, vid)
#         return self.transform(vid)

#     def __len__(self):
#         return self.num_instances

