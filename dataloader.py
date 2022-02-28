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

# TODO: The class is implemented now for random, 
# Do if we have both the actual images and labels
class Dataloader():

    def __init__(self, video_dir_path, num_classes = 5, is_random = True):

        self.video_dir_path = video_dir_path
        self.instances = []     # Tensor of image frames
        self.labels = []        # Class labels
        self.num_classes = num_classes

        self.is_random = is_random

        self.videos = os.listdir(self.video_dir_path)
        self.get_frames()

        self.instances = torch.stack(self.instances)
        self.num_instances = len(self.instances)

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