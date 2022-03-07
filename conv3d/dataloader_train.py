# List of imports
import os
import pickle

import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset

from PIL import Image
from tqdm.notebook import tqdm


class VideoLogitDataset(Dataset):

    def __init__(self, video_dir_path, logits_file, transform=None):

        self.video_dir_path = video_dir_path
        self.instances = []  # Tensor of image frames
        # self.logits = np.array(list(x[0] for x in pickle.load(open(logits_file, 'rb'))))
        self.logits = pickle.load(open(logits_file, 'rb'))

        self.videos = os.listdir(self.video_dir_path)
        self.get_frames()

        self.instances = torch.stack(self.instances)
        self.num_instances = len(self.instances)
        self.transform = transform

    def get_frames(self):
        for video in tqdm(self.videos, position=0, leave=True):
            image_frames = []
            video_dir = os.path.join(self.video_dir_path, video)
            images = os.listdir(video_dir)

            for image_name in images:
                image = Image.open(os.path.join(video_dir, image_name))
                image = np.array(image, dtype=np.float32)
                image_frames.append(torch.tensor(image))

            self.instances.append(torch.stack(image_frames))

    def __getitem__(self, idx):
        vid = self.instances[idx]
        vid = vid.swapaxes(0, 3)
        # vid = custom_rotate_transform(vid)
        if self.transform:
            vid = self.transform(vid)
        return vid, self.logits[idx]

    def __len__(self):
        return len(self.instances)
