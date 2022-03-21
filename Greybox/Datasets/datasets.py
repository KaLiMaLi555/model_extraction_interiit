import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class VideoLogitDataset(Dataset):

    def _init_(self, video_dir_path, video_name_file, logits_file, size: tuple = (224,224), transform=None, aug_list=None):

        self.video_dir_path = video_dir_path
        self.instances = []  # Tensor of image frames
        self.logits = pickle.load(open(logits_file, 'rb'))

        with open(video_name_file) as f:
            self.videos = [os.path.join(video_dir_path, x[:-1]) for x in f.readlines()]

        self.num_instances = len(self.videos)
        self.transform = transform
        self.size = size
        self.aug_list = aug_list

    def get_frames(self, video_path):
        images = sorted(os.listdir(video_path))
        image_frames = []

        for image_name in images:
            image = Image.open(os.path.join(video_path, image_name))
            image = image.resize(self.size)
            image = np.array(image, dtype=np.float32)
            image = image / 255.0
            image_frames.append(torch.tensor(image))

        return torch.stack(image_frames)

    def _getitem_(self, idx):
        vid = self.get_frames(self.videos[idx])
        if self.aug_list is not None:
            aug_idx = np.random.randint(len(self.aug_list))
            vid = torch.tensor(self.aug_list[aug_idx](vid.numpy()))
        if self.transform:
            vid = vid.permute(0, 3, 1, 2)
            vid = self.transform(vid)
            vid = vid.permute(0, 2, 3, 1)
        vid = vid.swapaxes(0, 3)
        return vid, self.logits[idx]

    def _len_(self):
        return self.num_instances


class VideoLabelDataset(Dataset):

    def _init_(self, video_dir_path, classes_file, labels_file, num_classes, size: tuple=(224,224), transform=None, aug_list=None):

        self.video_dir_path = video_dir_path
        self.classes_file = classes_file
        self.labels_file = labels_file
        self.transform = transform

        self.videos = sorted([str(x.name) for x in Path(self.video_dir_path).iterdir() if x.is_dir()])
        self.num_instances = len(self.videos)
        self.num_classes = num_classes

        self.label_dict = pd.read_csv(self.labels_file, header=None, index_col=1, squeeze=False).to_dict()
        self.label_dict = self.label_dict[0]

        self.classes_dict = pd.read_csv(self.classes_file, header=None, index_col=1, squeeze=False).to_dict()
        self.classes_dict = self.classes_dict[0]

        self.new_classes_dict = {}
        self.size = size
        for index, (id, label) in enumerate(self.classes_dict.items()):
            if index == 0:
                continue
            self.new_classes_dict[id] = self.label_dict[label]
        
        self.aug_list = aug_list

    def get_id(self, video_name):
        k = 0
        rev = video_name[::-1]
        for x in range(len(video_name)):
            if rev[x] == '_':
                k = k + 1
            if k >= 2:
                k = x
                break

        id = video_name[0:len(video_name) - k - 1]

        return id

    def get_label(self, idx):
        video_name = self.videos[idx]
        video_id = self.get_id(video_name)
        label = self.new_classes_dict[video_id]
        one_hot = F.one_hot(torch.tensor(int(label)), self.num_classes)
        return one_hot

    def get_frames(self, video_path):
        images = os.listdir(video_path)
        image_frames = []

        for image_name in images:
            image = Image.open(os.path.join(video_path, image_name))
            image = image.resize(self.size)
            image = np.array(image, dtype=np.float32)
            image = image / 255.0
            image_frames.append(torch.tensor(image))

        return torch.stack(image_frames)

    def _getitem_(self, idx):
        video_path = os.path.join(self.video_dir_path, self.videos[idx])
        vid = self.get_frames(video_path)
        if self.aug_list is not None:
            aug_idx = np.random.randint(len(self.aug_list))
            vid = torch.tensor(self.aug_list[aug_idx](vid.numpy()))
        if self.transform:
            vid = vid.permute(0, 3, 1, 2)
            vid = self.transform(vid)
            vid = vid.permute(0, 2, 3, 1)
        vid = vid.swapaxes(0, 3)
        return vid, self.get_label(idx)

    def _len_(self):
        return self.num_instances