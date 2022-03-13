# List of imports
import os
from pathlib import Path

import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image


class ValDataset(Dataset):

    def __init__(self, video_dir_path, classes_file, labels_file, num_classes, transform=None, scale=1, shift=0):

        self.video_dir_path = video_dir_path
        self.classes_file = classes_file
        self.labels_file = labels_file

        self.transform = transform
        self.scale = scale
        self.shift = shift

        self.videos = sorted([str(x.name) for x in Path(self.video_dir_path).iterdir() if x.is_dir()])
        self.num_instances = len(self.videos)
        self.num_classes = num_classes

        self.label_dict = pd.read_csv(self.labels_file, header=None, index_col=1, squeeze=False).to_dict()
        self.label_dict = self.label_dict[0]

        self.classes_dict = pd.read_csv(self.classes_file, header=None, index_col=1, squeeze=False).to_dict()
        self.classes_dict = self.classes_dict[0]

        self.new_classes_dict = {}
        for index, (id, label) in enumerate(self.classes_dict.items()):
            if index == 0:
                continue
            self.new_classes_dict[id] = self.label_dict[label]
        # print(self.new_classes_dict)
        # print(self.num_instances, len(self.new_classes_dict))

    def get_id(self, video_name):
        # k = 0
        # rev = video_name[::-1]
        # id = ''.join(rev.split('_')[2:])[::-1]
        # for x in range(len(video_name)):
        #     if rev[x] == '_':
        #         k = k + 1
        #     if k >= 2:
        #         k = x
        #         break

        id = video_name[:-18]
        return id

    def get_label(self, idx):
        video_name = self.videos[idx]
        video_id = self.get_id(video_name)
        label = self.new_classes_dict[video_id]
        # one_hot = F.one_hot(torch.tensor(int(label)), self.num_classes)
        # return one_hot
        return label

    def get_frames(self, video_path):
        images = sorted(os.listdir(video_path))
        image_frames = []

        for image_name in images:
            image = Image.open(os.path.join(video_path, image_name))
            image = np.array(image, dtype=np.float32)
            image_frames.append(torch.tensor(image))

        return torch.stack(image_frames) * self.scale + self.shift

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir_path, self.videos[idx])
        vid = self.get_frames(video_path)
        # vid = custom_resize_transform(vid)
        if self.transform:
            vid = vid.permute(0, 3, 1, 2)  # f, h, w, c
            vid = self.transform(vid)  # f, c, h, w
            vid = vid.permute(0, 2, 3, 1)  # f, h, w, c
        # vid = vid.swapaxes(0, 3)  # <C3D Transform>
        return vid, self.get_label(idx)

    def __len__(self):
        return self.num_instances
