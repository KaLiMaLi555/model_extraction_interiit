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

    def __init__(self, video_dir_path, video_name_file, logits_file, size: tuple = (224,224), transform=None, va_augments=None):

        self.video_dir_path = video_dir_path
        if logits_file is not None:
            self.logits = pickle.load(open(logits_file, 'rb'))
        else:
            self.logits = None

        with open(video_name_file) as f:
            self.videos = [os.path.join(video_dir_path, x[:-1]) for x in f.readlines()]

        self.num_instances = len(self.videos)
        self.transform = transform
        self.size = size
        self.va_augments = va_augments
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

    def __getitem__(self, idx):
        vid = self.get_frames(self.videos[idx])
        if self.va_augments is not None:
            vid = torch.tensor(np.array(self.va_augments(vid.numpy())))
        if self.transform:
            vid = vid.permute(0, 3, 1, 2)
            vid = self.transform(vid)
            vid = vid.permute(0, 2, 3, 1)
        vid = vid.swapaxes(0, 3)
        return vid, self.logits[idx]

    def __len__(self):
        return self.num_instances


class VideoLabelDataset(Dataset):

    def __init__(self, video_dir_path, classes_file, labels_file, num_classes, size: tuple=(224,224), transform=None, va_augments=None):

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
        
        self.va_augments = va_augments

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

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir_path, self.videos[idx])
        vid = torch.tensor(np.array(self.va_augments(vid.numpy())))
        if self.va_augments is not None:
            vid = torch.tensor(self.va_augments(vid.numpy()))
        if self.transform:
            vid = vid.permute(0, 3, 1, 2)
            vid = self.transform(vid)
            vid = vid.permute(0, 2, 3, 1)
        vid = vid.swapaxes(0, 3)
        return vid, self.get_label(idx)

    def __len__(self):
        return self.num_instances


class VideoOnlyDataset(Dataset):

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