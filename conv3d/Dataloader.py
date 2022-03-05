import torch
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import csv


class ValDataset(Dataset):

    def __init__(self, video_dir_path, classes_file, labels_file, num_classes):

        self.video_dir_path = video_dir_path
        self.classes_file = classes_file
        self.labels_file = labels_file

        self.videos = os.listdir(self.video_dir_path)
        self.num_instances = len(self.videos)
        self.num_classes = num_classes

        with open(self.labels_file, mode='r') as infile:
            self.label_dict = {rows[0]: rows[1] for rows in csv.reader(infile)}
            self.label_dict = {value: key for (key, value) in self.label_dict.items()}

        with open(self.classes_file, mode='r') as infile:
            self.classes_dict = {rows[0]: rows[1] for rows in csv.reader(infile)}
            self.classes_dict = {value: key for (key, value) in self.classes_dict.items()}

        self.new_classes_dict = {}
        for index, (id, label) in enumerate(self.classes_dict.items()):
            if index == 0:
                continue
            self.new_classes_dict[id] = self.label_dict[label]
        print(self.new_classes_dict)

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
        if video_name[0:2] == '--':
            id = video_name[2:len(video_name) - k - 1]
        elif video_name[0] == '-':
            id = video_name[1:len(video_name) - k - 1]
        return id

    def get_label(self, idx):
        video_name = self.videos[idx]
        video_id = get_id(video_name)
        # print("*****************")
        # print("*****************")
        # print(video_id)
        # print(video_name)
        # # print("*****************")
        # # print("*****************")
        label = self.new_classes_dict[video_id]
        # print(label)
        # print("*****************")
        # print("*****************")
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1
        one_hot = torch.from_numpy(one_hot)

    def get_frames(self, video_path):
        images = os.listdir(video_path)
        image_frames = []

        for image_name in images:
            image = Image.open(os.path.join(video_path, image_name))
            image = np.array(image, dtype=np.float32)
            image_frames.append(torch.tensor(image))

        return torch.stack(image_frames)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir_path, self.videos[idx])
        return self.get_frames(video_path), self.get_label(idx)

    def __len__(self):
        return self.num_instances




