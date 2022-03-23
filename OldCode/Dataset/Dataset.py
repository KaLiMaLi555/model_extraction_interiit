import shutil
from tqdm import tqdm
import pickle
from tqdm.notebook import tqdm
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

    def __init__(self, video_dir_path, logits_file, transform=None):

        self.video_dir_path = video_dir_path
        self.instances = []  # Tensor of image frames
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
        if self.transform:
            vid = self.transform(vid)
        return vid, self.logits[idx]

    def __len__(self):
        return len(self.instances)


class ValDataset(Dataset):

    def __init__(self, video_dir_path, classes_file, labels_file, num_classes, transform=None):

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
        for index, (id, label) in enumerate(self.classes_dict.items()):
            if index == 0:
                continue
            self.new_classes_dict[id] = self.label_dict[label]
        print(self.new_classes_dict)
        print(len(self.new_classes_dict))

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
        images = sorted(os.listdir(video_path))
        image_frames = []

        for image_name in images:
            image = Image.open(os.path.join(video_path, image_name))
            image = np.array(image, dtype=np.float32)
            image_frames.append(torch.tensor(image))

        return torch.stack(image_frames)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir_path, self.videos[idx])
        vid = self.get_frames(video_path)
        if self.transform:
            vid = vid.permute(0, 3, 1, 2)
            vid = self.transform(vid)
            vid = vid.permute(0, 2, 3, 1)
        vid = vid.swapaxes(0, 3)  # <C3D Transform>
        return vid, self.get_label(idx)

    def __len__(self):
        return self.num_instances


def extrapolate(input_dir, output_dir, out_frames: int = 16):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    error_count = 0
    videos = sorted(os.listdir(input_dir))

    for video in tqdm(videos):

        frames = sorted(os.listdir(input_dir / video))
        if len(frames) == 0:
            print(f'----> Skipping {video}: video has no frames')
            continue
        else:
            frames = sorted(frames * (out_frames // len(frames)))
            new_vid_frames = frames
            length = len(new_vid_frames)
            add_frames = out_frames % length
            x = length // (add_frames + 1)
            a = x

        if add_frames % 2 != 0:
            new_vid_frames.append(frames[length // 2])
            add_frames = add_frames - 1

        while add_frames != 0:
            new_vid_frames.append(frames[a - 1])
            new_vid_frames.append(frames[length - a])
            a = a + x
            add_frames = add_frames - 2

        new_vid_frames.sort()

        out_path = output_dir / video
        out_path.mkdir(parents=True, exist_ok=True)

        k = 0
        for idx, frame in enumerate(new_vid_frames):
            src = input_dir / video / frame
            dst = output_dir / video / (str(idx) + ".jpg")
            shutil.copy(src, dst)
        if len(os.listdir(output_dir / video)) != 16:
            print(len(new_vid_frames))
            error_count += 1
    if error_count > 0:
        print(f'----> {error_count} videos were not copied')
    else:
        print('----> All videos were copied')
