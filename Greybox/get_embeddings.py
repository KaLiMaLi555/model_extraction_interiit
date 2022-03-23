import os
import torch
import pickle
import random
import argparse
import numpy as np
import wget
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch.functional as F
import torch.nn as nn
from Datasets.transforms import MovinetTransform, SwinTransform
from Datasets.datasets import VideoOnlyDataset
from torch.utils.data import DataLoader

from options.embedding_options import *
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
from PIL import Image

import warnings

warnings.filterwarnings("ignore")

"""
    Function to generate data for movinet

    Args:
        video_dir_path: Path to the directory containing the videos
        video_names_file: Path to the file containing the video names
        transforms: List of transforms to be applied to the video
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for the dataloader
        num_classes: Number of classes in the dataset
        
"""


class TensorflowDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_dir_path, classes_file, labels_file, num_classes, transform=None, shuffle=False):
        self.shuffle = shuffle
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
            newsize = (224, 224)
            image = image.resize(newsize)
            image = np.array(image, dtype=np.float32)
            image = image / 255.0
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
        vid = torch.permute(vid, (3, 1, 2, 0))
        # b= self.get_label(idx)
        vid = vid.numpy()
        # b=b.numpy()
        vid = tf.convert_to_tensor(vid)
        # b= tf.convert_to_tensor(b)
        return vid

    def __len__(self):
        return self.num_instances

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.train_imgs)


def make_gen_callable(_gen):
    def gen():
        for x, y in _gen:
            yield x, y

    return gen


"""
    Function to get embeddings from a model

    Args:
        model: Model to be used for embedding extraction
        model_name: Name of the model
        dataloader: Dataloader to be used for embedding extraction
        device: Computational Device to be used for embedding extraction

    Returns:
        logits: Embeddings extracted from the model
        
"""


def get_logits(model, model_name, dataloader, device):
    logits = []
    # labels = []
    model.eval()
    with torch.no_grad():
        for idx, video_frames in tqdm(enumerate(dataloader), position=0, leave=True):
            video_frames.to(device)
            if model_name == "swin_transformer":
                outputs = model(video_frames, return_loss=False)
            else:
                outputs = model(video_frames)
            del video_frames
            logits.extend(outputs)
            # labels.extend(label)
            if model_name == "movinet":
                model.clean_activation_buffers()

    if model_name != "movinet":
        logits = torch.from_numpy(np.array(logits))
    else:
        logits = torch.stack(logits)

    # labels = torch.stack(labels)

    return logits  # , labels


def main():
    opt = EmbeddingOptions()
    cfg = opt.initialize()
    args = cfg["embeddings"]

    if not os.path.exists(args.logit_dir):
        os.makedirs(args.logit_dir)

    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model_name == "movinet":
        train_gen = TensorflowDataGenerator(args.video_dir_path, args.classes_file, args.labels_file, args.num_classes)
        train_ = make_gen_callable(train_gen)
        ot = (tf.float32, tf.int64)
        ds = tf.data.Dataset.from_generator(train_, ot)
        batched_dataset = ds.batch(args.batch_size)

        hub_url = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"
        encoder = hub.KerasLayer(hub_url, trainable=False)
        inputs = tf.keras.layers.Input(shape=[None, None, None, 3], dtype=tf.float32, name='image')
        outputs = encoder(dict(image=inputs))
        model = tf.keras.Model(inputs, outputs, name='movinet')
        logits = get_logits(model, args.model_name, batched_dataset, device)

    elif args.model_name == "swin_transformer":
        print("\n")
        tensor_dataset = VideoOnlyDataset(args.video_dir_path, args.video_names_file,
                                          transforms=SwinTransform())
        tensor_dataloader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        this_dir_name, _ = os.path.split(os.path.abspath(__file__))
        url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth"
        config = os.path.join(this_dir_name,
                              "models/VST/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py")
        wget.download(url, out=os.path.join(this_dir_name, "Assets"))
        checkpoint = os.path.join(this_dir_name, "/Assets/swin_tiny_patch244_window877_kinetics400_1k.pth")
        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location=device)
        logits = get_logits(model, args.model_name, tensor_dataloader, device)

    else:
        print("Model name not recognized")
        raise NotImplementedError

    pickle.dump(logits, open(os.path.join(args.logit_dir, args.model_name + "_" + args.dataset_type + ".pkl"), "wb"))


if __name__ == "__main__":
    main()
