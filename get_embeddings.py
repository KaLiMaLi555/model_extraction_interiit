import os
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from env import set_seed
from dataloader import Dataloader, VideoDatasetFromDisk
from torch.utils.data import TensorDataset, DataLoader, Dataset

from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint

# from movinets import MoViNet
# from movinets.config import _C

import warnings
warnings.filterwarnings("ignore")

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default = 16, type = int)
    parser.add_argument('--logit_dir', default = "/content/gdrive/MyDrive/logits", type = str)
    parser.add_argument('--is_random', default = True, type = bool)
    parser.add_argument('--num_classes', default = 5, type = int)
    parser.add_argument('--seed', default = 42, type = int)
    parser.add_argument('--model_name', default = "swin_transformer", type = str, choices = ["movinet", "swin_transformer"])
    parser.add_argument('--dataset_type', default = "noise", type = str)
    parser.add_argument('--video_dir_path', default = "./data/data", type = str)
    parser.add_argument('--from_folder', action='store_true')

    args = parser.parse_args()

    return args

def get_logits(model, model_name, dataloader, device):
    
    logits = []
    # labels = []
    model.eval()
    with torch.no_grad():
        for idx, video_frames in tqdm(enumerate(dataloader), position = 0, leave = True):
            shape = video_frames.shape
            video_frames = video_frames.view((shape[0], shape[4], shape[1], shape[2], shape[3]))
            
            video_frames.to(device)
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

    return logits #, labels

if __name__ == "__main__":

    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.logit_dir):
        os.makedirs(args.logit_dir)

    print("\n######################## Loading Data ########################\n")
    # dataloader = Dataloader(args.video_dir_path, num_classes = args.num_classes)

    if args.from_folder:
        tensor_dataset = VideoDatasetFromDisk(args.video_dir_path)
    else:
        tensor_dataset = VideoDataset(args.video_dir_path)
    tensor_dataloader = DataLoader(tensor_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 0)

    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model_name == "movinet":
        model = MoViNet(_C.MODEL.MoViNetA2, causal = True, pretrained = True)
    elif args.model_name == "swin_transformer":
        print("\n")
        config = "./VST/configs/_base_/models/swin/swin_tiny.py"
        checkpoint = "/content/swin_tiny_patch244_window877_kinetics400_1k.pth"
        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location=device)

    print("\n######################## Getting Logits ########################\n")
    logits = get_logits(model, args.model_name, tensor_dataloader, device)
    pickle.dump(logits, open(os.path.join(args.logit_dir, args.model_name + "_" + args.dataset_type + ".pkl"), "wb"))
