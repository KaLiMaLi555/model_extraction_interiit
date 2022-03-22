import os
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from env import seed_all
from dataloader import MovinetTransform, SwinTransform, VideoDataset, VideoDatasetFromDisk
from torch.utils.data import TensorDataset, DataLoader, Dataset

from mmcv import Config
from VST.mmaction.models import build_model
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
    parser.add_argument('--video_names_file', default= "/content/video_names_file.txt", type=str)
    parser.add_argument('--from_folder', action='store_true')

    args = parser.parse_args()

    return args

def get_logits(model, model_name, dataloader, device):
    
    logits = []
    # labels = []
    model.eval()
    with torch.no_grad():
        for idx, video_frames in tqdm(enumerate(dataloader), position = 0, leave = True):            
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

    return logits #, labels

if __name__ == "__main__":

    args = parse_args()
    seed_all(args.seed)

    # if not os.path.exists(args.logit_dir):
    #     os.makedirs(args.logit_dir)

    print("\n######################## Loading Data ########################\n")
    # dataloader = Dataloader(args.video_dir_path, num_classes = args.num_classes)

    if args.from_folder:
        if args.model_name == "swin_transformer":
            tensor_dataset = VideoDatasetFromDisk(args.video_dir_path, args.video_names_file, transforms=SwinTransform())
        else:
            tensor_dataset = VideoDatasetFromDisk(args.video_dir_path, args.video_names_file, transforms=MovinetTransform())
    else:
        if args.model_name == "swin_transformer":
            tensor_dataset = VideoDataset(args.video_dir_path, args.video_names_file, transforms=SwinTransform())
        else:
            tensor_dataset = VideoDataset(args.video_dir_path, args.video_names_file, transforms=MovinetTransform())
    tensor_dataloader = DataLoader(tensor_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 0)

    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model_name == "movinet":
        model = MoViNet(_C.MODEL.MoViNetA2, causal = True, pretrained = True)
    elif args.model_name == "swin_transformer":
        print("\n")
        config = "./Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
        checkpoint = "/content/swin_tiny_patch244_window877_kinetics400_1k.pth"
        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location=device)

    print("\n######################## Getting Logits ########################\n")
    logits = get_logits(model, args.model_name, tensor_dataloader, device)
    pickle.dump(logits, open(os.path.join(args.logit_dir, args.model_name + "_" + args.dataset_type + ".pkl"), "wb"))