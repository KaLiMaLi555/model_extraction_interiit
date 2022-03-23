import argparse
import os
import os.path as osp
import datetime
from config.cfg_parser import cfg_parser

class TrainOptions():
    """
    Training options for commandline
    """

    def initialize(self):

        parser = argparse.ArgumentParser(description="GreyBox")

        parser.add_argument("--train_vid_dir", type=str, help="Path to the training videos")
        parser.add_argument("--train_logits_file", type=str, help="Path to the training logits file")
        parser.add_argument("--train_vid_names_file", type=str, help="Path to the training video names file")
        parser.add_argument("--val_vid_dir", type=str, help="Path to the training videos")
        parser.add_argument("--val_logits_file", type=str, help="Path to the training logits file")
        parser.add_argument("--val_classes_file", type=str, help="Path to the training classes file")
        parser.add_argument("--val_labels_file", type=str, help="Path to the training labels file")
        parser.add_argument("--test_vid_dir", type=str, help="Path to the testing videos")
        parser.add_argument("--ckpts_dir", type=str, help="Path to save the checkpoints") 
        parser.add_argument("--train_mode", type=str, default="finetune", help="Mode for training: finetuning or pretraining")
        parser.add_argument("--pretrained_ckpt_path", type=str, help="Path to the pretrained checkpoint")
        parser.add_argument("--augmentations", type=list, default=[], help="Augmentation list")

        parser.add_argument("--epochs", type=int, help="Number of epochs to train")
        parser.add_argument("--batch_size", type=int, help="Batch size")
        parser.add_argument("--lr", type=float, help="Learning rate")
        parser.add_argument("--num_workers", type=int, help="Num_workers for dataloader")
        parser.add_argument("--lr_decay", type=float, help="Learning rate decay")
        parser.add_argument("--lr_decay_step", type=int, help="Learning rate decay step")
        parser.add_argument("--momentum", type=float, help="Momentum")
        parser.add_argument("--weight_decay", type=float, help="Weight decay")

        parser.add_argument("--dataset", type=str, help="Dataset use")
        parser.add_argument("--num_classes", type=int, help="Number of classes")

        args = parser.parse_args()

        cfg = cfg_parser(osp.join("config", 'params.json'))


        if args.train_vid_dir: cfg['experiment'].train_vid_dir = args.train_vid_dir
        if args.train_logits_file: cfg['experiment'].train_logits_file = args.train_logits_file
        if args.train_vid_names_file: cfg['experiment'].train_vid_names_file = args.train_vid_names_file
        if args.val_vid_dir: cfg['experiment'].val_vid_dir = args.val_vid_dir
        if args.val_logits_file:cfg['experiment'].val_logits_file = args.val_logits_file
        if args.val_classes_file:cfg['experiment'].val_classes_file = args.val_classes_file
        if args.val_labels_file:cfg['experiment'].val_labels_file = args.val_labels_file
        if args.test_vid_dir:cfg['experiment'].test_vid_dir = args.test_vid_dir
        if args.ckpts_dir:cfg['experiment'].ckpts_dir = args.ckpts_dir
        if args.epochs:cfg['experiment'].epochs = args.epochs
        if args.batch_size:cfg['experiment'].batch_size = args.batch_size
        if args.lr:cfg['experiment'].lr = args.lr
        if args.num_workers:cfg['experiment'].num_workers = args.num_workers
        if args.lr_decay:cfg['experiment'].lr_decay = args.lr_decay
        if args.lr_decay_step:cfg['experiment'].lr_decay_step = args.lr_decay_step
        if args.momentum:cfg['experiment'].momentum = args.momentum
        if args.weight_decay:cfg['experiment'].weight_decay = args.weight_decay
        if args.dataset:cfg['experiment'].dataset = args.dataset
        if args.num_classes:cfg['experiment'].num_classes = args.num_classes
        if args.train_mode:cfg['experiment'].train_mode = args.train_mode
        if args.augmentations:cfg['experiment'].augmentations = args.augmentations
        if args.pretrained_ckpt_path:cfg['experiment'].pretrained_ckpt_path = args.pretrained_ckpt_path

        return cfg

    def print_options(self, args):
        """
        Function that prints and saves the output
        """

        message = ''
        message += f'----------------- Options ----------------\n'
        for k, v in sorted(vars(args['experiment']).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------\n'
        print(message)

