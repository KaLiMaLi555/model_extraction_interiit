import argparse
import datetime
import os
import os.path as osp

import repackage

repackage.up()
from config.cfg_parser import cfg_parser


class TrainOptions():
    """
    Training options for commandline
    """

    def initialize(self):
        """
        Parameter definitions
        Returns:
            ArgumentParser.parse_args: Params values for training
            Command line arguments:
            --train_vid_dir: Path to the training videos
            --wandb [bool]: Log to wandb or not
            ...
        """

        parser = argparse.ArgumentParser(description="GreyBox")

        parser.add_argument("--train_vid_dir", type=str, default="", help="Path to the training videos")
        parser.add_argument("--train_logits_file", type=str, default="", help="Path to the training logits file")
        parser.add_argument("--val_vid_dir", type=str, default="", help="Path to the training videos")
        parser.add_argument("--val_logits_file", type=str, default="", help="Path to the training logits file")
        parser.add_argument("--test_vid_dir", type=str, default="", help="Path to the testing videos")
        parser.add_argument("--ckpts_dir", type=str, default="", help="Path to the testing logits file")

        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
        parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
        parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
        parser.add_argument("--num_workers", type=int, default=4, help="Num_workers for dataloader")
        parser.add_argument("--lr_decay", type=float, default=0.5, help="Learning rate decay")
        parser.add_argument("--lr_decay_step", type=int, default=5, help="Learning rate decay step")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
        parser.add_argument("--weight_decay", type=float, default=10, help="Weight decay")

        parser.add_argument("--dataset", type=str, default="k400", help="Dataset use")
        parser.add_argument("--num_classes", type=int, default=400, help="Number of classes")

        parser.add_argument("--config", type=str, default="config/params.json")

        args = parser.parse_args()

        cfg = cfg_parser(args.config)
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

        # save to the disk
        if not os.path.exists(args['experiment'].snapshot_dir):
            os.makedirs(args['experiment'].snapshot_dir)

        t = datetime.datetime.now()

        name = f'opt_{args["experiment"].model}_{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}.txt'
        file_name = osp.join(args['experiment'].snapshot_dir, name)

        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')
