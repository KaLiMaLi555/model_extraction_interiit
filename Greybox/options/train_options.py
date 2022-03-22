import argparse
import os
import os.path as osp
import datetime
from Greybox.config.cfg_parser import cfg_parser


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

        parser.add_argument("dataset", type=str, default="k400", help="Dataset use")
        parser.add_argument("num_classes", type=int, default=400, help="Number of classes")

        args = parser.parse_args()

        cfg = cfg_parser(osp.join("config", args.version + '.json'))


        cfg['experiment'].train_vid_dir = args.train_vid_dir
        cfg['experiment'].train_logits_file = args.train_logits_file
        cfg['experiment'].val_vid_dir = args.val_vid_dir
        cfg['experiment'].val_logits_file = args.val_logits_file
        cfg['experiment'].test_vid_dir = args.test_vid_dir
        cfg['experiment'].ckpts_dir = args.ckpts_dir
        cfg['experiment'].epochs = args.epochs
        cfg['experiment'].batch_size = args.batch_size
        cfg['experiment'].lr = args.lr
        cfg['experiment'].num_workers = args.num_workers
        cfg['experiment'].lr_decay = args.lr_decay
        cfg['experiment'].lr_decay_step = args.lr_decay_step
        cfg['experiment'].momentum = args.momentum
        cfg['experiment'].weight_decay = args.weight_decay
        cfg['experiment'].dataset = args.dataset
        cfg['experiment'].num_classes = args.num_classes

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
