import argparse
import os
import os.path as osp
import datetime
from config.cfg_parser import cfg_parser

class TestOptions():
    def initialize(self):

        parser = argparse.ArgumentParser(description="Test the model")
        parser.add_argument("--videos_dir", type=str, help="Path to the videos")
        parser.add_argument("--classes_file", type=str, help="Path to the classes file")
        parser.add_argument("--labels_file", type=str, help="Path to the labels file")
        parser.add_argument("--num-classes", type=int, help="Number of classes")
        parser.add_argument("--batch_size", type=int, help="Batch size")
        parser.add_argument("--num_workers", type=int,  help="Number workers for dataloader")
        parser.add_argument("--restore-from", type=str, help="restore model parameters from")
        parser.add_argument("--save", type=str, help="Path to save result.")

        args = parser.parse_args()

        cfg = cfg_parser(osp.join("config", 'params.json'))

        if args.videos_dir: 
            cfg["test"].video_dir = args.videos_dir
        if args.classes_file:
            cfg["test"].classes_file = args.classes_file
        if args.labels_file:
            cfg["test"].labels_file = args.labels_file
        if args.num_classes:
            cfg["test"].num_classes = args.num_classes
        if args.batch_size:
            cfg["test"].batch_size = args.batch_size
        if args.num_workers:
            cfg["test"].num_workers = args.num_workers
        if args.restore_from:
            cfg["test"].restore_from = args.restore_from
        if args.save:
            cfg["test"].save = args.save

        return cfg

    def print_options(self, args):
        """ Function that prints and saves the output
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------\n'
        print(message)