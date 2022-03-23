import argparse
import os
import os.path as osp
import datetime
from config.cfg_parser import cfg_parser


class TestOptions():
    def initialize(self):
        """ Definition of Test cmd line parameters
        Returns:
            ArgumentParser.parse_args: Params values for training
            Command line arguments:
            --model [model-key]: Available (dks/micronet), Defaults to micronet
            --device [device]: cpu/cuda, Defaults to cpu
            --data-dir-test [path]: Path to test set
            --num-workers [int]: Number workers for dataloader, Defaults to 43
            --restore-from [path]: Checkpoint dir to restore training, Defaults to None
            --save [path]: Path to save the metrics and outputs
        """

        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--model", type=str, default='micronet', help="available options : dks/micronet")
        parser.add_argument("--device", type=str, default='cpu', help="which device to use")

        parser.add_argument("--data_dir_test", type=str, default='../dataset/GTSRB/test',
                            help="Path to the directory containing the target dataset.")
        parser.add_argument("--num_classes", type=int, default=19, help="Number of classes for cityscapes.")

        parser.add_argument("--restore_from", type=str, default=None, help="restore model parameters from")

        parser.add_argument("--save", type=str, default='results', help="Path to save result.")

        args = parser.parse_args()

        cfg = cfg_parser(osp.join("config", args.version + '.json'))

        cfg["experiment"].model = args.model
        cfg["experiment"].device = args.device
        cfg["experiment"].data_dir_test = args.data_dir_test
        cfg["experiment"].num_classes = args.num_classes
        cfg["experiment"].restore_from = args.restore_from
        cfg["experiment"].save = args.save


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