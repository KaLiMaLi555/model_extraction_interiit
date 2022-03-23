import argparse
import os
import os.path as osp
import datetime
from Greybox.config.cfg_parser import cfg_parser

class EmbeddingOptions():
    def initialize(self):
        """ Definition of Test cmd line parameters
        Returns:
            ArgumentParser.parse_args: Params values for training
            Command line arguments:
            --batch_size: Batch size
            --logit_dir: Directory to save the logits
        """

        parser = argparse.ArgumentParser(description="test segmentation network")

        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--logit_dir', default=None, type=str)
        parser.add_argument('--is_random', default=True, type=bool)
        parser.add_argument('--num_classes', default=5, type=int)
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--model_name', default="swin_transformer", type=str, choices=["movinet", "swin_transformer"])
        # parser.add_argument('--dataset_type', default="noise", type=str)
        parser.add_argument('--video_dir_path', default="./data/data", type=str)
        parser.add_argument('--video_names_file', default=" ", type=str)
        parser.add_argument('--classes_file', default=" ", type=str)
        parser.add_argument('--labels_file', default=" ", type=str)
        # parser.add_argument('--from_folder', action='store_true')

        args = parser.parse_args()

        cfg = cfg_parser(osp.join("config", args.version + '.json'))

        cfg["embeddings"].batch_size = args.batch_size
        cfg["embeddings"].logit_dir = args.logit_dir
        cfg["embeddings"].is_random = args.is_random
        cfg["embeddings"].num_classes = args.num_classes
        cfg["embeddings"].seed = args.seed
        cfg["embeddings"].model_name = args.model_name
        # cfg["embeddings"].dataset_type = args.dataset_type
        cfg["embeddings"].video_dir_path = args.video_dir_path
        cfg["embeddings"].video_names_file = args.video_names_file
        cfg["embeddings"].classes_file = args.classes_file
        cfg["embeddings"].labels_file = args.labels_file
        # cfg["embeddings"].from_folder = args.from_folder

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

        # save to the disk
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        t = datetime.datetime.now()
        name = f'opt_{args.model}_{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}.txt'
        file_name = osp.join(args.snapshot_dir, name)
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')