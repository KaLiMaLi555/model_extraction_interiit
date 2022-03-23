import argparse
import datetime
import os
import os.path as osp


class EmbeddingOptions():
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

        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--logit_dir', default="/path-to-logit-directory/", type=str)
        parser.add_argument('--is_random', default=True, type=bool)
        parser.add_argument('--num_classes', default=400, type=int)
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--model_name', default="swin_transformer", type=str, choices=["movinet", "swin_transformer"])
        parser.add_argument('--video_dir_path', default="/path-to-video-directory/", type=str)
        parser.add_argument('--video_names_file', default="/path-to-video_names_file.txt", type=str)
        # parser.add_argument('--from_folder', action='store_true')

        return parser.parse_args()

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