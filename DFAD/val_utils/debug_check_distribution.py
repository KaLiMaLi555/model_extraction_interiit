import argparse
from collections import Counter

import numpy as np
from tqdm import tqdm

from custom_transformations import CustomResizeTransform
from dataloader_val import ValDataset

parser = argparse.ArgumentParser(description='Check data distribution')
parser.add_argument('--val_data_dir', type=str, default='/content/val_data/k400_16_frames_uniform')
parser.add_argument('--val_classes_file', type=str, default='/content/val_data/k400_16_frames_uniform/classes.csv')
parser.add_argument('--val_labels_file', type=str, default='/content/val_data/k400_16_frames_uniform/labels.csv')
parser.add_argument('--val_num_classes', type=int, default=400)
args = parser.parse_args()


def main():
    val_data = ValDataset(args.val_data_dir, args.val_classes_file,
                          args.val_labels_file, args.val_num_classes,
                          transform=CustomResizeTransform())

    distribution = [0] * val_data.num_classes
    for i in tqdm(range(val_data.num_instances)):
        one_hot = val_data.get_label(i)
        distribution[np.argmax(one_hot)] += 1
    print('Distribution')
    print(distribution)
    print(min(distribution), max(distribution))
    print('Meta-distribution')
    c = Counter(distribution)
    print(c.elements())


if __name__ == '__main__':
    main()
