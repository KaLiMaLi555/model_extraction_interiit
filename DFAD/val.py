import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from network import models
from val_utils import metrics
from val_utils.custom_transformations import CustomResizeTransform
from val_utils.dataloader_val import ValDataset

# Note: Not seeding here since importing metrics.py already seeds.
# Add seeding if not using metrics


parser = argparse.ArgumentParser(description='DFAD Validation')

parser.add_argument('--val_data_dir', type=str, default='/content/val_data/k400_16_frames_uniform')
parser.add_argument('--val_classes_file', type=str, default='/content/val_data/k400_16_frames_uniform/classes.csv')
parser.add_argument('--val_labels_file', type=str, default='/content/val_data/k400_16_frames_uniform/labels.csv')

parser.add_argument('--val_num_classes', type=int, default=400)
# Scale and shift config for Movinet: (1/127.5, -1)
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--scale_inv', type=float, default=1)
parser.add_argument('--shift', type=float, default=0)

parser.add_argument('--val_batch_size', type=int, default=16)
parser.add_argument('--checkpoint_path', type=str, default='/content/model.pth')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()


def val(student, dataloader, device):
    accuracy_1, accuracy_5 = [], []
    for (x, y) in tqdm(dataloader, total=len(dataloader)):
        x, y = x.to(device), y.to(device)
        x_shape = x.shape
        # b, f, h, w, c
        x = x.reshape(x_shape[0], x_shape[1], x_shape[4], x_shape[2], x_shape[3])
        # without swapaxis: b, c, f, h, w
        # with swapaxis:    b, f, c, h, w
        # print(x_shape, x.shape)
        logits = student(x)
        accuracy_1.append(metrics.topk_accuracy(logits, y, 1))
        accuracy_5.append(metrics.topk_accuracy(logits, y, 5))

    return sum(accuracy_1) / len(accuracy_1), sum(accuracy_5) / len(accuracy_5)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    student = models.ResCNNRNN()
    student.load_state_dict(checkpoint['student'])
    student = student.to(device)
    student.eval()

    if args.scale == 1:
        args.scale = 1 / args.scale_inv
    val_data = ValDataset(args.val_data_dir, args.val_classes_file,
                          args.val_labels_file, args.val_num_classes,
                          transform=CustomResizeTransform(),
                          scale=args.scale, shift=args.shift)

    val_loader = DataLoader(val_data, batch_size=args.val_batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=args.num_workers)
    accuracy_1, accuracy_5 = val(student, val_loader, device)
    print(f'{str(100 * accuracy_1.detach().cpu().numpy())}%')
    print(f'{str(100 * accuracy_5.detach().cpu().numpy())}%')


if __name__ == '__main__':
    main()
