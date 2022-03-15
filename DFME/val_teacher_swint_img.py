import argparse
import random

import torch
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from approximate_gradients_swint_img import *
from val_utils import metrics
from val_utils.custom_transformations import CustomStudentImageTransform
from val_utils.dataloader_val import ValDataset

print("torch version", torch.__version__)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Swin-T Image Validation')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--val_data_dir', type=str,
                        default='/content/val_data/k400_16_frames_uniform')
    parser.add_argument('--val_classes_file', type=str,
                        default='/content/val_data/k400_16_frames_uniform/classes.csv')
    parser.add_argument('--val_labels_file', type=str,
                        default='/content/val_data/k400_16_frames_uniform/labels.csv')
    parser.add_argument('--val_num_workers', type=int, default=2)

    parser.add_argument('--val_epoch', type=int, default=5)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--val_scale', type=float, default=1)
    parser.add_argument('--val_scale_inv', type=float, default=255)
    parser.add_argument('--val_shift', type=float, default=0)

    return parser.parse_args()


def val(teacher, dataloader, device):
    accuracy_1, accuracy_5 = [], []
    for (x, y) in tqdm(dataloader, total=len(dataloader)):
        # TODO: Consider printing x to make sure scaling is working correctly
        x, y = x.to(device), y.to(device)
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], x_shape[4], x_shape[2], x_shape[3])
        # print(x_shape, x.shape)

        # Expects and returns: N, C, L, S, S
        x = network.swin.swin_transform(x)
        logits = torch.Tensor(teacher(x, return_loss=False)).to(device)
        del x
        accuracy_1.append(metrics.topk_accuracy(logits, y, 1))
        accuracy_5.append(metrics.topk_accuracy(logits, y, 5))
        del y, logits

    return sum(accuracy_1) / len(accuracy_1), sum(accuracy_5) / len(accuracy_5)


def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    args.device = device
    args.num_classes = 400

    print(args)

    config = "./Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    checkpoint = "/content/swin_tiny_patch244_window877_kinetics400_1k.pth"
    cfg = Config.fromfile(config)
    teacher = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(teacher, checkpoint, map_location=device)

    teacher.eval()
    teacher = teacher.to(device)

    if args.val_scale == 1:
        args.val_scale = 1 / args.val_scale_inv
    # val_data = ValDataset(args.val_data_dir, args.val_classes_file,
    #                       args.val_labels_file, args.num_classes,
    #                       transform=CustomMobilenetTransform(size=args.image_size),
    #                       scale=args.val_scale, shift=args.val_shift)
    val_data = ValDataset(
        args.val_data_dir, args.val_classes_file, args.val_labels_file,
        args.num_classes, transform=CustomStudentImageTransform(size=224),
        scale=args.val_scale, shift=args.val_shift)

    val_loader = DataLoader(
        val_data, batch_size=args.val_batch_size, shuffle=False,
        drop_last=False, num_workers=args.val_num_workers)

    # Validating teacher on K400
    with torch.no_grad():
        teacher.eval()
        acc_1, acc_5 = val(teacher, val_loader, device)
        acc_1 = 100 * acc_1.detach().cpu().numpy()
        acc_5 = 100 * acc_5.detach().cpu().numpy()
    print(f'Top-1: {str(acc_1)}, Top-5: {str(acc_5)}\n')


if __name__ == '__main__':
    main()
