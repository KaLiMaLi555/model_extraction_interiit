# Import required libraries
import os
import torch
import wandb
import random
import network
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import tensorflow as tf
import torch.optim as optim
import tensorflow_hub as hub
import torch.nn.functional as F
from collections import Counter

# Import for Swin-T
# from mmcv import Config
# from VST.mmaction.models import build_model
# from mmcv.runner import load_checkpoint
from utils.wandb_utils import init_wandb, save_ckp
from val_utils import metrics
from val_utils.custom_transformations import CustomResizeTransform, CustomMobilenetTransform
from val_utils.dataloader_val import ValDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_mobilenet_transform(inp):
    t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t(inp)


def train_epoch(args, teacher, student, generator, device, optimizers, epoch, step_S, step_G):
    # teacher.eval()

    device_tf = tf.test.gpu_device_name()
    if device_tf != '/device:GPU:0':
        print('GPU not found!')
        device_tf = '/device:CPU:0'

    debug_distribution = True
    dist_t, dist_s = [], []
    optimizer_S, optimizer_G = optimizers

    for i in tqdm(range(args.epoch_itrs), position=0, leave=True):

        print(f"Outer Epoch: {epoch}, Inner Epoch: {i + 1}")

        total_loss_S = 0
        total_loss_G = 0

        student.train()
        generator.eval()

        for k in tqdm(range(step_S), position=0, leave=True):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
            optimizer_S.zero_grad()

            # Returns: b, c, w, h
            fake = torch.sigmoid(generator(z).detach())
            # b, 1, c, w, h
            fake_shape = fake.shape

            # TODO: Change as needed for Swin-T
            # t_logit = torch.tensor(teacher(fake)).to(device)

            # Reshaped to: b, h, w, c
            fake_tf = fake.reshape(fake_shape[0], fake_shape[2], fake_shape[3], fake_shape[1])
            # Unsqueezed to: b, 1, h, w, c
            fake_tf = torch.unsqueeze(fake_tf, dim=1)
            with tf.device(device_tf):
                tf_tensor = tf.convert_to_tensor(fake_tf.cpu().numpy())
                # Teacher expects: b, f, h, w, c
                t_logit = teacher(tf_tensor).numpy()
                t_logit = torch.tensor(t_logit).to(device)

            # Preprocess expects: b, c, h, w
            transformed = apply_mobilenet_transform(fake)
            # Student expects: b, c, h, w
            s_logit = student(transformed).to(device)

            loss_S = F.l1_loss(s_logit, t_logit)
            total_loss_S += loss_S.item()
            wandb.log({'Loss_S_inner_verbose': loss_S.item()})
            loss_S.backward()
            optimizer_S.step()

            if debug_distribution:
                dist_t.append(torch.argmax(t_logit.detach(), dim=1).cpu().numpy())
                dist_s.append(torch.argmax(s_logit.detach(), dim=1).cpu().numpy())

        wandb.log({'Loss_S': total_loss_S, 'epoch': epoch})
        wandb.log({'Loss_S_inner': total_loss_S})
        print('Loss on Student model:', total_loss_S / step_S)

        student.eval()
        generator.train()

        for k in tqdm(range(step_G), position=0, leave=True):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
            optimizer_G.zero_grad()
            generator.train()

            # Returns: b, c, w, h
            fake = torch.sigmoid(generator(z))
            # b, 1, c, w, h
            fake_shape = fake.shape

            # TODO: Change as needed for Swin-T
            # t_logit = torch.tensor(teacher(fake)).to(device)

            fake_tf = torch.empty_like(fake).copy_(fake).detach()
            # Reshaped to: b, h, w, c
            fake_tf = fake_tf.reshape(fake_shape[0], fake_shape[2], fake_shape[3], fake_shape[1])
            # Unsqueezed to: b, 1, h, w, c
            fake_tf = torch.unsqueeze(fake_tf, dim=1)
            with tf.device(device_tf):
                tf_tensor = tf.convert_to_tensor(fake_tf.cpu().numpy())
                # Teacher expects: b, f, h, w, c
                t_logit = teacher(tf_tensor).numpy()
                t_logit = torch.tensor(t_logit).to(device)

            # Preprocess expects: b, c, h, w
            transformed = apply_mobilenet_transform(fake)
            # Student expects: b, c, h, w
            s_logit = student(transformed).to(device)

            loss_G = - F.l1_loss(s_logit, t_logit)
            total_loss_G += loss_G.item()
            wandb.log({'Loss_G_inner_verbose': loss_G.item()})
            loss_G.backward()
            optimizer_G.step()

            if debug_distribution:
                dist_t.append(torch.argmax(t_logit.detach(), dim=1).cpu().numpy())
                dist_s.append(torch.argmax(s_logit.detach(), dim=1).cpu().numpy())

        wandb.log({'Loss_G': total_loss_G, 'epoch': epoch})
        wandb.log({'Loss_G_inner': total_loss_G})
        print('Loss on Generator model:', total_loss_G)

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))

    checkpoint = {
        'outer_epoch': epoch,
        'inner_epoch': i + 1,
        'optimizer_S': optimizer_S.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'generator': generator.state_dict(),
        'student': student.state_dict(),
        # 'scheduler':scheduler.state_dict(),
        # 'criterion': criterion.state_dict()
    }

    if args.wandb_save:
        save_ckp(checkpoint, epoch, args.checkpoint_path, args.checkpoint_base, args.wandb_save)

    if debug_distribution:
        c_t = Counter(list(np.array(dist_t).flatten())).most_common()
        c_s = Counter(list(np.array(dist_s).flatten())).most_common()
        wandb.log({f'Teacher distribution epoch {epoch}': c_t})
        wandb.log({f'Student distribution epoch {epoch}': c_s})
        print(f'Teacher distribution epoch {epoch}:', c_t)
        print(f'Student distribution epoch {epoch}:', c_s)


def val(student, dataloader, device):
    accuracy_1, accuracy_5 = [], []
    for (x, y) in tqdm(dataloader, total=len(dataloader)):
        x, y = x.to(device), y.to(device)
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        # print(x_shape, x.shape)

        # Student expects: b, c, h, w
        logits = student(x).detach()
        del x
        accuracy_1.append(metrics.topk_accuracy(logits, y, 1))
        accuracy_5.append(metrics.topk_accuracy(logits, y, 5))
        del y, logits

    return sum(accuracy_1) / len(accuracy_1), sum(accuracy_5) / len(accuracy_5)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD MNIST')
    parser.add_argument('--model_name', default="movinet", type=str)
    parser.add_argument('--num_classes', type=int, default=600)
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--step_S', default=5, type=int)
    parser.add_argument('--step_G', default=1, type=int)

    parser.add_argument('--lr_S', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=False)

    parser.add_argument('--val_data_dir', type=str,
                        default='/content/val_data/content/k600_val_16_frames_uniform')
    parser.add_argument('--val_classes_file', type=str,
                        default='/content/val_data/classes.csv')
    parser.add_argument('--val_labels_file', type=str,
                        default='/content/val_data/labels.csv')
    parser.add_argument('--val_num_workers', type=int, default=2)

    parser.add_argument('--val_epoch', type=int, default=5)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--val_scale', type=float, default=1)
    parser.add_argument('--val_scale_inv', type=float, default=255)
    parser.add_argument('--val_shift', type=float, default=0)

    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--wandb_project', type=str, default="model_extraction")
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--wandb_resume', action="store_true")
    parser.add_argument('--wandb_watch', action="store_true")
    parser.add_argument('--checkpoint_base', type=str, default="/content")
    parser.add_argument('--checkpoint_path', type=str,
                        default="/drive/MyDrive/DFAD_video_ckpts")
    parser.add_argument('--wandb_save', action="store_true")

    print("\nArguments are as follows:\n")

    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device = "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.model_name == "swin-t":
        print()
        config = "./VST/configs/_base_/models/swin/swin_tiny.py"
        checkpoint = "./swin_tiny_patch244_window877_kinetics400_1k.pth"
        cfg = Config.fromfile(config)
        teacher = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(teacher, checkpoint, map_location=device)

    elif args.model_name == "movinet":
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        print("\n######################## Loading Model ########################\n")
        hub_url = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"

        encoder = hub.KerasLayer(hub_url, trainable=False)
        inputs = tf.keras.layers.Input(shape=[None, None, None, 3], dtype=tf.float32, name='image')

        # [batch_size, 600]
        outputs = encoder(dict(image=inputs))
        teacher = tf.keras.Model(inputs, outputs, name='movinet')
    print('\nLoaded teacher')

    # NOTE: Uncomment for using CNN LSTM
    # student = network.models.ResCNNRNN()
    # NOTE: Lightweight CNN model (Mobile Net)
    student = torchvision.models.mobilenet_v2()
    student.classifier[1] = torch.nn.Linear(in_features=student.classifier[1].in_features, out_features=args.num_classes)
    print('Loaded student')
    # TODO: Unhardcode ngpu
    generator = network.models.ImageGenerator(ngpu=1, nz=args.nz, ngf=args.image_size)
    print('Loaded generator\n')

    if args.wandb:
        init_wandb(student, args.wandb_api_key, args.wandb_resume, args.wandb_name, args.wandb_project, args.wandb_run_id, args.wandb_watch)

    if args.model_name == "swin-t":
        teacher = teacher.to(device)
    student = student.to(device)
    generator = generator.to(device)

    print(count_parameters(generator))
    # exit(0)

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    if args.scheduler:
        scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, args.step_size, 0.1)
        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, args.step_size, 0.1)

    if args.val_scale == 1:
        args.val_scale = 1 / args.val_scale_inv
    val_data = ValDataset(args.val_data_dir, args.val_classes_file,
                          args.val_labels_file, args.num_classes,
                          transform=CustomMobilenetTransform(size=args.image_size),
                          scale=args.val_scale, shift=args.val_shift)

    val_loader = DataLoader(val_data, batch_size=args.val_batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=args.val_num_workers)

    for epoch in range(1, args.epochs + 1):
        # Train
        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()

        print("################### Training Student and Generator Models ###################\n")
        train_epoch(args, teacher=teacher, student=student, generator=generator,
                    device=device, optimizers=[optimizer_S, optimizer_G],
                    epoch=epoch, step_S=args.step_S, step_G=args.step_G)

        # Validate student model
        if epoch % args.val_epoch == 0:
            print("################### Evaluating Student Model ###################\n")
            student.eval()
            acc_1, acc_5 = val(student, val_loader, device)
            acc_1 = 100 * acc_1.detach().cpu().numpy()
            acc_5 = 100 * acc_5.detach().cpu().numpy()
            print(f'\nEpoch {epoch}')
            print(f'Top-1: {str(acc_1)}, Top-5: {str(acc_5)}\n')
            wandb.log({'val_T1': acc_1, 'epoch': epoch})
            wandb.log({'val_T5': acc_5, 'epoch': epoch})


if __name__ == '__main__':
    main()
