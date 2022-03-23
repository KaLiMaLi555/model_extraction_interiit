import argparse
import json
import os
import random
from pprint import pprint

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.optim as optim
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from tqdm.notebook import tqdm

# TODO: Fix import in the final code
from model_extraction_interiit.Blackbox.approximate_gradients import approximate_gradients_conditional
from models import ConditionalGenerator


# TODO: convert to cfg args
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD Swin-T Image')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=20, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--g_iter', type=int, default=10, help='Number of generator iterations per epoch_iter')

    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
    parser.add_argument('--nz', type=int, default=256, help='Size of random noise input to generator')

    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')

    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'], )
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', 'none'], )
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float, help='Percentage epochs at which to take next step')
    parser.add_argument('--scale', type=float, default=3e-1, help='Fractional decrease in lr')

    # parser.add_argument('--dataset', type=str, default='cifar10', choices=['svhn', 'cifar10'], help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--model_id', type=str, default='debug')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='results')

    # Gradient approximation parameters
    parser.add_argument('--approx_grad', type=int, default=1, help='Always set to 1')
    parser.add_argument('--grad_m', type=int, default=5, help='Number of steps to approximate the gradients')
    parser.add_argument('--grad_epsilon', type=float, default=1e-3)

    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])

    parser.add_argument('--rec_grad_norm', type=int, default=1)

    parser.add_argument('--MAZE', type=int, default=0)

    parser.add_argument('--store_checkpoints', type=int, default=1)
    parser.add_argument('--val_epoch', type=int, default=5)

    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='model_extraction')
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--wandb_resume', action='store_true')
    parser.add_argument('--wandb_watch', action='store_true')
    parser.add_argument('--checkpoint_base', type=str, default='/content')
    parser.add_argument('--checkpoint_path', type=str, default='/drive/MyDrive/DFAD_video_ckpts')
    parser.add_argument('--wandb_save', action='store_true')

    return parser.parse_args()


def pretrain(args, victim_model, generator, device, device_tf, optimizer):
    """Main Loop for one epoch of Pretraining Generator"""
    if args.model == 'swin-t':
        victim_model.eval()

    for i in tqdm(range(args.epoch_itrs), position=0, leave=True):
        # Repeat epoch_itrs times per epoch
        # Sample Random Noise
        labels = torch.argmax(torch.randn((args.batch_size, args.num_classes)), dim=1).to(device)
        labels_onehot = torch.nn.functional.one_hot(labels, args.num_classes)
        z = torch.randn((args.batch_size, args.nz)).to(device)
        optimizer.zero_grad()
        generator.train()

        # Get fake image from generator
        # pre_x returns the output of G before applying the activation
        fake = generator(z, label=labels_onehot, pre_x=True)
        fake = fake.unsqueeze(dim=2)

        # Perform gradient approximation
        approx_grad_wrt_x, loss = approximate_gradients_conditional(
            args, victim_model, fake, labels=labels,
            epsilon=args.grad_epsilon, m=args.grad_m,
            device=device, device_tf=device_tf, pre_x=True)

        fake.backward(approx_grad_wrt_x)
        optimizer.step()


def main():
    # TODO: Replace with cfg parser stuff
    args = parse_args()

    # TODO: Use common set_env util
    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # REVIEW: Decide if we're keeping this query budget stuff
    args.query_budget *= 10 ** 6
    args.query_budget = int(args.query_budget)

    threat_options = ['rescnnlstm', 'mars', 'mobilenet']
    victim_options = ['swin-t', 'movinet']
    mode_options = ['image', 'video']

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.device if use_cuda else 'cpu')
    device_tf = '/device:GPU:%d' % args.device if use_cuda else '/device:CPU'
    args.device, args.device_tf = device, device_tf

    # TODO: cfg parser stuff
    args.normalization_coefs = None
    args.G_activation = torch.sigmoid
    args.num_classes = 400

    pprint(args, width=80)

    if args.victim_model == 'swin-t':
        config = args.swin_t_config
        checkpoint = args.swin_t_checkpoint
        cfg = Config.fromfile(config)
        victim_model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(victim_model, checkpoint, map_location=device)
        victim_model.eval()
        victim_model = victim_model.to(device)
    else:
        hub_url = 'https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3'

        encoder = hub.KerasLayer(hub_url, trainable=False)
        inputs = tf.keras.layers.Input(shape=[None, None, None, 3], dtype=tf.float32, name='image')

        # [batch_size, 600]
        outputs = encoder(dict(image=inputs))
        victim_model = tf.keras.Model(inputs, outputs, name='movinet')

    generator = ConditionalGenerator(nz=args.nz, nc=3, img_size=224, num_classes=400, activation=args.G_activation)
    generator = generator.to(device)

    optimizer = optim.Adam(generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * args.epochs) for step in args.steps])
    print('Learning rate scheduling at steps: ', steps)
    print()

    if args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, args.scale)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Train loop
    for epoch in range(1, args.epochs + 1):
        if args.scheduler != 'none':
            scheduler.step()

        pretrain(args, victim_model=victim_model, generator=generator,
                 device=device, device_tf=device_tf, optimizer=optimizer)
        checkpoint = {
            'outer_epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'generator': generator.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        # TODO: Save checkpoint


if __name__ == '__main__':
    main()
