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

from config.cfg_parser import cfg_parser
# TODO: Fix import in the final code
from approximate_gradients import approximate_gradients_conditional
from cGAN.models import ConditionalGenerator


def get_config():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/params.yaml")

    cfg = cfg_parser(parser.parse_args().config)

    return cfg


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
    args = get_config()["experiment"]

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    generator = ConditionalGenerator(
        nz=args.nz, nc=3, img_size=224, num_classes=args.num_classes,
        activation=args.G_activation)
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

        # Generate and save checkpoint
        checkpoint = {
            'outer_epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'generator': generator.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        f_path = os.path.join(
            args.checkpoint_path, 'Epoch_' + str(epoch) + '.pth')
        torch.save(checkpoint, f_path)


if __name__ == '__main__':
    main()
