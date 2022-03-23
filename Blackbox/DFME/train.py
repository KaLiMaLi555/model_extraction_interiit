import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import repackage
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn.functional as F
import torch.optim as optim
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from torchmetrics.functional import accuracy
from tqdm.notebook import tqdm

from models.video_gan import VideoGAN

repackage.up()
from MARS.model import generate_model
from approximate_gradients import approximate_gradients
from config.cfg_parser import cfg_parser
from utils_common import set_seed, swin_transform


def get_config():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/params_dfme_swint.yaml")

    cfg = cfg_parser(parser.parse_args().config)

    return cfg


def threat_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for Threat Model"""
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


def train(args, victim_model, threat_model, generator, device, device_tf,
          optimizer):
    """Main Loop for one epoch of Training Generator and Threat Models"""
    if args.victim_model == 'swin-t':
        victim_model.eval()
    threat_model.train()

    optimizer_T, optimizer_G = optimizer

    total_loss_T = 0
    total_loss_G = 0

    # Repeat epoch_itrs times per outer epoch
    for i in tqdm(range(args.epoch_itrs), position=0, leave=True):
        for _ in range(args.g_iter):
            # Sample Random Noise
            z = torch.randn((args.batch_size, args.nz)).to(device)
            optimizer_G.zero_grad()
            generator.train()
            # Get fake image from generator
            # pre_x returns the output of G before applying the activation
            fake = generator(z, pre_x=args.approx_grad)
            # fake = fake.unsqueeze(dim=2)

            # Approximate gradients
            approx_grad_wrt_x, loss_G = approximate_gradients(
                args, victim_model, threat_model, fake,
                epsilon=args.grad_epsilon, m=args.grad_m,
                device=device, device_tf=device_tf, pre_x=True)

            fake.backward(approx_grad_wrt_x)
            optimizer_G.step()

            total_loss_G += loss_G.item()

        # Print Generator loss
        print(f'Total loss Generator:', total_loss_G / (i + 1))

        # Arrays to accumulate accuracies
        t_t1_list, t_t5_list = [], []

        for _ in range(args.d_iter):
            # Sample Random Noise
            z = torch.randn((args.batch_size, args.nz)).to(device)

            fake = generator(z).detach()
            N, C, L, S = fake.shape[:4]
            optimizer_T.zero_grad()

            with torch.no_grad():
                if args.victim_model == 'swin-t':
                    fake_swin = swin_transform(fake.detach())
                    logits_victim = victim_model(fake_swin, return_loss=False)
                else:
                    fake_tf = fake.reshape(N, L, S, S, C).cpu().numpy()
                    with tf.device(device_tf):
                        tf_tensor = tf.convert_to_tensor(fake_tf)
                        logits_victim = victim_model(tf_tensor).numpy()
                logits_victim = torch.tensor(logits_victim).to(device)

            logits_threat = torch.nn.Softmax(dim=1)(threat_model(fake))

            # Print accuracy of Threat Model via Victim's logits
            victim_argmax = logits_victim.argmax(axis=1)
            t_t1 = 100 * accuracy(logits_threat, victim_argmax, top_k=1)
            t_t5 = 100 * accuracy(logits_threat, victim_argmax, top_k=5)
            t_t1_list.append(t_t1.detach().cpu().numpy())
            t_t5_list.append(t_t5.detach().cpu().numpy())

            # Correction for the fake logits
            if args.loss == "l1" and args.no_logits:
                logits_victim = torch.log(logits_victim)
                if args.logit_correction == 'min':
                    logits_victim -= logits_victim.min(dim=1).values.view(-1, 1)
                elif args.logit_correction == 'mean':
                    logits_victim -= logits_victim.mean(dim=1).view(-1, 1)
                logits_victim = logits_victim.detach()

            # Compute Threat loss and backpropagate
            loss_T = threat_loss(args, logits_threat, logits_victim)
            loss_T.backward()
            optimizer_T.step()

            total_loss_T += loss_T.item()

        # Print Threat model loss
        print(f'Total loss Threat:', total_loss_T / (i + 1))

        # Print inner epoch accuracies
        t_t1, t_t5 = np.array(t_t1_list).mean(), np.array(t_t5_list).mean()
        print(f'Threat accuracy. T1: {t_t1}, T5: {t_t5}')


def main():
    # Prepare the environment
    args = get_config()["experiment"]
    set_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.device if use_cuda else 'cpu')
    device_tf = '/device:GPU:%d' % args.device if use_cuda else '/device:CPU'
    args.device, args.device_tf = device, device_tf

    args.normalization_coefs = None
    args.G_activation = torch.sigmoid

    if args.victim_model == 'swin-t':
        args.num_classes = 400
        args.no_logits = 1
        config = args.swin_t_config
        checkpoint = args.swin_t_checkpoint
        cfg = Config.fromfile(config)
        victim_model = build_model(cfg.model, train_cfg=None,
                                   test_cfg=cfg.get('test_cfg'))
        load_checkpoint(victim_model, checkpoint, map_location=device)
        victim_model.eval()
        victim_model = victim_model.to(device)
    else:
        args.num_classes = 600
        args.no_logits = 0
        hub_url = 'https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3'

        encoder = hub.KerasLayer(hub_url, trainable=False)
        inputs = tf.keras.layers.Input(shape=[None, None, None, 3],
                                       dtype=tf.float32, name='image')

        # [batch_size, 600]
        outputs = encoder(dict(image=inputs))
        victim_model = tf.keras.Model(inputs, outputs, name='movinet')

    generator = VideoGAN(zdim=args.nz)
    generator = generator.to(device)

    threat_model, threat_parameters = generate_model(args.num_classes)
    threat_model = threat_model.to(device)
    # Allow loading a pretrained checkpoint to resume training
    if args.threat_checkpoint:
        threat_model.module.load_state_dict(torch.load(args.threat_checkpoint))

    optimizer_T = optim.SGD(threat_parameters, lr=args.lr_T,
                            weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * args.epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)
    print()

    if args.scheduler == "multistep":
        scheduler_T = optim.lr_scheduler.MultiStepLR(optimizer_T, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_T = optim.lr_scheduler.CosineAnnealingLR(optimizer_T, args.epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.epochs)

    if args.generator_checkpoint:
        checkpoint = torch.load(args.generator_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        for g in optimizer_G.param_groups:
            g['lr'] = args.lr_G

    # Train loop
    for epoch in range(1, args.epochs + 1):
        if args.scheduler != 'none':
            scheduler_T.step()
            scheduler_G.step()

        train(args, victim_model=victim_model, threat_model=threat_model,
              generator=generator, device=device, device_tf=device_tf,
              optimizer=[optimizer_T, optimizer_G])

        # Generate and save checkpoint
        checkpoint = {
            'outer_epoch': epoch,
            'optimizer_T': optimizer_T.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'generator': generator.state_dict(),
            'threat': threat_model.state_dict(),
            'scheduler_T': scheduler_T.state_dict(),
            'scheduler_G': scheduler_G.state_dict(),
        }
        f_path = os.path.join(
            args.checkpoint_path, 'Epoch_' + str(epoch) + '.pth')
        torch.save(checkpoint, f_path)


if __name__ == '__main__':
    main()
