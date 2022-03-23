import argparse
import os

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

from models import ConditionalGenerator

repackage.up()
from MARS.model import generate_model
from approximate_gradients import approximate_gradients
from config.cfg_parser import cfg_parser
from utils_common import set_seed, swin_transform


def get_config():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/params.yaml")

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

    total_loss_S = 0
    total_loss_G = 0

    # Repeat epoch_itrs times per outer epoch
    for i in tqdm(range(args.epoch_itrs), position=0, leave=True):
        for _ in range(args.g_iter):
            # Generate labels for Conditional Generator
            labels = torch.randn((args.batch_size, args.num_classes)).to(device)
            labels = torch.argmax(labels, dim=1)
            labels_onehot = torch.nn.functional.one_hot(labels, args.num_classes)
            # Sample Random Noise
            z = torch.randn((args.batch_size, args.nz)).to(device)
            optimizer_G.zero_grad()
            generator.train()
            # Get fake image from generator
            # pre_x returns the output of G before applying the activation
            fake = generator(z, label=labels_onehot, pre_x=args.approx_grad)
            fake = fake.unsqueeze(dim=2)

            # Approximate gradients
            approx_grad_wrt_x, loss_G = approximate_gradients(
                args, victim_model, threat_model, fake,
                epsilon=args.grad_epsilon, m=args.grad_m,
                device=device, device_tf=device_tf, pre_x=True)

            fake.backward(approx_grad_wrt_x)
            optimizer_G.step()

            total_loss_G += loss_G.item()

        print(f'Total loss G:', total_loss_G / (i + 1))

        for _ in range(args.d_iter):
            # Generate labels for Conditional Generator
            labels = torch.randn((args.batch_size, args.num_classes)).to(device)
            labels = torch.argmax(labels, dim=1)
            labels_onehot = torch.nn.functional.one_hot(labels, args.num_classes)
            # Sample Random Noise
            z = torch.randn((args.batch_size, args.nz)).to(device)

            fake = generator(z, label=labels_onehot).detach()
            fake = fake.unsqueeze(dim=2)
            N, C, L, S = fake.shape[:4]
            optimizer_T.zero_grad()

            with torch.no_grad():
                if args.victim_model == 'swin-t':
                    fake_swin = swin_transform(fake.detach())
                    logits_victim = victim_model(fake_swin, return_loss=False)
                else:
                    fake_tf = fake.reshape(N, L, S, S, C)
                    with tf.device(device_tf):
                        tf_tensor = tf.convert_to_tensor(fake_tf.cpu().numpy())
                        logits_victim = victim_model(tf_tensor).numpy()
                logits_victim = torch.tensor(logits_victim).to(device)

            # Correction for the fake logits
            if args.loss == "l1" and args.no_logits:
                logits_victim = torch.log(logits_victim)
                if args.logit_correction == 'min':
                    logits_victim -= logits_victim.min(dim=1).values.view(-1, 1)
                elif args.logit_correction == 'mean':
                    logits_victim -= logits_victim.mean(dim=1).view(-1, 1)
                logits_victim = logits_victim.detach()

            logits_threat = torch.nn.Softmax(dim=1)(threat_model(fake))
            loss_S = threat_loss(args, logits_threat, logits_victim)
            loss_S.backward()
            optimizer_T.step()

            total_loss_S += loss_S.item()

            # Print accuracy of Generator via Victim's logits
            g_t1 = 100 * accuracy(logits_victim, labels, top_k=1)
            g_t5 = 100 * accuracy(logits_victim, labels, top_k=5)
            print(f'Generator accuracy. T1: {g_t1}, T5: {g_t5}')

            # Print accuracy of Threat Model via Victim's logits
            victim_argmax = logits_victim.argmax(axis=1)
            t_t1 = 100 * accuracy(logits_threat, victim_argmax, top_k=1)
            t_t5 = 100 * accuracy(logits_threat, victim_argmax, top_k=5)
            print(f'Threat accuracy. T1: {t_t1}, T5: {t_t5}')

        print(f'Total loss S:', total_loss_S / (i + 1))


def main():
    # Prepare the environment
    args = get_config()["experiment"]
    set_seed(args.seed)
    print(args.victim_model)
    print(args.victim_model == 'swin-t')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.device if use_cuda else 'cpu')
    device_tf = '/device:GPU:%d' % args.device if use_cuda else '/device:CPU'
    args.device, args.device_tf = device, device_tf

    args.normalization_coefs = None
    args.G_activation = torch.sigmoid

    if args.victim_model == 'swin-t':
        args.num_classes = 400
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
        hub_url = 'https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3'

        encoder = hub.KerasLayer(hub_url, trainable=False)
        inputs = tf.keras.layers.Input(shape=[None, None, None, 3],
                                       dtype=tf.float32, name='image')

        # [batch_size, 600]
        outputs = encoder(dict(image=inputs))
        victim_model = tf.keras.Model(inputs, outputs, name='movinet')

    generator = ConditionalGenerator(
        nz=args.nz, nc=3, img_size=224, num_classes=args.num_classes,
        activation=args.G_activation)
    generator = generator.to(device)

    threat_model, threat_parameters = generate_model(args.num_classes)
    threat_model = threat_model.to(device)

    # REVIEW: Decide if we're adding functionality to load a threat_model from checkpoint
    #   probably need this somewhere, at least in some eval script
    # if args.student_load_path :
    #     # "checkpoint/student_no-grad/cifar10-resnet34_8x.pt"
    #     threat_model.load_state_dict( torch.load( args.student_load_path ) )
    #     myprint("Student initialized from %s"%(args.student_load_path))
    #     acc = test(args, threat_model=threat_model, generator=generator, device = device, test_loader = test_loader)

    optimizer_T = optim.SGD(threat_parameters, lr=args.lr_S,
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
