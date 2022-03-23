import argparse
import json
import os
import random
from pprint import pprint

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# from approximate_gradients import *
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from config.cfg_parser import cfg_parser
from torchmetrics import accuracy
from tqdm.notebook import tqdm

from approximate_gradients import approximate_gradients
from utils_common import swin_transform
from cGAN.models import ConditionalGenerator
# from utils_common import


# TODO: Get MARS working once added
# from model_extraction_interiit.prod.BlackBox import MARS

def config():
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


def train(args, victim_model, threat_model, generator, device, device_tf, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Threat Models"""
    if args.model == 'swin-t':
        victim_model.eval()
    threat_model.train()

    optimizer_S, optimizer_G = optimizer

    total_loss_S = 0
    total_loss_G = 0

    for i in tqdm(range(args.epoch_itrs), position=0, leave=True):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(args.g_iter):
            # Sample Random Noise
            labels = torch.argmax(torch.randn((args.batch_size, args.num_classes)), dim=1).to(device)
            labels_onehot = torch.nn.functional.one_hot(labels, args.num_classes)
            z = torch.randn((args.batch_size, args.nz)).to(device)
            optimizer_G.zero_grad()
            generator.train()
            # Get fake image from generator
            fake = generator(z, label=labels_onehot, pre_x=args.approx_grad)  # pre_x returns the output of G before applying the activation
            fake = fake.unsqueeze(dim=2)

            # Approximate gradient
            approx_grad_wrt_x, loss_G = approximate_gradients(
                args, victim_model, threat_model, fake,
                epsilon=args.grad_epsilon, m=args.grad_m,
                device=device, device_tf=device_tf, pre_x=True)

            fake.backward(approx_grad_wrt_x)
            optimizer_G.step()

            total_loss_G += loss_G.item()

        print(f'Total loss G:', total_loss_G / (i + 1))

        for _ in range(args.d_iter):
            labels = torch.argmax(torch.randn((args.batch_size, args.num_classes)), dim=1).to(device)
            labels_onehot = torch.nn.functional.one_hot(labels, args.num_classes)
            z = torch.randn((args.batch_size, args.nz)).to(device)

            fake = generator(z, label=labels_onehot).detach()
            fake = fake.unsqueeze(dim=2)
            optimizer_S.zero_grad()

            with torch.no_grad():
                fake_swin = swin_transform(fake.detach())
                logits = victim_model(fake_swin, return_loss=False)
                logits = torch.Tensor(logits).to(device)
                t_argmax = logits.argmax(axis=1)

                # REVIEW: Are we printing this stuff? This is some of the only
                #         val we can do as it doesn't involve loading real data
                # t_t1 = 100 * accuracy(logits, labels, top_k=1)
                # t_t5 = 100 * accuracy(logits, labels, top_k=5)

            # Correction for the fake logits
            if args.loss == "l1" and args.no_logits:
                logits = torch.log(logits).detach()
                if args.logit_correction == 'min':
                    logits -= logits.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    logits -= logits.mean(dim=1).view(-1, 1).detach()

            s_logit = torch.nn.Softmax(dim=1)(threat_model(fake[:, :, 0, :, :]))
            loss_S = threat_loss(args, s_logit, logits)
            loss_S.backward()
            optimizer_S.step()

            total_loss_S += loss_S.item()

            # REVIEW: Are we printing this stuff? This is some of the only
            #         val we can do as it doesn't involve loading real data
            t1 = 100 * accuracy(s_logit, t_argmax, top_k=1)
            t5 = 100 * accuracy(s_logit, t_argmax, top_k=5)

        print(f'Total loss S:', total_loss_S / (i + 1))


def main():
    args = config()["experiment"]

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.query_budget *= 10 ** 6
    args.query_budget = int(args.query_budget)

    threat_options = ['rescnnlstm', 'mars', 'mobilenet']
    victim_options = ['swin-t', 'movinet']
    mode_options = ['image', 'video']

    pprint(args, width=80)
    print(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.store_checkpoints:
        os.makedirs(args.log_dir + "/checkpoint", exist_ok=True)

    # Save JSON with parameters
    with open(args.log_dir + "/parameters.json", "w") as f:
        json.dump(vars(args), f)

    with open(args.log_dir + "/loss.csv", "w") as f:
        f.write("epoch,loss_G,loss_S\n")

    if args.rec_grad_norm:
        with open(args.log_dir + "/norm_grad.csv", "w") as f:
            f.write("epoch,G_grad_norm,S_grad_norm,grad_wrt_X\n")

    with open("latest_experiments.txt", "a") as f:
        f.write(args.log_dir + "\n")


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.device if use_cuda else 'cpu')
    device_tf = '/device:GPU:%d' % args.device if use_cuda else '/device:CPU'
    args.device, args.device_tf = device, device_tf

    # Preparing checkpoints for the best Student
    global file
    model_dir = f"checkpoint/student_{args.model_id}";
    args.model_dir = model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    file = open(f"{args.model_dir}/logs.txt", "w")
    print(args)

    # Eigen values and vectors of the covariance matrix
    # _, test_loader = get_dataloader(args)

    args.normalization_coefs = None
    args.G_activation = torch.sigmoid
    args.num_classes = 400

    if args.victim_model == 'swin-t':
        # TODO: cfg parser for VST file paths
        config = args.swin_t_config 
        checkpoint = args.swin_t_checkpoint
        cfg = Config.fromfile(config)
        victim_model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(victim_model, checkpoint, map_location=device)
        victim_model.eval()
        victim_model = victim_model.to(device)
    else:
        hub_url = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"

        encoder = hub.KerasLayer(hub_url, trainable=False)
        inputs = tf.keras.layers.Input(shape=[None, None, None, 3], dtype=tf.float32, name='image')

        # [batch_size, 600]
        outputs = encoder(dict(image=inputs))
        victim_model = tf.keras.Model(inputs, outputs, name='movinet')

    # TODO: Load MARS as threat model
    threat_model = torchvision.models.mobilenet_v2()
    threat_model = threat_model.to(device)

    generator = ConditionalGenerator(nz=args.nz, nc=3, img_size=224, num_classes=400, activation=args.G_activation)
    generator = generator.to(device)

    args.generator = generator
    args.threat_model = threat_model
    args.victim_model = victim_model

    # REVIEW: Decide if we're adding functionality to load a threat_model from checkpoint
    #   probably need this somewhere, at least in some eval script
    # if args.student_load_path :
    #     # "checkpoint/student_no-grad/cifar10-resnet34_8x.pt"
    #     threat_model.load_state_dict( torch.load( args.student_load_path ) )
    #     myprint("Student initialized from %s"%(args.student_load_path))
    #     acc = test(args, threat_model=threat_model, generator=generator, device = device, test_loader = test_loader)

    ## Compute the number of epochs with the given query budget:
    # REVIEW: Decide if we're keeping this query budget stuff
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m + 1) + args.d_iter)

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    optimizer_S = optim.SGD(threat_model.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)
    print()

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    if args.generator_checkpoint:
        checkpoint = torch.load(args.generator_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        for g in optimizer_G.param_groups:
            g['lr'] = args.lr_G

    for epoch in range(1, number_epochs + 1):
        # Train
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

        train(args, victim_model=victim_model, threat_model=threat_model,
              generator=generator, device=device, device_tf=device_tf,
              optimizer=[optimizer_S, optimizer_G], epoch=epoch)

        checkpoint = {
            'outer_epoch': epoch,
            'optimizer_S': optimizer_S.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'generator': generator.state_dict(),
            'student': threat_model.state_dict(),
            'scheduler_S': scheduler_S.state_dict(),
            'scheduler_G': scheduler_G.state_dict(),
            # 'criterion': criterion.state_dict()
        }
        # TODO: Add checkpoint saving stuff, possibly check wandb_utils
        # TODO: Get rid of all validation/dataloader stuff as not allowed


if __name__ == '__main__':
    main()