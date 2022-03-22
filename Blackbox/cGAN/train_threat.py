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
from approximate_gradients_swint_img import *
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from config import cfg_parser
from torchmetrics import accuracy
from tqdm.notebook import tqdm

from model_extraction_interiit.Blackbox.approximate_gradients import approximate_gradients
from model_extraction_interiit.Blackbox.utils_common import swin_transform
from models import ConditionalGenerator
# from utils_common import 


# TODO: Get MARS working once added
# from model_extraction_interiit.prod.BlackBox import MARS

def config():
    # TODO: Replace with cfg parser stuff
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/params.yaml")
    # parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 256)')
    # parser.add_argument('--query_budget', type=float, default=20, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
    # parser.add_argument('--epoch_itrs', type=int, default=50)
    # parser.add_argument('--g_iter', type=int, default=10, help="Number of generator iterations per epoch_iter")
    # parser.add_argument('--d_iter', type=int, default=5, help="Number of discriminator iterations per epoch_iter")

    # parser.add_argument('--lr_S', type=float, default=0.001, metavar='LR', help='Student learning rate (default: 0.1)')
    # parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
    # parser.add_argument('--nz', type=int, default=256, help="Size of random noise input to generator")

    # parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    # parser.add_argument('--generator_checkpoint', type=str, default='')
    # parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'], )
    # parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
    # parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float, help="Percentage epochs at which to take next step")
    # parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")

    # # parser.add_argument('--dataset', type=str, default='cifar10', choices=['svhn', 'cifar10'], help='dataset name (default: cifar10)')
    # parser.add_argument('--data_root', type=str, default='data')
    # parser.add_argument('--model', type=str, default='resnet34_8x', choices=classifiers, help='Target model name (default: resnet34_8x)')
    # parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=42, metavar='S',
    #                     help='random seed (default: 1)')
    # # parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')

    # parser.add_argument('--student_load_path', type=str, default=None)
    # parser.add_argument('--model_id', type=str, default="debug")

    # parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--log_dir', type=str, default="results")

    # # Gradient approximation parameters
    # parser.add_argument('--approx_grad', type=int, default=1, help='Always set to 1')
    # parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
    # parser.add_argument('--grad_epsilon', type=float, default=1e-3)

    # parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')

    # # Eigenvalues computation parameters
    # parser.add_argument('--no_logits', type=int, default=1)
    # parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])

    # parser.add_argument('--rec_grad_norm', type=int, default=1)

    # parser.add_argument('--MAZE', type=int, default=0)

    # parser.add_argument('--store_checkpoints', type=int, default=1)

    # parser.add_argument('--val_data_dir', type=str,
    #                     default='/content/val_data/k400_16_frames_uniform')
    # parser.add_argument('--val_classes_file', type=str,
    #                     default='/content/val_data/k400_16_frames_uniform/classes.csv')
    # parser.add_argument('--val_labels_file', type=str,
    #                     default='/content/val_data/k400_16_frames_uniform/labels.csv')
    # parser.add_argument('--val_num_workers', type=int, default=2)

    # parser.add_argument('--val_epoch', type=int, default=5)
    # parser.add_argument('--val_batch_size', type=int, default=8)
    # parser.add_argument('--val_scale', type=float, default=1)
    # parser.add_argument('--val_scale_inv', type=float, default=255.0)
    # parser.add_argument('--val_shift', type=float, default=0)

    # parser.add_argument('--wandb_api_key', type=str)
    # parser.add_argument('--wandb', action="store_true")
    # parser.add_argument('--wandb_project', type=str, default="model_extraction")
    # parser.add_argument('--wandb_name', type=str)
    # parser.add_argument('--wandb_run_id', type=str, default=None)
    # parser.add_argument('--wandb_resume', action="store_true")
    # parser.add_argument('--wandb_watch', action="store_true")
    # parser.add_argument('--checkpoint_base', type=str, default="/content")
    # parser.add_argument('--checkpoint_path', type=str, default="/drive/MyDrive/DFAD_video_ckpts")
    # parser.add_argument('--wandb_save', action="store_true")

    # parser.add_argument('--student_model', type=str, default='resnet18_8x',
    #                     help='Student model architecture (default: resnet18_8x)')

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
    # TODO: Replace with cfg parser stuff
    args = config()["experiment"]

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
    # TODO: Add asserts that cfg options are from this list
    # if args.student_model not in classifiers:
    #     if "wrn" not in args.student_model:
    #         raise ValueError("Unknown model")

    # REVIEW: Decide if we're keeping or throwing this logging dir stuff
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

    # with open(args.log_dir + "/accuracy.csv", "w") as f:
    #     f.write("epoch,accuracy\n")

    if args.rec_grad_norm:
        with open(args.log_dir + "/norm_grad.csv", "w") as f:
            f.write("epoch,G_grad_norm,S_grad_norm,grad_wrt_X\n")

    with open("latest_experiments.txt", "a") as f:
        f.write(args.log_dir + "\n")

    # REVIEW: Decide if we're keeping or throwing logging dir stuff above

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.device if use_cuda else 'cpu')
    device_tf = '/device:GPU:%d' % args.device if use_cuda else '/device:CPU'
    args.device, args.device_tf = device, device_tf

    # REVIEW: Decide if we're keeping or throwing this stuff
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

    # TODO: cfg parser stuff
    args.normalization_coefs = None
    args.G_activation = torch.sigmoid
    args.num_classes = 400

    # TODO: Convert this to cfg parser
    if args.victim_model == 'swin-t':
        # TODO: cfg parser for VST file paths
        config = "./Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
        checkpoint = "/content/swin_tiny_patch244_window877_kinetics400_1k.pth"
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
