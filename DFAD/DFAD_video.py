# Import required libraries
from __future__ import print_function
import os
import torch
import random
import network
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

# Import for Swin-T
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint

def train(args, teacher, student, generator, device, optimizer, epoch):

    teacher.eval()
    student.train()
    generator.train()
    
    optimizer_S, optimizer_G = optimizer

    for i in tqdm(range( args.epoch_itrs), position = 0, leave = True):

        print(f"Outer Epoch: {epoch}, Inner Epoch: {i+1}")

        total_loss_S = 0
        total_loss_G = 0
        
        for k in tqdm(range(5), position = 0, leave = True):
        
            z = torch.randn( (args.batch_size, args.nz) ).to(device)
            optimizer_S.zero_grad()
            
            fake = generator(z).detach()
            fake_shape = fake.shape
            
            t_logit = torch.tensor(teacher(fake)).to(device)
            
            fake = fake.view(fake_shape[0], fake_shape[2], fake_shape[1], fake_shape[3], fake_shape[4])
            s_logit = student(fake).to(device)

            loss_S = F.l1_loss(s_logit, t_logit)
            total_loss_S += loss_S.item()
            
            loss_S.backward()
            optimizer_S.step()

        print("Loss on Student model:", total_loss_S / (5 * (i+1)))

        z = torch.randn( (args.batch_size, args.nz) ).to(device)
        optimizer_G.zero_grad()
        generator.train()
        fake = generator(z)
        fake_shape = fake.shape

        t_logit = torch.tensor(teacher(fake)).to(device)

        fake = fake.view(fake_shape[0], fake_shape[2], fake_shape[1], fake_shape[3], fake_shape[4]) 
        s_logit = student(fake).to(device)
    
        loss_G = - F.l1_loss( s_logit, t_logit ) 
        total_loss_G += loss_G.item()

        print("Loss on Generator model: ", total_loss_G / (i+1))

        loss_G.backward()
        optimizer_G.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100*float(i)/float(args.epoch_itrs), loss_G.item(), loss_S.item()))

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='DFAD MNIST')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr_S', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--model_name', default = "swin-t", type = str)
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--image_size', default = 32, type = int)

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
    
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    if args.model_name == "swin-t":
        print()
        config = "./VST/configs/_base_/models/swin/swin_tiny.py"
        checkpoint = "./swin_tiny_patch244_window877_kinetics400_1k.pth"
        cfg = Config.fromfile(config)
        teacher = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(teacher, checkpoint, map_location=device)

    student = network.models.ResCNNRNN()
    print("\nLoaded student and teacher")
    generator = network.models.VideoGAN(zdim = args.nz)
    print("Loaded student, generator and teacher\n")

    teacher = teacher.to(device)
    student = student.to(device)
    generator = generator.to(device)

    optimizer_S = optim.SGD( student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9 )
    optimizer_G = optim.Adam( generator.parameters(), lr=args.lr_G )
    
    if args.scheduler:
        scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, args.step_size, 0.1)
        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, args.step_size, 0.1)

    for epoch in range(1, args.epochs + 1):
        # Train
        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()

        print("################### Training Student and Generator Models ###################\n")
        train(args, teacher=teacher, student=student, generator=generator, device=device, optimizer=[optimizer_S, optimizer_G], epoch=epoch)
        
        # Test
        # TODO: Validate after we get the sample validation set

if __name__ == '__main__':
    main()