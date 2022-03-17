import argparse
import json
import os
import random
from pprint import pprint

import torch
import torch.optim as optim
import wandb
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from approximate_gradients_swint_img_pretrain import *
# from my_utils import *
from utils.wandb_utils import init_wandb, save_ckp

print("torch version", torch.__version__)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD Swin-T Image')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=20, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--g_iter', type=int, default=10, help="Number of generator iterations per epoch_iter")

    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
    parser.add_argument('--nz', type=int, default=256, help="Size of random noise input to generator")

    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')

    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'], )
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float, help="Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")

    # parser.add_argument('--dataset', type=str, default='cifar10', choices=['svhn', 'cifar10'], help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')

    parser.add_argument('--model_id', type=str, default="debug")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="results")

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
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--wandb_project', type=str, default="model_extraction")
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--wandb_resume', action="store_true")
    parser.add_argument('--wandb_watch', action="store_true")
    parser.add_argument('--checkpoint_base', type=str, default="/content")
    parser.add_argument('--checkpoint_path', type=str, default="/drive/MyDrive/DFAD_video_ckpts")
    parser.add_argument('--wandb_save', action="store_true")

    # parser.add_argument('--student_model', type=str, default='resnet18_8x',
    #                     help='Student model architecture (default: resnet18_8x)')

    return parser.parse_args()


# def generator_loss(args, s_logit, t_logit,  z = None, z_logit = None, reduction="mean"):
#     assert 0
#     loss = - F.l1_loss( s_logit, t_logit , reduction=reduction)
#     return loss


def pretrain(args, teacher, generator, device, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""
    teacher.eval()

    debug_distribution = True
    distribution = []
    total_loss = 0

    for i in tqdm(range(args.epoch_itrs), position=0, leave=True):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(args.g_iter):
            # Sample Random Noise
            labels = torch.argmax(torch.randn((args.batch_size, args.num_classes)), dim=1).to(device)
            labels_oh = torch.nn.functional.one_hot(labels, args.num_classes)
            z = torch.randn((args.batch_size, args.nz)).to(device)
            optimizer.zero_grad()
            generator.train()
            # Get fake image from generator
            fake = generator(z, label=labels_oh, pre_x=True)  # pre_x returns the output of G before applying the activation
            fake = fake.unsqueeze(dim=2)

            ## APPOX GRADIENT
            approx_grad_wrt_x, loss = estimate_gradient_objective(
                args, teacher, fake, labels=labels, epsilon=args.grad_epsilon,
                m=args.grad_m, num_classes=args.num_classes,
                device=device, pre_x=True)

            # grad_wrt_x, loss_conf = compute_gradient(args, teacher, fake, labels=labels, device=device, pre_x=True)

            print()
            print(approx_grad_wrt_x.shape)
            # print(grad_wrt_x.shape)
            print(loss)
            # print(loss_conf)
            # print(loss - loss_conf)
            # print(np.sum(loss - loss_conf))
            # print(approx_grad_wrt_x - grad_wrt_x)
            # print(np.sum(approx_grad_wrt_x - grad_wrt_x))

            fake.backward(approx_grad_wrt_x)
            optimizer.step()

            total_loss += loss.item()
            wandb.log({'loss_G_verbose': loss.item()})

        wandb.log({'loss_G_inner': total_loss / (i + 1)})
        print(f'Total loss G:', total_loss / (i + 1))

        # if debug_distribution:
        # TODO: Also print confidence, possibly for T-5 predicted classes
        #  might be useful to understand the exact nature of mode collapse
        # distribution.append(torch.argmax(t_logit.detach(), dim=1).cpu().numpy())

        # Log Results
        if i % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss.item():.6f}')

        # update query budget
        args.query_budget -= args.cost_per_iteration

        if args.query_budget < args.cost_per_iteration:
            break

        checkpoint = {
            'outer_epoch': epoch,
            'inner_epoch': i + 1,
            'optimizer_G': optimizer.state_dict(),
            'generator': generator.state_dict(),
            # 'scheduler':scheduler.state_dict(),
            # 'criterion': criterion.state_dict()
        }
        if args.wandb_save:
            save_ckp(checkpoint, epoch, args.checkpoint_path, args.checkpoint_base, args.wandb_save)

    # if debug_distribution:
    #     c_t = Counter(list(np.array(distribution).flatten())).most_common()
    #     c_s = Counter(list(np.array(dist_s).flatten())).most_common()
    #     wandb.run.summary[f'Teacher distribution epoch {epoch}'] = c_t
    #     wandb.run.summary[f'Student distribution epoch {epoch}'] = c_s
    #     print(f'Teacher distribution epoch {epoch}:', c_t)
    #     print(f'Student distribution epoch {epoch}:', c_s)


def compute_grad_norms(generator):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            G_grad.append(p.grad.norm().to("cpu"))
    return np.mean(G_grad)


def main():
    args = parse_args()

    args.query_budget *= 10 ** 6
    args.query_budget = int(args.query_budget)

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
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    args.device = device

    args.normalization_coefs = None
    args.G_activation = torch.sigmoid

    args.num_classes = 400

    pprint(args, width=80)
    config = "./Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    checkpoint = "../swin_tiny_patch244_window877_kinetics400_1k.pth"
    cfg = Config.fromfile(config)
    teacher = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(teacher, checkpoint, map_location=device)

    teacher.eval()
    teacher = teacher.to(device)

    generator = network.gan.GeneratorC(nz=args.nz, nc=3, img_size=224, num_classes=400, activation=args.G_activation)

    generator = generator.to(device)

    args.generator = generator
    args.teacher = teacher

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m + 1))

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)
    print()

    if args.scheduler == "multistep":
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    if args.wandb:
        init_wandb(generator, args.wandb_api_key, args.wandb_resume, args.wandb_name, args.wandb_project, args.wandb_run_id, args.wandb_watch)

    for epoch in range(1, number_epochs + 1):
        # Train
        if args.scheduler != "none":
            scheduler_G.step()

        pretrain(args, teacher=teacher, generator=generator, device=device, optimizer=optimizer_G, epoch=epoch)

        # Validating student on K400
        # if epoch % args.val_epoch == 0:
        #     print("################### Evaluating Student Model ###################\n")


if __name__ == '__main__':
    main()
