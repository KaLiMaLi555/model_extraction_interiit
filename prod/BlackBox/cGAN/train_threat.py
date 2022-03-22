import argparse
import json
import os
import random
from collections import Counter
from pprint import pprint

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from tqdm import tqdm

from approximate_gradients_swint_img import *
# from my_utils import *
from utils.wandb_utils import init_wandb, save_ckp
from val_utils import metrics
from val_utils.custom_transformations import CustomStudentImageTransform
from val_utils.dataloader_val import ValDataset


# TODO: Replace with cfg parser stuff
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD Swin-T Image')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=20, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--g_iter', type=int, default=10, help="Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help="Number of discriminator iterations per epoch_iter")

    parser.add_argument('--lr_S', type=float, default=0.001, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
    parser.add_argument('--nz', type=int, default=256, help="Size of random noise input to generator")

    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--generator_checkpoint', type=str, default='')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'], )
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float, help="Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")

    # parser.add_argument('--dataset', type=str, default='cifar10', choices=['svhn', 'cifar10'], help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model', type=str, default='resnet34_8x', choices=classifiers, help='Target model name (default: resnet34_8x)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')

    parser.add_argument('--student_load_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default="debug")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="results")

    # Gradient approximation parameters
    parser.add_argument('--approx_grad', type=int, default=1, help='Always set to 1')
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
    parser.add_argument('--grad_epsilon', type=float, default=1e-3)

    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])

    parser.add_argument('--rec_grad_norm', type=int, default=1)

    parser.add_argument('--MAZE', type=int, default=0)

    parser.add_argument('--store_checkpoints', type=int, default=1)

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
    parser.add_argument('--val_scale_inv', type=float, default=255.0)
    parser.add_argument('--val_shift', type=float, default=0)

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

    device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

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

    args.device = device

    # Eigen values and vectors of the covariance matrix
    # _, test_loader = get_dataloader(args)

    args.normalization_coefs = None
    args.G_activation = torch.sigmoid

    # num_classes = 10 if args.dataset in ['cifar10', 'svhn'] else 100
    args.num_classes = 400

    # if args.model == 'resnet34_8x':
    #     teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    #     if args.dataset == 'svhn':
    #         print("Loading SVHN TEACHER")
    #         args.ckpt = 'checkpoint/teacher/svhn-resnet34_8x.pt'
    #     teacher.load_state_dict( torch.load( args.ckpt, map_location=device) )
    # else:
    #     teacher = get_classifier(args.model, pretrained=True, num_classes=args.num_classes)

    config = "./Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    checkpoint = "/content/swin_tiny_patch244_window877_kinetics400_1k.pth"
    cfg = Config.fromfile(config)
    teacher = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(teacher, checkpoint, map_location=device)

    teacher.eval()
    teacher = teacher.to(device)
    # myprint("Teacher restored from %s"%(args.ckpt))
    # print(f"\n\t\tTraining with {args.model} as a Target\n")
    # correct = 0
    # with torch.no_grad():
    #     for i, (data, target) in enumerate(test_loader):
    #         data, target = data.to(device), target.to(device)
    #         output = teacher(data)
    #         pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    # accuracy = 100. * correct / len(test_loader.dataset)
    # print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset),accuracy))

    # student = get_classifier(args.student_model, pretrained=False, num_classes=args.num_classes)
    student = torchvision.models.mobilenet_v2()
    student.classifier[1] = torch.nn.Linear(in_features=student.classifier[1].in_features, out_features=400)

    generator = network.gan.GeneratorC(nz=args.nz, nc=3, img_size=224, num_classes=400, activation=args.G_activation)

    student = student.to(device)
    generator = generator.to(device)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    # if args.student_load_path :
    #     # "checkpoint/student_no-grad/cifar10-resnet34_8x.pt"
    #     student.load_state_dict( torch.load( args.student_load_path ) )
    #     myprint("Student initialized from %s"%(args.student_load_path))
    #     acc = test(args, student=student, generator=generator, device = device, test_loader = test_loader)

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m + 1) + args.d_iter)

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)

    if args.MAZE:
        optimizer_G = optim.SGD(generator.parameters(), lr=args.lr_G, weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    if args.generator_checkpoint:
        checkpoint = torch.load(args.generator_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        for g in optimizer_G.param_groups:
            g['lr'] = args.lr_G

    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)
    print()

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    best_acc = 0
    acc_list = []

    if args.val_scale == 1:
        args.val_scale = 1 / args.val_scale_inv
    # val_data = ValDataset(args.val_data_dir, args.val_classes_file,
    #                       args.val_labels_file, args.num_classes,
    #                       transform=CustomMobilenetTransform(size=args.image_size),
    #                       scale=args.val_scale, shift=args.val_shift)
    val_data = ValDataset(args.val_data_dir, args.val_classes_file,
                          args.val_labels_file, args.num_classes,
                          transform=CustomStudentImageTransform(size=224),
                          scale=args.val_scale, shift=args.val_shift)

    val_loader = DataLoader(val_data, batch_size=args.val_batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=args.val_num_workers)
    if args.wandb:
        init_wandb(student, args.wandb_api_key, args.wandb_resume, args.wandb_name, args.wandb_project, args.wandb_run_id, args.wandb_watch)

    # print("################### Evaluating Student Model ###################\n")
    # student.eval()
    # with torch.no_grad():
    #     acc_1, acc_5 = val(student, val_loader, device)
    #     acc_1 = 100 * acc_1.detach().cpu().numpy()
    #     acc_5 = 100 * acc_5.detach().cpu().numpy()
    # print(f'\nEpoch {epoch}')
    # print(f'Top-1: {str(acc_1)}, Top-5: {str(acc_5)}\n')
    # wandb.log({'val_T1': acc_1, 'epoch': epoch})
    # wandb.log({'val_T5': acc_5, 'epoch': epoch})

    for epoch in range(1, number_epochs + 1):
        # Train
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

        train(args, teacher=teacher, student=student, generator=generator, device=device, optimizer=[optimizer_S, optimizer_G], epoch=epoch)

        # Validating student on K400
        if epoch % args.val_epoch == 0:
            print("################### Evaluating Student Model ###################\n")
            student.eval()
            with torch.no_grad():
                acc_1, acc_5 = val(student, val_loader, device)
                acc_1 = 100 * acc_1.detach().cpu().numpy()
                acc_5 = 100 * acc_5.detach().cpu().numpy()
            print(f'\nEpoch {epoch}')
            print(f'Top-1: {str(acc_1)}, Top-5: {str(acc_5)}\n')
            wandb.log({'val_T1': acc_1, 'epoch': epoch})
            wandb.log({'val_T5': acc_5, 'epoch': epoch})

        # Test
        # acc = test(args, student=student, generator=generator, device = device, test_loader = test_loader, epoch=epoch)
        # acc_list.append(acc)
        # if acc>best_acc:
        #     best_acc = acc
        #     name = 'mobilenetV2'
        #     torch.save(student.state_dict(),f"checkpoint/student_{args.model_id}/{args.dataset}-{name}.pt")
        #     torch.save(generator.state_dict(),f"checkpoint/student_{args.model_id}/{args.dataset}-{name}-generator.pt")
        # # vp.add_scalar('Acc', epoch, acc)
        # if args.store_checkpoints:
        #     torch.save(student.state_dict(), args.log_dir + f"/checkpoint/student.pt")
        #     torch.save(generator.state_dict(), args.log_dir + f"/checkpoint/generator.pt")
    # myprint("Best Acc=%.6f"%best_acc)

    # with open(args.log_dir + "/Max_accuracy = %f"%best_acc, "w") as f:
    #     f.write(" ")


if __name__ == '__main__':
    main()
