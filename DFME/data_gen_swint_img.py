import argparse
import random
from collections import Counter
from pprint import pprint
from PIL import Image
import numpy as np
import torch
import wandb
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from approximate_gradients_swint_img_pretrain import *
from utils.wandb_utils import init_wandb

print("torch version", torch.__version__)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD Swin-T Image')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 256)')
    parser.add_argument('--nz', type=int, default=256, help="Size of random noise input to generator")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')

    parser.add_argument('--model_id', type=str, default="debug")
    parser.add_argument('--generator_checkpoint', type=str, default='')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="results")

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

    parser.add_argument('--vid_dir_path', type=str, default="/content")
    parser.add_argument('--logit_dir_path', type=str, default="/content")

    return parser.parse_args()


def gen_examples(args, generator, teacher, device, epoch):
    # TODO: Make sure running with no_grad!!
    counter = 0
    teacher.eval()
    generator.eval()
    distribution = []
    logs = []
    labs = []
    for i in tqdm(range(50), position=0, leave=True):
        labels = torch.argmax(torch.randn((args.batch_size, args.num_classes)), dim=1).to(device)
        labels_oh = torch.nn.functional.one_hot(labels, args.num_classes)
        z = torch.randn((args.batch_size, args.nz)).to(device)

        fake = generator(z, label=labels_oh, pre_x=True).to(device)
        fake = fake.unsqueeze(dim=2)
        fake = args.G_activation(fake)
        vid = fake.squeeze(dim=2).cpu().numpy()
        print(f'Vid_shape = {vid.shape}')
        x_swin = network.swin.swin_transform(fake)
        logits = torch.Tensor(teacher(x_swin, return_loss=False)).to(device)
        logits_argmax = torch.argmax(logits.detach(), dim=1)
        distribution.append(logits_argmax.cpu().numpy())

        for img in vid:
            img = img.transpose(1, 2, 0)
            print(img.shape)
            PIL_image = Image.fromarray(np.uint8(img*255)).convert('RGB')
            print(PIL_image.size)
            break
            os.mkdir(os.path.join(dir_path, str(counter)))
            PIL_image.save(os.path.join(args.vid_dir_path, str(counter), str(counter) + ".png"))
            counter += 1

        for i in logits:
            logs.extend(i.cpu())
        
        for j in logits_argmax:
            labs.extend(j.cpu().numpy())
        logs = torch.stack(logs)
        pickle.dump(logs, open(os.path.join(args.logit_dir_path, "SwinT" + "_logits_" + "Cgan_gen_1" + ".pkl"), "wb"))
        pickle.dump(labs, open(os.path.join(args.logit_dir_path, "SwinT" + "_logits_argmax_" + "Cgan_gen_1" + ".pkl"), "wb"))



        print('Expected')
        print(labels)
        print('Predictions')
        print(logits_argmax)
        print('Confidences')
        print(logits.max(dim=1)[0])
        

        # TODO: mult image by 255 and save as png
        # TODO: save logits file

    c = Counter(list(np.array(distribution).flatten())).most_common()
    wandb.run.summary[f'Distribution epoch {epoch}'] = c
    print('\n\n---------Distribution---------')
    print(c)
    print('---------Distribution---------\n\n')


def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")

    args.device = device
    args.num_classes = 400
    args.G_activation = torch.sigmoid

    pprint(args, width=80)
    config = "./Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    checkpoint = "../swin_tiny_patch244_window877_kinetics400_1k.pth"
    cfg = Config.fromfile(config)
    teacher = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(teacher, checkpoint, map_location=device)

    generator = network.gan.GeneratorC(nz=args.nz, nc=3, img_size=224, num_classes=400, activation=args.G_activation)

    teacher.eval()
    generator.eval()
    generator = generator.to(device)
    teacher = teacher.to(device)

    args.generator = generator
    args.teacher = teacher

    if args.generator_checkpoint:
        checkpoint = torch.load(args.generator_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator'])

    if args.wandb:
        init_wandb(generator, args.wandb_api_key, args.wandb_resume, args.wandb_name, args.wandb_project, args.wandb_run_id, args.wandb_watch)

    with torch.no_grad():
        for epoch in range(1, 25 + 1):
            gen_examples(args, generator, teacher, device, epoch)


if __name__ == '__main__':
    main()
