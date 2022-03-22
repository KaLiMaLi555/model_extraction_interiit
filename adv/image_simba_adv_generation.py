# -*- coding: utf-8 -*-
# !rm -rf /content/model_extraction_interiit
# !git clone -b swint https://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@github.com/KaLiMaLi555/model_extraction_interiit

# !ls model_extraction_interiit

# !pip install -q -r /content/model_extraction_interiit/conv3d/requirements.txt
# !pip install -q -r /content/model_extraction_interiit/Video-Swin-Transformer/requirements.txt
# !pip install -q cleverhans

# # Commented out IPython magic to ensure Python compatibility.
# # %cd model_extraction_interiit/Video-Swin-Transformer/
# !pip install -v -e . --user

# !wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth

# !pip install -q mmcv==1.3.1

import numpy as np
import torch
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import pandas as pd
from simba_utils import swin_transform
from simba import SimBA
from io_utils import get_noise_dataset_image, save_gen_image


def model_wrapper(inp):
    with torch.no_grad():
        return torch.tensor(teacher(inp, return_loss=False))


def model_wrapper_image(inp):  # b,3,224,224 assumes input with 0...1
    b_t = []
    for inp_i in inp:
        lim = [inp_i * 255 for _ in range(16)]
        b_t.append(torch.stack(lim, dim=2))  # b,3,16,224,224

    with torch.no_grad():
        return torch.tensor(
            teacher(swin_transform(torch.stack(b_t, dim=0).cuda()), return_loss=False)
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = "/content/model_extraction_interiit/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    checkpoint = "/content/model_extraction_interiit/Video-Swin-Transformer/swin_tiny_patch244_window877_kinetics400_1k.pth"
    cfg = Config.fromfile(config)
    teacher = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(teacher, checkpoint, map_location=device)
    teacher.to(device)
    teacher.eval()
    attacker = SimBA(model_wrapper_image, "k400", 224)
    video_dir_path = "/content/drive/MyDrive/Noise_Gen_Expt_Rohit_k400/"
    csv = pd.read_csv("/content/drive/MyDrive/Noise_Expt_Rohit_k400.csv")
    videos = csv["FileNames"]
    videos = [video_dir_path + x + ".pkl" for x in videos]
    labels = csv["Labels"]
    num_instances = len(csv)
    while True:
        batched_imgs, batched_labels = get_noise_dataset_image(
            np.random.randint(0, 540), num_instances, videos, labels
        )
        batched_targets = batched_labels[torch.randperm(batched_labels.shape[0])]
        expanded, probs, succs, queries, l2_norms, linf_norms = attacker.simba_batch(
            batched_imgs,
            batched_labels,
            3000,
            224,
            7,
            0.2,
            log_every=100,
            targeted=True,
        )
        final_labels = attacker.get_preds(expanded)
        print(final_labels)
        save_gen_image(expanded, final_labels)

    print(expanded.shape)
    print(probs.shape)
    print(succs.shape)
    print(queries.shape)
    print(l2_norms.shape)
    print(linf_norms.shape)
    print(probs[:, -1])
    print(succs[:, -1])
    print(queries[:, -1])
    print(l2_norms[:, -1])

    print(final_labels.shape)
    print(final_labels)
    print(batched_labels)

    # img_test = get_random_img(0)
    # img_test = img_test.unsqueeze(0)
    # test_p_y, test_y = model_wrapper_image(img_test).max(1)
    # print(test_p_y, test_y)
    # img_test = img_test.squeeze(0)
    # img_untargeted = attacker.simba_single(img_test, test_y, num_iters=3000)
    # img_untargeted = img_untargeted.unsqueeze(0)
    # out_test_p_y, out_test_y = model_wrapper_image(img_untargeted).max(1)
    # print(out_test_p_y, out_test_y)

    # img_untargeted = img_untargeted.squeeze(0)
    # img_untargeted = attacker.simba_single(img_untargeted, out_test_y, num_iters=3000)
    # img_untargeted = img_untargeted.unsqueeze(0)
    # out_test_p_y, out_test_y = model_wrapper_image(img_untargeted).max(1)
    # print(out_test_p_y, out_test_y)

    # img_untargeted = img_untargeted.squeeze(0)
    # img_untargeted = attacker.simba_single(
    #     img_untargeted, torch.tensor(5), num_iters=7000, targeted=True
    # )
    # img_untargeted = img_untargeted.unsqueeze(0)
    # out_test_p_y, out_test_y = model_wrapper_image(img_untargeted).max(1)
    # print(out_test_p_y, out_test_y)

    # adv_dict = {}

    # def run_attack(vid, vid_y, base):
    #     print(f"generating {vid_y} from {base}")
    #     vid_adv = attacker.simba_single(
    #         vid.cpu(), vid_y.cpu(), epsilon=0.2, num_iters=10000, targeted=True
    #     )
    #     vid_adv = vid_adv.unsqueeze(0)
    #     p_adv, y_adv = model_wrapper_image(vid_adv).max(1)
    #     y = int(y_adv.cpu().numpy())
    #     y_adv = y_adv.cuda()
    #     if y in adv_dict:
    #         adv_dict[y].append(vid_adv.cpu().detach().numpy())
    #     else:
    #         adv_dict[y] = [vid_adv.cpu().detach().numpy()]
    #     vid_adv = vid_adv.cuda()
    #     print(f"generated {y}")
    #     return vid_adv, y_adv, p_adv

    # def generate_from_given_video(vid_base, vid_base_y):
    #     y = int(vid_base_y.cpu().numpy())
    #     vid_base_y = vid_base_y.cuda()
    #     print(f"started for {vid_base.shape} and {y}")
    #     vid_base = vid_base.squeeze(0)
    #     if y in adv_dict:
    #         adv_dict[y].append(vid_base.cpu().detach().numpy())
    #     else:
    #         adv_dict[y] = [vid_base.cpu().detach().numpy()]
    #     vid_base = vid_base.cuda()
    #     for i in range(10):
    #         run_attack(vid_base, torch.tensor(int(np.random.rand() * 399)), vid_base_y)
    #     vid_base.cpu().detach()
    #     vid_base_y.cpu().detach()

    # for img_base, img_base_y in zip(batched_imgs, img_y):
    #     generate_from_given_video(img_base, img_base_y)

    # # from concurrent.futures import ProcessPoolExecutor
    # # with ProcessPoolExecutor() as executor:
    # #     executor.map(generate_from_given_video,zip(batched_vids,vid_y))

    # for x in adv_dict:
    #     print(x, len(adv_dict[x]))
