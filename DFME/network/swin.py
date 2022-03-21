import torch
import torchvision.transforms.functional as TF


def swin_transform(fake):  # N, C, L, S, S
    fake_shape = fake.shape
    N, C, L, S, _ = fake.shape
    fake_teacher = fake.reshape((-1, fake_shape[2], fake_shape[1], *fake_shape[3:]))  # N, L, C, S, S
    fake_teacher_batch = []
    for vid in list(fake_teacher):
        vid = torch.stack(
            [
                TF.normalize(frame * 255, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]).permute(1, 2, 0)
                for frame in vid
            ]
        )
        # vid = vid.reshape((-1, 1, 16, 224, 224, 3)) # 1, 1, L, S, S, C
        vid = vid.reshape((-1, 1, L, S, S, C))  # 1, 1, L, S, S, C
        vid = vid.permute(0, 1, 5, 2, 3, 4)  # 1, 1, C, L, S, S
        vid = vid.reshape((-1, C, L, S, S))  # 1, C, L, S, S
        fake_teacher_batch.append(vid)
    fake_teacher = torch.stack(fake_teacher_batch)  # N, 1, C, L, S, S
    return fake_teacher
