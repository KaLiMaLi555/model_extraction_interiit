import torch
import torchvision.transforms.functional as TF


class SwinTransform:

    def __init__(self, n_clips=2, clip_len=16):
        self.n_clips = n_clips
        self.clip_len = clip_len
        self.total_frames = self.n_clips * self.clip_len

    def transform(self, vid):
        x = list(vid)
        n = self.total_frames // len(x)
        x = x * n + x[:(self.total_frames % len(x))]
        assert len(x) == self.total_frames, "clip length mismatch"
        video_transform = []
        for im in x:
            im = im.permute(2, 0, 1)
            im = TF.resize(im, 224)
            im = TF.center_crop(im, (224, 224))
            im = im.float()
            im = TF.normalize(im, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
            im = im.permute(1, 2, 0)
            video_transform.append(im)
        vid = torch.stack(video_transform)
        vid = vid.reshape((-1, self.n_clips, self.clip_len, 224, 224, 3))
        vid = vid.permute(0, 1, 5, 2, 3, 4)
        vid = vid.reshape((-1, 3, self.clip_len, 224, 224))
        return vid

    def __call__(self, vid):
        vid = self.transform(vid)  # Resize
        return vid


def transform(vid):
    x = list(vid)
    video_transform = []
    for im in x:
        im = im.permute(2, 0, 1)
        im = TF.resize(im, 224)
        im = TF.center_crop(im, (224, 224))
        im = im.float()
        im = im / 255.0
        im = im.permute(1, 2, 0)
        video_transform.append(im)
    vid = torch.stack(video_transform)
    return vid


class MovinetTransform:

    def __init__(self):
        pass

    def __call__(self, vid):
        vid = transform(vid)  # Resize
        return vid
