import numpy as np
import torchvision.transforms.functional as TF
import torch
import random
import imgaug.augmenters as iaa


# Ref: https://pytorch.org/vision/stable/transforms.html#functional-transforms

def get_preview(images, augmentationList, probability):
    """
    Accepts a list of images and augmentationList as input.
    Provides a list of augmented images in that order as ouptut.
    """
    augmentations = []
    for augmentation in augmentationList:
        aug_id = augmentation['id']
        params = augmentation['params']
        if aug_id == 1:
            aug = iaa.SaltAndPepper(p=params[0], per_channel=params[1])
        elif aug_id == 2:
            aug = iaa.imgcorruptlike.GaussianNoise(severity=(params[0], params[1]))
        elif aug_id == 3:
            aug = iaa.Rain(speed=(params[0], params[1]), drop_size=(params[2], params[3]))
        elif aug_id == 4:
            aug = iaa.imgcorruptlike.Fog(severity=(params[0], params[1]))
        elif aug_id == 5:
            aug = iaa.imgcorruptlike.Snow(severity=(params[0], params[1]))
        elif aug_id == 6:
            aug = iaa.imgcorruptlike.Spatter(severity=(params[0], params[1]))
        elif aug_id == 7:
            aug = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1))
        elif aug_id == 8:
            aug = iaa.Rotate(rotate=(params[0], params[1]))
        elif aug_id == 9:
            aug = iaa.Affine()
        elif aug_id == 10:
            aug = iaa.MotionBlur(k=params[0], angle=(params[1], params[2]))
        elif aug_id == 11:
            aug = iaa.imgcorruptlike.ZoomBlur(severity=(params[0], params[1]))
        elif aug_id == 12:
            aug = iaa.AddToBrightness(add=(params[0], params[1]))
        elif aug_id == 13:
            aug = iaa.ChangeColorTemperature(kelvin=(params[0], params[1]))
        elif aug_id == 14:
            aug = iaa.SigmoidContrast(gain=(params[0], params[1]), cutoff=(params[2], params[3]), per_channel=params[4])
        elif aug_id == 15:
            aug = iaa.Cutout(nb_iterations=(params[0], params[1]), size=params[2], squared=params[3])
        else:
            print("Not implemented")
        augmentations.append(aug)

    images_augmented = iaa.Sometimes(p=probability, then_list=augmentations)(images=images)
    return images_augmented


def custom_rotate_transform(vid, angle=None):
    x = vid
    video_transform = []
    for i, image in enumerate(x):
        if random.random() > 0.5:
            if angle is None:
                angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            video_transform.append(image)
    return torch.stack(video_transform)


def custom_resize_transform(vid, size=224):
    x = vid
    video_transform = []
    for i, image in enumerate(x):
        image = TF.resize(image, (size, size))
        video_transform.append(image)
    return torch.stack(video_transform)


class CustomTransform:

    def __init__(self):
        pass

    def __call__(self, vid):
        vid = custom_resize_transform(vid)  # Resize
        return vid


rotation_transform = CustomTransform()


class CustomResizeTransform:

    def __init__(self, size=224):
        self.size = size

    def custom_resize_transform(self, vid):
        x = vid
        video_transform = []
        for i, image in enumerate(list(x)):
            image = TF.resize(image, self.size + 32)
            image = TF.center_crop(image, (self.size, self.size))
            video_transform.append(image)
        return torch.stack(video_transform)

    def __call__(self, vid):
        vid = self.custom_resize_transform(vid)  # Resize
        return vid
