import numpy as np
import cv2
import torch

def get_random_vid(idx):
    colors = [
        (
            int(np.random.random() * 255),
            int(np.random.random() * 255),
            int(np.random.random() * 255),
        )
        for _ in range(9)
    ]

    pos = [
        [
            [int(np.random.random() * 224), int(np.random.random() * 224)],
            [int(np.random.random() * 224), int(np.random.random() * 224)],
        ]
        for _ in range(9)
    ]
    bgc = (
        int(np.random.random() * 255),
        int(np.random.random() * 255),
        int(np.random.random() * 255),
    )
    imgs = []
    for _ in range(16):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.rectangle(img, (0, 0), (224, 224), bgc, -1)
        for i in range(9):
            img = cv2.rectangle(
                img,
                tuple(pos[i][0]),
                (pos[i][0][0] + pos[i][1][0], pos[i][0][1] + pos[i][1][0]),
                colors[i],
                -1,
            )
            shiftX = int(np.random.random() * int(np.random.random() * 224)) - int(
                np.random.random() * 224
            )
            shiftY = int(np.random.random() * int(np.random.random() * 224)) - int(
                np.random.random() * 224
            )
            pos[i][0][0] = (
                pos[i][0][0]
                + shiftX
                + int(np.random.random() * int(np.random.random() * 224))
                - int(np.random.random() * 224)
            )
            pos[i][1][0] = (
                pos[i][1][0]
                + int(np.random.random() * int(np.random.random() * 224))
                - int(np.random.random() * 224)
            )
            pos[i][0][1] = (
                pos[i][0][1]
                + shiftY
                + int(np.random.random() * int(np.random.random() * 224))
                - int(np.random.random() * 224)
            )
            pos[i][1][1] = (
                pos[i][1][1]
                + int(np.random.random() * int(np.random.random() * 224))
                - int(np.random.random() * 224)
            )
        imgs.append(img)
    vid = torch.tensor(np.array(imgs), dtype=torch.float)
    vid = vid.permute(3, 0, 1, 2)
    if idx % 2:
        vid /= 255
    return vid


def get_random_img(idx):
    colors = [
        (
            int(np.random.random() * 255),
            int(np.random.random() * 255),
            int(np.random.random() * 255),
        )
        for _ in range(9)
    ]

    pos = [
        [
            [int(np.random.random() * 224), int(np.random.random() * 224)],
            [int(np.random.random() * 224), int(np.random.random() * 224)],
        ]
        for _ in range(9)
    ]
    bgc = (
        int(np.random.random() * 255),
        int(np.random.random() * 255),
        int(np.random.random() * 255),
    )
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img = cv2.rectangle(img, (0, 0), (224, 224), bgc, -1)
    for i in range(9):
        img = cv2.rectangle(
            img,
            tuple(pos[i][0]),
            (pos[i][0][0] + pos[i][1][0], pos[i][0][1] + pos[i][1][0]),
            colors[i],
            -1,
        )
        shiftX = int(np.random.random() * int(np.random.random() * 224)) - int(
            np.random.random() * 224
        )
        shiftY = int(np.random.random() * int(np.random.random() * 224)) - int(
            np.random.random() * 224
        )
        pos[i][0][0] = (
            pos[i][0][0]
            + shiftX
            + int(np.random.random() * int(np.random.random() * 224))
            - int(np.random.random() * 224)
        )
        pos[i][1][0] = (
            pos[i][1][0]
            + int(np.random.random() * int(np.random.random() * 224))
            - int(np.random.random() * 224)
        )
        pos[i][0][1] = (
            pos[i][0][1]
            + shiftY
            + int(np.random.random() * int(np.random.random() * 224))
            - int(np.random.random() * 224)
        )
        pos[i][1][1] = (
            pos[i][1][1]
            + int(np.random.random() * int(np.random.random() * 224))
            - int(np.random.random() * 224)
        )
    img = torch.tensor(np.array(img), dtype=torch.float)
    img = img.permute(2, 0, 1)
    img /= 255
    return img

