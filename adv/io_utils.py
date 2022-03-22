import pickle
import torch
import os
import numpy as np

def get_noise_dataset_image(idx, num_instances, videos, labels, batch_size=16):
    frames_list = []
    labels_list = []
    for i in range(int(batch_size)):
        id = (idx + i) % num_instances
        frame = torch.tensor(pickle.load(open(videos[id], "rb"))).to(torch.float32)
        frame1 = frame[0]
        frame1_ = frame1[:, 0, :, :]
        frame1_ /= 255
        label = torch.tensor(labels[id])
        frames_list.append(frame1_)
        labels_list.append(label)

    frames_list = torch.stack(frames_list)
    labels_list = torch.stack(labels_list)
    return frames_list, labels_list


def save_gen_image(
    image_batch, label_batch, dir_path="/content/drive/MyDrive/Adv_gen_from_noise/"
):
    counter_gen_images = len(os.listdir(dir_path)) - 1
    for i in range(image_batch.shape[0]):
        image = image_batch[i]
        filename = os.path.join(
            dir_path, str(label_batch[i].item()) + "_" + str(counter_gen_images)
        )
        np.save(filename, image)
        counter_gen_images += 1
