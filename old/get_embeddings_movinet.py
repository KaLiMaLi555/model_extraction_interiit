import argparse
import os
import pickle
import warnings

import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

from dataloader import Dataloader
from env import set_seed

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--logit_dir', default="./data/logits", type=str)
    parser.add_argument('--is_random', default=True, type=bool)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_type', default="noise", type=str, choices=["noise", "actual"])
    parser.add_argument('--video_dir_path', default="./data/data", type=str)

    args = parser.parse_args()

    return args


def load_model():
    hub_url = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"

    encoder = hub.KerasLayer(hub_url, trainable=False)
    inputs = tf.keras.layers.Input(shape=[None, None, None, 3], dtype=tf.float32, name='image')

    # [batch_size, 600]
    outputs = encoder(dict(image=inputs))

    model = tf.keras.Model(inputs, outputs, name='movinet')
    return model


def get_logits(model, dataloader, device):
    logits, labels = [], []

    with tf.device(device):
        for idx, (video_frames, label) in tqdm(enumerate(dataloader), position=0, leave=True):
            outputs = tf.stop_gradient(model(video_frames))

            del video_frames
            logits.extend(outputs)
            labels.extend(label)

    logits = tf.stack(logits)
    labels = tf.stack(labels)

    return logits, labels


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.logit_dir):
        os.makedirs(args.logit_dir)

    print("\n######################## Loading Data ########################\n")
    dataloader = Dataloader(args.video_dir_path, num_classes=args.num_classes)
    tensor_dataset = tf.data.Dataset.from_tensor_slices((dataloader.instances, dataloader.labels))
    batched_dataset = tensor_dataset.batch(args.batch_size)

    print("\n######################## Loading Model ########################\n")
    model = load_model()

    device = tf.test.gpu_device_name()
    if device != '/device:GPU:0':
        print('GPU not found, running on CPU.')
        device = '/device:CPU:0'

    print("\n######################## Getting Logits ########################\n")
    logits, labels = get_logits(model, batched_dataset, device)

    print("\n######################## Saving Logits ########################\n")
    combined = zip(logits, labels)
    pickle.dump(combined, open(os.path.join(args.logit_dir, args.model_name + "_" + args.dataset_type + ".pkl"), "wb"))


if __name__ == "__main__":
    main()
