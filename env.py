import torch
import numpy as np
import random as rn
import pytorch_lightning as pl
import tensorflow as tf

SEED_ = 42
def seed_all(SEED=SEED_):
    rn.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(SEED)
    tf.random.set_seed(SEED)

seed_all(SEED_)
