import wandb
import os
import numpy as np
from sklearn.metrics import confusion_matrix


def init_wandb(model, args=None, pytorch=True) -> None:
    """
    Initialize project on Weights & Biases

    Args:
        model (Model): Model for Training
        args (dict,optional): dict with wandb config. Defaults to None.
        wandb_api_key : add your api key
        wandb_name : add a unique descriptive name for the run
        project : name of wandb project
        Sample args : args = {'wandb_api_key': '','wandb_name' : 'test', 'project' : 'test_project'}
        pytorch : whether model is in pytorch or tensorflow
    """
    wandb.login(key=args['wandb_api_key'])
    wandb.init(
        name=args['wandb_name'],
        project=args['project'],
        resume=True,
        dir="./"
    )
    # if args:
    #     wandb.config.update(args)
    if pytorch:
      	wandb.watch(model, log="all")