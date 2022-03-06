import wandb
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

def init_wandb(model, wandb_api_key, wandb_resume, wandb_name, wandb_project, wandb_run_id, wandb_watch):
    os.environ["WANDB_API_KEY"] = wandb_api_key
    if wandb_resume:
        wandb.login()
        wandb.init(project=wandb_project, name=wandb_name, id=wandb_run_id, resume=True)
        print("---------------------------------------------------------------------------------------------------")
        print("Session Resumed")
        print("---------------------------------------------------------------------------------------------------")
    else:
        wandb.init(project=wandb_project, name=wandb_name)
    if wandb_resume and wandb_watch:
        wandb.watch(model, log="all")


def wandb_log(train_loss: float, val_loss: float, train_acc: float, val_acc: float, epoch: int):
    """
    Logs the accuracy and loss to wandb

    Args:
        train_loss (float): Training loss
        val_loss (float): Validation loss
        train_acc (float): Training Accuracy
        val_acc (float): Validation Accuracy
        epoch (int): Epoch Number
    """
    wandb.log({
        'Training loss': train_loss,
        'Validation loss': val_loss,
        'Training Accuracy': train_acc,
        'Validation Accuracy': val_acc
    }, step=epoch)


def wandb_save_summary(test_acc: float, test_f1: float, test_precision: float, test_recall: float):
    """[summary]

    Args:
        test_acc (float): Test Accuracy
        test_f1 (float): Test F1 Score
        test_precision (float): Test Precision
        test_recall (float): Test Recall
    """

    wandb.run.summary["test_accuracy"] = test_acc
    wandb.run.summary["test_f1_score"] = test_f1
    wandb.run.summary["test_precision"] = test_precision
    wandb.run.summary["test_recall"] = test_recall


def wandb_log_conf_matrix(y_true: list, y_pred: list):
    """
    Logs the confusion matrix

    Args:
        y_true (list): ground truth labels
        y_pred (list): predicted labels
    """
    num_classes = len(set(y_true))

    wandb.log({'confusion_matrix': wandb.plots.HeatMap(list(np.arange(0, num_classes)), list(
        np.arange(0, num_classes)), confusion_matrix(y_true, y_pred, normalize="true"), show_text=True)})


def save_model_wandb(save_path):
    """ 
    Saves model to wandb

    Args:
        save_path (str): Path to save the wandb model
    """

    wandb.save(os.path.abspath(save_path))


def save_ckp(state, epoch, checkpoint_path, checkpoint_base, wandb_save):
    """
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    f_path = os.path.join(checkpoint_path, "Epoch_" + str(epoch) + '.pth')
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    if wandb_save:
        wandb.save(f_path, base_path=checkpoint_base, policy='live')
