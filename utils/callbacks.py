'''
    Generic Callbacks and Logger
'''
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

def init_logger(project,name,id="",resume=True, watch=True,model=None,log_freq=50):
    wandb_logger=WandbLogger(
        project=project,
        name=name,
        id=id,
        log_model='all',
        resume=None
    )
    if resume:
        wandb_logger=WandbLogger(
            project=project,
            name=name,
            id=id,
            log_model='all',
            resume='allow'
        )
    if watch:
        wandb_logger.watch(
            model,
            log='all',
            log_freq=log_freq,
            log_graph=True
        )
    return wandb_logger
    

def init_lr_monitor_epoch():
    return LearningRateMonitor(
        logging_interval='epoch'
    )
def init_lr_monitor_step():
    return LearningRateMonitor(
        logging_interval='step'
    )

def init_model_checkpoint(checkpoint_path=None):
    '''
    pass path for directory
    '''
    return ModelCheckpoint(
        dirpath=checkpoint_path
        monitor='val_loss',
        save_top_k=2,
        mode='min',
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_last=True,
        filename='{epoch}-{val_loss:.2f}-{val_accuracy:.2f}'
        auto_insert_metric_name=True,
    )

def load_from_wandb_artifact(project,resume,checkpoint_path,ModelClass,num_classes):
    run = wandb.init(project=project, resume=resume)
    artifact = run.use_artifact(checkpoint_path, type='model')
    artifact_dir = artifact.download()
    model = ModelClass.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", num_classes=num_classes)
    return model

def get_trainer(resume,epochs,logger,num_gpus,lr_monitor,checkpoint_callback,checkpoint_path=None)
    if resume:
        return pl.Trainer(
            max_epochs=epochs,
            progress_bar_refresh_rate=1,
            log_every_n_steps=1,
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor],
            resume_from_checkpoint=checkpoint_path,
            gpus=num_gpus,
        )
    return pl.Trainer(
        max_epochs=epochs,
        progress_bar_refresh_rate=1,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gpus=num_gpus,
    )

