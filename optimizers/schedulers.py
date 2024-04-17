from typing import Dict
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


##################################################################################################
def warmup_lr_scheduler(config, optimizer):
    """
    Linearly ramps up the learning rate within X
    number of epochs to the working epoch.
    Args:
        optimizer (_type_): _description_
        warmup_epochs (_type_): _description_
        warmup_lr (_type_): warmup lr should be the starting lr we want.
    """
    lambda1 = lambda epoch: (
        (epoch + 1) * 1.0 / config["warmup_scheduler"]["warmup_epochs"]
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, verbose=False)
    return scheduler


##################################################################################################
def training_lr_scheduler(config, optimizer):
    """
    Wraps a normal scheuler
    """
    scheduler_type = config["train_scheduler"]["scheduler_type"]
    if scheduler_type == "reducelronplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            mode=config["train_scheduler"]["mode"],
            patience=config["train_scheduler"]["patience"],
            verbose=False,
            min_lr=config["train_scheduler"]["scheduler_args"]["min_lr"],
        )
        return scheduler
    elif scheduler_type == "cosine_annealing_wr":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["train_scheduler"]["scheduler_args"]["t_0_epochs"],
            T_mult=config["train_scheduler"]["scheduler_args"]["t_mult"],
            eta_min=config["train_scheduler"]["scheduler_args"]["min_lr"],
            last_epoch=-1,
            verbose=False,
        )
        return scheduler
    elif scheduler_type == "poly_lr":
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer=optimizer,
            total_iters=5,
            power=config["train_scheduler"]["scheduler_args"]["power"],
            last_epoch=-1,
        )
        return scheduler
    else:
        raise NotImplementedError("Specified Scheduler Is Not Implemented")


##################################################################################################
def build_scheduler(
    optimizer: optim.Optimizer, scheduler_type: str, config
) -> LRScheduler:
    """generates the learning rate scheduler

    Args:
        optimizer (optim.Optimizer): pytorch optimizer
        scheduler_type (str): type of scheduler

    Returns:
        LRScheduler: _description_
    """
    if scheduler_type == "warmup_scheduler":
        scheduler = warmup_lr_scheduler(config=config, optimizer=optimizer)
        return scheduler
    elif scheduler_type == "training_scheduler":
        scheduler = training_lr_scheduler(config=config, optimizer=optimizer)
        return scheduler
    else:
        raise ValueError("Invalid Input -- Check scheduler_type")
