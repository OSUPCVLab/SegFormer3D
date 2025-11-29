from typing import Dict, Union
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


##################################################################################################
def warmup_lr_scheduler(config: Dict, optimizer: optim.Optimizer) -> LRScheduler:
    """Create linear warmup learning rate scheduler.
    
    Linearly ramps up the learning rate from 0 to the initial learning rate
    over the specified number of warmup epochs. This helps stabilize training
    in the early stages.
    
    Args:
        config: Configuration dictionary containing warmup_scheduler parameters
        optimizer: PyTorch optimizer to schedule
        
    Returns:
        Configured LambdaLR scheduler for warmup
    """
    warmup_epochs = config["warmup_scheduler"]["warmup_epochs"]
    lambda1 = lambda epoch: (epoch + 1) * 1.0 / warmup_epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, verbose=False)
    return scheduler


##################################################################################################
def training_lr_scheduler(config: Dict, optimizer: optim.Optimizer) -> LRScheduler:
    """Create training learning rate scheduler.
    
    Supports multiple scheduler types:
    - reducelronplateau: Reduces LR when metric plateaus
    - cosine_annealing_wr: Cosine annealing with warm restarts
    - poly_lr: Polynomial learning rate decay
    
    Args:
        config: Configuration dictionary containing train_scheduler parameters
        optimizer: PyTorch optimizer to schedule
        
    Returns:
        Configured learning rate scheduler
        
    Raises:
        NotImplementedError: If scheduler_type is not supported
    """
    scheduler_type = config["train_scheduler"]["scheduler_type"]
    scheduler_args = config["train_scheduler"]["scheduler_args"]
    
    if scheduler_type == "reducelronplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            mode=config["train_scheduler"]["mode"],
            patience=config["train_scheduler"]["patience"],
            verbose=False,
            min_lr=scheduler_args["min_lr"],
        )
        return scheduler
    
    elif scheduler_type == "cosine_annealing_wr":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_args["t_0_epochs"],
            T_mult=scheduler_args["t_mult"],
            eta_min=scheduler_args["min_lr"],
            last_epoch=-1,
            verbose=False,
        )
        return scheduler
    
    elif scheduler_type == "poly_lr":
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer=optimizer,
            total_iters=5,
            power=scheduler_args["power"],
            last_epoch=-1,
        )
        return scheduler
    
    else:
        raise NotImplementedError(
            f"Scheduler type '{scheduler_type}' is not implemented. "
            "Supported types: ['reducelronplateau', 'cosine_annealing_wr', 'poly_lr']"
        )


##################################################################################################
def build_scheduler(
    optimizer: optim.Optimizer, 
    scheduler_type: str, 
    config: Dict
) -> LRScheduler:
    """Factory function to build learning rate schedulers.
    
    Creates either a warmup scheduler (for initial training phase) or a
    training scheduler (for main training phase with decay).
    
    Args:
        optimizer: PyTorch optimizer to schedule
        scheduler_type: Type of scheduler ('warmup_scheduler' or 'training_scheduler')
        config: Configuration dictionary containing scheduler parameters
        
    Returns:
        Configured learning rate scheduler
        
    Raises:
        ValueError: If scheduler_type is not supported
    """
    scheduler_registry = {
        "warmup_scheduler": warmup_lr_scheduler,
        "training_scheduler": training_lr_scheduler,
    }
    
    if scheduler_type not in scheduler_registry:
        raise ValueError(
            f"Invalid scheduler type: {scheduler_type}. "
            f"Supported types: {list(scheduler_registry.keys())}"
        )
    
    return scheduler_registry[scheduler_type](config=config, optimizer=optimizer)
