from typing import Dict
import torch.optim as optim
import torch.nn as nn


######################################################################
def optim_adam(model: nn.Module, optimizer_args: Dict) -> optim.Adam:
    """Create Adam optimizer.
    
    Args:
        model: PyTorch model
        optimizer_args: Dictionary containing lr and optional weight_decay
        
    Returns:
        Configured Adam optimizer
    """
    return optim.Adam(
        model.parameters(),
        lr=optimizer_args["lr"],
        weight_decay=optimizer_args.get("weight_decay", 0.0),
    )


######################################################################
def optim_sgd(model: nn.Module, optimizer_args: Dict) -> optim.SGD:
    """Create SGD optimizer with momentum.
    
    Args:
        model: PyTorch model
        optimizer_args: Dictionary containing lr, optional weight_decay and momentum
        
    Returns:
        Configured SGD optimizer
    """
    return optim.SGD(
        model.parameters(),
        lr=optimizer_args["lr"],
        weight_decay=optimizer_args.get("weight_decay", 0.0),
        momentum=optimizer_args.get("momentum", 0.9),
    )


######################################################################
def optim_adamw(model: nn.Module, optimizer_args: Dict) -> optim.AdamW:
    """Create AdamW optimizer (Adam with decoupled weight decay).
    
    Args:
        model: PyTorch model
        optimizer_args: Dictionary containing lr and weight_decay
        
    Returns:
        Configured AdamW optimizer
    """
    return optim.AdamW(
        model.parameters(),
        lr=optimizer_args["lr"],
        weight_decay=optimizer_args["weight_decay"],
        # amsgrad=True,  # Can be enabled for more robust convergence
    )


######################################################################
def build_optimizer(
    model: nn.Module, 
    optimizer_type: str, 
    optimizer_args: Dict
) -> optim.Optimizer:
    """Factory function to build optimizers.
    
    Args:
        model: PyTorch model to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        optimizer_args: Dictionary containing optimizer hyperparameters
        
    Returns:
        Instantiated optimizer
        
    Raises:
        ValueError: If optimizer_type is not supported
    """
    optimizer_registry = {
        "adam": optim_adam,
        "adamw": optim_adamw,
        "sgd": optim_sgd,
    }
    
    if optimizer_type not in optimizer_registry:
        raise ValueError(
            f"Unsupported optimizer type: {optimizer_type}. "
            f"Supported types: {list(optimizer_registry.keys())}"
        )
    
    return optimizer_registry[optimizer_type](model, optimizer_args)
