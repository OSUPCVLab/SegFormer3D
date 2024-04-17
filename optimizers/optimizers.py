from typing import Dict
import torch.optim as optim


######################################################################
def optim_adam(model, optimizer_args):
    adam = optim.Adam(
        model.parameters(),
        lr=optimizer_args["lr"],
        weight_decay=optimizer_args.get("weight_decay"),
    )
    return adam


######################################################################
def optim_sgd(model, optimizer_args):
    adam = optim.SGD(
        model.parameters(),
        lr=optimizer_args["lr"],
        weight_decay=optimizer_args.get("weight_decay"),
        momentum=optimizer_args.get("momentum"),
    )
    return adam


######################################################################
def optim_adamw(model, optimizer_args):
    adam = optim.AdamW(
        model.parameters(),
        lr=optimizer_args["lr"],
        weight_decay=optimizer_args["weight_decay"],
        # amsgrad=True,
    )
    return adam


######################################################################
def build_optimizer(model, optimizer_type: str, optimizer_args: Dict):
    if optimizer_type == "adam":
        return optim_adam(model, optimizer_args)
    elif optimizer_type == "adamw":
        return optim_adamw(model, optimizer_args)
    elif optimizer_type == "sgd":
        return optim_sgd(model, optimizer_args)
    else:
        raise ValueError("must be adam or adamw for now")
