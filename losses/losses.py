import torch
import monai
import torch.nn as nn
from typing import Dict
from monai import losses

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss


###########################################################################
class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions, tragets):
        loss = self._loss(predictions, tragets)
        return loss
###########################################################################
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceCELoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


###########################################################################
def build_loss_fn(loss_type: str, loss_args: Dict = None):
    if loss_type == "crossentropy":
        return CrossEntropyLoss()

    elif loss_type == "binarycrossentropy":
        return BinaryCrossEntropyWithLogits()

    elif loss_type == "dice":
        return DiceLoss()

    elif loss_type == "diceCE":
        return DiceCELoss()
        
    else:
        raise ValueError("must be cross entropy or soft dice loss for now!")
