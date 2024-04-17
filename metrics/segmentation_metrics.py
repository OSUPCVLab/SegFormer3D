import torch
import torch.nn as nn
from typing import Dict
from monai.metrics import DiceMetric
from monai.transforms import Compose
from monai.data import decollate_batch
from monai.transforms import Activations
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference


################################################################################
class SlidingWindowInference:
    def __init__(self, roi: tuple, sw_batch_size: int):
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean_batch", get_not_nans=False
        )
        self.post_transform = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(argmax=False, threshold=0.5),
            ]
        )
        self.sw_batch_size = sw_batch_size
        self.roi = roi

    def __call__(
        self, val_inputs: torch.Tensor, val_labels: torch.Tensor, model: nn.Module
    ):
        self.dice_metric.reset()
        logits = sliding_window_inference(
            inputs=val_inputs,
            roi_size=self.roi,
            sw_batch_size=self.sw_batch_size,
            predictor=model,
            overlap=0.5,
        )
        val_labels_list = decollate_batch(val_labels)
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [
            self.post_transform(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]
        self.dice_metric(y_pred=val_output_convert, y=val_labels_list)
        # compute accuracy per channel
        acc = self.dice_metric.aggregate().cpu().numpy()
        avg_acc = acc.mean()
        # To access individual metric 
        # TC acc: acc[0]
        # WT acc: acc[1]
        # ET acc: acc[2]
        return avg_acc * 100


def build_metric_fn(metric_type: str, metric_arg: Dict = None):
    if metric_type == "sliding_window_inference":
        return SlidingWindowInference(
            roi=metric_arg["roi"],
            sw_batch_size=metric_arg["sw_batch_size"],
        )
    else:
        raise ValueError("must be cross sliding_window_inference!")
