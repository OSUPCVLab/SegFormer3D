import torch
import torch.nn as nn
from typing import Dict, Tuple, List
import numpy as np
from monai.metrics import DiceMetric
from monai.transforms import Compose
from monai.data import decollate_batch
from monai.transforms import Activations
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference


################################################################################
class SlidingWindowInference:
    """Efficient sliding window inference for volumetric segmentation.
    
    Uses MONAI's optimized sliding window implementation with memory-efficient
    batch processing and overlap handling.
    """
    
    def __init__(self, roi: Tuple[int, int, int], sw_batch_size: int) -> None:
        """Initialize sliding window inference.
        
        Args:
            roi: Region of interest size (D, H, W)
            sw_batch_size: Batch size for sliding window patches
        """
        self.dice_metric = DiceMetric(
            include_background=True, 
            reduction="mean_batch", 
            get_not_nans=False
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
        self, 
        val_inputs: torch.Tensor, 
        val_labels: torch.Tensor, 
        model: nn.Module
    ) -> float:
        """Compute Dice metric using sliding window inference.
        
        Args:
            val_inputs: Input volume (B, C, D, H, W)
            val_labels: Ground truth labels (B, C, D, H, W)
            model: Segmentation model
            
        Returns:
            Average Dice score across all classes (percentage)
        """
        self.dice_metric.reset()
        
        # Perform sliding window inference
        with torch.inference_mode():  # More efficient than no_grad for inference
            logits = sliding_window_inference(
                inputs=val_inputs,
                roi_size=self.roi,
                sw_batch_size=self.sw_batch_size,
                predictor=model,
                overlap=0.5,
            )
        
        # Decollate and post-process predictions
        val_labels_list = decollate_batch(val_labels)
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [
            self.post_transform(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]
        
        # Compute Dice metric
        self.dice_metric(y_pred=val_output_convert, y=val_labels_list)
        
        # Aggregate results - compute accuracy per channel
        acc = self.dice_metric.aggregate().cpu().numpy()
        avg_acc = float(acc.mean())  # Explicit conversion for clarity
        
        # To access individual metric:
        # TC acc: acc[0]
        # WT acc: acc[1]
        # ET acc: acc[2]
        return avg_acc * 100.0


def build_metric_fn(metric_type: str, metric_arg: Dict) -> SlidingWindowInference:
    """Factory function to build metric computation modules.
    
    Args:
        metric_type: Type of metric ('sliding_window_inference')
        metric_arg: Dictionary containing metric configuration
        
    Returns:
        Instantiated metric module
        
    Raises:
        ValueError: If metric_type is not supported
    """
    if metric_type == "sliding_window_inference":
        return SlidingWindowInference(
            roi=metric_arg["roi"],
            sw_batch_size=metric_arg["sw_batch_size"],
        )
    else:
        raise ValueError(
            f"Unsupported metric type: {metric_type}. "
            "Supported types: ['sliding_window_inference']"
        )
