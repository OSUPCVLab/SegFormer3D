import monai.transforms as transforms
from typing import Any

#######################################################################################
def build_augmentations(train: bool = True) -> transforms.Compose:
    """Build data augmentation pipeline for 3D medical image segmentation.
    
    Training augmentations include:
    - Random spatial cropping with 4 samples per volume
    - Random horizontal flipping (30% probability)
    - Random rotation around x-axis (50% probability, ±20.6°)
    - Coarse dropout for regularization (50% probability)
    - Gibbs noise to simulate MRI artifacts
    
    Validation uses minimal transforms (type conversion only).
    
    Args:
        train: If True, returns training augmentations. If False, returns validation transforms.
        
    Returns:
        Composed MONAI transform pipeline
    """
    if train:
        train_transform = [
            # Random spatial cropping - generates 4 crops per volume for data efficiency
            transforms.RandSpatialCropSamplesd(
                keys=["image", "label"], 
                roi_size=(96, 96, 96), 
                num_samples=4, 
                random_center=True, 
                random_size=False
            ),
            # Random horizontal flip for geometric augmentation
            transforms.RandFlipd(
                keys=["image", "label"], 
                prob=0.30, 
                spatial_axis=1
            ),
            # Random rotation around x-axis (sagittal plane)
            transforms.RandRotated(
                keys=["image", "label"], 
                prob=0.50, 
                range_x=0.36,  # ±20.6 degrees
                range_y=0.0, 
                range_z=0.0
            ),
            # Coarse dropout for robustness
            transforms.RandCoarseDropoutd(
                keys=["image", "label"], 
                holes=20, 
                spatial_size=(-1, 7, 7), 
                fill_value=0, 
                prob=0.5
            ),
            # Gibbs ringing artifact simulation (MRI-specific)
            transforms.GibbsNoised(keys=["image"]),
            # Ensure proper tensor types without metadata tracking (faster)
            transforms.EnsureTyped(
                keys=["image", "label"], 
                track_meta=False
            ),
        ]
        return transforms.Compose(train_transform)
    else:
        # Minimal validation transforms - only ensure type consistency
        val_transform = [
            transforms.EnsureTyped(
                keys=["image", "label"], 
                track_meta=False
            ),
        ]
        return transforms.Compose(val_transform)
