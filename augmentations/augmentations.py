import monai.transforms as transforms

#######################################################################################
def build_augmentations(train: bool = True):
    if train:
        train_transform = [
            transforms.RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(96, 96, 96), num_samples=4, random_center=True, random_size=False),
            transforms.RandFlipd(keys=["image", "label"], prob=0.30, spatial_axis=1),
            transforms.RandRotated(keys=["image", "label"], prob=0.50, range_x=0.36, range_y=0.0, range_z=0.0),
            transforms.RandCoarseDropoutd(keys=["image", "label"], holes=20, spatial_size=(-1, 7, 7), fill_value=0, prob=0.5),
            transforms.GibbsNoised(keys=["image"]),
            transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
        return transforms.Compose(train_transform)
    else:
        val_transform = [
            transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
        return transforms.Compose(val_transform)
