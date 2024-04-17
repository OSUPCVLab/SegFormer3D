import sys

sys.path.append("../")

from typing import Dict
from monai.data import DataLoader
from augmentations.augmentations import build_augmentations


######################################################################
def build_dataset(dataset_type: str, dataset_args: Dict):
    if dataset_type == "brats2021_seg":
        from .brats2021_seg import Brats2021Task1Dataset

        dataset = Brats2021Task1Dataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            transform=build_augmentations(dataset_args["train"]),
            fold_id=dataset_args["fold_id"],
        )
        return dataset
    elif dataset_type == "brats2017_seg":
        from .brats2017_seg import Brats2017Task1Dataset

        dataset = Brats2017Task1Dataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            transform=build_augmentations(dataset_args["train"]),
            fold_id=dataset_args["fold_id"],
        )
        return dataset
    else:
        raise ValueError(
            "only brats2021 and brats2017 segmentation is currently supported!"
        )


######################################################################
def build_dataloader(
    dataset, dataloader_args: Dict, config: Dict = None, train: bool = True
) -> DataLoader:
    """builds the dataloader for given dataset

    Args:
        dataset (_type_): _description_
        dataloader_args (Dict): _description_
        config (Dict, optional): _description_. Defaults to None.
        train (bool, optional): _description_. Defaults to True.

    Returns:
        DataLoader: _description_
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader_args["batch_size"],
        shuffle=dataloader_args["shuffle"],
        num_workers=dataloader_args["num_workers"],
        drop_last=dataloader_args["drop_last"],
        pin_memory=True,
    )
    return dataloader
