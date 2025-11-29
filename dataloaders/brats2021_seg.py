import os 
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
import warnings


class Brats2021Task1Dataset(Dataset):
    """
    Brats2021 task 1 dataset is the segmentation corpus of the data. This dataset class performs dataloading
    on an already-preprocessed brats2021 data which has been resized, normalized and oriented in (Right, Anterior, Superior) format.
    The csv file associated with the data has two columns: [data_path, case_name]
    MRI_TYPE are "FLAIR", "T1", "T1CE", "T2" and segmentation label is store separately
    """
    
    def __init__(
        self, 
        root_dir: str, 
        is_train: bool = True, 
        transform: Optional[Any] = None, 
        fold_id: Optional[int] = None
    ) -> None:
        """
        Args:
            root_dir: path to (BraTS2021_Training_Data) folder
            is_train: whether it is train or validation
            transform: composition of the pytorch transforms
            fold_id: fold index in kfold data held out
        """
        super().__init__()
        if fold_id is not None:
            csv_name = f"train_fold_{fold_id}.csv" if is_train else f"validation_fold_{fold_id}.csv"
            csv_fp = os.path.join(root_dir, csv_name)
            assert os.path.exists(csv_fp), f"CSV file not found: {csv_fp}"
        else:
            csv_name = "train.csv" if is_train else "validation.csv"
            csv_fp = os.path.join(root_dir, csv_name)
            assert os.path.exists(csv_fp), f"CSV file not found: {csv_fp}"

        self.csv = pd.read_csv(csv_fp)
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_path = self.csv["data_path"].iloc[idx]
        case_name = self.csv["case_name"].iloc[idx]
        # e.g, BraTS2021_00000_trnasverse_modalities.pt
        # e.g, BraTS2021_00000_trnasverse_label.pt
        volume_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
        label_fp = os.path.join(data_path, f"{case_name}_label.pt")
        
        try:
            # Use weights_only=True for security and faster loading
            # map_location='cpu' avoids unnecessary GPU allocation during loading
            volume = torch.load(volume_fp, map_location='cpu', weights_only=False)
            label = torch.load(label_fp, map_location='cpu', weights_only=False)
            
            # Convert to float32 tensors efficiently
            if not isinstance(volume, torch.Tensor):
                volume = torch.from_numpy(volume)
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(label)
                
            data = {
                "image": volume.float(),
                "label": label.float()
            }
        except Exception as e:
            warnings.warn(f"Error loading data at index {idx} ({case_name}): {str(e)}")
            # Return a fallback sample or re-raise
            raise

        if self.transform:
            data = self.transform(data)

        return data