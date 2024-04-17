import os 
import torch
import pandas as pd
from torch.utils.data import Dataset 

class Brats2021Task1Dataset(Dataset):
	"""
	Brats2021 task 1 dataset is the segmentation corpus of the data. This dataset class performs dataloading
	on an already-preprocessed brats2021 data which has been resized, normalized and oriented in (Right, Anterior, Superior) format.  
	The csv file associated with the data has two columns: [data_path, case_name]
    MRI_TYPE are "FLAIR", "T1", "T1CE", "T2" and segmentation label is store separately
	"""
	def __init__(self, root_dir: str, is_train: bool = True, transform = None,  fold_id: int = None):
		"""
		root_dir: path to (BraTS2021_Training_Data) folder
		is_train: whether or nor it is train or validation  
		transform: composition of the pytorch transforms 
		fold_id: fold index in kfold dataheld out 
		"""
		super().__init__()
		if fold_id is not None:
			csv_name = f"train_fold_{fold_id}.csv" if is_train else f"validation_fold_{fold_id}.csv"
			csv_fp = os.path.join(root_dir, csv_name)
			assert os.path.exists(csv_fp)
		else:
			csv_name = "train.csv" if is_train else "validation.csv"
			csv_fp = os.path.join(root_dir, csv_name)
			assert os.path.exists(csv_fp)

		self.csv = pd.read_csv(csv_fp)
		self.transform = transform

	def __len__(self):
		return self.csv.__len__()

	def __getitem__(self, idx):
		data_path = self.csv["data_path"][idx] 
		case_name = self.csv["case_name"][idx] 
		# e.g, BraTS2021_00000_trnasverse_modalities.pt
		# e.g, BraTS2021_00000_trnasverse_label.pt
		volume_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
		label_fp = os.path.join(data_path, f"{case_name}_label.pt")
		# load the preprocessed tensors
		volume = torch.load(volume_fp)
		label = torch.load(label_fp)
		data = {"image": torch.from_numpy(volume).float(), "label": torch.from_numpy(label).float()}
		
		if self.transform:
			data = self.transform(data)

		return data