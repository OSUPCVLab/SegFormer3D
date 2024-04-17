import os
import torch
import nibabel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from monai.data import MetaTensor
from multiprocessing import Process, Pool
from sklearn.preprocessing import MinMaxScaler 
from monai.transforms import (
    Orientation, 
    EnsureType,
    ConvertToMultiChannelBasedOnBratsClasses,
)

# whoever wrote this knew what he was doing (hint: It was me!)

"""
data 
 │
 ├───train
 │      ├──BraTS2021_00000 
 │      │      └──BraTS2021_00000_flair.nii.gz
 │      │      └──BraTS2021_00000_t1.nii.gz
 │      │      └──BraTS2021_00000_t1ce.nii.gz
 │      │      └──BraTS2021_00000_t2.nii.gz
 │      │      └──BraTS2021_00000_seg.nii.gz
 │      ├──BraTS2021_00002
 │      │      └──BraTS2021_00002_flair.nii.gz
 │      ...    └──...
"""


class Brats2021Task1Preprocess:
    def __init__(
        self,
        root_dir: str,
        train_folder_name: str = "train",
        save_dir: str = "../BraTS2021_Training_Data",
    ):
        """
        root_dir: path to the data folder where the raw train folder is
        train_folder_name: name of the folder of the training data
        save_dir: path to directory where each case is going to be saved as a single file containing four modalities
        """
        self.train_folder_dir = os.path.join(root_dir, train_folder_name)
        assert os.path.exists(self.train_folder_dir)
        # walking through the raw training data and list all the folder names, i.e. case name
        self.case_name = next(os.walk(self.train_folder_dir), (None, None, []))[1]
        # MRI type
        self.MRI_TYPE = ["flair", "t1", "t1ce", "t2", "seg"]
        self.save_dir = save_dir
        
    def __len__(self):
        return self.case_name.__len__()

    def get_modality_fp(self, case_name: str, mri_type: str)->str:
        """
        return the modality file path
        case_name: patient ID
        mri_type: any of the ["flair", "t1", "t1ce", "t2", "seg"]
        """
        modality_fp = os.path.join(
            self.train_folder_dir,
            case_name,
            case_name + f"_{mri_type}.nii.gz",
        )
        return modality_fp

    def load_nifti(self, fp)->list:
        """
        load a nifti file
        fp: path to the nifti file with (nii or nii.gz) extension
        """
        nifti_data = nibabel.load(fp)
        # get the floating point array
        nifti_scan = nifti_data.get_fdata()
        # get affine matrix
        affine = nifti_data.affine
        return nifti_scan, affine

    def normalize(self, x:np.ndarray)->np.ndarray:
        # Transform features by scaling each feature to a given range.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # (H, W, D) -> (H * W, D)
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_data = normalized_1D_array.reshape(x.shape)
        return normalized_data

    def orient(self, x: MetaTensor) -> MetaTensor:
        # orient the array to be in (Right, Anterior, Superior) scanner coordinate systems
        assert type(x) == MetaTensor
        return Orientation(axcodes="RAS")(x)

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert type(x) == MetaTensor
        return EnsureType(data_type="numpy", track_meta=False)(x)

    def crop_brats2021_zero_pixels(self, x: np.ndarray)->np.ndarray:
        # get rid of the zero pixels around mri scan and cut it so that the region is useful
        # crop (1, 240, 240, 155) to (1, 128, 128, 128)
        return x[:, 56:184, 56:184, 13:141]

    def preprocess_brats_modality(self, data_fp: str, is_label: bool = False)->np.ndarray:
        """
        apply preprocess stage to the modality
        data_fp: directory to the modality
        """
        data, affine = self.load_nifti(data_fp)
        # label do not the be normalized 
        if is_label:
            # Binary mask does not need to be float64! For saving storage purposes!
            data = data.astype(np.uint8)
            # categorical -> one-hot-encoded 
            # (240, 240, 155) -> (3, 240, 240, 155)
            data = ConvertToMultiChannelBasedOnBratsClasses()(data)
        else:
            data = self.normalize(x=data)
            # (240, 240, 155) -> (1, 240, 240, 155)
            data = data[np.newaxis, ...]
        
        data = MetaTensor(x=data, affine=affine)
        # for oreinting the coordinate system we need the affine matrix
        data = self.orient(data)
        # detaching the meta values from the oriented array
        data = self.detach_meta(data)
        # (240, 240, 155) -> (128, 128, 128)
        data = self.crop_brats2021_zero_pixels(data)
        return data

    def __getitem__(self, idx):
        case_name = self.case_name[idx]
        # e.g: train/BraTS2021_00000/BraTS2021_00000_flair.nii.gz
        
        # preprocess Flair modality
        FLAIR = self.get_modality_fp(case_name, self.MRI_TYPE[0])
        flair = self.preprocess_brats_modality(data_fp=FLAIR, is_label=False)
        flair_transv = flair.swapaxes(1, 3) # transverse plane
        
        # # preprocess T1 modality
        # T1 = self.get_modality_fp(case_name, self.MRI_TYPE[1])
        # t1 = self.preprocess_brats_modality(data_fp=T1, is_label=False)
        # t1_transv = t1.swapaxes(1, 3) # transverse plane
        
        # preprocess T1ce modality
        T1ce = self.get_modality_fp(case_name, self.MRI_TYPE[2])
        t1ce = self.preprocess_brats_modality(data_fp=T1ce, is_label=False)
        t1ce_transv = t1ce.swapaxes(1, 3) # transverse plane
        
        # preprocess T2
        T2 = self.get_modality_fp(case_name, self.MRI_TYPE[3])
        t2 = self.preprocess_brats_modality(data_fp=T2, is_label=False)
        t2_transv = t2.swapaxes(1, 3) # transverse plane
        
        # preprocess segmentation label
        Label = self.get_modality_fp(case_name, self.MRI_TYPE[4])
        label = self.preprocess_brats_modality(data_fp=Label, is_label=True)
        label_transv = label.swapaxes(1, 3) # transverse plane

        # stack modalities along the first dimension 
        modalities = np.concatenate(
            (flair_transv, t1ce_transv, t2_transv),
            axis=0,
        )
        label = label_transv
        return modalities, label, case_name

    def __call__(self):
        print("started preprocessing brats2021...")
        with Pool(processes=os.cpu_count()) as multi_p:
            multi_p.map_async(func=self.process, iterable=range(self.__len__()))
            multi_p.close()
            multi_p.join()
        print("finished preprocessing brats2021...")


    def process(self, idx):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # get the 4D modalities along with the label
        modalities, label, case_name = self.__getitem__(idx)
        # creating the folder for the current case id
        data_save_path = os.path.join(self.save_dir, case_name)
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        # saving the preprocessed 4D modalities containing all the modalities to save path
        modalities_fn = data_save_path + f"/{case_name}_modalities.pt"
        torch.save(modalities, modalities_fn)
        # saving the preprocessed segmentation label to save path
        label_fn = data_save_path + f"/{case_name}_label.pt"
        torch.save(label, label_fn)



def animate(input_1, input_2):
    """animate pairs of image sequences of the same length on two conjugate axis"""
    assert len(input_1) == len(
        input_2
    ), f"two inputs should have the same number of frame but first input had {len(input_1)} and the second one {len(input_2)}"
    # set the figure and axis
    fig, axis = plt.subplots(1, 2, figsize=(8, 8))
    axis[0].set_axis_off()
    axis[1].set_axis_off()
    sequence_length = input_1.__len__()
    sequence = []
    for i in range(sequence_length):
        im_1 = axis[0].imshow(input_1[i], cmap="gray", animated=True)
        im_2 = axis[1].imshow(input_2[i], cmap="gray", animated=True)
        if i == 0:
            axis[0].imshow(input_1[i], cmap="gray")  # show an initial one first
            axis[1].imshow(input_2[i], cmap="gray")  # show an initial one first

        sequence.append([im_1, im_2])
    return animation.ArtistAnimation(
        fig,
        sequence,
        interval=25,
        blit=True,
        repeat_delay=100,
    )

def viz(volume_indx: int = 1, label_indx: int = 1)->None:
    """
    pair visualization of the volume and label
    volume_indx: index for the volume. ["flair", "t1", "t1ce", "t2"]
    label_indx: index for the label segmentation ["TC" (Tumor core), "WT" (Whole tumor), "ET" (Enhancing tumor)]
    """
    assert volume_indx in [0, 1, 2]
    assert label_indx in [0, 1, 2]
    x = volume[volume_indx, ...]
    y = label[label_indx, ...]
    ani = animate(input_1=x, input_2=y)
    plt.show()


if __name__ == "__main__":
    brats2021_task1_prep = Brats2021Task1Preprocess(
            root_dir="./", 
            save_dir="../BraTS2021_Training_Data"
        )
    # start preprocessing 
    brats2021_task1_prep()

    # visualization    
    # volume, label, _ = brats2021_task1_prep[100]
    # viz(volume_indx = 0, label_indx = 2)

