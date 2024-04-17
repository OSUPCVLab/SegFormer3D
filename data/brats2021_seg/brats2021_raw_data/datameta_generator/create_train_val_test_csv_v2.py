import os
import random
import numpy as np
import pandas as pd


def create_pandas_df(data_dict: dict) -> pd.DataFrame:
    """
    create a pandas dataframe out of data dictionary
    data_dict: key values of the data to be data-framed
    """
    data_frame = pd.DataFrame(
        data=data_dict,
        index=None,
        columns=None,
    )
    return data_frame


def save_pandas_df(dataframe: pd.DataFrame, save_path: str, header: list) -> None:
    """
    save a dataframe to the save_dir with specified header
    dataframe: pandas dataframe to be saved
    save_path: the directory in which the dataframe is going to be saved
    header: list of headers of the to be saved csv file
    """
    assert save_path.endswith("csv")
    assert isinstance(dataframe, pd.DataFrame)
    assert (dataframe.columns.__len__() == header.__len__())
    dataframe.to_csv(path_or_buf=save_path, header=header, index=False)


def create_train_val_test_csv_from_data_folder(
    folder_dir: str,
    append_dir: str = "",
    save_dir: str = "./",
    train_split_perc: float = 0.80,
    val_split_perc: float = 0.05,
) -> None:
    """
    create train/validation/test csv file out of the given directory such that each csv file has its split percentage count
    folder_dir: path to the whole corpus of the data
    append_dir: path to be appended to the begining of the directory filed in the csv file
    save_dir: directory to which save the csv files
    train_split_perc: the percentage of the train set by which split the data
    val_split_perc: the percentage of the validation set by which split the data
    """
    assert os.path.exists(folder_dir), f"{folder_dir} does not exist"
    assert (
        train_split_perc < 1.0 and train_split_perc > 0.0
    ), "train split should be between 0 and 1"
    assert (
        val_split_perc < 1.0 and val_split_perc > 0.0
    ), "train split should be between 0 and 1"

    # set the seed
    np.random.seed(0)
    random.seed(0)

    # iterate through the folder to list all the filenames
    case_name = next(os.walk(folder_dir), (None, None, []))[1]

    # appending append_dir to the case name based on the anatamical planes 
    planes = ["sagittal", "coronal", "transverse"]
    data_fp = []
    label_fp = []
    for case in case_name:
        for plane_name in planes:
            case_data = f"{case}_{plane_name}_modalities.pt"
            case_label = f"{case}_{plane_name}_label.pt"
            # BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_sagittal_modalities.pt
            data_fp.append(os.path.join(append_dir, case, case_data).replace("\\", "/"))
            # BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_sagittal_label.pt
            label_fp.append(os.path.join(append_dir, case, case_label).replace("\\", "/"))

    # we have three anatomical plane for each cse 
    cropus_sample_count = case_name.__len__() * 3

    idx = np.arange(0, cropus_sample_count)
    # shuffling idx (inplace operation)
    np.random.shuffle(idx)

    # spliting the data into train/val split percentage respectively for train, val and test (test set is inferred automatically)
    train_idx, val_idx, test_idx = np.split(
        idx,
        [
            int(train_split_perc * cropus_sample_count),
            int((train_split_perc + val_split_perc) * cropus_sample_count),
        ],
    )

    # get the corresponding id from the train,validation and test set
    train_sample_data_fp = np.array(data_fp)[train_idx]
    train_sample_label_fp = np.array(label_fp)[train_idx]

    # we do not need test split so we can merge it with validation 
    val_idx = np.concatenate((val_idx, test_idx), axis=0)
    validation_sample_data_fp = np.array(data_fp)[val_idx]
    validation_sample_label_fp = np.array(label_fp)[val_idx]


    # dictionary object to get converte to dataframe
    train_data = {"data_path": train_sample_data_fp, "label_path": train_sample_label_fp}
    valid_data = {"data_path": validation_sample_data_fp, "label_path": validation_sample_label_fp}

    train_df = create_pandas_df(train_data)
    valid_df = create_pandas_df(valid_data)

    save_pandas_df(
        dataframe=train_df,
        save_path=f"./train_v2.csv",
        header=list(train_data.keys()),
    )
    save_pandas_df(
        dataframe=valid_df,
        save_path=f"./validation_v2.csv",
        header=list(valid_data.keys()),
    )


if __name__ == "__main__":
    create_train_val_test_csv_from_data_folder(
        # path to the raw train data folder
        folder_dir="../train",
        # this is inferred from where the actual experiments are run relative to the data folder
        append_dir="../../../data/brats2021_seg/BraTS2021_Training_Data/",
        # where to save the train, val and test csv file relative to the current directory
        save_dir="../../",
        train_split_perc=0.85,
        val_split_perc=0.10,
    )
