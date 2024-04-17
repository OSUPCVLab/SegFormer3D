import os
import random
import numpy as np
import pandas as pd


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
    cropus_sample_count = case_name.__len__()

    # appending append_dir to the case name
    data_dir = []
    for case in case_name:
        data_dir.append(os.path.join(append_dir, case))

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
    train_sample_base_dir = np.array(data_dir)[train_idx]
    train_sample_case_name = np.array(case_name)[train_idx]

    # we do not need test split so we can merge it with validation 
    val_idx = np.concatenate((val_idx, test_idx), axis=0)
    validation_sample_base_dir = np.array(data_dir)[val_idx]
    validation_sample_case_name = np.array(case_name)[val_idx]

    # create a pandas data frame
    train_df = pd.DataFrame(
        data={"base_dir": train_sample_base_dir, "case_name": train_sample_case_name},
        index=None,
        columns=None,
    )

    validation_df = pd.DataFrame(
        data={
            "base_dir": validation_sample_base_dir,
            "case_name": validation_sample_case_name,
        },
        index=None,
        columns=None,
    )

    # write csv files to the drive!
    train_df.to_csv(
        save_dir + "/train.csv",
        header=["data_path", "case_name"],
        index=False,
    )
    validation_df.to_csv(
        save_dir + "/validation.csv",
        header=["data_path", "case_name"],
        index=False,
    )



if __name__ == "__main__":
    create_train_val_test_csv_from_data_folder(
        # path to the train data folder
        folder_dir="../../BraTS2017_Training_Data",
        # this is inferred from where the actual experiments are run relative to the data folder
        append_dir="../../data/brats2017_seg/BraTS2017_Training_Data/",
        # where to save the train, val and test csv file relative to the current directory
        save_dir=".",
        train_split_perc=0.85,
        val_split_perc=0.10,
    )
