import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


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


def create_train_val_kfold_csv_from_data_folder(
    folder_dir: str,
    append_dir: str = "",
    save_dir: str = "./",
    n_k_fold: int = 5,
    random_state: int = 42,
) -> None:
    """
    create k fold train validation csv files
    folder_dir: path to the whole corpus of the data
    append_dir: path to be appended to the begining of the directory filed in the csv file
    save_dir: directory to which save the csv files
    n_k_fold: number of folds
    random_state: random seed ID
    """
    assert os.path.exists(folder_dir), f"{folder_dir} does not exist"

    header = ["data_path", "case_name"]

    # iterate through the folder to list all the filenames
    case_name = next(os.walk(folder_dir), (None, None, []))[1]
    case_name = np.array(case_name)
    np.random.seed(random_state)
    np.random.shuffle(case_name)

    # setting up k-fold module
    kfold = KFold(n_splits=n_k_fold, random_state=random_state, shuffle=True)
    # generating k-fold train and validation set
    for i, (train_fold_id, validation_fold_id) in enumerate(kfold.split(case_name)):
        # getting the corresponding case out of the fold index
        train_fold_cn = case_name[train_fold_id]
        valid_fold_cn = case_name[validation_fold_id]
        # create data path pointing to the case name
        train_dp = [
            os.path.join(append_dir, case).replace("\\", "/") for case in train_fold_cn
        ]
        valid_dp = [
            os.path.join(append_dir, case).replace("\\", "/") for case in valid_fold_cn
        ]
        # dictionary object to get converte to dataframe
        train_data = {"data_path": train_dp, "case_name": train_fold_cn}
        valid_data = {"data_path": valid_dp, "case_name": valid_fold_cn}

        train_df = create_pandas_df(train_data)
        valid_df = create_pandas_df(valid_data)

        save_pandas_df(
            dataframe=train_df,
            save_path=f"./train_fold_{i+1}.csv",
            header=header,
        )
        save_pandas_df(
            dataframe=valid_df,
            save_path=f"./validation_fold_{i+1}.csv",
            header=header,
        )


if __name__ == "__main__":
    create_train_val_kfold_csv_from_data_folder(
        # path to the raw train data folder
        folder_dir="../train",
        # this is inferred from where the actual experiments are run relative to the data folder
        append_dir="../../../data/brats2021_seg/BraTS2021_Training_Data/",
        # where to save the train, val and test csv file relative to the current directory
        save_dir="../../",
    )
