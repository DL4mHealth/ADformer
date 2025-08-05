import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
)
from utils.tools import get_channel_index
import warnings
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.signal import resample

from data_provider.dataset_loader.cognision_rseeg_loader import COGrsEEGLoader
from data_provider.dataset_loader.adftd_loader import ADFTDLoader
from data_provider.dataset_loader.cnbpm_loader import CNBPMLoader
from data_provider.dataset_loader.brainlat_loader import BrainLatLoader
from data_provider.dataset_loader.p_adic_loader import PADICLoader
from data_provider.dataset_loader.caueeg_loader import CAUEEGLoader

# data folder dict to loader mapping
data_folder_dict = {
    # should use the same name as the dataset folder
    'Cognision-rsEEG': COGrsEEGLoader,
    'ADFTD': ADFTDLoader,
    'ADFTD-RS': ADFTDLoader,
    'BrainLat': BrainLatLoader,
    'CNBPM': CNBPMLoader,
    'P-ADIC': PADICLoader,
    'CAUEEG': CAUEEGLoader,
}
warnings.filterwarnings('ignore')


class SingleDatasetLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path

        print(f"Loading {flag} samples from single dataset...")
        if flag == 'PRETRAIN':
            data_folder_list = args.pretraining_datasets.split(",")
        elif flag == 'TRAIN':
            data_folder_list = args.training_datasets.split(",")
        elif flag == 'TEST' or flag == 'VAL':
            data_folder_list = args.testing_datasets.split(",")
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")
        if len(data_folder_list) > 1:
            raise ValueError("Only one dataset should be given here")
        print(f"Datasets used ", data_folder_list[0])
        data = data_folder_list[0]
        if data not in data_folder_dict.keys():
            raise Exception("Data not matched, "
                            "please check if the data folder name in data_folder_dict.")
        else:
            Data = data_folder_dict[data]
            data_set = Data(
                root_path=os.path.join(args.root_path, data),
                args=args,
                flag=flag,
            )
            print(f"{data} data shape: {data_set.X.shape}, {data_set.y.shape}")
            # only one dataset, dataset ID is 1
            data_set.y = np.concatenate((data_set.y, np.full(data_set.y[:, 0].shape, 1).reshape(-1, 1)), axis=1)
            self.X, self.y = data_set.X, data_set.y

        # mask single channel for channel importance analysis
        if args.single_channel_mask == 'none':
            pass
        else:
            if self.X.shape[1] != 19:
                raise ValueError("Only 19 channels of 10-20 system in order are supported for single channel mask")
            channel_index = get_channel_index(args.single_channel_mask)
            self.X[:, :, channel_index] = 0

        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        self.max_seq_len = self.X.shape[1]
        # print(f"Unique subjects used in {flag}: ", len(np.unique(self.y[:, 1])))
        print()

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class MultiDatasetsLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path

        print(f"Loading {flag} samples from multiple datasets...")
        if flag == 'PRETRAIN':
            data_folder_list = args.pretraining_datasets.split(",")
        elif flag == 'TRAIN':
            data_folder_list = args.training_datasets.split(",")
        elif flag == 'TEST' or flag == 'VAL':
            data_folder_list = args.testing_datasets.split(",")
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")
        print(f"Datasets used ", data_folder_list)
        self.X, self.y = None, None
        global_ids_range = 1  # count global subject number to avoid duplicate IDs in multiple datasets
        for i, data in enumerate(data_folder_list):
            if data not in data_folder_dict.keys():
                raise Exception("Data not matched, "
                                "please check if the data folder name in data_folder_dict.")
            else:
                Data = data_folder_dict[data]
                data_set = Data(
                    root_path=os.path.join(args.root_path, data),
                    args=args,
                    flag=flag,
                )
                # add dataset ID to the third column of y, id starts from 1
                data_set.y = np.concatenate((data_set.y, np.full(data_set.y[:, 0].shape, i + 1).reshape(-1, 1)), axis=1)
                print(f"{data} data shape: {data_set.X.shape}, {data_set.y.shape}")
                if self.X is None or self.y is None:
                    self.X, self.y = data_set.X, data_set.y
                    global_ids_range = max(len(data_set.all_ids), max(data_set.all_ids))
                else:
                    # number of subjects or max subject ID in the current dataset
                    current_ids_range = max(len(data_set.all_ids), max(data_set.all_ids))
                    # update subject IDs in the current dataset by adding global_ids_range
                    data_set.y[:, 1] += global_ids_range
                    # update global subject number
                    global_ids_range += current_ids_range
                    # concatenate data from different datasets
                    self.X, self.y = (np.concatenate((self.X, data_set.X), axis=0),
                                      np.concatenate((self.y, data_set.y), axis=0))

        # mask single channel for channel importance analysis
        if args.single_channel_mask == 'none':
            pass
        else:
            if self.X.shape[2] != 19:
                raise ValueError("Only 19 channels of 10-20 system in order are supported for single channel mask")
            channel_index = get_channel_index(args.single_channel_mask)
            # none check
            if channel_index is None:
                raise ValueError("Channel name not found in the dataset")
            self.X[:, :, channel_index] = 0

        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        self.max_seq_len = self.X.shape[1]
        # print(f"Unique subjects used in {flag}: ", len(np.unique(self.y[:, 1])))
        print()

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)
