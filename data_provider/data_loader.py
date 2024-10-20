import copy
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import (
    subsample,
    interpolate_missing,
    Normalizer,
    normalize_batch_ts,
    bandpass_filter_func,
)
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from natsort import natsorted

warnings.filterwarnings('ignore')


# Subject-Dependent Loader
class DependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        # load data in subject-dependent manner
        self.X, self.y = self.load_ad(self.data_path, self.label_path, flag=flag)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_ad(self, data_path, label_path, flag=None):
        '''
        Loads data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        # print(filenames)
        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)

        # 60 : 20 : 20
        X_train, y_train = np.array(feature_list), np.array(label_list)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train[:, 0]
        elif flag == 'VAL':
            return X_val, y_val[:, 0]
        elif flag == 'TEST':
            return X_test, y_test[:, 0]
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


# Subject-Independent Loaders

class ADSZIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        a, b = 0.6, 0.8
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(self.label_path, a, b)

        self.X, self.y = self.load_adsz(self.data_path, self.label_path, flag=flag)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        '''
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ad_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # Alzheimer's disease IDs

        train_ids = hc_list[:int(a * len(hc_list))] + ad_list[:int(a * len(ad_list))]
        val_ids = hc_list[int(a * len(hc_list)):int(b * len(hc_list))] + ad_list[int(a * len(ad_list)):int(b * len(ad_list))]
        test_ids = hc_list[int(b * len(hc_list)):] + ad_list[int(b * len(ad_list)):]

        return train_ids, val_ids, test_ids

    def load_adsz(self, data_path, label_path, flag=None):
        '''
        Loads ADSZ data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        else:
            ids = subject_label[:, 1]
            print('all ids:', ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class APAVAIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)

        all_ids = list(data_list[:, 1])  # id of all samples
        val_ids = [15, 16, 19, 20]  # 15, 19 are AD; 16, 20 are HC
        test_ids = [1, 2, 17, 18]  # 1, 17 are AD; 2, 18 are HC
        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]
        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y = self.load_apava(self.data_path, self.label_path, flag=flag)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_apava(self, data_path, label_path, flag=None):
        '''
        Loads APAVA data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADFDIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_adfd(self.data_path, self.label_path, flag=flag)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        '''
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ftd_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Frontotemporal Dementia IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ftd_list[: int(a * len(ftd_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ftd_list[int(a * len(ftd_list)) : int(b * len(ftd_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ftd_list[int(b * len(ftd_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_adfd(self, data_path, label_path, flag=None):
        '''
        Loads adfd data from npy files in data_path based on flag and ids in label_path
        Channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class CNBPMIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_cnbpm(self.data_path, self.label_path, flag=flag)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        '''
        random.seed(42)
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        mci_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Mild Cognitive Impairment IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        # Shuffle the subject IDs since last 20% subjects have fewer samples.
        # Split on current ids order with 6:2:2 ratios will cause much fewer number of samples in test set.
        random.shuffle(cn_list)
        random.shuffle(mci_list)
        random.shuffle(ad_list)

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + mci_list[: int(a * len(mci_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + mci_list[int(a * len(mci_list)) : int(b * len(mci_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + mci_list[int(b * len(mci_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_cnbpm(self, data_path, label_path, flag=None):
        '''
        Loads CNBPM data from npy files in data_path based on flag and ids in label_path
        Channels: Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2

        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class COGERPIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(self.label_path, a, b)
        self.X, self.y = self.load_cog_erp(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = bandpass_filter_func(self.X, 125, 0.5, 45)
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        '''
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ad_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # Alzheimer's disease IDs

        train_ids = hc_list[:int(a * len(hc_list))] + ad_list[:int(a * len(ad_list))]
        val_ids = hc_list[int(a * len(hc_list)):int(b * len(hc_list))] + ad_list[int(a * len(ad_list)):int(b * len(ad_list))]
        test_ids = hc_list[int(b * len(hc_list)):] + ad_list[int(b * len(ad_list)):]

        return train_ids, val_ids, test_ids

    def load_cog_erp(self, data_path, label_path, flag=None):
        '''
        Loads ad data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        else:
            ids = subject_label[:, 1]
            print('all ids:', ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class COGrsEEGIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(self.label_path, a, b)
        self.X, self.y = self.load_cog_rseeg(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = bandpass_filter_func(self.X, 125, 0.5, 45)
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        '''
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ad_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # Alzheimer's disease IDs

        train_ids = hc_list[:int(a * len(hc_list))] + ad_list[:int(a * len(ad_list))]
        val_ids = hc_list[int(a * len(hc_list)):int(b * len(hc_list))] + ad_list[int(a * len(ad_list)):int(b * len(ad_list))]
        test_ids = hc_list[int(b * len(hc_list)):] + ad_list[int(b * len(ad_list)):]

        return train_ids, val_ids, test_ids

    def load_cog_rseeg(self, data_path, label_path, flag=None):
        '''
        Loads ad data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        else:
            ids = subject_label[:, 1]
            print('all ids:', ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class ADFDBinaryIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_adfd(self.data_path, self.label_path, flag=flag)
        # convert the class of label 2 to 1 in y
        self.y = np.where(self.y == 2, 1, self.y)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        '''
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_adfd(self, data_path, label_path, flag=None):
        '''
        Loads adfd data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class CNBPMBinaryIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_cnbpm(self.data_path, self.label_path, flag=flag)
        # convert the class of label 2 to 1 in y
        self.y = np.where(self.y == 2, 1, self.y)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        '''
        random.seed(42)
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        # Shuffle the subject IDs since last 20% subjects have fewer samples.
        # Split on current ids order with 6:2:2 ratios will cause much fewer number of samples in test set.
        random.shuffle(cn_list)
        random.shuffle(ad_list)

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_cnbpm(self, data_path, label_path, flag=None):
        '''
        Loads CNBPM data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADFD7ChannelsIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_adfd(self.data_path, self.label_path, flag=flag)
        # convert the class of label 2 to 1 in y
        self.y = np.where(self.y == 2, 1, self.y)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        # Raw channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2
        # Use the only 7 channels: F3, Fz, F4, Cz, P3, Pz, and P4
        self.X = self.X[:, :, [3, 4, 5, 9, 13, 14, 15]]

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        '''
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_adfd(self, data_path, label_path, flag=None):
        '''
        Loads adfd data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class CNBPM7ChannelsIndependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_cnbpm(self.data_path, self.label_path, flag=flag)
        # convert the class of label 2 to 1 in y
        self.y = np.where(self.y == 2, 1, self.y)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        # Raw channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2
        # Use the only 7 channels: F3, Fz, F4, Cz, P3, Pz, and P4
        self.X = self.X[:, :, [3, 4, 5, 9, 13, 14, 15]]

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        '''
        random.seed(42)
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        # Shuffle the subject IDs since last 20% subjects have fewer samples.
        # Split on current ids order with 6:2:2 ratios will cause much fewer number of samples in test set.
        random.shuffle(cn_list)
        random.shuffle(ad_list)

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_cnbpm(self, data_path, label_path, flag=None):
        '''
        Loads CNBPM data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADFDLeaveSubjectsOutLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a = 4

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a
        )
        self.X, self.y = self.load_adfd(self.data_path, self.label_path, flag=flag)
        # convert the class of label 2 to 1 in y
        self.y = np.where(self.y == 2, 1, self.y)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=2):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: subject number of each class in test set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set (here test set is the same as validation set)
        '''
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs
        random.shuffle(cn_list)
        random.shuffle(ad_list)

        train_ids = (
            cn_list[: -a*2]
            + ad_list[: -a*2]
        )
        val_ids = (
            cn_list[-a*2: -a]
            + ad_list[-a*2: -a]
        )
        test_ids = (
            cn_list[-a: ]
            + ad_list[-a: ]
        )

        return train_ids, val_ids, test_ids

    def load_adfd(self, data_path, label_path, flag=None):
        '''
        Loads adfd data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class CNBPMLeaveSubjectsOutLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a = 10

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a
        )
        self.X, self.y = self.load_cnbpm(self.data_path, self.label_path, flag=flag)
        # convert the class of label 2 to 1 in y
        self.y = np.where(self.y == 2, 1, self.y)

        # pre_process
        if not self.no_normalize:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=2):
        '''
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: subject number of each class in test set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set (here test set is the same as validation set)
        '''
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs
        random.shuffle(cn_list)
        random.shuffle(ad_list)

        train_ids = (
            cn_list[: -a*2]
            + ad_list[: -a*2]
        )
        val_ids = (
            cn_list[-a*2: -a]
            + ad_list[-a*2: -a]
        )
        test_ids = (
            cn_list[-a: ]
            + ad_list[-a: ]
        )

        return train_ids, val_ids, test_ids

    def load_cnbpm(self, data_path, label_path, flag=None):
        '''
        Loads cnbpm data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)