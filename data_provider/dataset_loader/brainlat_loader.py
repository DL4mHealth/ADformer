import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
    load_data_by_ids,
)
import warnings
import random

warnings.filterwarnings('ignore')


def get_id_list_brainlat(args, label_path, a=0.6, b=0.8):
    '''
    Loads subject IDs for all, training, validation, and test sets for BrainLat data
    Only use healthy and Alzheimer's disease subjects
    Args:
        args: arguments
        label_path: directory of label.npy file
        a: ratio of ids in training set
        b: ratio of ids in training and validation set
    Returns:
        all_ids: list of all IDs
        train_ids: list of IDs for training set
        val_ids: list of IDs for validation set
        test_ids: list of IDs for test set
    '''
    # random shuffle to break the potential influence of human named ID order,
    # e.g., put all healthy subjects first or put subjects with more samples first, etc.
    # (which could cause data imbalance in training, validation, and test sets)
    data_list = np.load(label_path)
    all_ids = list(data_list[:, 1])  # all subjects
    hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
    ad_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # Alzheimer's disease IDs
    ftd_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])  # behavioral variant frontotemporal dementia IDs
    pd_list = list(data_list[np.where(data_list[:, 0] == 3)][:, 1])  # Parkinson's disease IDs
    ms_list = list(data_list[np.where(data_list[:, 0] == 4)][:, 1])  # multiple sclerosis IDs
    if args.cross_val == 'fixed' or args.cross_val == 'mccv':  # fixed split or Monte Carlo cross-validation
        if args.cross_val == 'fixed':
            random.seed(42)  # fixed seed for fixed split
        else:
            random.seed(args.seed)  # random seed for Monte Carlo cross-validation

        random.shuffle(hc_list)
        random.shuffle(ad_list)
        random.shuffle(ftd_list)
        random.shuffle(pd_list)
        random.shuffle(ms_list)

        train_ids = (hc_list[:int(a * len(hc_list))] +
                     ad_list[:int(a * len(ad_list))] +
                     ftd_list[:int(a * len(ftd_list))] +
                     pd_list[:int(a * len(pd_list))] +
                     ms_list[:int(a * len(ms_list))])
        val_ids = (hc_list[int(a * len(hc_list)):int(b * len(hc_list))] +
                   ad_list[int(a * len(ad_list)):int(b * len(ad_list))] +
                   ftd_list[int(a * len(ftd_list)):int(b * len(ftd_list))] +
                   pd_list[int(a * len(pd_list)):int(b * len(pd_list))] +
                   ms_list[int(a * len(ms_list)):int(b * len(ms_list))])
        test_ids = (hc_list[int(b * len(hc_list)):] +
                    ad_list[int(b * len(ad_list)):] +
                    ftd_list[int(b * len(ftd_list)):] +
                    pd_list[int(b * len(pd_list)):] +
                    ms_list[int(b * len(ms_list)):])

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    elif args.cross_val == '5-fold':  # 5-fold cross-validation
        random.seed(42)
        # split data into 5 folds
        all_ids = list(data_list[:, 1])  # all subjects, including subjects with other labels beyond AD and HC
        random.shuffle(hc_list)
        random.shuffle(ad_list)
        random.shuffle(ftd_list)
        random.shuffle(pd_list)
        random.shuffle(ms_list)
        fold_size_hc = len(hc_list) / 5
        fold_size_ad = len(ad_list) / 5
        fold_size_ftd = len(ftd_list) / 5
        fold_size_pd = len(pd_list) / 5
        fold_size_ms = len(ms_list) / 5
        seed = (args.seed-41) % 5
        # take 1-fold as test set
        test_ids = (hc_list[int(seed * fold_size_hc):int((seed + 1) * fold_size_hc)] +
                    ad_list[int(seed * fold_size_ad):int((seed + 1) * fold_size_ad)] +
                    ftd_list[int(seed * fold_size_ftd):int((seed + 1) * fold_size_ftd)] +
                    pd_list[int(seed * fold_size_pd):int((seed + 1) * fold_size_pd)] +
                    ms_list[int(seed * fold_size_ms):int((seed + 1) * fold_size_ms)])
        # take another 1-fold as validation set
        if seed == 4:  # take the first fold as validation set when seed is 4
            val_ids = ((hc_list[:int(fold_size_hc)] + ad_list[:int(fold_size_ad)]) +
                       ftd_list[:int(fold_size_ftd)]) + pd_list[:int(fold_size_pd)] + ms_list[:int(fold_size_ms)]
        else:
            val_ids = (hc_list[int((seed + 1) * fold_size_hc):int((seed + 2) * fold_size_hc)] +
                       ad_list[int((seed + 1) * fold_size_ad):int((seed + 2) * fold_size_ad)] +
                       ftd_list[int((seed + 1) * fold_size_ftd):int((seed + 2) * fold_size_ftd)] +
                       pd_list[int((seed + 1) * fold_size_pd):int((seed + 2) * fold_size_pd)] +
                       ms_list[int((seed + 1) * fold_size_ms):int((seed + 2) * fold_size_ms)])
        # take the remaining ids not in test and validation sets as training set
        train_ids = [id for id in all_ids if id not in test_ids and id not in val_ids]

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    elif args.cross_val == 'loso':  # leave-one-subject-out cross-validation
        if args.classify_choice == 'ad_vs_hc':
            hc_ad_list = sorted(hc_list + ad_list)  # all subjects with AD and HC labels
            # take subject ID with index (args.seed-41) % len(all_ids) as test set, random seed start from 41
            test_ids = [hc_ad_list[(args.seed - 41) % len(hc_ad_list)]]
            train_ids = [id for id in hc_ad_list if id not in test_ids]
        else:
            # take subject ID with index (args.seed-41) % len(all_ids) as test set, random seed start from 41
            test_ids = [all_ids[(args.seed - 41) % len(all_ids)]]
            train_ids = [id for id in all_ids if id not in test_ids]
        # randomly take 20% of the training set as validation set
        random.seed(args.seed)
        random.shuffle(train_ids)
        val_ids = train_ids[int(0.8 * len(train_ids)):]

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)
    else:
        raise ValueError('Invalid cross_val. Please use fixed, mccv, or loso.')


class BrainLatLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        a, b = 0.6, 0.8
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = get_id_list_brainlat(args, self.label_path, a, b)

        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        elif flag == 'PRETRAIN':
            ids = self.all_ids
            print('all ids:', ids)
        else:
            raise ValueError('Invalid flag. Please use TRAIN, VAL, TEST, or ALL.')

        self.X, self.y = load_data_by_ids(self.data_path, self.label_path, ids, args.segment_length, args.overlapping)
        print('X shape:', self.X.shape)
        if args.classify_choice == 'ad_vs_hc':  # 1 vs 0
            # delete all other diseases except AD and HC
            print('Delete bvFTD, PD, and MS subjects in train ids list for AD vs HC classification')
            mask = (self.y[:, 0] < 2)
            self.X = self.X[mask]
            self.y = self.y[mask]
        elif args.classify_choice == 'ad_vs_nonad':
            # all other diseases are also 0, change labels to 0
            print('Change bvFTD, PD, and MS subjects from 2 to 0 in train ids list for AD vs non-AD classification')
            self.y[:, 0] = np.where(self.y[:, 0] > 1, 0, self.y[:, 0])
        elif args.classify_choice == 'hc_vs_abnormal':
            # change all other diseases to 1
            print('Change bvFTD, PD, and MS subjects from 2 to 1 in train ids list for HC vs abnormal classification')
            self.y[:, 0] = np.where(self.y[:, 0] > 1, 1, self.y[:, 0])
        elif args.classify_choice == 'multi_class':
            print('Keep all subjects in train ids list for multi-class classification')
            pass
        # self.X = bandpass_filter_func(self.X, fs=args.sampling_rate, lowcut=args.low_cut, highcut=args.high_cut)
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)
