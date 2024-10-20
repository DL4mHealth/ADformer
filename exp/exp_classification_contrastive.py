from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from utils.losses import id_contrastive_loss
from utils.eval_protocols import fit_lr
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import time
import warnings
import numpy as np
import pdb
import sklearn
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')


class Exp_Classification_Contrastive(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification_Contrastive, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)  # redefine seq_len
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = train_data.X.shape[2]  # redefine enc_in
        self.args.num_class = len(np.unique(train_data.y))
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def encode(self, loader):
        reprs_list = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                reprs = self.model(batch_x, padding_mask, None, None)
                reprs = reprs.reshape(reprs.shape[0], -1)
                reprs_list.append(reprs)

            reprs_array = torch.cat(reprs_list, dim=0).cpu().numpy()

        self.model.train()
        return reprs_array

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                _, _, outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = torch.nn.functional.one_hot(trues.reshape(-1, ).to(torch.long),
                                                   num_classes=self.args.num_class).float().cpu().numpy()
        # print(trues_onehot.shape)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {'Accuracy': accuracy_score(trues, predictions),
                        'Precision': precision_score(trues, predictions, average='macro'),
                        'Recall': recall_score(trues, predictions, average='macro'),
                        'F1': f1_score(trues, predictions, average='macro'),
                        'AUROC': roc_auc_score(trues_onehot, probs, multi_class='ovr'),
                        'AUPRC': average_precision_score(trues_onehot, probs, average="macro")}

        self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')
        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)
        print(test_data.X.shape)
        print(test_data.y.shape)

        path = './checkpoints/' + self.args.task_name + '/' + self.args.model_id + '/' \
               + self.args.model + '/' + setting + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion1 = id_contrastive_loss
        criterion2 = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                reprs_1, reprs_2, outputs = self.model(batch_x, padding_mask, None, None)
                loss1 = criterion1(reprs_1, reprs_2, label.long().squeeze(-1))
                loss2 = criterion2(outputs, label.long().squeeze(-1))
                loss = loss1 + loss2
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion2)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion2)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, Accuracy: {val_metrics_dict['Accuracy']:.5f} "
                f"Precision: {val_metrics_dict['Precision']:.5f}, Recall: {val_metrics_dict['Recall']:.5f} "
                f"F1: {val_metrics_dict['F1']:.5f}, AUROC: {val_metrics_dict['AUROC']:.5f} "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, Accuracy: {test_metrics_dict['Accuracy']:.5f} "
                f"Precision: {test_metrics_dict['Precision']:.5f}, Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, AUROC: {test_metrics_dict['AUROC']:.5f} "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            early_stopping(-val_metrics_dict['F1'], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)"""

        best_model_path = path + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            path = './checkpoints/' + self.args.task_name + '/' + self.args.model_id + '/' \
                   + self.args.model + '/' + setting + '/'
            model_path = path + 'checkpoint.pth'
            if not os.path.exists(model_path):
                raise Exception('No model found at %s' % model_path)
            self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = './results/' + self.args.task_name + '/' + self.args.model_id + '/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, Accuracy: {val_metrics_dict['Accuracy']:.5f} "
            f"Precision: {val_metrics_dict['Precision']:.5f}, Recall: {val_metrics_dict['Recall']:.5f} "
            f"F1: {val_metrics_dict['F1']:.5f}, AUROC: {val_metrics_dict['AUROC']:.5f} "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, Accuracy: {test_metrics_dict['Accuracy']:.5f} "
            f"Precision: {test_metrics_dict['Precision']:.5f}, Recall: {test_metrics_dict['Recall']:.5f} "
            f"F1: {test_metrics_dict['F1']:.5f}, AUROC: {test_metrics_dict['AUROC']:.5f} "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, Accuracy: {val_metrics_dict['Accuracy']:.5f} "
            f"Precision: {val_metrics_dict['Precision']:.5f}, Recall: {val_metrics_dict['Recall']:.5f} "
            f"F1: {val_metrics_dict['F1']:.5f}, AUROC: {val_metrics_dict['AUROC']:.5f} "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, Accuracy: {test_metrics_dict['Accuracy']:.5f} "
            f"Precision: {test_metrics_dict['Precision']:.5f}, Recall: {test_metrics_dict['Recall']:.5f} "
            f"F1: {test_metrics_dict['F1']:.5f}, AUROC: {test_metrics_dict['AUROC']:.5f} "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write('\n')
        f.write('\n')
        f.close()
        return
