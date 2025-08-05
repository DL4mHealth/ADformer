from torch.utils.data import Sampler
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import os
import random
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Metric score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        try:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        except Exception as e:
            print(f"Error saving model: {e}")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_channel_index(channel_name):
    """
    Get the channel index based on the channel name.
    :param channel_name: The name of the channel.
    :return: The index of the channel.
    """
    #                  0  , 1  , 2 , 3 , 4 , 5 , 6,  7 , 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17,     18
    # 19 channels are Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2 in order
    # T3, T4, T5, T6 is the same as T7, T8, P7, P8 in the 10-20 system
    channel_dict = {
        'Fp1': 0, 'Fp2': 1, 'F7': 2, 'F3': 3, 'Fz': 4, 'F4': 5, 'F8': 6, 'T3': 7, 'T7': 7, 'C3': 8,
        'Cz': 9, 'C4': 10, 'T4': 11, 'T8': 11, 'T5': 12, 'P7': 12, 'P3': 13, 'Pz': 14, 'P4': 15, 'T6': 16, 'P8': 16,
        'O1': 17, 'O2': 18
    }

    return channel_dict.get(channel_name, None)


class CustomGroupSampler(Sampler):
    """ A custom Sampler sort samples by subject IDs first,
        group xxx samples into groups, shuffle the groups,
        then concatenate the groups into a list of indices,
        finally we put then into batches and shuffle the samples within each batch.

        This is used for subject-level contrastive pretraining to guarantee that there are samples from the same subject in a batch,
        avoiding all the samples from the same subjects, and maintaining the different order of samples each epoch.

    """
    def __init__(self, dataset, batch_size=128, group_size=2):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.indices = self.create_indices()

    def create_indices(self):
        # sort by subject IDs
        subject_ids = self.dataset.y[:, 1]
        sorted_indices = np.argsort(subject_ids)

        # group samples
        N = len(sorted_indices)
        indices = sorted_indices
        groups = [indices[i:i + self.group_size] for i in range(0, N, self.group_size)]
        # shuffle groups
        np.random.shuffle(groups)
        """# shuffle samples in the group
        for group in groups:
            np.random.shuffle(group)"""
        # concatenate groups
        final_indices = np.concatenate(groups)

        # split into batches
        batches = [final_indices[i:i + self.batch_size] for i in range(0, len(final_indices), self.batch_size)]
        # shuffle samples in the batch
        for batch in batches:
            np.random.shuffle(batch)
        # concatenate batches
        final_indices = np.concatenate(batches)

        return final_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.dataset)


def calculate_subject_level_metrics(predictions, true_labels, subject_ids, num_classes):
    # Step 1: Get unique subject_ids
    unique_subjects = np.unique(subject_ids)

    # Step 2: Aggregate predictions and true labels for each subject_id
    subject_predictions = []
    subject_trues = []

    for subject in unique_subjects:
        # Find all sample indices for the current subject
        indices = np.where(subject_ids == subject)[0]

        # Get predictions and true labels for the current subject
        subject_preds = predictions[indices]
        subject_true = true_labels[indices][0]  # The true_label should be the same for all samples of a subject

        # Determine the majority vote prediction for the subject
        majority_label = Counter(subject_preds).most_common(1)[0][0]

        # Record subject-level results
        subject_predictions.append(majority_label)
        subject_trues.append(subject_true)

    # Convert to numpy arrays
    subject_predictions = np.array(subject_predictions)
    subject_trues = np.array(subject_trues)

    # Step 3: Calculate metrics
    # Convert true labels to one-hot encoding for AUROC and AUPRC
    subject_true_onehot = label_binarize(subject_trues, classes=list(range(num_classes)))
    subject_probs = label_binarize(subject_predictions, classes=list(range(num_classes)))  # Simplified assumption

    metrics = {
        "Accuracy": accuracy_score(subject_trues, subject_predictions),
        # "Precision": precision_score(subject_trues, subject_predictions, average="macro"),
        # "Recall": recall_score(subject_trues, subject_predictions, average="macro"),
        # "F1": f1_score(subject_trues, subject_predictions, average="macro"),
        # "AUROC": roc_auc_score(subject_true_onehot, subject_probs, multi_class="ovr"),
        # "AUPRC": average_precision_score(subject_true_onehot, subject_probs, average="macro"),
    }
    # Check how many unique classes are present in the true labels
    unique_labels = np.unique(subject_trues)
    if len(unique_labels) < 2:
        # If there is only one class(e,g, leave-one-subject-out validation),
        # subject-level AUROC and AUPRC are meaningless
        metrics["Precision"] = -1
        metrics["Recall"] = -1
        metrics["F1"] = -1
        metrics["AUROC"] = -1
        metrics["AUPRC"] = -1
    else:
        metrics["Precision"] = precision_score(subject_trues, subject_predictions, average="macro")
        metrics["Recall"] = recall_score(subject_trues, subject_predictions, average="macro")
        metrics["F1"] = f1_score(subject_trues, subject_predictions, average="macro")
        metrics["AUROC"] = roc_auc_score(subject_true_onehot, subject_probs, multi_class="ovr")
        metrics["AUPRC"] = average_precision_score(subject_true_onehot, subject_probs, average="macro")

    return metrics


def compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                    sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params):
    print('>>>>>>>average testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    sample_val_metrics_dict_avg_std = {}
    subject_val_metrics_dict_avg_std = {}
    sample_test_metrics_dict_avg_std = {}
    subject_test_metrics_dict_avg_std = {}

    # Compute average and std
    for key in sample_val_metrics_dict_list[0].keys():
        # convert to percentage
        sample_val_avg = np.mean([val_metrics_dict[key] for val_metrics_dict in sample_val_metrics_dict_list]) * 100
        sample_val_std = np.std([val_metrics_dict[key] for val_metrics_dict in sample_val_metrics_dict_list]) * 100
        subject_val_avg = np.mean([val_metrics_dict[key] for val_metrics_dict in subject_val_metrics_dict_list]) * 100
        subject_val_std = np.std([val_metrics_dict[key] for val_metrics_dict in subject_val_metrics_dict_list]) * 100
        sample_test_avg = np.mean([test_metrics_dict[key] for test_metrics_dict in sample_test_metrics_dict_list]) * 100
        sample_test_std = np.std([test_metrics_dict[key] for test_metrics_dict in sample_test_metrics_dict_list]) * 100
        subject_test_avg = np.mean([test_metrics_dict[key] for test_metrics_dict in subject_test_metrics_dict_list]) * 100
        subject_test_std = np.std([test_metrics_dict[key] for test_metrics_dict in subject_test_metrics_dict_list]) * 100

        sample_val_metrics_dict_avg_std[key] = (sample_val_avg, sample_val_std)
        subject_val_metrics_dict_avg_std[key] = (subject_val_avg, subject_val_std)
        sample_test_metrics_dict_avg_std[key] = (sample_test_avg, sample_test_std)
        subject_test_metrics_dict_avg_std[key] = (subject_test_avg, subject_test_std)

    # Format results into a single line with two decimal places and percentages
    sample_val_results = "Validation results --- " + ", ".join(
        [f"{key}: {sample_val_metrics_dict_avg_std[key][0]:.2f}+-{sample_val_metrics_dict_avg_std[key][1]:.2f}%"
         for key in sample_val_metrics_dict_avg_std.keys()]
    )
    sample_test_results = "Test results --- " + ", ".join(
        [f"{key}: {sample_test_metrics_dict_avg_std[key][0]:.2f}+-{sample_test_metrics_dict_avg_std[key][1]:.2f}%"
         for key in sample_test_metrics_dict_avg_std.keys()]
    )
    subject_val_results = "Validation results --- " + ", ".join(
        [f"{key}: {subject_val_metrics_dict_avg_std[key][0]:.2f}+-{subject_val_metrics_dict_avg_std[key][1]:.2f}%"
         for key in subject_val_metrics_dict_avg_std.keys()]
    )
    subject_test_results = "Test results --- " + ", ".join(
        [f"{key}: {subject_test_metrics_dict_avg_std[key][0]:.2f}+-{subject_test_metrics_dict_avg_std[key][1]:.2f}%"
         for key in subject_test_metrics_dict_avg_std.keys()]
    )

    folder_path = (
        "./results/"
        + args.method
        + "/"
        + args.task_name
        + "/"
        + args.model
        + "/"
        + args.model_id
        + "/"
    )
    file_name = "results.txt"
    file_path = os.path.join(folder_path, file_name)
    # Write results to file
    with open(file_path, 'a') as f:
        if args.is_training == 1:
            if 'pretrain' in args.task_name:
                f.write(f"Pretraining Datasets: {args.pretraining_datasets}\n")
            f.write(f"Training Datasets: {args.training_datasets}\n")
        f.write(f"Test Datasets: {args.testing_datasets}\n")
        f.write(f"Model_id: {args.model_id}; Model: {args.model}; Total Params: {total_params}\n")
        f.write('Average and std of validation and testing results over {} runs\n'.format(args.itr))
        f.write('Sample-level results: \n')
        f.write(sample_val_results + "\n")
        f.write(sample_test_results + "\n")
        f.write('Subject-level results after majority voting: \n')
        f.write(subject_val_results + "\n")
        f.write(subject_test_results + "\n")
        f.write("\n\n\n")

    print('Sample-level results: \n')
    print(sample_val_results)
    print(sample_test_results)
    print('Subject-level results after majority voting: \n')
    print(subject_val_results)
    print(subject_test_results)
