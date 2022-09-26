import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations import DataTransform_FD, DataTransform_TD
import torch.fft as fft
from scipy.io import loadmat



class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, params, training_mode, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)

        X_train, y_train = np.stack(list(X_train)), np.stack(list(y_train))
        X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
        # X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # make sure the Channels in second dim
        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :1, :int(params["TSlength_aligned"])]

        """Subset for debugging"""
        # 如果Subset是True，则采用仅有十倍的目标数据集大小的源训练集
        if subset == True:
            subset_size = target_dataset_size * 10
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size] #
            y_train = y_train[:subset_size]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        window_length = self.x_data.shape[-1]
        # /(window_length) # rfft for real value inputs.
        self.x_data_f = fft.fft(self.x_data).abs()
        # self.x_data_f = self.x_data_f[:, :, 1:] # not a problem.

        self.len = X_train.shape[0]
        """Augmentation"""
        # no need to apply Augmentations in other modes
        if training_mode == "pre_train":
            self.aug1 = DataTransform_TD(self.x_data, params)
            self.aug1_f = DataTransform_FD(self.x_data_f, params)

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


# subset 调试子集
def data_generator(sourcedata_path, targetdata_path, params, training_mode, subset=True):

    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))
    test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))
    """ Dataset notes:
    Paderborn: train_dataset['samples'].shape = torch.Size([6000, 1, 5120]); binary labels [6000] 
    valid: [60, 1, 5120]
    test: [6000, 1, 5120]. In test set, 1835 are positive sampels, the positive rate is 0.7978"""
    """sleepEDF: finetune_dataset['samples']: [7786, 1, 3000]"""

    # subset = True # if true, use a subset for debugging.
    # for self-supervised, the data are augmented here
    train_dataset = Load_Dataset(train_dataset, params, training_mode, target_dataset_size=params["batch_size"], subset=subset)
    finetune_dataset = Load_Dataset(finetune_dataset, params, training_mode, target_dataset_size=params["target_batch_size"], subset=subset)
    if test_dataset['labels'].shape[0] > 10*params["target_batch_size"]:
        test_dataset = Load_Dataset(test_dataset, params, training_mode, target_dataset_size=params["target_batch_size"]*10, subset=subset)
    else:
        test_dataset = Load_Dataset(test_dataset, params, training_mode, target_dataset_size=params["target_batch_size"], subset=subset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params["batch_size"],
                                               shuffle=True, drop_last=params["drop_last"],
                                               num_workers=0)

    """the valid and test loader would be finetuning set and test set."""
    valid_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=params["target_batch_size"],
                                               shuffle=True, drop_last=params["drop_last"],
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params["target_batch_size"],
                                              shuffle=True, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    sourcedata_path="./datasets/CWRU/group1/FD-A"
    targetdata_path="./datasets/CWRU/group1/FD-B"
    """定义参数字典"""
    params = {'input_channels': 1,
              'kernel_size': 32,
              'stride': 4,
              'final_out_channels': 128,
              'features_len': 162,
              'num_classes': 10,
              'num_epoch': 40,
              'batch_size': 64,
              'optimizer': "adam",
              'drop_last': True,
              'TSlength_aligned': 5120,
              'target_batch_size': 64,
              'increased_dim': 1,
              'num_classes_target': 10,
              'features_len_f': 162,
              'CNNoutput_channel': 162,
              'hidden_dim': 64,
              'timesteps': 50,
              'temperature': 0.2,
              'use_cosine_similarity': True,
              'dropout': 0.35,
              'corruption_prob': 0.3,
              'beta1': 0.9,
              'beta2': 0.99,
              'lr': 0.0001,
              'lr_f': 0.0001,
              'jitter_scale_ratio': 0.5,
              'jitter_ratio': 0.0,
              'max_seg': 5
              }

    training_mode = "pre_train"
    train_loader, valid_loader, test_loader = data_generator(sourcedata_path, targetdata_path, params, training_mode, subset=True)

    print("0")
