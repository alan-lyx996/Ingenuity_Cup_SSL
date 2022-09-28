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
    def __init__(self, dataset, params, params_keep, training_mode, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        # 获取训练模式，预训练还是微调
        self.training_mode = training_mode
        # 获取数据和标签
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # 打乱
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        # 转换成tensor格式
        X_train, y_train = np.stack(list(X_train)), np.stack(list(y_train))
        X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)

        # 增加一个channel维度
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # 确保channel是在第二个维度
        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)

        # 在源数据集和目标数据集之间调整TS长度
        X_train = X_train[:, :1, :int(params_keep["TSlength_aligned"])]

        """Subset for debugging"""
        # 如果Subset是True，则采用仅有十倍的目标数据集大小的源训练集，选取100倍的源测试数据集
        if subset == True:
            subset_size = target_dataset_size * 10

            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        """将x_data传输到频域。如果使用fft。fft，输出具有相同的形状；如果使用fft。rfft，输出形状是时间窗口的一半。"""
        window_length = self.x_data.shape[-1]
        # /(window_length) # rfft for real value inputs.
        self.x_data_f = fft.fft(self.x_data).abs()
        self.len = X_train.shape[0]

        """进行数据增强"""
        # 无需在除预训练模式下的其他模式下应用增强，
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


def data_generator(sourcedata_path, targetdata_path, params, params_keep, training_mode, subset=True):
    """数据生成操作"""
    # 加载训练数据集、微调数据集和测试数据集
    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))
    test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))
    # subset = True
    # subset是调试开关，打开调试开关后，目标域的数据集大小仅为batch的10倍
    # 对于自监督学习，数据集在此进行了扩充
    train_dataset = Load_Dataset(train_dataset, params, params_keep, training_mode, target_dataset_size=params_keep["batch_size"], subset=subset)
    finetune_dataset = Load_Dataset(finetune_dataset, params, params_keep, training_mode, target_dataset_size=params_keep["target_batch_size"], subset=subset)
    # 将测试数据集的大小设置为目标域训练数据集大小的10倍或以下
    if test_dataset['labels'].shape[0] > 10*params_keep["target_batch_size"]:
        test_dataset = Load_Dataset(test_dataset, params, params_keep, training_mode, target_dataset_size=params_keep["target_batch_size"]*10, subset=subset)
    else:
        test_dataset = Load_Dataset(test_dataset, params, params_keep,training_mode, target_dataset_size=params_keep["target_batch_size"], subset=subset)
    # 将训练数据集载入Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params_keep["batch_size"],
                                               shuffle=True, drop_last=params_keep["drop_last"],
                                               num_workers=0)

    """需要注意的是，此处命名时，将常用的验证数据集用于微调数据集的命名"""
    # 将微调数据集和测试数据集载入Dataloader,且只在微调阶段使用
    valid_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=params_keep["target_batch_size"],
                                               shuffle=True, drop_last=params_keep["drop_last"],
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params_keep["target_batch_size"],
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

    training_mode = "fine_tune_test"
    train_loader, valid_loader, test_loader = data_generator(sourcedata_path, targetdata_path, params, training_mode, subset=False)

    print("0")
