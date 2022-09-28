import torch
import nni
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
# from models.TC import TC
from utils import _calc_metrics, copy_Files
from model import * # base_Model, base_Model_F, target_classifier

from dataloader import data_generator
from trainer import Trainer, model_pretrain, model_finetune, model_test
import warnings
warnings.filterwarnings("ignore")

# Args selections
start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
# parser.add_argument('--experiment_description', default='Exp1', type=str,
#                     help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int, help='seed value')

# 1. self_supervised; 2. finetune (itself contains finetune and test)
parser.add_argument('--training_mode', default='fine_tune_test', type=str,
                    help='pre_train, fine_tune_test')

parser.add_argument('--pretrain_dataset', default='CWRU/group1/FD-A', type=str,
                    help='Dataset of choice: Paderborn/FD-A,CWRU/group1/FD-A')
parser.add_argument('--target_dataset', default='CWRU/group1/FD-B', type=str,
                    help='Dataset of choice: Paderborn/FD-B, CWRU/group1/FD-B、FD-C、FD-D')

parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

"""定义参数字典"""
# 固定参数集合
params_keep = {'input_channels': 1, 'kernel_size': 32, 'stride': 4, 'final_out_channels': 128, 'features_len': 162,
               'num_classes': 10, 'num_epoch': 40, 'batch_size': 64,'optimizer': "adam", 'drop_last': True,
               'TSlength_aligned': 5120, 'target_batch_size': 64, 'increased_dim': 1, 'num_classes_target': 10,
               'features_len_f': 162, 'CNNoutput_channel': 162, 'hidden_dim': 64, 'timesteps': 50, 'temperature': 0.2,
               'use_cosine_similarity': True, 'dropout': 0.35, 'corruption_prob': 0.3}
# 微调参数集合
params = {
          'beta1': 0.9,
          'beta2': 0.99,
          'lr': 0.0001,
          'lr_f': 0.0001,
          'jitter_scale_ratio': 1.5,
          'jitter_ratio': 0.1,
          'f_remove': 0.1,
          'f_add': 0.1,
          'max_seg': 5
}


# 获取最新参数
optimized_params = nni.get_next_parameter()
# #获取下一个实验id
# id = nni.get_experiment_id()

# 更新参数字典
params.update(optimized_params)
print(params)

# 定义device为cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 源域和目标域数据集地址
sourcedata = args.pretrain_dataset
targetdata = args.target_dataset

# ####### 固定随机种子确保可重复性 ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################


# 加载数据集
sourcedata_path = f"./datasets/{sourcedata}"
targetdata_path = f"./datasets/{targetdata}"

# 对于自监督学习，这里的数据被扩充。只有自监督学习需要进行数据增强
# 如果subset=true，则使用子集进行调试。
# 调试开关
subset = True

training_mode = args.training_mode
# 利用data_generator分割数据集
train_dl, valid_dl, test_dl = data_generator(sourcedata_path, targetdata_path, params, params_keep,
                                             training_mode, subset=subset)

# Load Model
"""Here are two models, one basemodel, another is temporal contrastive model"""
# model = Time_Model(params).to(device)
# model_F = Frequency_Model(params).to(device) #base_Model_F(params).to(device) """here is right. No bug in this line.
# 加载模型
# 时频一致性模型
TFC_model = TFC(params_keep).to(device)
# 下游分类器模型
classifier = target_classifier(params_keep).to(device)

# 设置优化算法
model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"]), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"]), weight_decay=3e-4)

epochs = params_keep["num_epoch"]


for t in range(40):
    loss1, _ = model_pretrain(TFC_model, model_optimizer, train_dl, params_keep, device)

for r in range(40):
    loss2, acc = model_finetune(TFC_model, valid_dl, params_keep, device, model_optimizer, classifier, classifier_optimizer)
    loss3, acc2 = model_test(TFC_model, test_dl, device, training_mode, classifier)
    acc2 = float(acc2)
    print(acc2)
    nni.report_intermediate_result(acc2)

nni.report_final_result(acc2)


