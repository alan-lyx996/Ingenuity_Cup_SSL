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
               'num_classes': 10, 'num_epoch': 40, 'batch_size': 64, 'optimizer': "adam", 'drop_last': True,
               'TSlength_aligned': 5120, 'target_batch_size': 64, 'increased_dim': 1, 'num_classes_target': 10,
               'features_len_f': 162, 'CNNoutput_channel': 162, 'hidden_dim': 64, 'timesteps': 50, 'temperature': 0.2,
               'use_cosine_similarity': True, 'dropout': 0.35, 'corruption_prob': 0.3}

"""复制参数"""

params = {
    "beta1": 0.9,
    "beta2": 0.999,
    "lr": 0.005157727215625317,
    "lr_f": 0.009661235073362693,
    "jitter_scale_ratio": 0.3399510299645554,
    "jitter_ratio": 0.28333406458213517,
    "f_remove": 0.05809377239044571,
    "f_add": 0.015084068128470497,
    "max_seg": 8
}



# 定义device为cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 源域和目标域数据集地址
sourcedata = args.pretrain_dataset
targetdata = args.target_dataset

# 实验的描述

experiment_description = str(sourcedata).split("/")[0]+"/"+str(sourcedata).split("/")[1]+"/"+str(sourcedata).split("/")[2]+'_2_'+str(targetdata).split("/")[2]
# 例：'CWRU/group1/FD-A_2_FD-B'

# 时频一致性
method = 'Time-Freq Consistency'

# 训练的模式 “pre_train” 或是 “fine_tune_test”
training_mode = args.training_mode

# 运行描述
run_description = args.run_description

# 存储目录
logs_save_dir = args.logs_save_dir

# 创建存储目录
os.makedirs(logs_save_dir, exist_ok=True)


# ####### 固定随机种子确保可重复性 ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

# 实验记录目录
experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
# 例：'experiments_logs/CWRU/group1/FD-A_2_FD-B/fine_tune_test_seed_0'

# 创建实验记录目录
os.makedirs(experiment_log_dir, exist_ok=True)

# 设置记录器存储位置
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'

logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Pre-training Dataset: {sourcedata}')
logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# 加载数据集
sourcedata_path = f"./datasets/{sourcedata}"
targetdata_path = f"./datasets/{targetdata}"

# 对于自监督学习，这里的数据被扩充。只有自监督学习需要进行数据增强
# 如果subset=true，则使用子集进行调试。
# 调试开关
subset = False

# 利用data_generator分割数据集
train_dl, valid_dl, test_dl = data_generator(sourcedata_path, targetdata_path, params, params_keep,
                                             training_mode, subset=subset)

# 时频一致性模型
TFC_model = TFC(params_keep).to(device)
# 下游分类器模型
classifier = target_classifier(params_keep).to(device)

if training_mode == "fine_tune_test":
    # 加载预训练的模型
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
                                          f"pre_train_seed_{SEED}", "saved_models"))
    # 'experiments_logs/Exp1/run1/self_supervised_seed_0/saved_models'
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    # two saved models: ['model_state_dict', 'temporal_contr_model_state_dict']

    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = TFC_model.state_dict()
    # pretrained_dict = remove_logits(pretrained_dict)
    model_dict.update(pretrained_dict)
    TFC_model.load_state_dict(model_dict)
    print("the pre-trained model is loaded")

# 设置优化算法
model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"]), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"]), weight_decay=3e-4)

# 训练器开启训练
Trainer(TFC_model, model_optimizer, train_dl, valid_dl, test_dl, device, logger, params_keep,
        experiment_log_dir, training_mode, classifier=classifier, classifier_optimizer=classifier_optimizer)

logger.debug(f"Training time is : {datetime.now()-start_time}")



