import os
import pickle
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier

from loss import *


def one_hot_encoding(X):
    """转换为onehot编码"""
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b


def Trainer(model, model_optimizer, train_dl, valid_dl, test_dl, device, logger, params_keep, experiment_log_dir,
            training_mode, classifier=None, classifier_optimizer=None):
    """
    Args:
        model: 时频一致性（TFC）自监督模型
        model_optimizer: TFC优化器
        train_dl: 源训练集
        valid_dl: 目标训练集
        test_dl: 目标测试集
        device: 设备，cuda/CPU
        logger: 记录器
        params: 参数空间
        experiment_log_dir:实验存储地址
        training_mode: 训练模式
        classifier: 分类器
        classifier_optimizer:分类器优化器
    """
    # 开始训练
    # 记录器记录
    logger.debug("Training Started ....")
    # 调整学习率
    # 当参考的评价指标停止改进时, 降低学习率, factor为每次下降的比例, 训练过程中, 当指标连续patience次数还没有改进时, 降低学习率;
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    """预训练模式"""

    if training_mode == "pre_train":
        print('Pretraining on source dataset')
        for epoch in range(1, params_keep["num_epoch"] + 1):
            # 预训练模式，并返回相关测试参数
            # 在预训练模式下，无需进行模型的测试
            # 所以只有无标签数据集的损失，准确率和auc均为0
            train_loss, feature_flat = model_pretrain(model, model_optimizer, train_dl, params_keep, device)
            if training_mode != 'self_supervised':  # use scheduler in all other modes.
                scheduler.step(train_loss)
            logger.debug(f'\nPre-training Epoch : {epoch}\n' f'Train Loss     : {train_loss:.4f}\t')

        # 存储预训练模型
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = {'model_state_dict': model.state_dict(),}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
        print('Pretrained model is stored at folder:{}'.format(experiment_log_dir+'saved_models'+'ckp_last.pt'))

    """微调模式"""
    if training_mode != 'pre_train':
        print('Fine-tune on Fine-tuning set')
        performance_list = []
        total_f1 = []
        for epoch in range(1, params_keep["num_epoch"] + 1):
            # 微调模型，并返回相关测试参数
            # 微调训练集损失、准确率、auc、prc、F1分数
            # 也返回了微调的embbing和label
            valid_loss, valid_acc = model_finetune(model, valid_dl, params_keep, device, model_optimizer, classifier, classifier_optimizer)

            if training_mode != 'pre_train':  # use scheduler in all other modes.
                scheduler.step(valid_loss)
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'finetune Loss  : {valid_loss:.4f}\t | \tfinetune Accuracy : {valid_acc:2.4f}\t | ')

            # save best fine-tuning model""
            # global arch
            # arch = 'sleepedf2eplipsy'
            # if len(total_f1) == 0 or F1 > max(total_f1):
            #     print('update fine-tuned model')
            #     os.makedirs('experiments_logs/finetunemodel/', exist_ok=True)
            #     torch.save(model.state_dict(), 'experiments_logs/finetunemodel/' + arch + '_model.pt')
            #     torch.save(classifier.state_dict(), 'experiments_logs/finetunemodel/' + arch + '_classifier.pt')
            # total_f1.append(F1)
            """在目标域测试集上进行测试"""
            logger.debug('\nTest on Target datasts test set')
            # model.load_state_dict(torch.load('experiments_logs/finetunemodel/' + arch + '_model.pt'))
            # classifier.load_state_dict(torch.load('experiments_logs/finetunemodel/' + arch + '_classifier.pt'))

            # 返回测试损失、准确率
            # performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
            test_loss, test_acc = model_test(model, test_dl, device, training_mode, classifier=classifier)
        print('Best Testing: test_loss=%.4f| test_acc = %.4f' % (test_loss, test_acc))

    logger.debug("\n################## Training is Done! #########################")


def model_pretrain(model, model_optimizer, train_loader, params_keep, device):
    """预训练模块"""
    total_loss = []
    model.train()

    # 开始一次循环
    for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_loader):
        # data:[batch,1,length],label:[batch]
        data, labels = data.float().to(device), labels.long().to(device)
        # aug1 = data : [batch, 1, length]
        aug1 = aug1.float().to(device)
        # data_f = aug1_f : [batch, 1, length]
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)

        # optimizer
        model_optimizer.zero_grad()

        """产生embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

        fea_concat = torch.cat((z_t, z_f), dim=1)
        # 将合并的特征图展平后输出，可用于其余分类器的训练
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)

        """计算预训练损失"""
        """NTXentLoss：归一化温度标度交叉熵损失。来自SimCLR"""
        nt_xent_criterion = NTXentLoss_poly(device, params_keep["batch_size"], params_keep["temperature"],
                                       params_keep["use_cosine_similarity"])

        # 计算时域的损失
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        # 计算频域的损失
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        # 计算时频空间里的时域和频域损失
        l_TF = nt_xent_criterion(z_t, z_f)
        # 计算时频空间里时域和频域增强的损失，时域增强和频域的损失，时域增强和频域增强的损失
        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
        # 根据Triplet loss，改进视频空间的总损失
        loss_c = (1+l_TF-l_1) + (1+l_TF-l_2) + (1+l_TF-l_3)
        # 对时域、频域和时频空间的损失进行加权和
        lam = 0.2
        loss = lam*(loss_t + loss_f) + (1-lam)*loss_c
        # 记录损失
        total_loss.append(loss.item())
        # 损失反向传递
        loss.backward()
        # 优化一步
        model_optimizer.step()

    # 求总损失，为所有损失的均值
    total_loss = torch.tensor(total_loss).mean()

    return total_loss, fea_concat_flat


def model_pretrain_test(emb_finetune, label_finetune, emb_test, label_test):
    """
    预训练阶段的测试函数
    Args:
        emb_finetune: 训练KNN的embbeddings
        label_finetune: 训练KNN的label
        emb_test: 测试KNN的embbeddings
        label_test: 测试KNN的label

    Returns:knn的测试准确率

    """
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(emb_finetune.detach().cpu().numpy(), label_finetune)
    knn_acc_train = neigh.score(emb_finetune.detach().cpu().numpy(), label_finetune)
    print('KNN finetune acc:', knn_acc_train)
    # 测试KNN下游分类器
    representation_test = emb_test.detach().cpu().numpy()
    knn_result = neigh.predict(representation_test)
    print(classification_report(label_test, knn_result, digits=4))
    print(confusion_matrix(label_test, knn_result))
    knn_acc = accuracy_score(label_test, knn_result)
    return knn_acc


def get_emb(model, dl, device):
    """
    获取embbeddings
    Args:
        model: 输入的模型
        dl: 需要映射的数据集
        device: 设备

    Returns: embbeddings和label

    """
    model.eval()
    emb_all = []
    trgs = np.array([])
    with torch.no_grad():
        for data, labels, aug1, data_f, aug1_f in dl:
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)
            h_t, z_t, h_f, z_f = model(data, data_f)

            fea_concat = torch.cat((z_t, z_f), dim=1)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            emb_all.append(fea_concat_flat)
            trgs = np.append(trgs, labels.data.cpu().numpy())
    return emb_all, trgs


def model_finetune(model, val_dl, params_keep, device, model_optimizer, classifier=None, classifier_optimizer=None):
    """微调模块"""
    model.train()
    classifier.train()

    total_loss = []
    total_acc = []
    criterion = nn.CrossEntropyLoss()

    for data, labels, aug1, data_f, aug1_f in val_dl:
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)

        # """if random initialization:"""
        # model_optimizer.zero_grad()
        # model_F_optimizer.zero_grad()

        """产生embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

        """计算微调损失"""
        """NTXentLoss：归一化温度标度交叉熵损失。来自SimCLR"""

        nt_xent_criterion = NTXentLoss_poly(device, params_keep["target_batch_size"], params_keep["temperature"],
                                            params_keep["use_cosine_similarity"])
        # 计算时域损失
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        # 计算频域损失
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        # 计算时频空间内的损失
        l_TF = nt_xent_criterion(z_t, z_f)
        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug,
                                                                                                            z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        """添加监督分类器：1）它是微调所特有的。2） 该分类器也将用于测试"""
        # 合并时域和频域的信息
        fea_concat = torch.cat((z_t, z_f), dim=1)
        # 此处定义为MLP
        predictions = classifier(fea_concat)

        # 计算监督性的多分类交叉熵损失
        loss_p = criterion(predictions, labels)

        # 所有的损失的加权和
        lam = 0.2
        loss = loss_p + (1-lam)*loss_c + lam*(loss_t + loss_f)

        # 计算准确率
        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()

        total_acc.append(acc_bs)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

    # 求取均值
    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_test(model, test_dl, device, training_mode, classifier=None):
    """测试模块"""
    # 将预训练模型和分类器置于评估模式，即不进行梯度传递
    model.eval()
    classifier.eval()
    total_loss = []
    total_acc = []
    # 定义多分类交叉熵损失
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels, _, data_f, _ in test_dl:
            # print('TEST: {} of target samples'.format(labels.shape[0]))
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            """添加监督分类器：1）它是微调所特有的。2） 该分类器也将用于测试"""
            h_t, z_t, h_f, z_f = model(data, data_f)
            # 合并模型输出特征图
            fea_concat = torch.cat((z_t, z_f), dim=1)

            # 输出MLP预测的结果
            predictions_test = classifier(fea_concat)

            if training_mode != "pre_train":
                # 计算分类损失
                loss = criterion(predictions_test, labels)
                # 计算准确度
                acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
                # 将acc,auc添加到各自的列表里
                total_acc.append(acc_bs)
                total_loss.append(loss.item())
    # 取均值
    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc