## 文档说明
### 相关脚本
    augmentations.py ---数据增强
    dataloader.py ---数据加载
    loss.py ---损失函数
    model.py ---模型架构
    trainer.py ---训练器
    utils.py ---记录器等工具脚本
### nni 架构文件
    main_nni.py ---nni架构超参数调优
    config.yaml ---nni配置文件

### 寻优后模型训练脚本
    main.py ---确定超参数后训练测试主程序


## 具体的流程
    终端输入 nnictl create --config config.yaml --port 8080
    等待参数寻优完成
    找到最优配置后，复制配置至main.py文件，执行main.py文件。
    查验结果后保存模型
