from torchvision import transforms
from utils import *
from datetime import datetime
import os

class setting_config:
    """
    训练设置的配置类
    """
    # 模型配置
    network = 'cdiunet'  # 使用的网络模型
    model_config = {
        'base_channels': 32,        # 基础通道数
        'reduction_ratio': 8,       # 降低通道压缩比
        'min_spatial_size': 8,      # 最小特征图大小
        'downsample_ratio': 4,      # 空间降采样比例
        'num_heads': 4,
        'kernel_size': 3,
        'attn_drop': 0.1,
        'proj_drop': 0.1,
    }

    # 数据集配置
    datasets = 'isic17'           
    if datasets == 'isic17':
        data_path = '.CDIUNet/data/isic2017/'
    elif datasets == 'isic18':
        data_path = '.CDIUNet/data/isic2018/'
    else:
        raise Exception('datasets is not right!')
    
    input_channels = 3             # 输入通道数
    num_classes = 1                # 类别数
    input_size_h = 256            # 输入高度
    input_size_w = 256            # 输入宽度

    # 训练配置
    epochs = 200                  # 总轮数
    batch_size = 2                  # 进一步减小批次大小
    num_workers = 2               # 数据加载线程数
    print_interval = 10           # 打印间隔
    save_interval = 50   
    base_dir = './CDIUNet/results'
    work_dir = os.path.join(base_dir, 
                           f'{network}_{datasets}_{datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss")}/')      # 工作目录
    gpu_id = '0'                  # GPU ID
    seed = 42                     # 随机种子
    distributed = False           # 是否使用分布式训练

    # 数据增强配置
    train_transformer = transforms.Compose([
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[-30, 30]),
        myNormalize(datasets, train=True),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])
    
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    # 优化器配置
    opt = 'AdamW'                # 使用AdamW优化器
    lr = 1e-4                    # 学习率
    betas = (0.9, 0.999)         # Adam/AdamW的beta参数
    eps = 1e-8                   # 数值稳定性参数
    weight_decay = 1e-2          # 权重衰减
    amsgrad = False              # 是否使用AMSGrad变体

    # 学习率调度器配置
    sch = 'CosineAnnealingLR'    # 使用余弦退火
    T_max = 50                   # 调整周期
    eta_min = 1e-6               # 最小学习率
    last_epoch = -1              # 上一轮epoch
    warm_up_epochs = 10          # 预热轮数

    # 损失函数配置
    criterion = GT_BceDiceLoss(wb=1.0, wd=1.0)  # 使用组合损失函数
    aux_loss_weights = {         # 辅助损失权重
        'level5': 0.5,
        'level4': 0.4,
        'level3': 0.3,
        'level2': 0.2,
        'level1': 0.1
    }

    # 训练优化配置
    gradient_accumulation_steps = 8  # 增加梯度累积步数
    max_grad_norm = 1.0             # 梯度裁剪阈值
    amp = True                      # 启用自动混合精度

    # 内存优化配置
    pin_memory = True
    prefetch_factor = 2
    persistent_workers = True       # 持久化工作进程
    clean_cache_interval = 5        # 缓存清理间隔
    tensorboard_log_freq = 20       # tensorboard记录频率
    empty_cache_freq = 50           # 定期清理GPU缓存的频率

    # 模型检查点配置
    use_checkpoint = True           # 使用梯度检查点
    save_best = True               # 是否保存最佳模型
    save_last = True               # 是否保存最新模型

    # 验证和测试配置
    val_interval = 1               # 验证间隔
    test_interval = 5              # 测试间隔

    
