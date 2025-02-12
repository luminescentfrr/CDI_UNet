import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_logger(name, log_dir):
    """
    创建日志记录器
    Args:
        name(str): 日志记录器名称
        log_dir(str): 日志保存路径
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(
        info_name,
        when='D',
        encoding='utf-8'
    )
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    return logger

def log_config_info(config, logger):
    """记录配置信息到日志"""
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if not k.startswith('_'):
            logger.info(f'{k}: {v}')

# 损失函数相关类
class BCELoss(nn.Module):
    """二元交叉熵损失"""
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, pred, aux_outputs, target):
        if aux_outputs is None:
            return self.bceloss(pred, target)
        
        # 计算主输出损失
        main_loss = self.bceloss(pred, target)
        
        # 计算辅助输出损失
        aux_loss = sum(self.bceloss(aux, target) for aux in aux_outputs)
        
        return main_loss + 0.4 * aux_loss

class DiceLoss(nn.Module):
    """Dice损失"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, aux_outputs, target):
        if aux_outputs is None:
            return self._dice_loss(pred, target)
        
        # 计算主输出损失
        main_loss = self._dice_loss(pred, target)
        
        # 计算辅助输出损失
        aux_loss = sum(self._dice_loss(aux, target) for aux in aux_outputs)
        
        return main_loss + 0.4 * aux_loss

    def _dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1.0
        
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        
        intersection = (pred_flat * target_flat).sum(1)
        unionset = pred_flat.sum(1) + target_flat.sum(1)
        loss = 1 - (2 * intersection + smooth) / (unionset + smooth)
        
        return loss.mean()

class GT_BceDiceLoss(nn.Module):
    """组合损失函数，支持CDIUNet的多输出"""
    def __init__(self, wb=1.0, wd=1.0):
        super(GT_BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, main_output, aux_outputs, target):
        if aux_outputs is None:
            bce_loss = self.bce(main_output, None, target)
            dice_loss = self.dice(main_output, None, target)
        else:
            bce_loss = self.bce(main_output, aux_outputs, target)
            dice_loss = self.dice(main_output, aux_outputs, target)
        
        return self.wb * bce_loss + self.wd * dice_loss

# 数据转换相关类
class myTransform:
    """基础转换类"""
    def __init__(self):
        pass

    def __call__(self, data):
        raise NotImplementedError

class myToTensor(myTransform):
    """转换为Tensor"""
    def __call__(self, data):
        image, mask = data
        
        # 处理图像
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        
        # 处理掩码
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        mask = torch.from_numpy(mask.transpose((2, 0, 1)))
        
        return image, mask

class myResize(myTransform):
    """调整大小"""
    def __init__(self, size_h, size_w):
        self.size_h = size_h
        self.size_w = size_w
        
    def __call__(self, data):
        image, mask = data
        return (TF.resize(image, [self.size_h, self.size_w]), 
                TF.resize(mask, [self.size_h, self.size_w]))

class myRandomHorizontalFlip(myTransform):
    """随机水平翻转"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        return image, mask

class myRandomVerticalFlip(myTransform):
    """随机垂直翻转"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        return image, mask

class myRandomRotation(myTransform):
    """随机旋转"""
    def __init__(self, p=0.5, degree=[-30, 30]):
        self.p = p
        self.degree = degree
        
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            return TF.rotate(image, angle), TF.rotate(mask, angle)
        return image, mask

class myNormalize(myTransform):
    """标准化"""
    def __init__(self, data_name, train=True):
        self.mean = 153.2975
        self.std = 29.364

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / (self.std + 1e-6)
        img_normalized = ((img_normalized - np.min(img_normalized)) / 
                        (np.max(img_normalized) - np.min(img_normalized) + 1e-6))
        return img_normalized, msk

def save_imgs(images, targets, pred, batch_idx, save_path, threshold=0.5):
    """保存图像、目标和预测结果的可视化
    Args:
        images: 输入图像
        targets: 目标掩码
        pred: 模型预测结果
        batch_idx: 批次索引
        save_path: 保存路径
        threshold: 二值化阈值，默认0.5
    """
    # 确保阈值是浮点数
    if isinstance(threshold, str):
        threshold = 0.5  # 如果传入字符串，使用默认值
    else:
        threshold = float(threshold)
    
    # 转换为 NumPy 数组并确保数据类型
    pred = pred.detach().cpu().numpy()
    pred = np.squeeze(pred)  # 移除 batch 和 channel 维度
    pred = pred.astype(np.float32)  # 确保使用 float32 类型
    pred_binary = (pred >= threshold).astype(np.uint8)
    
    if torch.is_tensor(images):
        images = images.detach().cpu().numpy()
        images = np.transpose(images[0], (1, 2, 0))  # 将 channel 维度移到最后
        images = images.astype(np.float32)  # 确保使用 float32 类型
    
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
        targets = np.squeeze(targets)  # 移除多余维度
        targets = targets.astype(np.uint8)  # 转换为 uint8 类型
    
    # 创建图像网格
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(images)
    plt.axis('off')
    
    # 显示目标掩码
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(targets, cmap='gray')
    plt.axis('off')
    
    # 显示预测结果
    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(pred_binary, cmap='gray')
    plt.axis('off')
    
    # 保存图像
    plt.savefig(os.path.join(save_path, f'batch_{batch_idx}.png'))
    plt.close()

def get_optimizer(config, model):
    """获取优化器"""
    if config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    else:
        raise ValueError(f'Optimizer {config.opt} not supported')

def get_scheduler(config, optimizer):
    """获取学习率调度器"""
    if config.sch == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    else:
        raise ValueError(f'Scheduler {config.sch} not supported')