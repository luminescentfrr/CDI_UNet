import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import torch.nn.functional as F
from collections import defaultdict

class AverageMeter:
    """跟踪指标的计量器类"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_metrics(pred, target):
    """计算评估指标"""
    # 确保输入是布尔值或0/1值
    pred = (pred > 0.5).float()
    target = target.float()
    
    # 计算混淆矩阵元素
    tp = torch.logical_and(pred, target).sum().float()
    tn = torch.logical_and(~pred.bool(), ~target.bool()).sum().float()
    fp = torch.logical_and(pred, ~target.bool()).sum().float()
    fn = torch.logical_and(~pred.bool(), target).sum().float()
    
    # 计算指标
    epsilon = 1e-7  # 防止除零
    
    iou = tp / (tp + fp + fn + epsilon)
    f1_score = 2 * tp / (2 * tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    
    return {
        'iou': iou.item(),
        'f1_score': f1_score.item(),
        'accuracy': accuracy.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item()
    }

def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer):
    """训练一个epoch"""
    losses = AverageMeter()
    metrics_sum = defaultdict(list)
    scaler = GradScaler(enabled=config.amp)
    
    # 确保模型在训练模式
    model.train()
    
    # 添加进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
    
    for iter, (images, targets) in enumerate(pbar):
        step += 1
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        with autocast(enabled=config.amp):
            outputs = model(images)
            
            # 处理主输出和辅助输出
            if isinstance(outputs, tuple):
                main_output, aux_outputs = outputs[0], outputs[1:]
                loss = criterion(main_output, aux_outputs, targets)
            else:
                loss = criterion(outputs, None, targets)
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        if config.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # 优化器步进
        scaler.step(optimizer)
        scaler.update()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 更新损失记录
        losses.update(loss.item())
        
        # 计算评估指标
        if isinstance(outputs, tuple):
            pred = torch.sigmoid(main_output).detach()
        else:
            pred = torch.sigmoid(outputs).detach()
        
        metrics = calculate_metrics(pred, targets)
        for k, v in metrics.items():
            metrics_sum[k].append(v)
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'IoU': f'{np.mean(metrics_sum["iou"]):.4f}'
        })
        
        # 记录训练信息
        if writer is not None and step % config.print_interval == 0:
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], step)
            for k, v in metrics.items():
                writer.add_scalar(f'Metrics/{k}', v, step)
    
    pbar.close()
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items()}
    
    # 记录训练信息
    log_info = f'Training Epoch {epoch}:\n' \
               f'Loss: {losses.avg:.4f}\n' \
               f'IoU: {avg_metrics["iou"]:.4f}\n' \
               f'F1/DSC: {avg_metrics["f1_score"]:.4f}\n' \
               f'Accuracy: {avg_metrics["accuracy"]:.4f}\n' \
               f'Sensitivity: {avg_metrics["sensitivity"]:.4f}\n' \
               f'Specificity: {avg_metrics["specificity"]:.4f}'
    logger.info(log_info)
    
    return losses.avg, step

def validate_one_epoch(val_loader, model, criterion, epoch, logger, config, writer=None):
    """验证一个epoch"""
    losses = AverageMeter()
    metrics_sum = defaultdict(list)
    
    # 切换到评估模式
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} Validation')
        for i, (images, targets) in enumerate(pbar):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            # 使用混合精度推理
            with autocast(enabled=config.amp):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    main_output, aux_outputs = outputs[0], outputs[1:]
                    loss = criterion(main_output, aux_outputs, targets)
                    pred = torch.sigmoid(main_output)
                else:
                    loss = criterion(outputs, None, targets)
                    pred = torch.sigmoid(outputs)
            
            losses.update(loss.item())
            
            # 计算评估指标
            metrics = calculate_metrics(pred, targets)
            for k, v in metrics.items():
                metrics_sum[k].append(v)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'IoU': f'{np.mean(metrics_sum["iou"]):.4f}'
            })
            
            # 保存预测结果
            if i % config.save_interval == 0:
                save_imgs(images, targets, pred, i, 
                         config.work_dir + 'outputs/', 
                         config.datasets)
            
            # 记录验证信息
            if writer is not None and i % config.print_interval == 0:
                writer.add_scalar('Loss/val', loss.item(), 
                                epoch * len(val_loader) + i)
        
        pbar.close()
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items()}
    
    # 记录验证信息
    log_info = f'Validation Results:\n' \
               f'Loss: {losses.avg:.4f}\n' \
               f'IoU: {avg_metrics["iou"]:.4f}\n' \
               f'F1/DSC: {avg_metrics["f1_score"]:.4f}\n' \
               f'Accuracy: {avg_metrics["accuracy"]:.4f}\n' \
               f'Sensitivity: {avg_metrics["sensitivity"]:.4f}\n' \
               f'Specificity: {avg_metrics["specificity"]:.4f}'
    logger.info(log_info)
    
    return losses.avg

def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    """测试函数"""
    model.eval()
    losses = AverageMeter()
    metrics_sum = defaultdict(list)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for i, (images, targets) in enumerate(pbar):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            # 使用混合精度推理
            with autocast(enabled=config.amp):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    main_output, aux_outputs = outputs[0], outputs[1:]
                    loss = criterion(main_output, aux_outputs, targets)
                    pred = torch.sigmoid(main_output)
                else:
                    loss = criterion(outputs, None, targets)
                    pred = torch.sigmoid(outputs)
            
            losses.update(loss.item())
            
            # 计算评估指标
            metrics = calculate_metrics(pred, targets)
            for k, v in metrics.items():
                metrics_sum[k].append(v)
            
            # 保存预测结果
            if i % config.save_interval == 0:
                save_imgs(images, targets, pred, i, 
                         config.work_dir + 'outputs/', 
                         config.datasets,
                         test_data_name=test_data_name)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'IoU': f'{np.mean(metrics_sum["iou"]):.4f}'
            })
        
        pbar.close()
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items()}
    
    # 记录测试信息
    log_info = f'Test Results:\n' \
               f'Loss: {losses.avg:.4f}\n' \
               f'IoU: {avg_metrics["iou"]:.4f}\n' \
               f'F1/DSC: {avg_metrics["f1_score"]:.4f}\n' \
               f'Accuracy: {avg_metrics["accuracy"]:.4f}\n' \
               f'Sensitivity: {avg_metrics["sensitivity"]:.4f}\n' \
               f'Specificity: {avg_metrics["specificity"]:.4f}'
    logger.info(log_info)
    
    return losses.avg