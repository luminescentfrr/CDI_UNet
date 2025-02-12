import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.cdiunet import CDI_UNet

from engine import *
import os
import sys
import time
import warnings
from utils import *
from configs.config_setting import *

warnings.filterwarnings("ignore")

def main(config):
    """
    主训练函数
    Args:
        config: 配置对象
    """
    print('#----------创建工作目录----------#')
    # 创建必要的目录
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)
    
    # 创建日志记录器和TensorBoard写入器
    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')
    
    # 记录配置信息
    log_config_info(config, logger)
    
    print('#----------GPU初始化----------#')
    # 设置GPU和随机种子
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()
    
    print('#----------准备数据集----------#')
    # 创建训练集数据加载器
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        drop_last=True,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    
    # 创建验证集数据加载器
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        drop_last=False,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    
    print('#----------创建模型----------#')
    # 创建模型
    model = CDI_UNet(
        in_channels=config.input_channels,
        num_classes=config.num_classes,
        base_c=config.model_config['base_channels']
    )
    
    # 移动模型到GPU
    model = model.cuda()
    
    # 如果使用分布式训练
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.gpu],
            find_unused_parameters=True
        )
    
    print('#----------准备训练----------#')
    # 创建优化器和学习率调度器
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # 获取损失函数
    criterion = config.criterion
    
    # 初始化训练状态
    start_epoch = 0
    best_loss = float('inf')
    best_epoch = 0
    step = 0
    
    # 如果存在检查点，则加载
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']
        logger.info(f'Resume from epoch {start_epoch}')
    
    print('#----------开始训练----------#')
    # 主训练循环
    for epoch in range(start_epoch, config.epochs):
        # 训练一个epoch
        train_loss, step = train_one_epoch(
            train_loader, model, criterion, optimizer,
            scheduler, epoch, step, logger, config, writer
        )
        
        # 验证
        if (epoch + 1) % config.val_interval == 0:
            val_loss = validate_one_epoch(
                val_loader, model, criterion,
                epoch, logger, config, writer
            )
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                if config.save_best:
                    torch.save(model.state_dict(),
                             os.path.join(checkpoint_dir, 'best.pth'))
        
        # 保存最新模型
        if config.save_last:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'step': step,
                'best_loss': best_loss,
                'best_epoch': best_epoch
            }, os.path.join(checkpoint_dir, 'latest.pth'))
        
        # 清理GPU缓存
        if (epoch + 1) % config.clean_cache_interval == 0:
            torch.cuda.empty_cache()
    
    print('#----------训练完成----------#')
    # 使用最佳模型进行测试
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------开始测试----------#')
        # 加载最佳模型
        best_weight = torch.load(
            os.path.join(checkpoint_dir, 'best.pth'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(best_weight)
        
        # 测试
        test_loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config
        )
        
        # 重命名最佳模型文件
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, 
                        f'best-epoch{best_epoch}-loss{best_loss:.4f}.pth')
        )
        
        # 记录最终结果
        log_info = f'训练完成. 最佳验证loss: {best_loss:.4f} ' \
                  f'(Epoch {best_epoch}). 测试loss: {test_loss:.4f}'
        logger.info(log_info)

def init_distributed_mode(args):
    """
    初始化分布式训练设置
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    torch.distributed.barrier()

if __name__ == '__main__':
    # 获取配置
    config = setting_config
    
    # 如果使用分布式训练
    if config.distributed:
        init_distributed_mode(config)
    
    # 开始训练
    main(config)