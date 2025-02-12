from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torch
import random
import cv2

class NPY_datasets(Dataset):
    """数据集加载类，支持CDI_UNet的图像分割任务"""
    def __init__(self, path_Data, config, train=True):
        """
        初始化数据集
        Args:
            path_Data: 数据集根目录
            config: 配置对象
            train: 是否为训练集
        """
        super(NPY_datasets, self).__init__()
        
        self.path_Data = path_Data
        self.train = train
        self.config = config
        
        # 验证数据目录
        self._validate_directories()
        
        # 预先验证所有图像
        self.valid_pairs = self._get_valid_pairs()
        print(f"Found {len(self.valid_pairs)} valid image pairs")

    def _validate_directories(self):
        """验证数据目录结构"""
        if self.train:
            self.img_path = os.path.join(self.path_Data, 'train/images/')
            self.mask_path = os.path.join(self.path_Data, 'train/masks/')
        else:
            self.img_path = os.path.join(self.path_Data, 'val/images/')
            self.mask_path = os.path.join(self.path_Data, 'val/masks/')
            
        if not os.path.exists(self.img_path) or not os.path.exists(self.mask_path):
            raise RuntimeError(f"Data directories not found: {self.img_path} or {self.mask_path}")

    def _get_valid_pairs(self):
        """预先验证所有图像对"""
        valid_pairs = []
        images = sorted([f for f in os.listdir(self.img_path) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))])
        masks = sorted([f for f in os.listdir(self.mask_path) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_name, mask_name in zip(images, masks):
            img_path = os.path.join(self.img_path, img_name)
            mask_path = os.path.join(self.mask_path, mask_name)
            
            try:
                # 验证图像是否可以正确加载
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(mask_path) as mask:
                    mask.verify()
                valid_pairs.append((img_path, mask_path))
            except Exception as e:
                print(f"Skipping invalid pair ({img_name}, {mask_name}): {str(e)}")
                continue
                
        return valid_pairs

    def __getitem__(self, index):
        """获取数据项，不使用递归"""
        if not 0 <= index < len(self.valid_pairs):
            raise IndexError(f"Index {index} out of range")
            
        img_path, mask_path = self.valid_pairs[index]
        
        try:
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # 转换为tensor
            img = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
            mask = torch.FloatTensor(np.array(mask)).unsqueeze(0) / 255.0
            
            return img, mask
            
        except Exception as e:
            # 发生错误时返回一个默认值，而不是递归
            print(f"Error loading data at index {index}: {str(e)}")
            # 返回零张量作为替代
            return torch.zeros((3, 256, 256)), torch.zeros((1, 256, 256))
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def _load_image(self, path):
        """加载图像文件"""
        try:
            # 使用PIL加载图像
            img = Image.open(path).convert('RGB')
            img = np.array(img)
            
            # 标准化到[0,1]范围
            if img.max() > 1:
                img = img / 255.0
                
            return img
            
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            raise
    
    def _load_mask(self, path):
        """加载掩码文件"""
        try:
            # 使用PIL加载掩码
            mask = Image.open(path).convert('L')
            mask = np.array(mask)
            
            # 将掩码转换为二值图像
            mask = (mask > 127).astype(np.float32)
            
            # 添加通道维度
            mask = np.expand_dims(mask, axis=2)
            
            return mask
            
        except Exception as e:
            print(f"Error loading mask {path}: {str(e)}")
            raise

def get_dataloader(config, train=True):
    """获取数据加载器，优化以适应CDI_UNet"""
    # 创建数据集
    dataset = NPY_datasets(config.data_path, config, train)
    
    # 创建数据加载器
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size if train else 1,
        shuffle=train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=train,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    
    return loader

def get_mean_std(dataset):
    """计算数据集的均值和标准差"""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    
    return mean, std

class AugmentedDataset(Dataset):
    """带有额外数据增强的数据集类"""
    def __init__(self, base_dataset, augment_factor=2):
        """
        初始化增强数据集
        Args:
            base_dataset: 基础数据集
            augment_factor: 增强倍数
        """
        self.base_dataset = base_dataset
        self.augment_factor = augment_factor
        
    def __getitem__(self, index):
        """获取增强后的数据样本"""
        # 计算原始数据集的索引
        base_idx = index // self.augment_factor
        aug_idx = index % self.augment_factor
        
        # 获取原始数据
        img, mask = self.base_dataset[base_idx]
        
        # 根据aug_idx应用不同的增强
        if aug_idx > 0:
            img, mask = self._apply_extra_augmentation(img, mask, aug_idx)
            
        return img, mask
    
    def __len__(self):
        """返回增强后的数据集大小"""
        return len(self.base_dataset) * self.augment_factor
    
    def _apply_extra_augmentation(self, img, mask, aug_type):
        """应用额外的数据增强"""
        if aug_type == 1:
            # 添加高斯噪声
            noise = torch.randn_like(img) * 0.1
            img = img + noise
            img = torch.clamp(img, 0, 1)
        
        return img, mask