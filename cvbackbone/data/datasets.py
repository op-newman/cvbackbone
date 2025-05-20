from torchvision import datasets
from torch.utils.data import DataLoader
from .transforms import build_transforms, get_default_transforms
from typing import Dict, Tuple

class MNISTDataLoader:
    @staticmethod
    def create(config):
        # 如果配置中没有transform，使用默认转换
        if 'transform' not in config:
            transforms = get_default_transforms()
        else:
            transforms = build_transforms(config['transform'])
        
        train_set = datasets.MNIST(
            root=config['root'],
            train=True,
            download=True,
            transform=transforms['train']
        )
        
        val_set = datasets.MNIST(
            root=config['root'],
            train=False,
            download=True,
            transform=transforms['val']
        )
        
        return (
            DataLoader(train_set, 
                      batch_size=config['batch_size'], 
                      shuffle=True,
                      num_workers=config.get('num_workers', 0)),
            DataLoader(val_set, 
                      batch_size=config['batch_size'], 
                      shuffle=False,
                      num_workers=config.get('num_workers', 0))
        )

# 数据集注册表
DATASETS = {
    'mnist': MNISTDataLoader
}

def get_dataloader(config: Dict):
    """
    根据配置获取数据加载器
    Args:
        config: 必须包含'name'字段指定数据集类型
    Returns:
        (train_loader, val_loader)
    """
    dataset_name = config['name'].lower()
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASETS[dataset_name].create(config)