import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from cvbackbone.models import ModelFactory
from cvbackbone.data.datasets import get_dataloader
from cvbackbone.engine.trainer import Trainer
from cvbackbone.utils.logger import TrainLogger
from datetime import datetime

plt.style.use('seaborn')

def deep_merge(base_dict, override_dict):
    """深度合并字典（支持嵌套）"""
    result = base_dict.copy()
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path):
    """加载并合并配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    if '_base_' in config:
        base_path = Path(config_path).parent / config.pop('_base_')
        base_config = load_config(base_path)
        config = deep_merge(base_config, config)
    
    return config

def override_config(config, args):
    """命令行参数覆盖配置"""
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['optimizer']['lr'] = args.lr
    if args.device is not None:
        config['training']['device'] = args.device
    return config

class Visualizer:
    def __init__(self, config, logger=None):
        self.save_dir = Path(config['visualization']['save_dir'])
        self.colors = config['visualization']['colors']
        self.figsize = config['visualization']['figure_size']
        self.dpi = config['visualization']['dpi']
        self.formats = config['visualization']['save_formats']
        self.logger = logger
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.logger:
            self.logger.log_message(f"Visualizer initialized at {self.save_dir}")

        model_name = config['model']['name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{model_name}_{timestamp}"
        self.save_dir = self.save_dir / exp_name
    def save_curves(self, train_losses, val_losses, val_accs, prefix="lenet"):
        """保存训练曲线（使用Matplotlib默认颜色）"""
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # 绘制训练/验证损失曲线 (ax1) - 不指定颜色，自动分配
        ax1.plot(epochs, train_losses, 'o-', label='Train Loss')  # 自动使用第一种颜色
        ax1.plot(epochs, val_losses, 'o-', label='Val Loss')     # 自动使用第二种颜色
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 绘制验证准确率曲线 (ax2) - 自动分配新颜色
        ax2.plot(epochs, val_accs, 'o-', label='Val Accuracy')   # 自动使用第三种颜色
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        for fmt in self.formats:
            save_path = self.save_dir / f"{prefix}_curves.{fmt}"
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            if self.logger:
                self.logger.log_message(f"Saved visualization: {save_path}")
        
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='configs/lenet.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Override training epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--device', help='Override device (e.g. "cuda:0")')
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        # 加载配置
        config = load_config(args.config)
        config = override_config(config, args)

        # setup the exp path
        model_name = config['model']['name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{model_name}_{timestamp}"
        
        logger = TrainLogger(
            log_dir=config['logging']['log_dir'],
            name=exp_name
        )
        logger.log_config(config)
        
        # 初始化其他组件
        visualizer = Visualizer(config, logger)
        model = ModelFactory.create(config['model'])
        logger.log_model_info(model)
        
        train_loader, val_loader = get_dataloader(config['data'])
        logger.log_message(
            f"Data loaded: {len(train_loader.dataset)} train, "
            f"{len(val_loader.dataset)} val samples"
        )
        
        # 训练循环
        trainer = Trainer(model, config['training'], logger)
        stats = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in tqdm(range(config['training']['epochs'])):
            stats['train_loss'].append(trainer.train_epoch(train_loader))
            val_loss, val_acc = trainer.validate(val_loader)
            stats['val_loss'].append(val_loss)
            stats['val_acc'].append(val_acc)
            logger.log_epoch(epoch+1, *[stats[k][-1] for k in ['train_loss', 'val_loss', 'val_acc']])
        
        # 保存结果
        visualizer.save_curves(stats['train_loss'], stats['val_loss'], stats['val_acc'])
        model_path = Path(config['visualization']['save_dir']) / exp_name / "final_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.log_message(f"Model saved to {model_path}")
        
    except Exception as e:
        if 'logger' in locals():
            logger.log_message(f"Fatal error: {str(e)}", level="error")
        else:
            print(f"Fatal error before logger initialized: {str(e)}")
        raise

if __name__ == "__main__":
    main()