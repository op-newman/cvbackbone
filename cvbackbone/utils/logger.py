import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import torch

class TrainLogger:
    def __init__(self, log_dir: str = "logs", name: Optional[str] = None):
        """
        初始化训练日志系统
        Args:
            log_dir: 日志保存目录
            name: 实验名称(用于创建子目录)
        """
        # 创建日志目录
        self.log_dir = Path(log_dir)
        if name:
            self.log_dir = self.log_dir / name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志格式
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        
        # 文件日志
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"train_{current_time}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 训练统计信息
        self.stats = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "best_acc": 0.0,
            "start_time": datetime.now()
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_acc: float):
        """记录每个epoch的训练信息"""
        self.stats["train_loss"].append(train_loss)
        self.stats["val_loss"].append(val_loss)
        self.stats["val_acc"].append(val_acc)
        
        if val_acc > self.stats["best_acc"]:
            self.stats["best_acc"] = val_acc
        
        self.logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"Best Acc: {self.stats['best_acc']:.2%}"
        )
    
    def log_config(self, config: Dict[str, Any]):
        """记录训练配置"""
        self.logger.info("Training Configuration:")
        for section, params in config.items():
            self.logger.info(f"[{section.upper()}]")
            for k, v in params.items():
                self.logger.info(f"{k}: {v}")
        self.logger.info("-" * 50)
    
    def log_model_info(self, model: torch.nn.Module):
        """记录模型信息"""
        self.logger.info("Model Architecture:")
        self.logger.info(str(model))
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total Parameters: {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.logger.info("-" * 50)
    
    def log_checkpoint(self, path: str, message: str = ""):
        """记录模型保存点"""
        self.logger.info(f"Checkpoint saved to {path} {message}")
    
    def log_message(self, message: str, level: str = "info"):
        """通用日志方法"""
        if level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message)
    
    def close(self):
        """结束训练时调用"""
        training_time = datetime.now() - self.stats["start_time"]
        self.logger.info(
            f"Training completed in {str(training_time).split('.')[0]} | "
            f"Best Val Acc: {self.stats['best_acc']:.2%}"
        )