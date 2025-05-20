import torch
from tqdm import tqdm
from typing import Dict, Optional
from cvbackbone.utils.logger import TrainLogger

class Trainer:
    def __init__(self, model, config, logger: Optional[TrainLogger] = None):
        """
        初始化训练器
        Args:
            model: 要训练的模型
            config: 训练配置字典
            logger: 日志记录器实例
        """
        self.model = model.to(config['device'])
        self.config = config
        self.logger = logger
        self.device = config['device']
        
        # 初始化损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # 构建优化器和调度器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler() if 'lr_scheduler' in config else None
        
        # 记录初始化信息
        self._log_init()

    def _log_init(self):
        """记录初始化信息"""
        if self.logger:
            self.logger.log_message(f"Training on device: {self.device}")
            self.logger.log_message(f"Optimizer: {self.config['optimizer']}")
            if self.scheduler:
                self.logger.log_message(f"LR Scheduler: {self.config['lr_scheduler']}")
            self.logger.log_message("-" * 50)

    def _build_optimizer(self):
        """构建优化器"""
        cfg = self.config['optimizer']
        if cfg['type'] == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg['lr'],
                weight_decay=cfg.get('weight_decay', 0)
            )
        elif cfg['type'] == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg['lr'],
                momentum=cfg.get('momentum', 0),
                weight_decay=cfg.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg['type']}")
        
        if self.logger:
            self.logger.log_message(f"Initialized {cfg['type']} optimizer with lr={cfg['lr']}")
        return optimizer

    def _build_scheduler(self):
        """构建学习率调度器"""
        cfg = self.config['lr_scheduler']
        if cfg['type'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg['step_size'],
                gamma=cfg.get('gamma', 0.1)
            )
        elif cfg['type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.get('T_max', 10),
                eta_min=cfg.get('eta_min', 0)
            )
        elif cfg['type'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=cfg.get('factor', 0.1),
                patience=cfg.get('patience', 5)
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg['type']}")
        
        if self.logger:
            self.logger.log_message(f"Initialized {cfg['type']} LR scheduler")
        return scheduler

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        if self.logger:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_message(f"Starting epoch with LR={current_lr:.6f}")
        
        progress_bar = tqdm(train_loader, desc="Training")
        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条描述
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        
        # 更新学习率
        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
            if self.logger:
                new_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_message(f"LR updated to {new_lr:.6f}")
        
        return avg_loss

    def validate(self, val_loader):
        """验证模型性能"""
        self.model.eval()
        correct = 0
        total_loss = 0
        
        if self.logger:
            self.logger.log_message("Running validation...")
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 计算损失
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 计算准确率
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        
        # 更新Plateau调度器
        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(accuracy)
            if self.logger:
                new_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_message(
                    f"Plateau scheduler updated (Acc: {accuracy:.2%}, New LR: {new_lr:.6f})"
                )
        
        if self.logger:
            self.logger.log_message(
                f"Validation results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}"
            )
        
        return avg_loss, accuracy