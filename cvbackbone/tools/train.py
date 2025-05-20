# file description:
# This file is used to train the LeNet model on the MNIST dataset.
# 🚀the file train.py has been created by PB on 2025/05/20 16:39:18

import yaml
import torch
from tqdm import tqdm
from pathlib import Path
from cvbackbone.models import ModelFactory
from cvbackbone.data.datasets import get_dataloader
from cvbackbone.engine.trainer import Trainer
from cvbackbone.utils.logger import TrainLogger

def main():
    # 初始化日志系统
    logger = TrainLogger(log_dir="../runs/lenet", name="lenet_mnist")
    
    # 加载配置
    try:
        config_path = "configs/lenet.yaml"
        with open(config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.log_message(f"Loaded config from {config_path}")
    except Exception as e:
        logger.log_message(f"Error loading config: {str(e)}", "error")
        raise
    
    # 记录配置和模型信息
    logger.log_config(config)
    
    # 初始化模型
    try:
        model = ModelFactory.create(config['model'])
        logger.log_model_info(model)
    except Exception as e:
        logger.log_message(f"Model initialization failed: {str(e)}", "error")
        raise
    
    # 获取数据加载器
    try:
        train_loader, val_loader = get_dataloader(config['data'])
        logger.log_message(
            f"Data loaded: {len(train_loader.dataset)} train, "
            f"{len(val_loader.dataset)} val samples"
        )
    except Exception as e:
        logger.log_message(f"Data loading failed: {str(e)}", "error")
        raise
    
    # 初始化训练器
    trainer = Trainer(model, config['training'])
    
    # 训练准备
    best_acc = 0.0
    save_dir = Path("../runs/lenet/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    logger.log_message("Starting training...")
    try:
        for epoch in tqdm(range(config['training']['epochs']), desc="Training"):
            # 训练阶段
            train_loss = trainer.train_epoch(train_loader)
            
            # 验证阶段
            val_loss, val_acc = trainer.validate(val_loader)
            
            # 记录日志
            logger.log_epoch(epoch+1, train_loss, val_loss, val_acc)
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                model_path = save_dir / f"best_model_epoch{epoch+1}_acc{val_acc:.4f}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }, model_path)
                logger.log_checkpoint(str(model_path), f"(Acc: {val_acc:.2%})")
    
    except Exception as e:
        logger.log_message(f"Training interrupted: {str(e)}", "error")
        raise
    
    finally:
        logger.close()

if __name__ == "__main__":
    main()