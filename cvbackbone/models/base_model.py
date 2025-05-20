# file description:
# This file contains the base model class. All models should inherit from this class.
# 🚀the file base_model.py has been created by PB on 2025/05/20 15:13:44


import torch.nn as nn
import torch
from typing import Dict, Tuple

class BaseModel(nn.Module):
    """
    所有模型的基类，定义标准接口和基础功能
    继承自该类的模型需要实现以下方法：
    1. forward(): 定义前向传播
    2. get_input_size(): 返回模型期望的输入尺寸
    """
    
    def __init__(self):
        super().__init__()
        self._initialized = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        必须实现的前向传播逻辑
        Args:
            x: 输入张量 (B, C, H, W)
        Returns:
            输出张量 (B, num_classes)
        """
        raise NotImplementedError("All models must implement forward()")
    
    def get_input_size(self) -> Tuple[int, int]:
        """
        返回模型期望的输入尺寸 (H, W)
        用于数据预处理和验证
        """
        raise NotImplementedError("All models must implement get_input_size()")
    
    def init_weights(self, init_cfg: Dict = None):
        """
        权重初始化 (可选实现)
        Args:
            init_cfg: 初始化配置字典，例如:
                {
                    'type': 'kaiming',  # 可选项: ['normal', 'xavier', 'kaiming', 'pretrained']
                    'mode': 'fan_out',
                    'pretrained_path': ''
                }
        """
        if init_cfg is None:
            return
            
        if init_cfg['type'] == 'pretrained':
            self.load_state_dict(torch.load(init_cfg['pretrained_path']))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if init_cfg['type'] == 'kaiming':
                        nn.init.kaiming_normal_(
                            m.weight, 
                            mode=init_cfg.get('mode', 'fan_out'),
                            nonlinearity='relu'
                        )
                    elif init_cfg['type'] == 'xavier':
                        nn.init.xavier_normal_(m.weight)
                    elif init_cfg['type'] == 'normal':
                        nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        self._initialized = True
    
    def get_num_params(self, trainable: bool = True) -> int:
        """
        计算模型参数量
        Args:
            trainable: 是否只计算可训练参数
        Returns:
            参数总量
        """
        return sum(p.numel() for p in self.parameters() if not trainable or p.requires_grad)
    
    def get_flops(self, input_size: Tuple[int, int] = None) -> float:
        """
        估算模型FLOPs (需要实现具体逻辑)
        Args:
            input_size: 输入尺寸 (H, W)
        Returns:
            浮点运算次数 (FLOPs)
        """
        raise NotImplementedError("FLOPs calculation not implemented")
        # 实际实现可以使用thop等库
    
    def save(self, path: str):
        """保存模型权重"""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, strict: bool = True):
        """加载模型权重"""
        self.load_state_dict(torch.load(path), strict=strict)