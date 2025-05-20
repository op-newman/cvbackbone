# file description:
# This file contains the implementation of LeNet model architecture.
# 🚀the file lenet.py has been created by PB on 2025/05/20 15:11:37

import torch.nn as nn
from .base_model import BaseModel
import torch

class LeNet(BaseModel):
    def __init__(self, config):
        super().__init__()
        in_channels = 1 if config.get('grayscale', False) else 3
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, config['num_classes'])
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x