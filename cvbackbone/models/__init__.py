# file description:
# 🚀the file __init__.py has been created by PB on 2025/05/20 15:10:40


import torch
from .lenet import LeNet

class ModelFactory:
    _models = {
        'lenet': LeNet
    }

    @staticmethod
    def create(config: dict):
        model_name = config['name'].lower()
        if model_name not in ModelFactory._models:
            raise ValueError(f"Unsupported model: {model_name}")
        return ModelFactory._models[model_name](config)