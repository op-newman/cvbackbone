# CV Backbone Framework

A modular deep learning framework for computer vision tasks.

## Features
- Support for LeNet, ResNet and custom models
- Configurable training pipeline
- Detailed logging system

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train LeNet on MNIST
python tools/train.py --config configs/lenet.yaml
```

## Project Structure
```
├── configs/       # Configuration files
├── data/          # Data loading modules
├── engine/        # Training core
├── models/        # Model implementations
├── tools/         # Scripts
└── utils/         # Utilities
```