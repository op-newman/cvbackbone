import torchvision.transforms as T
from typing import Dict, List, Optional

def build_transforms(transform_cfg: Dict[str, List[Dict]]) -> Dict[str, T.Compose]:
    """
    安全构建数据增强流程，自动处理转换顺序
    改进点：
    1. 自动确保ToTensor在最前面
    2. 自动确保Normalize在ToTensor之后
    3. 更严格的参数检查
    """
    transforms = {}
    
    # 预定义转换优先级
    TRANSFORM_PRIORITY = {
        'tensor': 0,       # 必须第一个
        'randomrotation': 1,
        'randomaffine': 1,
        'normalize': 2     # 必须最后一个
    }

    for phase in ['train', 'val']:
        if phase not in transform_cfg:
            continue
            
        # 按优先级排序转换
        sorted_transforms = sorted(
            transform_cfg[phase],
            key=lambda x: TRANSFORM_PRIORITY.get(x['name'].lower(), 1)
        )
        transform_list = []
        has_tensor = False
        has_normalize = False

        for aug in sorted_transforms:
            name = aug['name'].lower()
            args = {k: v for k, v in aug.items() if k != 'name'}

            if name == 'tensor':
                transform_list.append(T.ToTensor())
                has_tensor = True
            elif name == 'normalize':
                if not has_tensor:
                    raise ValueError("Normalize must be used after ToTensor")
                transform_list.append(T.Normalize(**args))
                has_normalize = True
            elif name == 'randomrotation':
                transform_list.append(T.RandomRotation(**args))
            elif name == 'randomaffine':
                transform_list.append(T.RandomAffine(**args))
            else:
                raise ValueError(f"Unsupported transform: {name}")

        # 自动补全必要转换
        if not has_tensor:
            transform_list.insert(0, T.ToTensor())
        if not has_normalize:
            transform_list.append(T.Normalize(mean=[0.1307], std=[0.3081]))

        transforms[phase] = T.Compose(transform_list)
    
    # 确保两个phase都有转换
    for phase in ['train', 'val']:
        if phase not in transforms:
            transforms[phase] = get_default_transforms(phase)
    
    return transforms

def get_default_transforms(phase: str) -> T.Compose:
    """
    获取默认转换流程
    Args:
        phase: 'train' 或 'val'
    """
    transforms = [T.ToTensor()]
    
    if phase == 'train':
        transforms.append(T.RandomRotation(degrees=15))
    
    transforms.append(T.Normalize(mean=[0.1307], std=[0.3081]))
    return T.Compose(transforms)