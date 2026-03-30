"""
CIFAR-10 数据加载模块
用于ID数据训练和测试
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10(batch_size):
    # 基础预处理（后续可以加 normalize / augmentation）
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_root = './../data'
    
    cifar_path = os.path.join(data_root, 'cifar-10-batches-py')

    # 如果目录不存在，则下载；否则不下载
    need_download = not os.path.exists(cifar_path)
    # 训练集
    train = datasets.CIFAR10(
        root='./data',
        train=True,
        download=need_download,
        transform=transform
    )

    # 测试集（用于ID评估）
    test = datasets.CIFAR10(
        root='./data',
        train=False,
        download=need_download,
        transform=transform
    )

    # DataLoader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader