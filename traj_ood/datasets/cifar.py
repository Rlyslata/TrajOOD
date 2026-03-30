"""
CIFAR-10 数据加载模块
用于ID数据训练和测试
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10(batch_size):
    # 基础预处理（后续可以加 normalize / augmentation）
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 训练集
    train = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 测试集（用于ID评估）
    test = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # DataLoader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader