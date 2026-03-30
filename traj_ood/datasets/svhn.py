"""
SVHN 数据集（作为OOD数据）
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_svhn(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # SVHN测试集作为OOD
    test = datasets.SVHN(
        root='./data',
        split='test',
        download=True,
        transform=transform
    )

    loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return loader