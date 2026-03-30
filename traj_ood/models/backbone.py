"""
Backbone模型定义
当前使用 ResNet18（后续可以替换为 ViT）
"""

import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 使用标准ResNet18结构
        self.model = resnet18(pretrained=False)

        # 替换最后分类层
        # 原输出 512 -> num_classes
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        输入:
            x: [B, 3, 32, 32]
        输出:
            logits: [B, num_classes]
        """
        return self.model(x)