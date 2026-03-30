"""
轨迹编码器
将 [B, L, D] → [B, d']
"""

import torch.nn as nn

class TrajMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, out_dim=128):
        super().__init__()

        """
        简单baseline：
        - flatten轨迹
        - MLP降维
        """

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, traj):
        """
        输入:
            traj: [B, L, D]

        输出:
            z: [B, out_dim]
        """
        B, L, D = traj.shape

        # 展平轨迹
        x = traj.view(B, L * D)

        return self.net(x)