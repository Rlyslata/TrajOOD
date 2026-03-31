"""
轨迹编码器
将 [B, L, D] → [B, d']
"""

import torch.nn as nn

class TrajMLP(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=128):
        super().__init__()

        self.net = None
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def build(self, input_dim):
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )

    def forward(self, traj):
        B, L, D = traj.shape
        
        x = traj.view(B, L * D)

        # ⭐ 动态初始化
        if self.net is None:
            # print(f"[TrajMLP] Building with input_dim = {L*D}")
            self.build(L * D)
            self.net.to(x.device)
        # print(f'[TrajMLP] traj device : {traj.device}  net device : {next(self.net.parameters()).device}')
        return self.net(x)