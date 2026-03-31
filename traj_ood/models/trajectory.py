"""
Trajectory构建模块
将多层特征转为轨迹表示
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryBuilder(nn.Module):
    def __init__(self, d_model=256, use_delta=True):
        super().__init__()

        self.use_delta = use_delta
        self.d_model = d_model

        self.proj = None  # 延迟初始化

    def build_proj(self, features):
        channel_list = [s.shape[1] for s in features]
        # print(f"[TrTrajectoryBuilder] Building projection layers for channels: {channel_list}")
        self.proj = nn.ModuleList([
            nn.Linear(c, self.d_model) for c in channel_list
        ])

    def forward(self, features):
        # for i, f in enumerate(features):
        #     print(f"[TrTrajectoryBuilder] Layer {i} shape: {f.shape}")
        # ⭐ 第一次运行时自动初始化
        if self.proj is None:
            self.build_proj(features)
            self.device = next(self.parameters()).device
            self.proj.to(self.device)

        processed = []

        for i, f in enumerate(features):
            f = f.to(self.device)
            if len(f.shape) == 4:
                f = F.adaptive_avg_pool2d(f, (1, 1))
                f = f.view(f.size(0), -1)

            f = self.proj[i](f)

            processed.append(f)

        traj = torch.stack(processed, dim=1)

        if self.use_delta:
            traj = traj[:, 1:] - traj[:, :-1]

        return traj