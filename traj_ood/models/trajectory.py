"""
Trajectory构建模块
将多层特征转为轨迹表示
"""

import torch

class TrajectoryBuilder:
    def __init__(self, use_delta=True):
        """
        参数:
            use_delta: 是否使用Δ特征（推荐开启）
        """
        self.use_delta = use_delta

    def build(self, features):
        """
        输入:
            features: list of [B, D]

        输出:
            traj: [B, L, D]
        """

        # 拼接成轨迹
        traj = torch.stack(features, dim=1)

        if self.use_delta:
            """
            Δ特征（核心思想）：
            捕捉层间变化，而不是绝对值
            """
            delta = traj[:, 1:] - traj[:, :-1]

            # 第一层保留原始特征
            traj = torch.cat([traj[:, :1], delta], dim=1)

        return traj