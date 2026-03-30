"""
高斯建模（Mahalanobis）
用于轨迹空间的OOD检测
"""

import torch

class GaussianModel:
    def __init__(self):
        self.means = {}     # 每类均值
        self.cov_inv = None # 共享协方差逆

    def fit(self, features, labels):
        """
        训练阶段：估计分布

        输入:
            features: [N, D]
            labels: [N]
        """

        classes = labels.unique()
        feats_all = []

        for c in classes:
            f = features[labels == c]
            self.means[int(c)] = f.mean(0)
            feats_all.append(f)

        # 计算共享协方差
        feats_all = torch.cat(feats_all, dim=0)
        cov = torch.cov(feats_all.T)

        # 数值稳定
        self.cov_inv = torch.inverse(cov + 1e-5 * torch.eye(cov.size(0)))

    def score(self, x):
        """
        计算Mahalanobis距离（OOD score）

        输入:
            x: [B, D]

        输出:
            score: [B]
        """

        scores = []

        for c in self.means:
            mu = self.means[c]

            diff = x - mu
            dist = torch.sum(diff @ self.cov_inv * diff, dim=1)

            scores.append(dist.unsqueeze(1))

        scores = torch.cat(scores, dim=1)

        # 取最小距离（closest class）
        return scores.min(dim=1)[0]