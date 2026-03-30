"""
主程序入口
一键跑通：
CIFAR10（ID） vs SVHN（OOD）
"""

import torch

from datasets.cifar import get_cifar10
from datasets.svhn import get_svhn

from models.backbone import ResNet18
from models.hook import FeatureHook
from models.trajectory import TrajectoryBuilder
from models.traj_encoder import TrajMLP

from methods.gaussian import GaussianModel
from utils.metrics import compute_auroc

from trainers.train_m1 import train_m1
from eval.ood_eval import evaluate_ood


def main():

    # ====== 基础设置 ======
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128

    # ====== 数据 ======
    train_loader, id_test_loader = get_cifar10(batch_size)
    ood_loader = get_svhn(batch_size)

    # ====== 模型 ======
    model = ResNet18(num_classes=10)

    print("\n=== Training M1 ===")
    model = train_m1(model, train_loader, device)

    # ====== Hook ======
    hook = FeatureHook(model)

    # ====== 构建轨迹 ======
    traj_builder = TrajectoryBuilder(use_delta=True)

    # ====== 轨迹编码器 ======
    # 先用简单MLP baseline
    traj_encoder = TrajMLP(input_dim=512 * len(hook.handles)).to(device)

    # ====== 提取训练特征（用于Gaussian） ======
    print("\n=== Extracting Trajectory Features ===")

    all_feats = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)

            hook.clear()

            _ = model(x)

            traj = traj_builder.build(hook.features)

            z = traj_encoder(traj)

            all_feats.append(z.cpu())
            all_labels.append(y)

    features = torch.cat(all_feats)
    labels = torch.cat(all_labels)

    # ====== 高斯建模 ======
    print("\n=== Fitting Gaussian ===")
    gaussian = GaussianModel()
    gaussian.fit(features, labels)

    # ====== OOD评估 ======
    print("\n=== OOD Evaluation ===")
    evaluate_ood(
        model,
        id_test_loader,
        ood_loader,
        hook,
        traj_builder,
        traj_encoder,
        gaussian,
        device,
        compute_auroc
    )


if __name__ == "__main__":
    main()