"""
OOD评估流程
"""

import torch
from tqdm import tqdm

from methods.energy import energy_score
from methods.fusion import fuse_score

def get_scores(model, loader, hook, traj_builder, traj_encoder, gaussian, device):
    model.eval()

    scores = []

    with torch.no_grad():
        for x, _ in tqdm(loader):
            x = x.to(device)

            # 清空hook
            hook.clear()

            logits = model(x)

            # S1: Energy
            s1 = energy_score(logits)

            # 构建轨迹
            traj = traj_builder.build(hook.features)

            # 编码
            z = traj_encoder(traj)

            # S2: Gaussian
            s2 = gaussian.score(z)

            # 融合
            s = fuse_score(s1, s2)

            scores.extend(s.cpu().numpy())

    return scores


def evaluate_ood(model, id_loader, ood_loader,
                 hook, traj_builder, traj_encoder,
                 gaussian, device, metric_fn):

    print("Computing ID scores...")
    id_scores = get_scores(model, id_loader, hook,
                           traj_builder, traj_encoder,
                           gaussian, device)

    print("Computing OOD scores...")
    ood_scores = get_scores(model, ood_loader, hook,
                            traj_builder, traj_encoder,
                            gaussian, device)

    auroc = metric_fn(id_scores, ood_scores)

    print(f"\nAUROC: {auroc:.4f}")

    return auroc