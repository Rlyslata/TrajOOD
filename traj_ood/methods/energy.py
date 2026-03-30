"""
Energy-based OOD score
"""

import torch

def energy_score(logits):
    """
    输入:
        logits: [B, C]

    输出:
        energy: [B]

    原理:
        E(x) = -logsumexp(logits)
    """
    return -torch.logsumexp(logits, dim=1)