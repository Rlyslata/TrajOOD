"""
融合策略
"""

def fuse_score(s1, s2, lam=0.5):
    """
    输入:
        s1: energy score
        s2: trajectory score

    输出:
        融合后的OOD score
    """
    # return lam * s1 + (1 - lam) * s2
    return s1