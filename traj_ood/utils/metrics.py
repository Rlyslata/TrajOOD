"""
评估指标
"""

from sklearn.metrics import roc_auc_score

def compute_auroc(id_scores, ood_scores):
    """
    输入:
        id_scores: list
        ood_scores: list

    输出:
        AUROC
    """

    y_true = [1]*len(id_scores) + [0]*len(ood_scores)
    y_score = list(id_scores) + list(ood_scores)

    return roc_auc_score(y_true, y_score)