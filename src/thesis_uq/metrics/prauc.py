import numpy as np
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import precision_recall_curve, auc as sk_auc


class PRAUC(Metric):
    """
    PR-AUC metric compatible with pytorch-tabnet's eval_metric API.

    IMPORTANT: in your fork, pass the *class* (PRAUC), not an instance (PRAUC()).
    """
    def __init__(self):
        self._name = "prauc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        # TabNet often returns shape (n,2) -> use class 1
        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_prob = y_score[:, 1]
        else:
            y_prob = y_score.reshape(-1)

        y = y_true.reshape(-1)

        # handle split with no positives
        if y.sum() == 0:
            return 0.0

        precision, recall, _ = precision_recall_curve(y, y_prob)
        return float(sk_auc(recall, precision))
