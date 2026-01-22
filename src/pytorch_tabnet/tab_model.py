import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import SparsePredictDataset, PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader
import scipy


class TabNetClassifier(TabModel):
    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = "classification"
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = "accuracy"

    def weight_updater(self, weights):
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.long())

    def update_fit_params(self, X_train, y_train, eval_set, weights):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ("auc" if self.output_dim == 2 else "accuracy")
        self.classes_ = train_labels
        self.target_mapper = {class_label: index for index, class_label in enumerate(self.classes_)}
        self.preds_mapper = {str(index): class_label for index, class_label in enumerate(self.classes_)}
        self.updated_weights = self.weight_updater(weights)

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    # ---------------------------
    # Standard deterministic proba
    # ---------------------------
    def predict_proba(self, X):
        """
        Deterministic predict_proba (dropout OFF).
        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results = []
        for _, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, _ = self.network(data)
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        return np.vstack(results)

    # ---------------------------
    # MC Dropout proba + uncertainty
    # ---------------------------
    def _enable_mc_dropout(self):
        """
        Enable MC Dropout at inference:
        - keep BatchNorm/GBN frozen (eval)
        - enable Dropout sampling (train) ONLY for dropout modules
        """
        self.network.eval()
        for m in self.network.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def predict_proba_mc(self, X, n_samples: int = 50, return_std: bool = True):
        """
        MC Dropout predictive probabilities (dropout ON, BN frozen).

        Returns
        -------
        mean_proba : np.ndarray, shape (n_rows, n_classes)
        std_proba  : np.ndarray, shape (n_rows, n_classes)  (if return_std=True)
        """
        self._enable_mc_dropout()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        means, stds = [], []

        for _, data in enumerate(dataloader):
            data = data.to(self.device).float()

            probs_T = []
            with torch.no_grad():
                for _ in range(n_samples):
                    logits, _ = self.network(data)                 # (B, C)
                    probs = torch.softmax(logits, dim=1)           # (B, C)
                    probs_T.append(probs)

            probs_T = torch.stack(probs_T, dim=0)                 # (T, B, C)
            mean_batch = probs_T.mean(dim=0).cpu().numpy()        # (B, C)
            means.append(mean_batch)

            if return_std:
                std_batch = probs_T.std(dim=0, unbiased=False).cpu().numpy()
                stds.append(std_batch)

        mean_proba = np.vstack(means)
        if return_std:
            std_proba = np.vstack(stds)
            return mean_proba, std_proba
        return mean_proba


class TabNetRegressor(TabModel):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = "regression"
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = "mse"

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(self, X_train, y_train, eval_set, weights):
        if len(y_train.shape) != 2:
            msg = (
                "Targets should be 2D : (n_samples, n_regression) "
                + f"but y_train.shape={y_train.shape} given.\n"
                + "Use reshape(-1, 1) for single regression."
            )
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score

    # ---------------------------
    # MC Dropout predictions + uncertainty
    # ---------------------------
    def _enable_mc_dropout(self):
        self.network.eval()
        for m in self.network.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def predict_mc(self, X, n_samples: int = 50, return_std: bool = True):
        """
        MC Dropout predictive mean/std for regression.

        Returns
        -------
        mean_pred : np.ndarray, shape (n_rows, out_dim)
        std_pred  : np.ndarray, shape (n_rows, out_dim) (if return_std=True)
        """
        self._enable_mc_dropout()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        means, stds = [], []

        for _, data in enumerate(dataloader):
            data = data.to(self.device).float()

            preds_T = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pred, _ = self.network(data)                  # (B, out_dim)
                    preds_T.append(pred)

            preds_T = torch.stack(preds_T, dim=0)                 # (T, B, out_dim)
            mean_batch = preds_T.mean(dim=0).cpu().numpy()
            means.append(mean_batch)

            if return_std:
                std_batch = preds_T.std(dim=0, unbiased=False).cpu().numpy()
                stds.append(std_batch)

        mean_pred = np.vstack(means)
        if return_std:
            std_pred = np.vstack(stds)
            return mean_pred, std_pred
        return mean_pred
