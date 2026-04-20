import numpy as np
from abc import ABC, abstractmethod

from models.tree_node import TreeNode


class BaseDecisionTree(ABC):
    """
    Common decision tree base for:
    - Simple Tree (ST)
    - Random Forest (RF)
    - Extra Trees (ET)

    Shared responsibilities:
    - recursive tree building
    - stopping criteria
    - leaf value creation
    - prediction traversal
    - paper-aligned score functions

    Child classes must implement:
    - _find_best_split(X, y)
    """

    def __init__(self, task="classification", min_samples_split=None):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")

        self.task = task
        self.min_samples_split = min_samples_split
        self.root = None

        if self.min_samples_split is None:
            self.min_samples_split = 2 if task == "classification" else 5

        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
            y_internal = np.array([label_to_idx[val] for val in y], dtype=int)
        else:
            y_internal = np.asarray(y, dtype=float)

        self.root = self._build_tree(X, y_internal)
        return self

    def predict(self, X):
        if self.root is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        preds = np.array([self._predict_one(self.root, x) for x in X])

        if self.task == "classification":
            idx = np.argmax(preds, axis=1)
            return self.classes_[idx]

        return preds

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification")

        if self.root is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(self.root, x) for x in X])

    def _build_tree(self, X, y):
        node = TreeNode()

        if self._should_stop(X, y):
            node.value = self._make_leaf_value(y)
            return node

        best_feature, best_threshold, best_score = self._find_best_split(X, y)

        if best_feature is None or best_score == -np.inf:
            node.value = self._make_leaf_value(y)
            return node

        left_mask = X[:, best_feature] < best_threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            node.value = self._make_leaf_value(y)
            return node

        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(X[left_mask], y[left_mask])
        node.right = self._build_tree(X[right_mask], y[right_mask])

        return node

    def _should_stop(self, X, y):
        if len(y) < self.min_samples_split:
            return True

        if self._output_constant(y):
            return True

        if len(self._non_constant_features(X)) == 0:
            return True

        return False

    def _output_constant(self, y):
        if len(y) == 0:
            return True
        return np.all(y == y[0])

    def _non_constant_features(self, X):
        return [j for j in range(X.shape[1]) if not np.all(X[:, j] == X[0, j])]

    def _make_leaf_value(self, y):
        if self.task == "regression":
            return float(np.mean(y))

        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / counts.sum()

    def _predict_one(self, node, x):
        while node.value is None:
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _regression_score(self, y, left_mask):
        """
        Relative variance reduction from the paper.
        """
        n = len(y)
        n_l = left_mask.sum()
        n_r = n - n_l

        if n_l == 0 or n_r == 0:
            return -np.inf

        var_y = y.var()
        if var_y == 0:
            return -np.inf

        var_l = y[left_mask].var()
        var_r = y[~left_mask].var()

        return (var_y - (n_l / n) * var_l - (n_r / n) * var_r) / var_y

    def _entropy(self, counts, n):
        probs = counts[counts > 0] / n
        return -np.sum(probs * np.log2(probs))

    def _classification_score(self, y, left_mask):
        """
        Normalized information gain from the paper:
        2 * I(s,c) / (H(s) + H(c))
        """
        n = len(y)
        n_l = left_mask.sum()
        n_r = n - n_l

        if n_l == 0 or n_r == 0:
            return -np.inf

        class_counts = np.bincount(y, minlength=self.n_classes_)
        Hc = self._entropy(class_counts, n)
        if Hc == 0:
            return -np.inf

        p_l, p_r = n_l / n, n_r / n
        Hs = (
            -p_l * np.log2(p_l) - p_r * np.log2(p_r)
            if (p_l > 0 and p_r > 0)
            else 0.0
        )

        counts_l = np.bincount(y[left_mask], minlength=self.n_classes_)
        counts_r = np.bincount(y[~left_mask], minlength=self.n_classes_)

        Hc_given_s = (
            p_l * self._entropy(counts_l, n_l) +
            p_r * self._entropy(counts_r, n_r)
        )

        I_sc = Hc - Hc_given_s
        denom = Hs + Hc

        if denom == 0:
            return -np.inf

        return 2 * I_sc / denom

    def _score_split(self, y, left_mask):
        if self.task == "regression":
            return self._regression_score(y, left_mask)
        return self._classification_score(y, left_mask)

    @abstractmethod
    def _find_best_split(self, X, y):
        """
        Must return:
        (best_feature, best_threshold, best_score)

        To be implemented differently by:
        - SimpleTree: all features, all valid thresholds
        - RandomForest: random subset of features, best threshold
        - ExtraTrees: random subset of features, random thresholds
        """
        pass