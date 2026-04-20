import numpy as np

from models.base_decision_tree import BaseDecisionTree


class SimpleTree(BaseDecisionTree):
    """
    Paper-aligned Single Tree (ST)

    - deterministic
    - unpruned
    - uses all non-constant features
    - searches all valid midpoint thresholds
    - uses shared score logic from BaseDecisionTree
    """

    def __init__(self, task="classification", min_samples_split=None):
        super().__init__(task=task, min_samples_split=min_samples_split)

    def _find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_score = -np.inf

        candidate_features = self._non_constant_features(X)

        for feature_idx in candidate_features:
            values = X[:, feature_idx]
            unique_values = np.unique(values)

            if len(unique_values) <= 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values < threshold
                n_left = left_mask.sum()
                n_right = len(values) - n_left

                if n_left == 0 or n_right == 0:
                    continue

                score = self._score_split(y, left_mask)

                if score > best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_score