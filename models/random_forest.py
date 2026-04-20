import numpy as np
from models.base_decision_tree import BaseDecisionTree


class RandomForestTree(BaseDecisionTree):
    """
    Single Random Forest tree.

    RF logic:
    - sample a random subset of features at each node
    - among only those features, search the best threshold
    - split score is handled by BaseDecisionTree
    """

    def __init__(
        self,
        task="classification",
        max_features=None,
        min_samples_split=None,
        random_state=None,
    ):
        super().__init__(task=task, min_samples_split=min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def _get_num_features_to_try(self, n_features):
        """
        Paper-style practical defaults:
        - classification: sqrt(p)
        - regression: p/3 (common RF default)
        """
        if self.max_features is not None:
            return max(1, min(int(self.max_features), n_features))

        if self.task == "classification":
            return max(1, int(np.sqrt(n_features)))

        return max(1, int(np.ceil(n_features / 3.0)))

    def _find_best_split(self, X, y):
        non_const = self._non_constant_features(X)
        if not non_const:
            return None, None, -np.inf

        m_try = min(self._get_num_features_to_try(X.shape[1]), len(non_const))
        chosen_features = self.rng.choice(non_const, size=m_try, replace=False)

        best_score = -np.inf
        best_feature = None
        best_threshold = None

        for j in chosen_features:
            values = np.unique(X[:, j])

            if len(values) <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, j] < threshold
                score = self._score_split(y, left_mask)

                if score > best_score:
                    best_score = score
                    best_feature = j
                    best_threshold = threshold

        return best_feature, best_threshold, best_score


class RandomForest:
    """
    Random Forest ensemble.

    Uses:
    - bootstrap samples per tree
    - RandomForestTree as base learner
    - average probabilities for classification
    - average predictions for regression
    """

    def __init__(
        self,
        task="classification",
        n_estimators=100,
        max_features=None,
        min_samples_split=None,
        random_state=None,
    ):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")

        self.task = task
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.trees_ = []
        self.classes_ = None
        self.n_classes_ = None

    def _bootstrap_sample(self, X, y, rng):
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.trees_ = []
        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap_sample(X, y, rng)

            tree_seed = None if self.random_state is None else self.random_state + i

            tree = RandomForestTree(
                task=self.task,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                random_state=tree_seed,
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        if self.task == "classification" and self.trees_:
            self.classes_ = self.trees_[0].classes_
            self.n_classes_ = self.trees_[0].n_classes_

        return self

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification")

        if not self.trees_:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        all_probs = np.array([tree.predict_proba(X) for tree in self.trees_])
        return np.mean(all_probs, axis=0)

    def predict(self, X):
        if not self.trees_:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if self.task == "classification":
            probs = self.predict_proba(X)
            idx = np.argmax(probs, axis=1)
            return self.classes_[idx]

        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(all_preds, axis=0)