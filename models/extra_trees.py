import numpy as np
from models.base_decision_tree import BaseDecisionTree


class ExtraTree(BaseDecisionTree):
    """
    Single Extra Tree.

    Common ET logic:
    - choose K random non-constant features at each node
    - draw ONE random threshold uniformly in the local [min, max]
    - score those random splits
    - keep the best sampled split
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
        Paper-style defaults:
        - classification: round(sqrt(p))
        - regression: p
        """
        if self.max_features is not None:
            return max(1, min(int(self.max_features), n_features))

        if self.task == "classification":
            return max(1, int(round(np.sqrt(n_features))))

        return n_features

    def _find_best_split(self, X, y):
        non_const = self._non_constant_features(X)
        if not non_const:
            return None, None, -np.inf

        k = min(self._get_num_features_to_try(X.shape[1]), len(non_const))
        chosen_features = self.rng.choice(non_const, size=k, replace=False)

        best_score = -np.inf
        best_feature = None
        best_threshold = None

        for j in chosen_features:
            a_min = X[:, j].min()
            a_max = X[:, j].max()

            if a_min == a_max:
                continue

            threshold = self.rng.uniform(a_min, a_max)
            left_mask = X[:, j] < threshold
            score = self._score_split(y, left_mask)

            if score > best_score:
                best_score = score
                best_feature = j
                best_threshold = float(threshold)

        return best_feature, best_threshold, best_score


class ExtraTrees:
    """
    Sequential Extra Trees ensemble.

    - no bootstrap by default
    - one random threshold per selected feature
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
        bootstrap=False,
    ):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")

        self.task = task
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.bootstrap = bootstrap

        if min_samples_split is None:
            self.min_samples_split = 2 if task == "classification" else 5
        else:
            self.min_samples_split = min_samples_split

        self.trees_ = []
        self.classes_ = None
        self.n_classes_ = None

    def _sample_data(self, X, y, rng):
        if not self.bootstrap:
            return X, y

        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.trees_ = []
        base_seed = self.random_state if self.random_state is not None else 12345

        for i in range(self.n_estimators):
            tree_seed = base_seed + i
            rng = np.random.RandomState(tree_seed)

            X_tree, y_tree = self._sample_data(X, y, rng)

            tree = ExtraTree(
                task=self.task,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                random_state=tree_seed,
            )
            tree.fit(X_tree, y_tree)
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