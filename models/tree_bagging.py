import numpy as np
from joblib import Parallel, delayed

from models.single_tree import SimpleTree


class TreeBagging:
    """
    Tree Bagging (TB)

    - trains many unpruned SimpleTree models
    - each tree sees a bootstrap sample
    - classification -> majority vote
    - regression -> average
    """

    def __init__(
        self,
        task="classification",
        n_estimators=100,
        min_samples_split=None,
        random_state=42,
        n_jobs=-1
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees_ = []

    def _fit_one_tree(self, X, y, seed):
        rng = np.random.default_rng(seed)
        n_samples = X.shape[0]

        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        tree = SimpleTree(
            task=self.task,
            min_samples_split=self.min_samples_split
        )
        tree.fit(X_boot, y_boot)
        return tree

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        seeds = [self.random_state + i for i in range(self.n_estimators)]

        self.trees_ = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self._fit_one_tree)(X, y, seed)
            for seed in seeds
        )

        return self

    def predict(self, X):
        if not self.trees_:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if self.task == "classification":
            all_preds = np.array([tree.predict(X) for tree in self.trees_])

            final_preds = []
            for j in range(all_preds.shape[1]):
                values, counts = np.unique(all_preds[:, j], return_counts=True)
                final_preds.append(values[np.argmax(counts)])

            return np.array(final_preds)

        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(all_preds, axis=0)