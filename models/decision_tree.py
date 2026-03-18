# ============================================================
# decision_tree_scratch.py
# Decision Tree Classifier from Scratch
# ============================================================

import numpy as np
from collections import Counter


class Node:
    """
    A single node in the decision tree.

    If the node is a leaf node:
        value stores the class label

    If the node is an internal node:
        feature_index stores which feature to split on
        threshold stores the split value
        left stores the left child
        right stores the right child
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    """
    Decision Tree Classifier built from scratch using Gini impurity.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def gini_impurity(self, y):
        """
        Compute Gini impurity for a target array y.

        Formula:
            Gini = 1 - sum(p_i^2)
        where p_i is the proportion of class i.
        """
        classes = np.unique(y)
        impurity = 1.0

        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            impurity -= p_cls ** 2

        return impurity

    def split_dataset(self, X, y, feature_index, threshold):
        """
        Split data into left and right subsets.

        Left child:  feature <= threshold
        Right child: feature > threshold
        """
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]
        X_right = X[right_mask]
        y_right = y[right_mask]

        return X_left, y_left, X_right, y_right

    def best_split(self, X, y):
        """
        Find the best feature and threshold that gives the lowest weighted Gini impurity.
        """
        best_feature = None
        best_threshold = None
        best_gini = float("inf")

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split_dataset(
                    X, y, feature_index, threshold
                )

                # Skip invalid splits
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self.gini_impurity(y_left)
                gini_right = self.gini_impurity(y_right)

                weighted_gini = (
                    (len(y_left) / n_samples) * gini_left
                    + (len(y_right) / n_samples) * gini_right
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def most_common_label(self, y):
        """
        Return the majority class in y.
        Used when creating a leaf node.
        """
        return Counter(y).most_common(1)[0][0]

    def build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping condition 1: all samples belong to one class
        if n_classes == 1:
            return Node(value=y[0])

        # Stopping condition 2: max depth reached
        if depth >= self.max_depth:
            return Node(value=self.most_common_label(y))

        # Stopping condition 3: too few samples to split
        if n_samples < self.min_samples_split:
            return Node(value=self.most_common_label(y))

        # Find best split
        feature_index, threshold = self.best_split(X, y)

        # If no valid split found, create leaf node
        if feature_index is None:
            return Node(value=self.most_common_label(y))

        # Split data
        X_left, y_left, X_right, y_right = self.split_dataset(
            X, y, feature_index, threshold
        )

        # Recursively create left and right subtrees
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree
        )

    def fit(self, X, y):
        """
        Train the decision tree.
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have same number of rows. Got X={len(X)}, y={len(y)}")
        
        self.root = self.build_tree(X, y)

    def predict_sample(self, x, node):
        """
        Predict the class label for one sample by traversing the tree.
        """
        # If leaf node, return prediction
        if node.value is not None:
            return node.value

        # Otherwise move left or right depending on split rule
        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        """
        Predict class labels for all samples in X.
        """
        return np.array([self.predict_sample(x, self.root) for x in X])