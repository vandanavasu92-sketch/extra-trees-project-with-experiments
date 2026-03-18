# ============================================================
# run_decision_tree.py
# Decision Tree experiment setup
# ============================================================

from models.decision_tree import DecisionTree
from utils.run_model import run_model_pipeline


def build_decision_tree(max_depth):
    return DecisionTree(max_depth=max_depth, min_samples_split=5)


def run_decision_tree():
    dt_results = run_model_pipeline(
        model_name="Decision Tree",
        model_builder=build_decision_tree,
        parameter_name="max_depth",
        parameter_values=[3, 5, 10]
    )
    return dt_results