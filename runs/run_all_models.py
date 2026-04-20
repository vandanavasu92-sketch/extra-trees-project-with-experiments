import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.single_tree import SimpleTree
from models.tree_bagging import TreeBagging
from models.random_forest import RandomForest
from models.extra_trees import ExtraTrees

from utils.data_loader import get_dataset_names, load_dataset
from utils.dataset_config import DATASET_CONFIG
from utils.evaluation import evaluate_predictions


RESULTS_DIR = "results"
OUTPUT_FILE = "all_models_results.csv"


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def safe_stratify_arg(y, task):
    """
    Use stratification only for classification.
    """
    return y if task == "classification" else None


def get_num_runs(dataset_name, X):
    """
    Paper uses:
    - usually 10 runs
    - sometimes 50 for smaller datasets
    """
    n_samples = X.shape[0]
    if n_samples < 500:
        return 50
    return 10


def build_models_for_task(task, run_seed):
    """
    Return all models to compare for the given task.
    """
    models = {
        "ST": SimpleTree(task=task),

        "TB": TreeBagging(
            task=task,
            n_estimators=100,
            random_state=run_seed
        ),
        "RF": RandomForest(
            task=task,
            n_estimators=100,
            random_state=run_seed
        ),
        "ET": ExtraTrees(
            task=task,
            n_estimators=100,
            random_state=run_seed
        ),
    }
    return models


def run_one_model(model_name, model, dataset_name, task, run,
                  X_train, X_test, y_train, y_test):
    """
    Fit one model, predict, evaluate, and return one result row.
    """
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_start

    pred_start = time.time()
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    predict_time = time.time() - pred_start

    train_metrics = evaluate_predictions(task, y_train, train_pred)
    test_metrics = evaluate_predictions(task, y_test, test_pred)

    row = {
        "dataset": dataset_name,
        "task": task,
        "run": run + 1,
        "model": model_name,

        "n_train": len(y_train),
        "n_test": len(y_test),

        "train_mse": train_metrics["mse"],
        "train_r2": train_metrics["r2"],
        "train_acc": train_metrics["acc"],
        "train_err": train_metrics["err"],

        "test_mse": test_metrics["mse"],
        "test_r2": test_metrics["r2"],
        "test_acc": test_metrics["acc"],
        "test_err": test_metrics["err"],

        "fit_time": float(fit_time),
        "predict_time": float(predict_time),

        "params": json.dumps(model.__dict__, default=str)
    }

    return row, train_pred, test_pred


def run_one_dataset(dataset_name):
    """
    Run all models for one dataset across repeated splits.
    """
    X, y = load_dataset(dataset_name)
    task = DATASET_CONFIG[dataset_name]["task"]
    n_runs = get_num_runs(dataset_name, X)

    print("\n" + "#" * 80)
    print(f"Running all models for dataset: {dataset_name}")
    print(f"Task      : {task}")
    print(f"X shape   : {X.shape}")
    print(f"y shape   : {y.shape}")
    print(f"Runs      : {n_runs}")
    print("#" * 80)

    rows = []

    for run in range(n_runs):
        print(f"\nRunning {dataset_name} | split {run + 1}/{n_runs}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=run,
            stratify=safe_stratify_arg(y, task)
        )

        models = build_models_for_task(task, run_seed=run)

        for model_name, model in models.items():
            print(f"  Fitting {model_name}...")

            try:
                row, train_pred, test_pred = run_one_model(
                    model_name=model_name,
                    model=model,
                    dataset_name=dataset_name,
                    task=task,
                    run=run,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test
                )
                rows.append(row)

                # Preview only on first run
                if run == 0:
                    if task == "classification":
                        print(f"    Train acc : {row['train_acc']:.4f} | Train err : {row['train_err']:.4f}")
                        print(f"    Test  acc : {row['test_acc']:.4f} | Test  err : {row['test_err']:.4f}")
                    else:
                        print(f"    Train mse : {row['train_mse']:.4f} | Train r2 : {row['train_r2']:.4f}")
                        print(f"    Test  mse : {row['test_mse']:.4f} | Test  r2 : {row['test_r2']:.4f}")

                    print(f"    Fit time  : {row['fit_time']:.4f}s")
                    print(f"    Pred time : {row['predict_time']:.4f}s")
                    print("First 10 test predictions:", np.round(test_pred[:10], 4))

            except Exception as e:
                print(f"    Error in {model_name} on {dataset_name}, run {run + 1}: {e}")

    if not rows:
        print(f"No results generated for dataset: {dataset_name}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print(f"Summary for dataset: {dataset_name}")
    print("=" * 80)

    if task == "classification":
        summary = (
            df.groupby("model", as_index=False)
              .agg(
                  mean_test_err=("test_err", "mean"),
                  std_test_err=("test_err", "std"),
                  mean_test_acc=("test_acc", "mean"),
                  std_test_acc=("test_acc", "std"),
                  mean_fit_time=("fit_time", "mean")
              )
              .sort_values("mean_test_err")
        )
        print(summary.to_string(index=False))
    else:
        summary = (
            df.groupby("model", as_index=False)
              .agg(
                  mean_test_mse=("test_mse", "mean"),
                  std_test_mse=("test_mse", "std"),
                  mean_test_r2=("test_r2", "mean"),
                  std_test_r2=("test_r2", "std"),
                  mean_fit_time=("fit_time", "mean")
              )
              .sort_values("mean_test_mse")
        )
        print(summary.to_string(index=False))

    return df


def run_all_models():
    ensure_results_dir()

    # use only the two selected datasets for now
    datasets = ["airfoil", "pima"]
    # datasets = get_dataset_names()

    all_results = []

    print("\n" + "#" * 80)
    print("Running unified comparison pipeline for all models")
    print("#" * 80)

    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIG:
            print(f"Skipping {dataset_name} (not found in DATASET_CONFIG)")
            continue

        try:
            dataset_df = run_one_dataset(dataset_name)
            if not dataset_df.empty:
                all_results.append(dataset_df)
        except Exception as e:
            print(f"Fatal error while processing dataset {dataset_name}: {e}")

    if not all_results:
        print("No results were generated.")
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=True)

    output_path = os.path.join(RESULTS_DIR, OUTPUT_FILE)
    final_df.to_csv(output_path, index=False)

    print("\n" + "#" * 80)
    print("Completed unified comparison pipeline")
    print(f"Saved raw results to: {output_path}")
    print("#" * 80)

    return final_df


if __name__ == "__main__":
    run_all_models()