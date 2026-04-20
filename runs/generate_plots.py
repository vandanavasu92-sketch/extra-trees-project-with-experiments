import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = ["ST", "TB", "RF", "ET"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prettify_dataset_name(name):
    return str(name).replace("_", " ").title()


def get_model_order(df):
    present = df["model"].dropna().unique().tolist()
    ordered = [m for m in MODEL_ORDER if m in present]
    extras = sorted([m for m in present if m not in MODEL_ORDER])
    return ordered + extras


def add_bar_labels(ax, bars, values, fmt="{:.3f}", pad=0.01):
    ymax = ax.get_ylim()[1]
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad * ymax,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=10
        )


def plot_classification_overall(df, output_dir):
    """
    Overall classification test error across all classification datasets/runs.
    Lower is better.
    """
    cls_df = df[df["task"] == "classification"].copy()
    if cls_df.empty:
        return

    summary = (
        cls_df.groupby("model", as_index=False)
        .agg(
            mean_test_err=("test_err", "mean"),
            std_test_err=("test_err", "std")
        )
    )

    model_order = get_model_order(summary)
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        summary["model"],
        summary["mean_test_err"],
        yerr=summary["std_test_err"],
        capsize=8
    )

    plt.title("Overall Classification Test Error (All Classification Datasets)")
    plt.xlabel("Model")
    plt.ylabel("Mean Test Error (lower is better)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    add_bar_labels(
        plt.gca(),
        bars,
        summary["mean_test_err"].values,
        fmt="{:.3f}",
        pad=0.01
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_overall_test_error.png"), dpi=300)
    plt.close()


def plot_classification_overall_accuracy(df, output_dir):
    """
    Overall classification test accuracy across all classification datasets/runs.
    Higher is better.
    """
    cls_df = df[df["task"] == "classification"].copy()
    if cls_df.empty or "test_acc" not in cls_df.columns:
        return

    summary = (
        cls_df.groupby("model", as_index=False)
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std")
        )
    )

    model_order = get_model_order(summary)
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        summary["model"],
        summary["mean_test_acc"],
        yerr=summary["std_test_acc"],
        capsize=8
    )

    plt.title("Overall Classification Accuracy (All Classification Datasets)")
    plt.xlabel("Model")
    plt.ylabel("Mean Test Accuracy (higher is better)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    add_bar_labels(
        plt.gca(),
        bars,
        summary["mean_test_acc"].values,
        fmt="{:.3f}",
        pad=0.01
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_overall_accuracy.png"), dpi=300)
    plt.close()


def plot_regression_overall(df, output_dir):
    """
    Overall regression test MSE across all regression datasets/runs.
    Lower is better.
    """
    reg_df = df[df["task"] == "regression"].copy()
    if reg_df.empty:
        return

    summary = (
        reg_df.groupby("model", as_index=False)
        .agg(
            mean_test_mse=("test_mse", "mean"),
            std_test_mse=("test_mse", "std")
        )
    )

    model_order = get_model_order(summary)
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        summary["model"],
        summary["mean_test_mse"],
        yerr=summary["std_test_mse"],
        capsize=8
    )

    plt.title("Overall Regression Test MSE (All Regression Datasets)")
    plt.xlabel("Model")
    plt.ylabel("Mean Test MSE (lower is better)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    add_bar_labels(
        plt.gca(),
        bars,
        summary["mean_test_mse"].values,
        fmt="{:.3f}",
        pad=0.01
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_overall_test_mse.png"), dpi=300)
    plt.close()


def plot_regression_overall_r2(df, output_dir):
    """
    Overall regression test R² across all regression datasets/runs.
    Higher is better.
    """
    reg_df = df[df["task"] == "regression"].copy()
    if reg_df.empty or "test_r2" not in reg_df.columns:
        return

    summary = (
        reg_df.groupby("model", as_index=False)
        .agg(
            mean_test_r2=("test_r2", "mean"),
            std_test_r2=("test_r2", "std")
        )
    )

    model_order = get_model_order(summary)
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        summary["model"],
        summary["mean_test_r2"],
        yerr=summary["std_test_r2"],
        capsize=8
    )

    plt.title("Overall Regression Test R² (All Regression Datasets)")
    plt.xlabel("Model")
    plt.ylabel("Mean Test R² (higher is better)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    add_bar_labels(
        plt.gca(),
        bars,
        summary["mean_test_r2"].values,
        fmt="{:.3f}",
        pad=0.01
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_overall_test_r2.png"), dpi=300)
    plt.close()


def plot_datasetwise_classification(df, output_dir):
    """
    Dataset-wise classification error plot.
    One bar cluster per dataset.
    """
    cls_df = df[df["task"] == "classification"].copy()
    if cls_df.empty:
        return

    summary = (
        cls_df.groupby(["dataset", "model"], as_index=False)
        .agg(mean_test_err=("test_err", "mean"))
    )

    pivot = summary.pivot(index="dataset", columns="model", values="mean_test_err")
    model_order = [m for m in MODEL_ORDER if m in pivot.columns] + sorted(
        [c for c in pivot.columns if c not in MODEL_ORDER]
    )
    pivot = pivot[model_order]
    pivot.index = [prettify_dataset_name(idx) for idx in pivot.index]

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Dataset-wise Classification Test Error")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean Test Error (lower is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "datasetwise_classification_test_error.png"), dpi=300)
    plt.close()


def plot_datasetwise_regression(df, output_dir):
    """
    Dataset-wise regression MSE plot.
    One bar cluster per dataset.
    """
    reg_df = df[df["task"] == "regression"].copy()
    if reg_df.empty:
        return

    summary = (
        reg_df.groupby(["dataset", "model"], as_index=False)
        .agg(mean_test_mse=("test_mse", "mean"))
    )

    pivot = summary.pivot(index="dataset", columns="model", values="mean_test_mse")
    model_order = [m for m in MODEL_ORDER if m in pivot.columns] + sorted(
        [c for c in pivot.columns if c not in MODEL_ORDER]
    )
    pivot = pivot[model_order]
    pivot.index = [prettify_dataset_name(idx) for idx in pivot.index]

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Dataset-wise Regression Test MSE")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean Test MSE (lower is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "datasetwise_regression_test_mse.png"), dpi=300)
    plt.close()


def plot_single_dataset_classification(df, dataset_name, output_dir):
    """
    Single dataset classification plot with mean ± std.
    """
    ds = df[(df["task"] == "classification") & (df["dataset"] == dataset_name)].copy()
    if ds.empty:
        return

    summary = (
        ds.groupby("model", as_index=False)
        .agg(
            mean_test_err=("test_err", "mean"),
            std_test_err=("test_err", "std")
        )
    )

    model_order = get_model_order(summary)
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")

    display_name = prettify_dataset_name(dataset_name)

    plt.figure(figsize=(9, 6))
    bars = plt.bar(
        summary["model"],
        summary["mean_test_err"],
        yerr=summary["std_test_err"],
        capsize=8
    )

    plt.title(f"Classification Test Error ({display_name} Dataset)")
    plt.xlabel("Model")
    plt.ylabel("Mean Test Error (lower is better)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    add_bar_labels(
        plt.gca(),
        bars,
        summary["mean_test_err"].values,
        fmt="{:.3f}",
        pad=0.01
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_classification_test_error.png"), dpi=300)
    plt.close()


def plot_single_dataset_regression(df, dataset_name, output_dir):
    """
    Single dataset regression plot with mean ± std.
    """
    ds = df[(df["task"] == "regression") & (df["dataset"] == dataset_name)].copy()
    if ds.empty:
        return

    summary = (
        ds.groupby("model", as_index=False)
        .agg(
            mean_test_mse=("test_mse", "mean"),
            std_test_mse=("test_mse", "std")
        )
    )

    model_order = get_model_order(summary)
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")

    display_name = prettify_dataset_name(dataset_name)

    plt.figure(figsize=(9, 6))
    bars = plt.bar(
        summary["model"],
        summary["mean_test_mse"],
        yerr=summary["std_test_mse"],
        capsize=8
    )

    plt.title(f"Regression Test MSE ({display_name} Dataset)")
    plt.xlabel("Model")
    plt.ylabel("Mean Test MSE (lower is better)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    add_bar_labels(
        plt.gca(),
        bars,
        summary["mean_test_mse"].values,
        fmt="{:.3f}",
        pad=0.01
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_regression_test_mse.png"), dpi=300)
    plt.close()


def generate_all_plots(results_csv="results/all_models_results.csv", output_dir="results/plots"):
    ensure_dir(output_dir)

    df = pd.read_csv(results_csv)

    # overall plots across all datasets of each task
    plot_classification_overall(df, output_dir)
    plot_classification_overall_accuracy(df, output_dir)
    plot_regression_overall(df, output_dir)
    plot_regression_overall_r2(df, output_dir)

    # dataset-wise plots
    plot_datasetwise_classification(df, output_dir)
    plot_datasetwise_regression(df, output_dir)

    # single dataset plots
    for dataset_name in sorted(df["dataset"].dropna().unique()):
        subset = df[df["dataset"] == dataset_name]
        if subset.empty:
            continue

        task = subset["task"].iloc[0]
        if task == "classification":
            plot_single_dataset_classification(df, dataset_name, output_dir)
        else:
            plot_single_dataset_regression(df, dataset_name, output_dir)

    print(f"Saved plots in: {output_dir}")