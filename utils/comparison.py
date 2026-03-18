# ============================================================
# comparison.py
# Creates final tables and plots for model comparison
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def compare_models(*results_lists):
    """
    Accepts multiple result lists (DT, RF, ET),
    combines them, creates tables and plots.
    """

    # --------------------------------------------------
    # 1. Combine all results
    # --------------------------------------------------
    all_results = []

    for result_list in results_lists:
        all_results.extend(result_list)

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("FULL COMBINED RESULTS")
    print("=" * 70)
    print(df)

    # --------------------------------------------------
    # 2. Save full table
    # --------------------------------------------------
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/all_results.csv", index=False)

    # --------------------------------------------------
    # 3. Get best result per dataset & model (by F1)
    # --------------------------------------------------
    best_df = df.loc[
        df.groupby(["Dataset", "Model"])["F1-score"].idxmax()
    ].reset_index(drop=True)

    print("\n" + "=" * 70)
    print("BEST RESULTS PER MODEL")
    print("=" * 70)
    print(best_df)

    best_df.to_csv("results/best_results.csv", index=False)

    # --------------------------------------------------
    # 4. Plot comparison for each dataset
    # --------------------------------------------------
    for dataset in best_df["Dataset"].unique():
        plot_dataset(best_df, dataset)

    # --------------------------------------------------
    # 5. Overfitting plot
    # --------------------------------------------------
    plot_overfitting(best_df)

    print("\n✅ Comparison completed. Results saved in 'results/' folder.")


# ----------------------------------------------------------
# Plot function (metrics)
# ----------------------------------------------------------
def plot_dataset(df, dataset_name):
    subset = df[df["Dataset"] == dataset_name]

    models = subset["Model"].tolist()
    acc = subset["Test Accuracy"].tolist()
    precision = subset["Precision"].tolist()
    recall = subset["Recall"].tolist()
    f1 = subset["F1-score"].tolist()

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, acc, width, label="Accuracy")
    plt.bar(x - 0.5 * width, precision, width, label="Precision")
    plt.bar(x + 0.5 * width, recall, width, label="Recall")
    plt.bar(x + 1.5 * width, f1, width, label="F1-score")

    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title(f"{dataset_name.upper()} Model Comparison")

    plt.legend()
    plt.tight_layout()

    plt.savefig(f"results/{dataset_name}_comparison.png")
    plt.show()


# ----------------------------------------------------------
# Overfitting plot
# ----------------------------------------------------------
def plot_overfitting(df):

    labels = df["Dataset"] + "-" + df["Model"]
    gaps = df["Overfitting Gap"]

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x, gaps)

    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Train - Test Accuracy")
    plt.title("Overfitting Comparison")

    plt.tight_layout()
    plt.savefig("results/overfitting.png")
    plt.show()