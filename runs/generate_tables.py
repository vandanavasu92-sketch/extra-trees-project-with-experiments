import os
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from utils.data_loader import load_dataset
from utils.dataset_config import DATASET_CONFIG


RESULTS_DIR = "results"
INPUT_FILE = "all_models_results.csv"
MODEL_ORDER = ["ST", "TB", "RF", "ET"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_to_csv(df, path):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        print(f"Permission denied while writing: {path}")
        print("Please close the file if it is open in Excel or another program.")
        raise


def _model_sort_key(model_name):
    if model_name in MODEL_ORDER:
        return MODEL_ORDER.index(model_name)
    return 999


def build_dataset_summary_table(datasets):
    """
    Dataset summary table for classification and regression datasets.
    """
    rows = []

    for dataset_name in datasets:
        X, y = load_dataset(dataset_name)
        task = DATASET_CONFIG[dataset_name]["task"]

        n_total = X.shape[0]
        n_features = X.shape[1]
        n_test = int(round(n_total * 0.2))
        n_train = n_total - n_test

        row = {
            "dataset": dataset_name,
            "task": task,
            "atts": n_features,
            "train_size": n_train,
            "test_size": n_test,
            "total_size": n_total,
        }

        if task == "classification":
            row["classes"] = int(len(np.unique(y)))
        else:
            row["classes"] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    cls_df = (
        df[df["task"] == "classification"]
        .drop(columns=["task"])
        [["dataset", "atts", "classes", "train_size", "test_size", "total_size"]]
        .sort_values("dataset")
        .reset_index(drop=True)
    )

    reg_df = (
        df[df["task"] == "regression"]
        .drop(columns=["task", "classes"])
        [["dataset", "atts", "train_size", "test_size", "total_size"]]
        .sort_values("dataset")
        .reset_index(drop=True)
    )

    return cls_df, reg_df


def build_detailed_per_dataset_tables(final_df):
    """
    Per-dataset results tables.
    Classification uses Test Error (mean ± std).
    Regression uses Test MSE (mean ± std).
    """
    cls = final_df[final_df["task"] == "classification"].copy()
    reg = final_df[final_df["task"] == "regression"].copy()

    cls_table = pd.DataFrame()
    reg_table = pd.DataFrame()

    if not cls.empty:
        cls_summary = (
            cls.groupby(["dataset", "model"], as_index=False)
            .agg(
                mean_test_err=("test_err", "mean"),
                std_test_err=("test_err", "std"),
                mean_test_acc=("test_acc", "mean"),
                std_test_acc=("test_acc", "std")
            )
        )

        cls_summary["Test Error (mean ± std)"] = cls_summary.apply(
            lambda r: f"{r['mean_test_err']:.4f} ± {r['std_test_err']:.4f}", axis=1
        )

        cls_table = (
            cls_summary.pivot(index="dataset", columns="model", values="Test Error (mean ± std)")
            .reset_index()
        )

        ordered_cols = ["dataset"] + [m for m in MODEL_ORDER if m in cls_table.columns]
        cls_table = cls_table[ordered_cols]

    if not reg.empty:
        reg_summary = (
            reg.groupby(["dataset", "model"], as_index=False)
            .agg(
                mean_test_mse=("test_mse", "mean"),
                std_test_mse=("test_mse", "std"),
                mean_test_r2=("test_r2", "mean"),
                std_test_r2=("test_r2", "std")
            )
        )

        reg_summary["Test MSE (mean ± std)"] = reg_summary.apply(
            lambda r: f"{r['mean_test_mse']:.4f} ± {r['std_test_mse']:.4f}", axis=1
        )

        reg_table = (
            reg_summary.pivot(index="dataset", columns="model", values="Test MSE (mean ± std)")
            .reset_index()
        )

        ordered_cols = ["dataset"] + [m for m in MODEL_ORDER if m in reg_table.columns]
        reg_table = reg_table[ordered_cols]

    return cls_table, reg_table


def build_table1_performance(final_df):
    """
    Overall performance summary by task.
    """
    cls = final_df[final_df["task"] == "classification"].copy()
    reg = final_df[final_df["task"] == "regression"].copy()

    cls_table = pd.DataFrame()
    reg_table = pd.DataFrame()

    if not cls.empty:
        cls_table = (
            cls.groupby("model", as_index=False)
            .agg(
                mean_test_err=("test_err", "mean"),
                std_test_err=("test_err", "std"),
                mean_test_acc=("test_acc", "mean"),
                std_test_acc=("test_acc", "std")
            )
        )
        cls_table["Test Error (mean ± std)"] = cls_table.apply(
            lambda r: f"{r['mean_test_err']:.4f} ± {r['std_test_err']:.4f}", axis=1
        )
        cls_table["Test Accuracy (mean ± std)"] = cls_table.apply(
            lambda r: f"{r['mean_test_acc']:.4f} ± {r['std_test_acc']:.4f}", axis=1
        )
        cls_table["sort_key"] = cls_table["model"].map(_model_sort_key)
        cls_table = cls_table.sort_values(["sort_key", "mean_test_err"]).drop(columns=["sort_key"])
        cls_table = cls_table[["model", "Test Error (mean ± std)", "Test Accuracy (mean ± std)"]]

    if not reg.empty:
        reg_table = (
            reg.groupby("model", as_index=False)
            .agg(
                mean_test_mse=("test_mse", "mean"),
                std_test_mse=("test_mse", "std"),
                mean_test_r2=("test_r2", "mean"),
                std_test_r2=("test_r2", "std")
            )
        )
        reg_table["Test MSE (mean ± std)"] = reg_table.apply(
            lambda r: f"{r['mean_test_mse']:.4f} ± {r['std_test_mse']:.4f}", axis=1
        )
        reg_table["Test R2 (mean ± std)"] = reg_table.apply(
            lambda r: f"{r['mean_test_r2']:.4f} ± {r['std_test_r2']:.4f}", axis=1
        )
        reg_table["sort_key"] = reg_table["model"].map(_model_sort_key)
        reg_table = reg_table.sort_values(["sort_key", "mean_test_mse"]).drop(columns=["sort_key"])
        reg_table = reg_table[["model", "Test MSE (mean ± std)", "Test R2 (mean ± std)"]]

    return cls_table, reg_table


def corrected_resampled_ttest(scores_row, scores_col, n_train, n_test):
    """
    Corrected resampled paired t-test.
    Table compares COLUMN algorithm versus ROW algorithm.

    Lower score is better.
    """
    scores_row = np.asarray(scores_row, dtype=float)
    scores_col = np.asarray(scores_col, dtype=float)

    if len(scores_row) != len(scores_col):
        raise ValueError("Run count mismatch between compared models")

    k = len(scores_row)
    if k < 2:
        return 0.0, 1.0

    d = scores_col - scores_row
    d_bar = np.mean(d)
    s2 = np.var(d, ddof=1)

    if s2 == 0:
        return 0.0, 1.0

    correction = (1.0 / k) + (n_test / n_train)
    t_stat = d_bar / np.sqrt(correction * s2)
    p_value = 2 * (1 - student_t.cdf(abs(t_stat), df=k - 1))
    return t_stat, p_value


def compare_column_vs_row(df_dataset, row_model, col_model, metric_col, alpha=0.05):
    """
    Returns:
    - 'win'  : column wins over row
    - 'draw' : no significant difference
    - 'loss' : column loses to row

    Here, 'win' means the COLUMN model is better than the ROW model.
    """
    row_df = df_dataset[df_dataset["model"] == row_model].sort_values("run")
    col_df = df_dataset[df_dataset["model"] == col_model].sort_values("run")

    if row_df.empty or col_df.empty:
        return None

    n_train = int(row_df["n_train"].iloc[0])
    n_test = int(row_df["n_test"].iloc[0])

    _, p_value = corrected_resampled_ttest(
        scores_row=row_df[metric_col].values,
        scores_col=col_df[metric_col].values,
        n_train=n_train,
        n_test=n_test
    )

    mean_row = row_df[metric_col].mean()
    mean_col = col_df[metric_col].mean()

    if p_value >= alpha:
        return "draw"

    if mean_col < mean_row:
        return "win"
    return "loss"


def build_table2_wdl(final_df, task, alpha=0.05):
    """
    Win/Draw/Loss records comparing algorithm in COLUMN versus algorithm in ROW.
    """
    df = final_df[final_df["task"] == task].copy()

    if df.empty:
        return pd.DataFrame()

    metric_col = "test_err" if task == "classification" else "test_mse"
    datasets = sorted(df["dataset"].unique())

    table = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER)

    for row_model in MODEL_ORDER:
        for col_model in MODEL_ORDER:
            if row_model == col_model:
                table.loc[row_model, col_model] = "-"
                continue

            wins = draws = losses = 0

            for dataset in datasets:
                df_dataset = df[df["dataset"] == dataset]
                outcome = compare_column_vs_row(
                    df_dataset=df_dataset,
                    row_model=row_model,
                    col_model=col_model,
                    metric_col=metric_col,
                    alpha=alpha
                )

                if outcome is None:
                    continue
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1
                else:
                    draws += 1

            table.loc[row_model, col_model] = f"{wins}/{draws}/{losses}"

    table.index.name = "row_model"
    return table.reset_index()


def build_table4_time(final_df):
    """
    Training time table in milliseconds.
    Uses median fit time to reduce outlier impact.
    """
    cls = final_df[final_df["task"] == "classification"].copy()
    reg = final_df[final_df["task"] == "regression"].copy()

    cls_table = pd.DataFrame()
    reg_table = pd.DataFrame()

    if not cls.empty:
        cls_table = (
            cls.groupby(["dataset", "model"], as_index=False)
            .agg(median_fit_time_sec=("fit_time", "median"))
        )
        cls_table["fit_time_ms"] = (
            cls_table["median_fit_time_sec"] * 1000.0
        ).round(0).astype(int)

        cls_table = (
            cls_table.pivot(index="dataset", columns="model", values="fit_time_ms")
            .reset_index()
        )
        ordered_cols = ["dataset"] + [m for m in MODEL_ORDER if m in cls_table.columns]
        cls_table = cls_table[ordered_cols]

    if not reg.empty:
        reg_table = (
            reg.groupby(["dataset", "model"], as_index=False)
            .agg(median_fit_time_sec=("fit_time", "median"))
        )
        reg_table["fit_time_ms"] = (
            reg_table["median_fit_time_sec"] * 1000.0
        ).round(0).astype(int)

        reg_table = (
            reg_table.pivot(index="dataset", columns="model", values="fit_time_ms")
            .reset_index()
        )
        ordered_cols = ["dataset"] + [m for m in MODEL_ORDER if m in reg_table.columns]
        reg_table = reg_table[ordered_cols]

    return cls_table, reg_table


def build_table8_overall(final_df):
    """
    Overall comparison summary using average rank and best-dataset count.
    Lower is better for both error and MSE.
    """
    rows = []

    for task in ["classification", "regression"]:
        df_task = final_df[final_df["task"] == task].copy()
        if df_task.empty:
            continue

        metric_col = "test_err" if task == "classification" else "test_mse"
        ranks_by_model = {m: [] for m in MODEL_ORDER}
        best_count = {m: 0 for m in MODEL_ORDER}

        for dataset in sorted(df_task["dataset"].unique()):
            df_dataset = (
                df_task[df_task["dataset"] == dataset]
                .groupby("model", as_index=False)
                .agg(mean_metric=(metric_col, "mean"))
                .sort_values("mean_metric")
                .reset_index(drop=True)
            )

            for rank, model_name in enumerate(df_dataset["model"], start=1):
                ranks_by_model[model_name].append(rank)

            best_model = df_dataset.iloc[0]["model"]
            best_count[best_model] += 1

        for model_name in MODEL_ORDER:
            model_ranks = ranks_by_model[model_name]
            if not model_ranks:
                continue

            rows.append({
                "task": task,
                "model": model_name,
                "avg_rank": round(float(np.mean(model_ranks)), 4),
                "best_dataset_count": int(best_count[model_name]),
                "num_datasets": int(len(model_ranks))
            })

    overall_df = pd.DataFrame(rows)
    if not overall_df.empty:
        overall_df["sort_key"] = overall_df["model"].map(_model_sort_key)
        overall_df = overall_df.sort_values(["task", "avg_rank"]).drop(columns=["sort_key"])

    return overall_df


def generate_paper_tables(final_df, datasets):
    """
    Create all required output tables.
    """
    ensure_dir(RESULTS_DIR)

    ds_cls, ds_reg = build_dataset_summary_table(datasets)
    safe_to_csv(ds_cls, os.path.join(RESULTS_DIR, "table_dataset_summary_classification.csv"))
    safe_to_csv(ds_reg, os.path.join(RESULTS_DIR, "table_dataset_summary_regression.csv"))

    t1_cls, t1_reg = build_table1_performance(final_df)
    safe_to_csv(t1_cls, os.path.join(RESULTS_DIR, "table1_performance_classification.csv"))
    safe_to_csv(t1_reg, os.path.join(RESULTS_DIR, "table1_performance_regression.csv"))

    td_cls, td_reg = build_detailed_per_dataset_tables(final_df)
    safe_to_csv(td_cls, os.path.join(RESULTS_DIR, "table_detailed_results_classification.csv"))
    safe_to_csv(td_reg, os.path.join(RESULTS_DIR, "table_detailed_results_regression.csv"))

    t2_cls = build_table2_wdl(final_df, task="classification", alpha=0.05)
    t2_reg = build_table2_wdl(final_df, task="regression", alpha=0.05)
    safe_to_csv(t2_cls, os.path.join(RESULTS_DIR, "table2_wdl_classification.csv"))
    safe_to_csv(t2_reg, os.path.join(RESULTS_DIR, "table2_wdl_regression.csv"))

    t4_cls, t4_reg = build_table4_time(final_df)
    safe_to_csv(t4_cls, os.path.join(RESULTS_DIR, "table4_time_classification.csv"))
    safe_to_csv(t4_reg, os.path.join(RESULTS_DIR, "table4_time_regression.csv"))

    t8 = build_table8_overall(final_df)
    safe_to_csv(t8, os.path.join(RESULTS_DIR, "table8_overall_summary.csv"))

    print("\n" + "#" * 80)
    print("Saved paper-style tables:")
    print("  table_dataset_summary_classification.csv")
    print("  table_dataset_summary_regression.csv")
    print("  table1_performance_classification.csv")
    print("  table1_performance_regression.csv")
    print("  table_detailed_results_classification.csv")
    print("  table_detailed_results_regression.csv")
    print("  table2_wdl_classification.csv")
    print("  table2_wdl_regression.csv")
    print("  table4_time_classification.csv")
    print("  table4_time_regression.csv")
    print("  table8_overall_summary.csv")
    print("#" * 80)


def generate_tables_from_csv(input_path):
    """
    Entry point to generate all tables from raw results CSV.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find {input_path}. Run run_all_models first."
        )

    final_df = pd.read_csv(input_path)
    datasets = sorted(final_df["dataset"].dropna().unique().tolist())

    generate_paper_tables(final_df, datasets)

    print("All tables generated successfully.")