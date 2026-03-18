import os
import pandas as pd


def get_dataset_names(processed_path="data/processed"):
    """
    Return dataset names from files like:
    X_train_stroke.csv -> stroke
    X_train_pima.csv   -> pima
    """
    files = os.listdir(processed_path)
    datasets = set()

    for file in files:
        if file.startswith("X_train_") and file.endswith(".csv"):
            name = file.replace("X_train_", "").replace(".csv", "")
            datasets.add(name)

    return sorted(list(datasets))


def load_dataset(dataset_name, processed_path="data/processed"):
    """
    Load train/test split for a given dataset name.
    Example dataset_name: 'stroke' or 'pima'
    """
    X_train = pd.read_csv(f"{processed_path}/X_train_{dataset_name}.csv").values
    X_test = pd.read_csv(f"{processed_path}/X_test_{dataset_name}.csv").values
    y_train = pd.read_csv(f"{processed_path}/y_train_{dataset_name}.csv").squeeze().values
    y_test = pd.read_csv(f"{processed_path}/y_test_{dataset_name}.csv").squeeze().values

    return X_train, X_test, y_train, y_test