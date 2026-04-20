import os
import pandas as pd


def get_dataset_names(processed_path="data/processed"):
    return sorted([
        d for d in os.listdir(processed_path)
        if os.path.isdir(os.path.join(processed_path, d))
    ])


def load_dataset(dataset_name, processed_path="data/processed"):
    X = pd.read_csv(f"{processed_path}/{dataset_name}/X.csv").values
    y = pd.read_csv(f"{processed_path}/{dataset_name}/y.csv").squeeze().values

    return X, y