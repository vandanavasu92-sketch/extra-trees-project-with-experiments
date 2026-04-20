"""
Preprocessing for Extra Trees project

✔ Handles datasets WITHOUT headers
✔ Cleans + encodes data
✔ Saves FULL dataset only (no train/test split)
"""

import numpy as np
import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ============================================================
# Helper functions
# ============================================================

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_dataset(name, X, y):
    out_dir = PROCESSED_DIR / name
    ensure_dir(out_dir)

    X.to_csv(out_dir / "X.csv", index=False)
    y.to_frame("target").to_csv(out_dir / "y.csv", index=False)

    print(f"Saved: {name}")


def one_hot_encode(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


# ============================================================
# CLASSIFICATION DATASETS
# ============================================================

def preprocess_pima():
    df = pd.read_csv(RAW_DIR / "pima.csv", header=None)

    df.columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]

    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)

    save_dataset("pima", X, y)


def preprocess_breast_cancer():
    df = pd.read_csv(RAW_DIR / "breast_cancer.csv", header=None)

    df.columns = [
        "id", "diagnosis",
        *[f"f{i}" for i in range(30)]
    ]

    df = df.dropna(how="all")
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip()

    df = df[df["diagnosis"].isin(["M", "B"])]
    df = df.drop(columns=["id"])

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].astype(int)

    save_dataset("breast_cancer", X, y)


def preprocess_ionosphere():
    df = pd.read_csv(RAW_DIR / "ionosphere.csv", header=None)

    df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["target"]
    df["target"] = df["target"].map({"g": 1, "b": 0})

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    save_dataset("ionosphere", X, y)


def preprocess_sonar():
    df = pd.read_csv(RAW_DIR / "sonar.csv", header=None)

    df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["target"]
    df["target"] = df["target"].map({"M": 1, "R": 0})

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    save_dataset("sonar", X, y)


def preprocess_vehicle():
    df = pd.read_csv(RAW_DIR / "vehicle.csv", header=None)

    # remove fully empty rows
    df = df.dropna(how="all")

    # set columns
    df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["target"]

    # clean target labels
    df["target"] = df["target"].astype(str).str.strip().str.lower()

    # convert feature columns to numeric
    feature_cols = df.columns[:-1]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop malformed rows
    df = df.dropna()

    # keep only valid class labels
    valid_labels = {"bus", "opel", "saab", "van"}
    df = df[df["target"].isin(valid_labels)]

    # map to integer labels
    class_map = {label: idx for idx, label in enumerate(sorted(valid_labels))}
    df["target"] = df["target"].map(class_map)

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    save_dataset("vehicle", X, y)


def preprocess_heart_disease():
    df = pd.read_csv(RAW_DIR / "heart_disease.csv", header=None)

    df = df.replace("?", np.nan)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    for col in df.columns:
        if df[col].isna().any():
            if str(df[col].dtype) != "object":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    target_col = df.columns[-1]

    df[target_col] = (df[target_col] > 0).astype(int)

    X = one_hot_encode(df.drop(columns=[target_col]))
    y = df[target_col].astype(int)

    save_dataset("heart_disease", X, y)


# ============================================================
# REGRESSION DATASETS
# ============================================================

def preprocess_abalone():
    df = pd.read_csv(RAW_DIR / "abalone.csv", header=None)

    df.columns = [
        "Sex", "Length", "Diameter", "Height",
        "Whole_weight", "Shucked_weight",
        "Viscera_weight", "Shell_weight", "Rings"
    ]

    X = one_hot_encode(df.drop(columns=["Rings"]))
    y = df["Rings"].astype(float)

    save_dataset("abalone", X, y)


def preprocess_concrete():
    possible_files = [
        RAW_DIR / "concrete.xlsx",
        RAW_DIR / "Concrete_Data.xlsx",
        RAW_DIR / "concrete.xls",
        RAW_DIR / "Concrete_Data.xls",
    ]

    file_path = None
    for f in possible_files:
        if f.exists():
            file_path = f
            break

    if file_path is None:
        raise FileNotFoundError("Concrete dataset not found in data/raw")

    if file_path.suffix == ".xls":
        df = pd.read_excel(file_path, engine="xlrd")
    else:
        df = pd.read_excel(file_path)

    target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target].astype(float)

    save_dataset("concrete", X, y)


def preprocess_energy_efficiency():
    df = pd.read_excel(RAW_DIR / "energy_efficiency.xlsx")

    target = df.columns[-2]

    X = df.drop(columns=[target])
    y = df[target].astype(float)

    save_dataset("energy_efficiency", X, y)


def preprocess_airfoil():
    df = pd.read_csv(RAW_DIR / "airfoil_self_noise.csv", sep=r"\s+", header=None)

    df.columns = ["f1", "f2", "f3", "f4", "f5", "target"]

    X = df.drop(columns=["target"])
    y = df["target"].astype(float)

    save_dataset("airfoil", X, y)


def preprocess_cpu_performance():
    df = pd.read_csv(RAW_DIR / "cpu_performance.csv")

    drop_cols = [c for c in ["vendor_name", "model_name"] if c in df.columns]

    target = df.columns[-1]

    X = one_hot_encode(df.drop(columns=drop_cols + [target]))
    y = df[target].astype(float)

    save_dataset("cpu_performance", X, y)


def preprocess_boston_housing():
    df = pd.read_csv(RAW_DIR / "boston_housing.csv")

    target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target].astype(float)

    save_dataset("boston_housing", X, y)


# ============================================================
# MAIN
# ============================================================

DATASETS = [
    preprocess_pima,
    preprocess_breast_cancer,
    preprocess_ionosphere,
    preprocess_sonar,
    preprocess_vehicle,
    preprocess_heart_disease,
    preprocess_abalone,
    preprocess_concrete,
    preprocess_energy_efficiency,
    preprocess_airfoil,
    preprocess_cpu_performance,
    preprocess_boston_housing,
]


if __name__ == "__main__":
    ensure_dir(PROCESSED_DIR)

    print("=" * 60)
    print("Running preprocessing...")
    print("=" * 60)

    for fn in DATASETS:
        try:
            fn()
        except Exception as e:
            print(f"Error in {fn.__name__}: {e}")

    print("=" * 60)
    print("Preprocessing complete.")