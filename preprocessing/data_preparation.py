# ===============================================================
# DATA PREPROCESSING SCRIPT FOR PROJECT
# Decision Tree vs Random Forest vs Extremely Randomized Trees
# ===============================================================
#
# This script prepares two datasets for the experiments:
#
# 1. Stroke Prediction Dataset
# 2. Pima Indians Diabetes Dataset
#
# The goal is to:
# - clean the data
# - encode categorical variables
# - separate features (X) and target (Y)
# - split into training and testing sets
#
# After this step, all team members will use the SAME datasets
# to ensure fair comparison between models.
#
# ===============================================================

# ---------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ---------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# ===============================================================
# PART 1: PREPROCESS STROKE DATASET
# ===============================================================

# Load the stroke dataset
# Make sure the file is placed in the data/ folder
# Example file name:
# data/healthcare-dataset-stroke-data.csv
stroke_df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

# ---------------------------------------------------------------
# STEP 1: REMOVE UNNECESSARY COLUMNS
# ---------------------------------------------------------------
# The 'id' column is only a patient identifier.
# It does not help the model learn any predictive pattern.
if "id" in stroke_df.columns:
    stroke_df = stroke_df.drop(columns=["id"])


# ---------------------------------------------------------------
# STEP 2: HANDLE MISSING VALUES
# ---------------------------------------------------------------
# In the stroke dataset, BMI often contains missing values.
# We fill missing BMI values with the median.
# Median is preferred because it is less affected by outliers.
stroke_df["bmi"] = stroke_df["bmi"].fillna(stroke_df["bmi"].median())


# ---------------------------------------------------------------
# STEP 3: ENCODE CATEGORICAL VARIABLES
# ---------------------------------------------------------------
# Machine learning models need numerical input.
# The following columns are categorical and must be converted
# into numeric form using one-hot encoding.
categorical_cols = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

stroke_df = pd.get_dummies(stroke_df, columns=categorical_cols, drop_first=False)


# ---------------------------------------------------------------
# STEP 4: SEPARATE FEATURES (X) AND TARGET (Y)
# ---------------------------------------------------------------
# X = predictor variables
# Y = target variable we want to predict
X_stroke = stroke_df.drop(columns=["stroke"])
y_stroke = stroke_df["stroke"]


# ---------------------------------------------------------------
# STEP 5: TRAIN-TEST SPLIT
# ---------------------------------------------------------------
# We split the data into:
# - 80% training data
# - 20% testing data
#
# stratify=y_stroke keeps the class distribution similar
# in both training and testing sets, which is important for
# classification problems.
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_stroke,
    y_stroke,
    test_size=0.2,
    random_state=42,
    stratify=y_stroke
)


# ---------------------------------------------------------------
# STEP 6: SAVE PROCESSED STROKE DATA
# ---------------------------------------------------------------
# Saving processed files ensures that all team members use
# the exact same cleaned dataset and split.
X_train_s.to_csv("data/processed/X_train_stroke.csv", index=False)
X_test_s.to_csv("data/processed/X_test_stroke.csv", index=False)
y_train_s.to_csv("data/processed/y_train_stroke.csv", index=False)
y_test_s.to_csv("data/processed/y_test_stroke.csv", index=False)

print("Stroke dataset preprocessing completed.")


# ===============================================================
# PART 2: PREPROCESS PIMA INDIANS DIABETES DATASET
# ===============================================================

# Load the Pima dataset
# If your CSV has no header row, use header=None and assign names.
# Example file name:
# data/pima-indians-diabetes.csv
pima_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

pima_df = pd.read_csv("data/pima-indians-diabetes.csv", header=None, names=pima_columns)


# ---------------------------------------------------------------
# STEP 1: CHECK FOR INVALID ZERO VALUES
# ---------------------------------------------------------------
# In this dataset, some medical variables contain 0 values that
# are not realistic and usually represent missing data.
#
# These columns should not normally be 0:
# - Glucose
# - BloodPressure
# - SkinThickness
# - Insulin
# - BMI
#
# We replace 0 with NaN so that they can be treated as missing values.
zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_as_missing_cols:
    pima_df[col] = pima_df[col].replace(0, np.nan)


# ---------------------------------------------------------------
# STEP 2: HANDLE MISSING VALUES
# ---------------------------------------------------------------
# After replacing invalid zeros with NaN, fill missing values
# using the median of each column.
# Median is a safe choice for medical variables because it is
# robust to extreme values.
for col in zero_as_missing_cols:
    pima_df[col] = pima_df[col].fillna(pima_df[col].median())


# ---------------------------------------------------------------
# STEP 3: SEPARATE FEATURES (X) AND TARGET (Y)
# ---------------------------------------------------------------
X_pima = pima_df.drop(columns=["Outcome"])
y_pima = pima_df["Outcome"]


# ---------------------------------------------------------------
# STEP 4: TRAIN-TEST SPLIT
# ---------------------------------------------------------------
# Again, use stratify to maintain class balance across train/test sets.
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_pima,
    y_pima,
    test_size=0.2,
    random_state=42,
    stratify=y_pima
)


# ---------------------------------------------------------------
# STEP 5: SAVE PROCESSED PIMA DATA
# ---------------------------------------------------------------
X_train_p.to_csv("data/processed/X_train_pima.csv", index=False)
X_test_p.to_csv("data/processed/X_test_pima.csv", index=False)
y_train_p.to_csv("data/processed/y_train_pima.csv", index=False)
y_test_p.to_csv("data/processed/y_test_pima.csv", index=False)

print("Pima Indians Diabetes dataset preprocessing completed.")


# ===============================================================
# FINAL MESSAGE
# ===============================================================
print("All datasets are now cleaned and ready for modeling.")
print("Team members can now load these datasets to train:")
print("- Decision Tree")
print("- Random Forest")
print("- Extra Trees")