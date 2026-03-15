```python
# ===============================================================
# DATA PREPROCESSING SCRIPT FOR PROJECT
# Decision Tree vs Random Forest vs Extremely Randomized Trees
# ===============================================================
#
# This script prepares two datasets for the experiments:
#
# 1. Stroke Prediction Dataset
# 2. Breast Cancer Wisconsin Dataset
#
# The goal is to:
# - clean the data
# - encode categorical variables
# - separate features (X) and target (Y)
# - split into training and testing sets
#
# ===============================================================

# Import required libraries
import pandas as pd
import numpy as np

# sklearn utilities for dataset splitting
from sklearn.model_selection import train_test_split

# sklearn dataset loader for Breast Cancer dataset
from sklearn.datasets import load_breast_cancer


# ===============================================================
# PART 1: PREPROCESS STROKE DATASET
# ===============================================================

# Load the stroke dataset
# Make sure stroke_raw.csv is placed in the data/ folder
stroke_df = pd.read_csv("data/stroke_raw.csv")

# ---------------------------------------------------------------
# STEP 1: Remove unnecessary columns
# ---------------------------------------------------------------
# The 'id' column is just a unique identifier and does not help
# the model learn any pattern, so we remove it.

if 'id' in stroke_df.columns:
    stroke_df = stroke_df.drop(columns=['id'])


# ---------------------------------------------------------------
# STEP 2: Handle missing values
# ---------------------------------------------------------------
# In the stroke dataset, the BMI column sometimes contains
# missing values. We replace them with the median value.
#
# Median is often preferred for medical data because it is
# robust to outliers.

stroke_df['bmi'] = stroke_df['bmi'].fillna(stroke_df['bmi'].median())


# ---------------------------------------------------------------
# STEP 3: Encode categorical variables
# ---------------------------------------------------------------
# Some columns contain text categories (e.g. gender, work_type).
# Machine learning models require numerical inputs.
#
# We convert categorical columns into numeric form using
# one-hot encoding.

categorical_cols = [
    'gender',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'
]

stroke_df = pd.get_dummies(stroke_df, columns=categorical_cols)


# ---------------------------------------------------------------
# STEP 4: Separate predictors (X) and target (Y)
# ---------------------------------------------------------------
# Y = variable we want to predict
# X = all input features used by the model

X_stroke = stroke_df.drop(columns=['stroke'])
y_stroke = stroke_df['stroke']


# ---------------------------------------------------------------
# STEP 5: Train-test split
# ---------------------------------------------------------------
# We divide the dataset into:
# - training set (80%)
# - testing set (20%)
#
# The training set is used to train models.
# The test set evaluates model performance.

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_stroke,
    y_stroke,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------------------
# STEP 6: Save the processed stroke dataset
# ---------------------------------------------------------------
# This allows all team members to load the same clean dataset.

X_train_s.to_csv("data/X_train_stroke.csv", index=False)
X_test_s.to_csv("data/X_test_stroke.csv", index=False)

y_train_s.to_csv("data/y_train_stroke.csv", index=False)
y_test_s.to_csv("data/y_test_stroke.csv", index=False)


print("Stroke dataset preprocessing completed.")


# ===============================================================
# PART 2: PREPROCESS BREAST CANCER DATASET
# ===============================================================

# Load the dataset directly from sklearn
cancer = load_breast_cancer()

# Convert dataset to pandas DataFrame
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Target variable (malignant or benign)
y_cancer = pd.Series(cancer.target)


# ---------------------------------------------------------------
# STEP 1: Train-test split
# ---------------------------------------------------------------
# Breast cancer dataset is already very clean,
# so we only need to split the data.

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer,
    y_cancer,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------------------
# STEP 2: Save processed breast cancer dataset
# ---------------------------------------------------------------

X_train_c.to_csv("data/X_train_cancer.csv", index=False)
X_test_c.to_csv("data/X_test_cancer.csv", index=False)

y_train_c.to_csv("data/y_train_cancer.csv", index=False)
y_test_c.to_csv("data/y_test_cancer.csv", index=False)


print("Breast Cancer dataset preprocessing completed.")


# ===============================================================
# FINAL MESSAGE
# ===============================================================

print("All datasets are now cleaned and ready for modeling.")

print("Team members can now load these datasets to train:")
print("- Decision Tree")
print("- Random Forest")
print("- Extra Trees")
```
