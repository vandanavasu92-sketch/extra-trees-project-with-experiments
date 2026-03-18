# ============================================================
# evaluator.py
# Common evaluation function for all classification models
# ============================================================

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X_train, y_train, X_test, y_test,
                   model_name, dataset_name, parameter_name=None, parameter_value=None):
    """
    Train the model, make predictions, calculate metrics,
    print results, and return them as a dictionary.
    """

    # ------------------------------------------------
    # 1. Train model
    # ------------------------------------------------
    model.fit(X_train, y_train)

    # ------------------------------------------------
    # 2. Predict on train and test
    # ------------------------------------------------
    y_train_pred = np.array(model.predict(X_train))
    y_test_pred = np.array(model.predict(X_test))

    # ------------------------------------------------
    # 3. Calculate metrics
    # ------------------------------------------------
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    unique, counts = np.unique(y_test_pred, return_counts=True)
    pred_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    cm = confusion_matrix(y_test, y_test_pred)

    # ------------------------------------------------
    # 4. Print results
    # ------------------------------------------------
    print("\n" + "=" * 60)

    if parameter_name is not None:
        print(f"{model_name} Evaluation for {parameter_name} = {parameter_value}")
    else:
        print(f"{model_name} Evaluation")

    print("=" * 60)
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-score       : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    # Sample predictions
    print("\nSample Predictions:")
    print("First 10 train predictions:", y_train_pred[:10])
    print("First 10 test predictions:", y_test_pred[:10])

    # Distribution (important for imbalance)
    print("Prediction distribution:", pred_dist)
    

    # ------------------------------------------------
    # 5. Observation
    # ------------------------------------------------
    gap = train_acc - test_acc

    if gap > 0.10:
        observation = "Possible overfitting."
    elif recall == 0 and f1 == 0:
        observation = "Model fails to detect minority class."
    elif recall < 0.20:
        observation = "Minority class detection is still very weak."
    else:
        observation = "No strong overfitting based on accuracy gap."

    print(f"\nObservation: {observation}")

    # ------------------------------------------------
    # 6. Return result dictionary
    # ------------------------------------------------
    result = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Parameter Name": parameter_name,
        "Parameter Value": parameter_value,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Overfitting Gap": gap
    }

    return result