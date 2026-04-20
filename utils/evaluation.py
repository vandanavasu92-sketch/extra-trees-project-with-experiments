from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def evaluate_predictions(task, y_true, y_pred):
    """
    Common evaluation for all models.

    Regression:
        - mse
        - r2

    Classification:
        - acc
        - err = 1 - acc
    """
    if task == "regression":
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            "mse": float(mse),
            "r2": float(r2),
            "acc": None,
            "err": None,
        }

    if task == "classification":
        acc = accuracy_score(y_true, y_pred)
        err = 1.0 - acc
        return {
            "mse": None,
            "r2": None,
            "acc": float(acc),
            "err": float(err),
        }

    raise ValueError("task must be 'classification' or 'regression'")