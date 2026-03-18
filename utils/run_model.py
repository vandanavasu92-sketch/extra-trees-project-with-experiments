# ============================================================
# run_model.py
# Common runner for Decision Tree, Random Forest, Extra Trees
# ============================================================

from utils.data_loader import get_dataset_names, load_dataset
from utils.evaluator import evaluate_model


def run_model_pipeline(model_name, model_builder, parameter_name=None, parameter_values=None):
    """
    Common runner for all models.

    Parameters
    ----------
    model_name : str
        Name of the model for printing and storing results

    model_builder : function
        A function that returns a model object.
        Example:
            lambda depth: DecisionTree(max_depth=depth, min_samples_split=5)

    parameter_name : str, optional
        Name of the parameter being tuned, e.g. 'max_depth'

    parameter_values : list, optional
        List of parameter values to try, e.g. [3, 5, 10]

    Returns
    -------
    all_results : list of dict
        List of evaluation results across all datasets and parameter values
    """

    all_results = []

    print("\n" + "#" * 70)
    print(f"Running {model_name} pipeline...")
    print("#" * 70)

    datasets = get_dataset_names()

    for dataset_name in datasets:
        print("\n" + "#" * 70)
        print(f"Running {model_name} for dataset: {dataset_name}")
        print("#" * 70)

        # --------------------------------------------
        # 1. Load dataset
        # --------------------------------------------
        X_train, X_test, y_train, y_test = load_dataset(dataset_name)

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        # --------------------------------------------
        # 2. If no parameter loop, run once
        # --------------------------------------------
        if parameter_values is None:
            model = model_builder()

            result = evaluate_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                dataset_name=dataset_name,
                parameter_name=None,
                parameter_value=None
            )

            all_results.append(result)

        # --------------------------------------------
        # 3. If parameter loop exists, run for each
        # --------------------------------------------
        else:
            print("\n" + "#" * 70)
            print(f"Dataset: {dataset_name}")
            print("#" * 70)

            for param_value in parameter_values:
                model = model_builder(param_value)

                result = evaluate_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    parameter_name=parameter_name,
                    parameter_value=param_value
                )

                all_results.append(result)

    print("\nPipeline execution completed successfully!")

    return all_results