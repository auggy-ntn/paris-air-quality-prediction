# Evaluation script for the models

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import constants.constants as cst

def evaluate_models(models: list[str]) -> pd.DataFrame:
    """Evaluate all models and print table of results"""

    metrics = {"mse": mean_squared_error, "mae": mean_absolute_error}
    res = {model: {} for model in models}
    for model in models:
        csv = pd.read_csv(cst.PREDICTIONS_PATH / f"{model}.csv")
        y_pred = csv[cst.PREDICTIONS_COL]
        y_true = csv[cst.TRUE_VALUES_COL]
        for metric in metrics:
            res[model][metric] = metrics[metric](y_true, y_pred)
    return pd.DataFrame(res) 