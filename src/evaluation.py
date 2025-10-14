# Evaluation script for the models

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# evaluate all models and print table of results

def evaluate_models(models: list[str], y_pred: pd.DataFrame, y_true: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all models and print table of results"""
    metrics = {"mse": mean_squared_error, "mae": mean_absolute_error}
    res = {model: {} for model in models}
    for model in models:
        for metric in metrics:
            res[model][metric] = metrics[metric](y_true, y_pred)
    return pd.DataFrame(res)