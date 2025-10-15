# Fill submission DataFrame with forecasted values for Kaggle
import pandas as pd

import constants.constants as cst


def fill_submission(
    submission: pd.DataFrame,
    forecast: pd.DataFrame,
    target: str,
    file_name: str = "submission.csv",
    save: bool = False,
):
    """Fills the submission DataFrame with forecasted values for a given target variable.

    Args:
        forecast (pd.DataFrame): Forecasted data
        target (str): Target variable name of forecasted data
        file_name (str, optional): File name to save the submission DataFrame. Defaults to "submission.csv".
        save (bool, optional): Whether to save the submission DataFrame. Defaults to False. (Save if all values filled)

    Returns:
        pd.DataFrame: Updated submission DataFrame
    """
    submission[target] = forecast["yhat"]

    if save:
        submission.to_csv(cst.PREDICTIONS_PATH / file_name)
    return submission
