import pandas as pd

import constants.constants as cst


def load_data(data_dir: str = cst.DATA_DIR) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Args:
        data_dir (str): Directory where the datasets are stored.
            The data directory must contain the following files:
            - train.csv
            - test.csv
            - sample_submission.csv

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Loaded datasets
            (train_data, test_data, sample_submission).
    """
    train_data_path = data_dir / "train.csv"
    train_data = pd.read_csv(train_data_path)

    test_data_path = data_dir / "test.csv"
    test_data = pd.read_csv(test_data_path)

    sample_submission_path = data_dir / "sample_submission.csv"
    sample_submission = pd.read_csv(sample_submission_path)

    return train_data, test_data, sample_submission
