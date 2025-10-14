import pandas as pd
import numpy as np
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

def preprocess_dates(df, drop=False):
    df['id'] = pd.to_datetime(df['id'])
    df.set_index('id', inplace=True)
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['doy'] = df.index.dayofyear
    for col, period in [('hour', 24), ('dow', 7), ('doy', 365)]:
        df[f"{col}_sin"] = np.sin(df[col] * 2 * np.pi / period)
        df[f"{col}_cos"] = np.cos(df[col] * 2 * np.pi / period)
        if drop:
            df.drop(columns=[col], inplace=True)
    return df

def get_training_data():
    df_train = pd.read_csv("./data/train.csv")
    df_train = preprocess_dates(df_train)
    return df_train

def get_test_data():
    df_test = pd.read_csv("./data/test.csv")
    df_test = preprocess_dates(df_test)
    return df_test

def get_daily_data(df: pd.DataFrame, columns=None):
    if not columns:
        columns = df.drop(columns=['date', 'hour'], errors='ignore').select_dtypes(include='number').columns.tolist()
    if columns:
        df = df[columns]
    return df.resample('D').mean()