# Preprocessing script for time series datasets

import pandas as pd
from qolmat.imputations import imputers

import constants.constants as cst

default_imputer = imputers.ImputerSimple()


def preprocess_data(
    df: pd.DataFrame, imputer: imputers._BaseImputer = default_imputer
) -> pd.DataFrame:
    """Preprocess the raw training dataset.

    Args:
        df (pd.DataFrame): Raw training dataset.
        imputer (imputers._BaseImputer): Imputer from QOLMAT to handle missing values.
            Default is ImputerSimple that imputes the mean for each column.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Rename columns
    df = df.rename(
        columns={
            cst.RAW_DATE: cst.DATE,
            cst.RAW_NO2: cst.NO2,
            cst.RAW_CO: cst.CO,
            cst.RAW_O3: cst.O3,
            cst.RAW_PM10: cst.PM10,
            cst.RAW_PM25: cst.PM25,
        }
    )

    # Convert date column to datetime
    df[cst.DATE] = pd.to_datetime(df[cst.DATE], format=cst.DATE_FORMAT)

    # Impute missing values using QOLMAT
    df[cst.TARGETS] = imputer.fit_transform(df[cst.TARGETS])

    return df
