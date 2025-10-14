import pandas as pd
import os

INPUT_FILE = "/home/rico/HEC/07-time_series/paris-air-quality-prediction/data/train.csv"
OUTPUT_FILE = "data/train_interpolated_small_gaps.csv"
DATETIME_COL = "id"
MAX_GAP = 2  # fill only sequences of 1–2 NaNs

def main():
    if not os.path.exists(INPUT_FILE):
        return

    df = pd.read_csv(INPUT_FILE)

    # Ensure chronological order for proper interpolation
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    # Interpolate only on numeric columns
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col] = df[col].interpolate(
            method="linear",
            limit=MAX_GAP,
            limit_direction="both"
        )

    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
