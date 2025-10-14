from pathlib import Path

# Project directory structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Date format used in the dataset
DATE_FORMAT = "%Y-%m-%d %H"

# RAW Column names in the dataset
RAW_DATE = "id"
RAW_NO2 = "valeur_NO2"
RAW_CO = "valeur_CO"
RAW_O3 = "valeur_O3"
RAW_PM10 = "valeur_PM10"
RAW_PM25 = "valeur_PM25"

RAW_TARGETS = [RAW_NO2, RAW_CO, RAW_O3, RAW_PM10, RAW_PM25]
RAW_COLS = [RAW_DATE, RAW_NO2, RAW_CO, RAW_O3, RAW_PM10, RAW_PM25]

# Processed Column names
DATE = "date"
NO2 = "NO2"
CO = "CO"
O3 = "O3"
PM10 = "PM10"
PM25 = "PM25"

TARGETS = [NO2, CO, O3, PM10, PM25]
COLS = [DATE, NO2, CO, O3, PM10, PM25]

# column names in the predictions csv file
PREDICTIONS_PATH = PROJECT_ROOT / "output"
PREDICTIONS_COL = "predictions"
TRUE_VALUES_COL = "true_values"

# Colors for plotting
COLOR_TRAIN = "#1f77b4"  # Blue
COLOR_PREDICTION = "#ff7f0e"  # Orange
