import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parents[2] / "spiges_meteo_joined.csv"

FEATURE_COLS = [
    "month", "day_of_week", "week_of_year", "is_weekend", "is_winter", "is_summer",
    "notfall_lag1", "notfall_lag7", "notfall_roll7",
    "pct_elderly", "mean_severity", "mean_nems", "ips_cases",
    "temperature_avg", "temperature_max", "temperature_min",
]

TARGET_COL = "target_notfall_next24h"

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)

n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]

train_features = train[FEATURE_COLS].reset_index(drop=True)
train_target   = train[TARGET_COL].reset_index(drop=True)

val_features   = val[FEATURE_COLS].reset_index(drop=True)
val_target     = val[TARGET_COL].reset_index(drop=True)

test_features  = test[FEATURE_COLS].reset_index(drop=True)
test_target    = test[TARGET_COL].reset_index(drop=True)
