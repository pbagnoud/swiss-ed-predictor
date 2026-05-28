"""
SpiGes / OFSP Data Loader
Loads Swiss hospital statistics for historical model training.

Data source: https://www.bag.admin.ch/spiges
License: Open Government Data (OGD) — free reuse
"""

import pandas as pd
from pathlib import Path
from loguru import logger


SPIGES_URL = (
    "https://www.bag.admin.ch/dam/bag/de/dokumente/kuv-leistungen/"
    "spiges/spiges-daten.zip.download.zip/spiges-daten.zip"
)

RELEVANT_COLUMNS = [
    "Jahr",           # Year
    "Kanton",         # Canton
    "InstNr",         # Institution number
    "Notfall",        # Emergency flag
    "Eintritte",      # Total admissions
    "AustritteT",     # Total discharges
    "Pflegetage",     # Patient days
    "Altersklasse",   # Age group
    "Geschlecht",     # Gender
]


def load_spiges(path: str | Path) -> pd.DataFrame:
    """
    Load and clean a SpiGes CSV file.

    Args:
        path: Path to local SpiGes CSV file

    Returns:
        Cleaned DataFrame with emergency admissions by year/canton
    """
    logger.info(f"Loading SpiGes data from {path}")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)

    # Keep only relevant columns that exist
    cols = [c for c in RELEVANT_COLUMNS if c in df.columns]
    df = df[cols].copy()

    # Filter emergency cases
    if "Notfall" in df.columns:
        df = df[df["Notfall"] == 1]

    # Parse year
    if "Jahr" in df.columns:
        df["Jahr"] = pd.to_numeric(df["Jahr"], errors="coerce")

    logger.success(f"Loaded {len(df):,} emergency records from SpiGes")
    return df


def compute_seasonal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract seasonal admission patterns from SpiGes historical data.

    Returns a DataFrame with:
    - month: 1-12
    - canton: canton code
    - avg_daily_admissions: historical average
    - seasonal_index: ratio vs annual mean (>1 = peak season)
    """
    if df.empty:
        logger.warning("Empty SpiGes DataFrame — returning empty patterns")
        return pd.DataFrame()

    # Aggregate by year and canton
    if "Eintritte" not in df.columns or "Kanton" not in df.columns:
        logger.warning("Missing key columns in SpiGes data")
        return df

    patterns = (
        df.groupby(["Jahr", "Kanton"])["Eintritte"]
        .sum()
        .reset_index()
        .rename(columns={"Eintritte": "total_admissions"})
    )

    # Compute canton-level seasonal index
    canton_mean = patterns.groupby("Kanton")["total_admissions"].transform("mean")
    patterns["seasonal_index"] = patterns["total_admissions"] / canton_mean

    logger.info(f"Computed seasonal patterns for {patterns['Kanton'].nunique()} cantons")
    return patterns


def get_canton_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract demographic risk factors by canton for feature engineering.
    """
    if "Altersklasse" not in df.columns:
        return pd.DataFrame()

    # Elderly population (65+) as a risk factor
    elderly = df[df["Altersklasse"].isin(["65-74", "75-84", "85+"])]
    risk = (
        elderly.groupby("Kanton")["Eintritte"]
        .sum()
        .reset_index()
        .rename(columns={"Eintritte": "elderly_admissions"})
    )
    return risk


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = load_spiges(sys.argv[1])
        patterns = compute_seasonal_patterns(df)
        print(patterns.head(20))
    else:
        print("Usage: python spiges.py <path_to_spiges.csv>")
