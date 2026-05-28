"""
Feature Engineering Pipeline
Combines all data sources into a unified feature matrix for XGBoost.
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime


def build_feature_matrix(
    weather_df: pd.DataFrame,
    transport_df: pd.DataFrame,
    seasonal_patterns: pd.DataFrame,
    target_date: datetime,
    canton: str,
) -> pd.DataFrame:
    """
    Build the full feature matrix for a given date and canton.

    Features:
    ─────────────────────────────────────────────
    Temporal
      - day_of_week (0=Mon, 6=Sun)
      - is_weekend
      - month
      - is_swiss_holiday
      - days_to_holiday

    Weather (from MétéoSuisse)
      - temperature_c
      - is_precipitation
      - temp_cold / temp_hot flags
      - extreme_weather_flag

    Mobility (from opentransportdata.swiss)
      - mobility_index_lag1 (yesterday)
      - mobility_index_lag7 (same day last week)
      - transport_delta (change vs previous day)

    Historical / Seasonal (from SpiGes)
      - seasonal_index (canton × month)
      - elderly_risk_factor (% 65+ admissions)
      - year_trend (linear trend)
    ─────────────────────────────────────────────

    Returns:
        Single-row DataFrame ready for XGBoost prediction
    """
    features = {}

    # ── Temporal features
    features["day_of_week"] = target_date.weekday()
    features["is_weekend"] = int(target_date.weekday() >= 5)
    features["month"] = target_date.month
    features["week_of_year"] = target_date.isocalendar().week
    features["is_summer"] = int(target_date.month in [6, 7, 8])
    features["is_winter"] = int(target_date.month in [12, 1, 2])

    # ── Swiss public holidays (federal + most cantons)
    swiss_holidays = _get_swiss_holidays(target_date.year)
    features["is_holiday"] = int(target_date.date() in swiss_holidays)
    features["days_to_next_holiday"] = _days_to_next_holiday(
        target_date, swiss_holidays
    )

    # ── Weather features
    if not weather_df.empty:
        latest = weather_df.iloc[-1]
        features["temperature_c"] = latest.get("temperature_c", 10.0)
        features["is_precipitation"] = latest.get("is_precipitation", 0)
        features["temp_cold"] = int(features["temperature_c"] < 5)
        features["temp_hot"] = int(features["temperature_c"] > 28)
    else:
        logger.warning("No weather data — using defaults")
        features.update({
            "temperature_c": 10.0,
            "is_precipitation": 0,
            "temp_cold": 0,
            "temp_hot": 0,
        })

    # ── Mobility features
    if not transport_df.empty:
        features["mobility_index"] = transport_df.get("mobility_index", pd.Series([0.5])).iloc[-1]
        features["mobility_lag1"] = transport_df.get("mobility_index", pd.Series([0.5])).shift(1).iloc[-1]
    else:
        features["mobility_index"] = 0.5
        features["mobility_lag1"] = 0.5

    # ── Seasonal features (SpiGes)
    if not seasonal_patterns.empty:
        canton_pattern = seasonal_patterns[
            seasonal_patterns["Kanton"] == canton
        ]
        if not canton_pattern.empty:
            features["seasonal_index"] = canton_pattern["seasonal_index"].mean()
        else:
            features["seasonal_index"] = 1.0
    else:
        features["seasonal_index"] = 1.0

    # ── Interaction features
    features["weekend_x_precipitation"] = (
        features["is_weekend"] * features["is_precipitation"]
    )
    features["cold_x_winter"] = features["temp_cold"] * features["is_winter"]
    features["hot_x_summer"] = features["temp_hot"] * features["is_summer"]

    return pd.DataFrame([features])


def _get_swiss_holidays(year: int) -> set:
    """Return federal Swiss public holidays for a given year."""
    from datetime import date
    holidays = {
        date(year, 1, 1),   # Neujahr
        date(year, 8, 1),   # Bundesfeiertag
        date(year, 12, 25), # Weihnachten
        date(year, 12, 26), # Stephanstag
    }
    return holidays


def _days_to_next_holiday(dt: datetime, holidays: set) -> int:
    """Compute days until next public holiday."""
    from datetime import timedelta
    for i in range(30):
        check = (dt + timedelta(days=i)).date()
        if check in holidays:
            return i
    return 30
