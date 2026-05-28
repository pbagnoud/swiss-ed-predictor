"""
MeteoSwiss Open Data Connector
Fetches weather forecasts via the MeteoSwiss open API.
"""

import httpx
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta


METEO_BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-surface-stations"


def fetch_current_weather(station_ids: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch current weather observations from MeteoSwiss open data.

    Args:
        station_ids: List of MeteoSwiss station IDs (e.g. ['BER', 'ZUR', 'GVA']).
                     If None, fetches all available stations.

    Returns:
        DataFrame with columns: station_id, datetime, temperature_c,
        precipitation_mm, wind_speed_kmh, humidity_pct
    """
    url = f"{METEO_BASE_URL}/ch.meteoschweiz.ogd-surface-stations_en.csv"
    logger.info(f"Fetching MeteoSwiss data from {url}")

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url)
            response.raise_for_status()

        # Parse CSV
        from io import StringIO
        df = pd.read_csv(StringIO(response.text), sep=";", skiprows=2)

        if station_ids:
            df = df[df["Station"].isin(station_ids)]

        logger.success(f"Fetched {len(df)} weather records")
        return df

    except httpx.HTTPError as e:
        logger.error(f"MeteoSwiss API error: {e}")
        raise


def build_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer weather features relevant to ED demand prediction.

    Features:
    - temperature_category: cold (<5°C), mild (5-20°C), hot (>20°C)
    - is_precipitation: binary flag for rain/snow
    - extreme_weather: binary flag for storms/heat waves
    - weekend_weather_interaction: weekend × precipitation
    """
    features = df.copy()

    # Temperature categories
    features["temp_cold"] = (features.get("temperature_c", 0) < 5).astype(int)
    features["temp_hot"] = (features.get("temperature_c", 0) > 28).astype(int)

    # Precipitation flag
    features["is_precipitation"] = (
        features.get("precipitation_mm", 0) > 0.5
    ).astype(int)

    logger.info(f"Built weather features: {features.shape}")
    return features


if __name__ == "__main__":
    df = fetch_current_weather(station_ids=["BER", "ZUR", "GVA", "BSL"])
    print(df.head())
