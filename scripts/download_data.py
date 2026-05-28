"""
Data Download Script
Downloads all open data sources needed for the Swiss ED Predictor.
"""

import httpx
import os
from pathlib import Path
from loguru import logger

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "meteo_stations": {
        "url": "https://data.geo.admin.ch/ch.meteoschweiz.ogd-surface-stations/ch.meteoschweiz.ogd-surface-stations_en.csv",
        "filename": "meteo_stations.csv",
        "description": "MétéoSuisse — surface station metadata",
    },
    "ofs_pop": {
        "url": "https://www.bfs.admin.ch/bfsstatic/dam/assets/canton-population.csv",
        "filename": "ofs_population.csv",
        "description": "OFS — cantonal population data",
    },
}


def download_all():
    for key, source in SOURCES.items():
        dest = RAW_DIR / source["filename"]
        if dest.exists():
            logger.info(f"Already downloaded: {source['filename']}")
            continue

        logger.info(f"Downloading {source['description']}...")
        try:
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                response = client.get(source["url"])
                response.raise_for_status()
                dest.write_bytes(response.content)
                logger.success(f"Saved {dest} ({len(response.content):,} bytes)")
        except Exception as e:
            logger.warning(f"Could not download {key}: {e}")
            logger.info(f"  Manual download: {source['url']}")

    logger.info("\n📌 Manual downloads required:")
    logger.info("  SpiGes: https://www.bag.admin.ch/spiges")
    logger.info("  opentransportdata.swiss API key: https://opentransportdata.swiss/en/register/")


if __name__ == "__main__":
    download_all()
