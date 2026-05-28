"""
opentransportdata.swiss Connector
Fetches public transport occupancy and flow data.
"""

import httpx
import pandas as pd
from loguru import logger


OTD_BASE_URL = "https://api.opentransportdata.swiss/ojp2020"


def fetch_transport_occupancy(
    stop_ids: list[str],
    api_key: str,
    hours_ahead: int = 24,
) -> pd.DataFrame:
    """
    Fetch transport flow data around major hospital stops.

    Args:
        stop_ids: List of OJP stop IDs near hospitals
        api_key: opentransportdata.swiss API key
        hours_ahead: Forecast horizon in hours

    Returns:
        DataFrame with transport occupancy metrics per stop per hour
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/xml",
    }

    logger.info(f"Fetching transport data for {len(stop_ids)} stops")

    records = []
    for stop_id in stop_ids:
        try:
            # OJP StopEventRequest
            xml_payload = _build_stop_event_request(stop_id)
            with httpx.Client(timeout=30) as client:
                response = client.post(OTD_BASE_URL, content=xml_payload, headers=headers)
                response.raise_for_status()

            parsed = _parse_stop_event_response(response.text, stop_id)
            records.extend(parsed)

        except httpx.HTTPError as e:
            logger.warning(f"Transport API error for stop {stop_id}: {e}")
            continue

    df = pd.DataFrame(records)
    logger.success(f"Fetched {len(df)} transport records")
    return df


def compute_mobility_index(df: pd.DataFrame) -> pd.Series:
    """
    Compute a mobility pressure index (0-1) as proxy for ED arrivals.
    Higher mobility = more people circulating = more potential ED cases.
    """
    if df.empty:
        return pd.Series(dtype=float)

    # Normalize departures + arrivals per stop
    mobility = (
        df.groupby("hour")["passenger_count"]
        .sum()
        .rolling(3, min_periods=1)
        .mean()
    )
    return (mobility - mobility.min()) / (mobility.max() - mobility.min())


def _build_stop_event_request(stop_id: str) -> bytes:
    """Build minimal OJP StopEventRequest XML payload."""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<OJP xmlns="http://www.vdv.de/ojp" xmlns:siri="http://www.siri.org.uk/siri" version="1.0">
  <OJPRequest>
    <siri:ServiceRequest>
      <OJPStopEventRequest>
        <Location>
          <PlaceRef>
            <StopPlaceRef>{stop_id}</StopPlaceRef>
          </PlaceRef>
        </Location>
        <Params>
          <NumberOfResults>20</NumberOfResults>
        </Params>
      </OJPStopEventRequest>
    </siri:ServiceRequest>
  </OJPRequest>
</OJP>"""
    return xml.encode("utf-8")


def _parse_stop_event_response(xml_text: str, stop_id: str) -> list[dict]:
    """Parse OJP response — stub for hackathon, extend as needed."""
    # TODO: implement full XML parsing
    return [{"stop_id": stop_id, "hour": 0, "passenger_count": 0}]
