"""Scratchpad for pulling and looking at the weather data.

Run from the repo root (the relative venv path needs that cwd):

    .venv/bin/python solar_generation_forecast/explore.py
"""

from pathlib import Path

import pandas as pd
import requests

# --- site and window ------------------------------------------------------
# Previous Runs archive starts Feb 2024. Earlier dates return HTTP 200 with
# all-null values, so they fail silently rather than erroring.
LAT, LON = 34.05, -118.24  # Los Angeles
TZ = "America/Los_Angeles"
START, END = "2024-03-01", "2025-08-31"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

CACHE = Path(__file__).parent / "cache"


def get_hourly(url: str, variables: list[str], label: str) -> pd.DataFrame:
    """GET an Open-Meteo endpoint and return its hourly block as a frame.

    Caches per label, site, and date range, so repeated runs skip the API but
    changing any of those triggers a fresh pull.
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START,
        "end_date": END,
        "hourly": ",".join(variables),
        "timezone": TZ,
    }

    # Cache key has to include everything that changes the request, or editing
    # the dates silently hands back the previous pull.
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{label}_{LAT}_{LON}_{START}_{END}.csv"
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["time"])

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.to_csv(cached, index=False)
    return df


def get_actuals() -> pd.DataFrame:
    """ERA5 reanalysis. This is the proxy for actuals, the target source."""
    return get_hourly(
        ARCHIVE_URL,
        [
            "shortwave_radiation",
            "direct_normal_irradiance",
            "diffuse_radiation",
            "temperature_2m",
            "wind_speed_10m",
            "cloud_cover",
        ],
        label="era5_actuals",
    )


def get_forecast() -> pd.DataFrame:
    """Day-ahead forecast fields. The feature source.

    TODO: same shape as get_actuals(), two differences.
      - hit PREVIOUS_RUNS_URL instead of ARCHIVE_URL
      - every variable needs a _previous_day1 suffix, which is what pins the
        lead time to one day, e.g. shortwave_radiation_previous_day1
    """
    raise NotImplementedError


# --- things to work out before modeling -----------------------------------
# TODO: do the two sources line up on time? ERA5 hourly values are period
#       aggregates, forecast fields are generally instantaneous. Check where
#       each one puts its daily peak relative to solar noon.
# TODO: build the target. pvlib PVWatts chain on the ERA5 fields.
# TODO: build the baseline. Same chain, forecast fields instead.
# TODO: features, models, evaluation.


if __name__ == "__main__":
    actuals = get_actuals()

    print(f"{len(actuals)} rows, {actuals.time.min()} to {actuals.time.max()}")
    print(f"\ncolumns: {list(actuals.columns)}")
    print(f"\nnulls per column:\n{actuals.isna().sum().to_string()}")
    print(f"\n{actuals.describe().round(2).to_string()}")
    print(f"\nfirst daylight hours:\n{actuals.head(24).to_string(index=False)}")
