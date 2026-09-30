"""Scratchpad for pulling and looking at the weather data.

Run from the repo root (the relative venv path needs that cwd):

    .venv/bin/python solar_generation_forecast/explore.py

Other helpful information:   
    GHI = DNI × cos(solar zenith) + DHI
"""

import re
from hashlib import md5
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

# --- site and window ------------------------------------------------------
# Previous Runs archive starts Feb 2024. Earlier dates return HTTP 200 with
# all-null values, so they fail silently rather than erroring.
LAT, LON = 34.05, -118.24  # Los Angeles
TZ = "America/Los_Angeles"
START, END = "2024-03-01", "2025-08-31"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

CACHE = Path(__file__).parent / "cache"

# Window for the diagnostic plot. Seaborn renders static, so a couple of weeks
# is about as much as stays readable at hourly resolution.
PLOT_START, PLOT_END = "2024-06-01", "2024-06-14"


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

    # Cache key has to cover everything that changes the request, variables
    # included, or editing any of them silently hands back the previous pull.
    CACHE.mkdir(exist_ok=True)
    fingerprint = md5(f"{url}{sorted(params.items())}".encode()).hexdigest()[:8]
    cached = CACHE / f"{label}_{START}_{END}_{fingerprint}.csv"
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
            "shortwave_radiation", # ghi, total sunlight on a flat horizontal surface 
            "direct_normal_irradiance", # dni, the beam straight from the sun's disc, measured on a surface held perpendicular to the rays
            "diffuse_radiation", # dhi, scattered skylight arriving from the whole sky dome on a horiz surface
            "temperature_2m",
            "wind_speed_10m",
            "cloud_cover",
        ],
        label="era5_actuals",
    )


def strip_lead_suffix(column: str) -> str:
    """Drop the _previous_dayN suffix so forecast columns match the actuals.

    The lead time belongs to the pull, not to every column name. Stripping it
    here is what lets the merge in data_checks() produce _actual / _forecast
    suffixes off a name collision.
    """
    return re.sub(r"_previous_day\d+$", "", column)


def get_forecasts() -> pd.DataFrame:
    """Day-ahead forecast fields. The feature source.
    
    
    TODO: same shape as get_actuals(), two differences.
      - hit PREVIOUS_RUNS_URL instead of ARCHIVE_URL
      - every variable needs a _previous_day1 suffix, which is what pins the
        lead time to one day, e.g. shortwave_radiation_previous_day1
    """
    return get_hourly(
        PREVIOUS_RUNS_URL,
        [
            "shortwave_radiation_previous_day1", # ghi, total sunlight on a flat horizontal surface 
            "direct_normal_irradiance_previous_day1", # dni, the beam straight from the sun's disc, measured on a surface held perpendicular to the rays
            "diffuse_radiation_previous_day1", # dhi, scattered skylight arriving from the whole sky dome on a horiz surface
            "temperature_2m_previous_day1",
            "wind_speed_10m_previous_day1",
            "cloud_cover_previous_day1",
        ],
        label="da_weather_forecasts",
    ).rename(columns=strip_lead_suffix)

def data_checks(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Merge the two sources and eyeball forecast temperature against actual.

    Temperature is the easy variable to start on: smooth, strongly diurnal, and
    the forecast should track it closely. That makes a timing problem obvious.
    Returns the merged frame.
    """
    # TODO Check that DNI * cos(zenith) + DHI ~ GHI within each df
    # TODO Check where daily peaks land relative to solar noon

    merged = actuals.merge(forecasts, on="time", suffixes=("_actual", "_forecast"))

    error = merged["temperature_2m_forecast"] - merged["temperature_2m_actual"]
    print(f"\ntemperature, forecast vs actual over {len(merged)} hours")
    print(f"  mean bias: {error.mean():+.2f} degC")
    print(f"  MAE:       {error.abs().mean():.2f} degC")
    print(f"  worst:     {error.abs().max():.2f} degC")

    window = merged[merged["time"].between(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))]

    fig, (ax_series, ax_error) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    sns.lineplot(data=window, x="time", y="temperature_2m_actual", ax=ax_series,
                 color="#4C78A8", label="actual (ERA5)")
    sns.lineplot(data=window, x="time", y="temperature_2m_forecast", ax=ax_series,
                 color="#E45756", label="forecast (day-1 lead)")
    ax_series.set_ylabel("temperature (degC)")
    ax_series.set_xlabel("")
    ax_series.set_title(f"Day-ahead forecast vs actual temperature, {PLOT_START} to {PLOT_END}")
    ax_series.legend(loc="upper left", fontsize=9)
    ax_series.grid(alpha=0.25)

    sns.lineplot(x=window["time"], y=window["temperature_2m_forecast"] - window["temperature_2m_actual"],
                 ax=ax_error, color="#54A24B")
    ax_error.axhline(0, color="gray", ls=":", lw=1.5)
    ax_error.set_ylabel("forecast - actual")
    ax_error.set_xlabel("")
    ax_error.grid(alpha=0.25)

    # A flat offset is bias. An error that swings with the daily cycle, crossing
    # zero twice a day, points at a timestamp offset instead.
    fig.tight_layout()
    plt.show()

    return merged


# --- things to work out before modeling -----------------------------------
# TODO: build the target. pvlib PVWatts chain on the ERA5 fields.
# TODO: build the baseline. Same chain, forecast fields instead.
# TODO: features, models, evaluation.


if __name__ == "__main__":
    actuals = get_actuals()

    print(f"Actuals:")
    print(f"{len(actuals)} rows, {actuals.time.min()} to {actuals.time.max()}")
    print(f"\ncolumns: {list(actuals.columns)}")
    print(f"\nnulls per column:\n{actuals.isna().sum().to_string()}")
    print(f"\n{actuals.describe().round(2).to_string()}")
    print(f"\nfirst daylight hours:\n{actuals.head(24).to_string(index=False)}")

    da_forecasts = get_forecasts()
    print("\nForecasts:")
    print(f"{len(da_forecasts)} rows, {da_forecasts.time.min()} to {da_forecasts.time.max()}")
    print(da_forecasts.head())

    merged = data_checks(da_forecasts, actuals)