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
import pvlib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

# --- site and window ------------------------------------------------------
# Previous Runs archive starts Feb 2024. Earlier dates return HTTP 200 with
# all-null values, so they fail silently rather than erroring.
LAT, LON = 34.05, -118.24  # Los Angeles
TZ = "America/Los_Angeles"  # display only; requests are made in UTC
START, END = "2024-02-01", "2026-09-29"
TEST_START = "2025-10-01"  # test set begins here: a full 12-month seasonal cycle
TILT, AZIMUTH = 34, 180 # assumes panel tilt~lat, facing due south
PDC0 = 1000 # nameplate DC in watts, setting to 1000 makes this per-kW
GAMMA_PDC = -0.004 # power lost per degC above 25*C

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

CACHE = Path(__file__).parent / "cache"

# Window for the diagnostic plot. Seaborn renders static, so a couple of weeks
# is about as much as stays readable at hourly resolution.
PLOT_START, PLOT_END = "2024-06-01", "2024-06-14"


def get_hourly(url: str, variables: list[str], label: str) -> pd.DataFrame:
    """GET an Open-Meteo endpoint and return its hourly block as a frame.

    Caches per label, site, and date range, so repeated runs skip the API but
    changing any of those triggers a fresh pull. Wind is converted to m/s on
    the way out; the cached file holds the raw km/h response.
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START,
        "end_date": END,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }

    # Cache key has to cover everything that changes the request, variables
    # included, or editing any of them silently hands back the previous pull.
    CACHE.mkdir(exist_ok=True)
    fingerprint = md5(f"{url}{sorted(params.items())}".encode()).hexdigest()[:8]
    cached = CACHE / f"{label}_{START}_{END}_{fingerprint}.csv"
    if cached.exists():
        df = pd.read_csv(cached, parse_dates=["time"])
    else:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        df = pd.DataFrame(response.json()["hourly"])
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
        df.to_csv(cached, index=False)

    # Open-Meteo serves wind in km/h. pvlib's cell temperature models want m/s.
    # Converting here rather than downstream means every caller gets m/s, and
    # the cache keeps the raw response so this can never be applied twice.
    for col in [c for c in df.columns if c.startswith("wind_speed")]:
        df[col] = df[col] / 3.6

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

    window = merged[
        merged["time"].between(pd.Timestamp(PLOT_START, tz=TZ), pd.Timestamp(PLOT_END, tz=TZ))
    ].copy()
    # Data is UTC. Convert for the axis so the diurnal cycle reads correctly.
    window["local_time"] = window["time"].dt.tz_convert(TZ)

    fig, (ax_series, ax_error) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    sns.lineplot(data=window, x="local_time", y="temperature_2m_actual", ax=ax_series,
                 color="#4C78A8", label="actual (ERA5)")
    sns.lineplot(data=window, x="local_time", y="temperature_2m_forecast", ax=ax_series,
                 color="#E45756", label="forecast (day-1 lead)")
    ax_series.set_ylabel("temperature (degC)")
    ax_series.set_xlabel("")
    ax_series.set_title(
        f"Day-ahead forecast vs actual temperature, {PLOT_START} to {PLOT_END} ({TZ})"
    )
    ax_series.legend(loc="upper left", fontsize=9)
    ax_series.grid(alpha=0.25)

    sns.lineplot(x=window["local_time"], y=window["temperature_2m_forecast"] - window["temperature_2m_actual"],
                 ax=ax_error, color="#54A24B")
    ax_error.axhline(0, color="gray", ls=":", lw=1.5)
    ax_error.set_ylabel("forecast - actual")
    ax_error.set_xlabel("")
    ax_error.grid(alpha=0.25)

    # A flat offset is bias. An error that swings with the daily cycle, crossing
    # zero twice a day, points at a timestamp offset instead.
    fig.tight_layout()

    return merged

def simulate_generation(merged: pd.DataFrame, suffix: str = "_actual") -> pd.Series:
    """Run the pvlib PVWatts chain over one source's columns, return hourly watts.

    Called twice on the same frame: suffix="_actual" builds the target,
    suffix="_forecast" builds the physical baseline. Same physics both times,
    which is the whole point of it being one function.

    TODO the four steps:
      1. Location(LAT, LON, tz="UTC").get_solarposition(merged["time"])
      2. irradiance.get_total_irradiance(...) for poa_global
      3. temperature.pvsyst_cell(poa_global, temp_air, wind_speed)
      4. pvsystem.pvwatts_dc(poa_global, temp_cell, pdc0, gamma_pdc)

    Still to pick: surface_tilt, surface_azimuth (180 is due south), pdc0,
    and gamma_pdc (negative, around -0.004).
    """
    # grab forecasts/actuals columns only
    selected_cols = {col: col.replace(suffix,"") for col in merged.columns if suffix in col}
    df = merged.set_index("time")[list(selected_cols)].rename(columns=selected_cols)

    # get sun location
    sun_location = pvlib.location.Location(LAT, LON, tz="UTC").get_solarposition(df.index)

    # how much light hits a tilted panel
    plane_of_array_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=TILT,
        surface_azimuth=AZIMUTH, 
        solar_zenith=sun_location['apparent_zenith'],
        solar_azimuth=sun_location['azimuth'],
        dni=df['direct_normal_irradiance'],
        ghi=df['shortwave_radiation'],
        dhi=df['diffuse_radiation']
    )

    # get cell temp
    cell_temp = pvlib.temperature.pvsyst_cell(
        poa_global=plane_of_array_irradiance['poa_global'],
        temp_air=df["temperature_2m"],
        wind_speed=df["wind_speed_10m"] # in m/s
    )

    # get pv generation in dc
    pv_dc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=plane_of_array_irradiance['poa_global'],
        temp_cell=cell_temp,
        pdc0=PDC0,
        gamma_pdc=GAMMA_PDC
    )

    return pv_dc


def plot_day(date: str, forecast: pd.Series, actual: pd.Series) -> None:
    """Plot one local day of DC generation, forecast against actual.

    date is a local calendar day, e.g. "2024-06-21". Both Series are indexed in
    UTC, so the index is converted before filtering, otherwise the day boundary
    lands seven hours off.
    """
    day = pd.Timestamp(date).date()
    frame = pd.DataFrame({"forecast": forecast, "actual": actual})
    frame.index = frame.index.tz_convert(TZ)
    frame = frame[frame.index.date == day]

    if frame.empty:
        raise ValueError(f"{date} is outside {START} to {END}")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=frame.index, y=frame["actual"], ax=ax,
                 marker="o", color="#4C78A8", label="actual (ERA5)")
    sns.lineplot(x=frame.index, y=frame["forecast"], ax=ax,
                 marker="o", color="#E45756", label="forecast (day-1 lead)")

    for col, color in [("actual", "#4C78A8"), ("forecast", "#E45756")]:
        peak = frame[col].idxmax()
        ax.axvline(peak, color=color, ls=":", lw=1.2, alpha=0.7)
        print(f"  {col:8s} peak {frame[col].max():6.1f} W at local hour {peak.hour}")

    ax.set_ylabel(f"DC power (W, pdc0={PDC0})")
    ax.set_xlabel(f"local time ({TZ})")
    ax.set_title(f"Simulated DC generation, {date}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()


def daylight_filter(df, max_zenith: float = 90.0):
    """Keep only the rows where the sun is high enough to generate.

    Takes anything with a tz-aware DatetimeIndex, frame or Series. Solar
    position is computed from the index, so this does not require a zenith
    column to already exist.

    max_zenith is degrees from straight overhead, so 90 is the horizon and
    keeps every hour the sun is up at all. A stricter cutoff like 85 also drops
    the dawn and dusk hours where output is nearly zero.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("needs a DatetimeIndex; set_index('time') first")

    solar_position = pvlib.location.Location(LAT, LON, tz="UTC").get_solarposition(df.index)
    return df[solar_position["apparent_zenith"].to_numpy() < max_zenith]


def create_regression_dict(y_true, y_pred, label: str = "") -> dict:
    """MAE, MedAE and RMSE, in watts and as a percentage of PDC0.

    Deliberately knows nothing about filtering, so the same function serves
    daylight hours, ramp hours, or any other subset. Returns a dict so several
    of these stack into a DataFrame for side-by-side printing.

    No MAPE: it divides by the true value, and generation approaches zero at
    dawn and dusk, so a few rows would swamp the metric.
    """
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {
        "label": label,
        "n": len(y_true),
        "MAE_W": round(mae, 1),
        "MedAE_W": round(medae, 1),
        "RMSE_W": round(rmse, 1),
        "MAE_pct": round(100 * mae / PDC0, 2),
        "MedAE_pct": round(100 * medae / PDC0, 2),
        "RMSE_pct": round(100 * rmse / PDC0, 2),
    }

def build_features(merged: pd.DataFrame, target: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    """Assemble features, target and baseline into one daylight-only frame.

    Features come from the _forecast columns only, plus quantities derived from
    time and geometry, which belong to neither source. The baseline is included
    as a feature on purpose: the model then only has to learn the correction to
    the physics rather than rediscover the physics.

    Keeping target and baseline in the same frame means one split instead of
    three objects to hold in alignment.
    """
    forecast_cols = {c: c.replace("_forecast", "") for c in merged.columns if c.endswith("_forecast")}
    frame = merged.set_index("time")[list(forecast_cols)].rename(columns=forecast_cols)

    site = pvlib.location.Location(LAT, LON, tz="UTC")
    solar_position = site.get_solarposition(frame.index)
    clearsky = site.get_clearsky(frame.index)

    frame["apparent_zenith"] = solar_position["apparent_zenith"]
    frame["azimuth"] = solar_position["azimuth"]
    frame["clearsky_ghi"] = clearsky["ghi"]

    # Fraction of a clear sky getting through, and how much of it is scattered.
    # These two carry most of the sky-condition signal, and more cleanly than
    # cloud_cover, which ERA5 reports almost bimodally.
    frame["clearsky_index"] = (
        frame["shortwave_radiation"] / clearsky["ghi"].where(clearsky["ghi"] > 0)
    ).clip(0, 1.2)
    frame["diffuse_fraction"] = (
        frame["diffuse_radiation"] / frame["shortwave_radiation"].where(frame["shortwave_radiation"] > 0)
    ).clip(0, 1)

    # Local hour, not UTC, so the number lines up with the diurnal cycle.
    frame["hour"] = frame.index.tz_convert(TZ).hour
    frame["day_of_year"] = frame.index.dayofyear

    frame["baseline"] = baseline
    frame["target"] = target

    frame = daylight_filter(frame).dropna()

    leaked = [c for c in frame.columns if "_actual" in c]
    assert not leaked, f"actual columns leaked into features: {leaked}"
    return frame


def split_by_time(frame: pd.DataFrame, cutoff: str = TEST_START):
    """Chronological train/test split at a calendar date.

    train_test_split with shuffle=False slices in order rather than randomising,
    which is what you want here: adjacent hours are highly correlated, so a
    random split would put near-duplicates of test rows into training and
    inflate every score. test_size is derived from the date so the boundary
    lands exactly where intended instead of at an arbitrary fraction.
    """
    test_fraction = float((frame.index >= pd.Timestamp(cutoff, tz="UTC")).mean())
    return train_test_split(frame, test_size=test_fraction, shuffle=False)


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

    # create pv_dc_forecast, a baseline of sorts with pure weather inputs; and pv_dc_actual, a target based on ERA5 weather reanalysis
    pv_dc_forecast = simulate_generation(merged, suffix="_forecast")
    pv_dc_actual = simulate_generation(merged, suffix="_actual")

    print("\nDC generation, 2024-06-21:")
    plot_day("2024-06-21", pv_dc_forecast, pv_dc_actual)

    frame = build_features(merged, pv_dc_actual, pv_dc_forecast)
    train, test = split_by_time(frame)
    feature_cols = [c for c in frame.columns if c != "target"]

    print(f"\n{len(feature_cols)} features: {feature_cols}")
    print(f"train {len(train)} rows, {train.index.min().date()} to {train.index.max().date()}")
    print(f"test  {len(test)} rows, {test.index.min().date()} to {test.index.max().date()}")

    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(train[feature_cols], train["target"])
    predictions = model.predict(test[feature_cols])

    results = pd.DataFrame([
        create_regression_dict(test["target"], test["baseline"], "baseline (physics on forecast)"),
        create_regression_dict(test["target"], predictions, "HistGradientBoosting"),
    ])
    print(f"\nTest set, daylight hours only:\n{results.to_string(index=False)}")

    baseline_mae, model_mae = results["MAE_W"].iloc[0], results["MAE_W"].iloc[1]
    print(f"\nskill vs baseline (MAE): {1 - model_mae / baseline_mae:+.1%}")

    # show every figure plotted
    plt.show()

    # print regression dict for baseline vs actual vs gradient boosted tree forecast

