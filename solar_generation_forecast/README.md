# Day-ahead solar generation forecast

Physics-based weather models publish temperature, cloud cover, irradiance and
other weather input forecasts. This project forecasts hourly solar generation
a day ahead from those weather inputs, and asks whether a statistical (ML)
model beats running the raw forecast straight through a physical panel model.

## Motivation

A device running on a solar panel and a battery has to budget power against
what the sun actually delivers. The operational question is "will the solar
power cover the load, and should the device throttle back tonight."

That makes the two error directions asymmetric. Forecast high and the battery
drains, taking the device offline until it recovers. Forecast low and you
throttle for no reason. A model with good average error can still be the wrong
model if it misses in the expensive direction on the days that matter.

## Setup

Weather forecast in, PV generation out. One model learns the conversion and
the forecast's bias together.

    forecast weather (day-1 lead)  ->  [model]  ->  hourly generation

- **Features:** forecast fields at a fixed one-day lead, from the Open-Meteo
  [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api). The
  `_previous_day1` suffix returns the forecast for an hour as it was issued a
  day earlier, which holds lead time constant across the whole sample.
  Irradiance components, temperature, wind, and cloud cover by layer.

- **Target:** generation from a pvlib PVWatts model chain run on ERA5
  reanalysis, via the Open-Meteo
  [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
  Both endpoints are Open-Meteo, but they serve different datasets: archived
  forecasts for the features, reanalysis for the target.

- **Baseline:** the same PVWatts chain run on the forecast weather. That is
  the naive physical answer, so the gap between it and the target is forecast
  error alone. The model earns its place only by learning systematic bias.
  Skill is reported as `1 - MAE_model / MAE_baseline`.

## Evaluation

Daylight hours only, filtered on solar zenith. Roughly half the rows are
nighttime zeros that every model predicts perfectly, and leaving them in
inflates the scores without saying anything.

Error is broken out three ways, with the third being the point of interest:

- by hour of day
- by sky condition, clear through overcast
- **by ramp versus plateau.** Sunrise and sunset transitions are where solar
  forecasts actually fail, and a single aggregate error number hides that
  completely.

## What this cannot tell you

The target is synthetic. It is a physical model run on reanalysis, not a
metered panel.

First, ERA5 is itself a model. Reanalysis is well constrained for variables
with dense instrument coverage, like temperature, but there is no comparable
global network of pyranometers (devices which measure irradiance), so
reanalysis irradiance is computed from the model's own cloud field rather than
measured. Cloud coverage is the hardest thing for weather models to get right.

Second, the synthetic, modeled target has no soiling, shading, inverter
clipping, snow, or outages. Real generation does. The target is therefore a
smooth deterministic function of weather with none of the irreducible noise a
real panel has, so error metrics here will look better than a production
model's would.

Third, and most relevant to the ramp analysis above: ERA5 is hourly and
roughly 31 km, which smooths out exactly the fast cloud transitions that cause
real ramp errors. The ramp problem is present in this data but damped.

Upgrading the target to measured irradiance from
[SURFRAD](https://gml.noaa.gov/grad/surfrad/), NOAA's ground pyranometer
network, would remove the first and third limits. It constrains the site to
seven US stations, none near Los Angeles.

## Open questions

- **Model family.** Gradient boosting first, then a two-part model in the
  shape of logistic-for-positive-generation plus beta regression for the
  proportion of capacity. Ridge belongs in the comparison too: if a linear
  correction captures most of the gain over the baseline, that is the finding,
  because it says the forecast error is smooth bias rather than structure.
- **Direct or residual.** Predict generation directly, or predict the residual
  against the physical baseline. Near-equivalent, but the residual framing puts
  the baseline at zero and makes skill obvious.
- **Time alignment.** ERA5 hourly values are period aggregates while forecast
  fields are generally instantaneous. A one-hour offset would still run and
  still look plausible. This needs checking before any score is trusted.
- **Site.** Starting in the Los Angeles area.

## Scope

**This pass:** one site, one year, hourly, models compared against the
physical baseline, error broken out by hour, sky condition, and ramp.

**Later:** several sites across climate regimes. Cost-asymmetric evaluation
and a duty-cycle threshold sweep. Skill decay at longer lead times, which the
same endpoint supports through `_previous_day2` and `_previous_day3`.

## Structure

The target is isolated behind a single function so it can be swapped from
reanalysis to measured data without touching the feature or evaluation code.

Raw pulls are cached to `cache/` and are not tracked.

```bash
uv pip install -r requirements.txt   # from the repo root
```
