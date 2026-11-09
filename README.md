# NYC Citi Bike 2024: where bikes are needed, and when

An end-to-end analysis of **44,158,430 Citi Bike trips** (all of 2024), joined to
NOAA weather and NYC's bike lane network, built around an operator's question:
where do bikes pile up and run out, and can tomorrow's demand be predicted
station by station?

**[View the live site](https://ancientabacus.github.io/NYC-CitiBike-Analysis/)**

![Ride volume by hour and day of week, 2024](docs/assets/heatmap.png)

## Key findings

- **The commute moves about 5,927 bikes the wrong way every weekday morning.**
  676 stations lose 2+ bikes each morning and 409 gain 2+; Brooklyn
  loses ~908 bikes a morning and Manhattan's office
  districts absorb them. The evening rush mirrors it (station-level correlation -0.92).
- **Weather sets the ceiling.** Holding temperature, weekday and system growth fixed, light rain
  cuts a day's rides by 16% and heavy rain by 44%.
  Dry-weekday ridership doubles from 5°C to 25°C, then plateaus.
- **Demand is forecastable a day ahead.** A gradient-boosted model of hourly departures for
  2,273 stations, trained on Jan–Oct and scored on unseen Nov–Dec, has
  28% lower error than "same hour last week"
  (MAE 0.97 vs. 1.35 rides per station-hour) and
  28% lower than each station's typical level.

## What's in the repo

| Path | What it does |
|---|---|
| [`docs/index.html`](docs/index.html) | The site: one scrolling page with the heatmap, seasons & weather, ridership maps, rebalancing map, forecast evaluation, and a written report. Plain HTML/SVG/JS charts ([`docs/assets/charts.js`](docs/assets/charts.js)), Leaflet maps. |
| [`scripts/pipeline.py`](scripts/pipeline.py) | Single source of truth for loading, cleaning, caching (parquet per month), and NOAA weather. |
| [`scripts/build_heatmap.py`](scripts/build_heatmap.py) | Hour × weekday ride volume. |
| [`scripts/build_geomap.py`](scripts/build_geomap.py) | Station totals, point-in-polygon borough assignment, bike lanes. |
| [`scripts/build_seasonality.py`](scripts/build_seasonality.py) | Daily rides vs. weather; log-linear model for rain/snow/holiday effects. |
| [`scripts/build_rebalancing.py`](scripts/build_rebalancing.py) | Weekday rush-hour net flow per station; bikes to move. |
| [`scripts/build_forecast.py`](scripts/build_forecast.py) | Station × hour demand forecast with a date-based split and three baselines. |
| [`CitiBike_Analysis.ipynb`](CitiBike_Analysis.ipynb) | Exploratory deep-dive on January 2024, including a member-vs-casual classifier and the class-imbalance accuracy trap. |

## Method notes

- **Cleaning** (in `pipeline.py`): drop duplicate rides, out-of-month spillover, trips missing
  station or coordinates, non-positive durations, and trips over 24h (Citi Bike's lost/stolen
  threshold). Removes 0.33% of 44.3M raw rows. Every section reads the same cleaned trips.
- **Forecast evaluation** is strictly by date: train Jan 8–Aug, early-stop on Sep–Oct, refit on
  Jan–Oct, test Nov–Dec. Every feature is known a day ahead (calendar, weather, lags of 24h/168h,
  same-hour 7-day mean, training-period station profile). A day-of-year feature was tried and
  removed: with a date split, test dates are values the trees never saw, and validation error
  rose as trees were added.
- **Limitations:** the forecast uses observed weather (an upper bound on what forecast weather
  would give); the trip data has no dock capacity, so "bikes to move" is a lower bound on
  rebalancing work; one year of data.

## Data sources

- Trips: [Citi Bike System Data](https://citibikenyc.com/system-data) (monthly zips on S3)
- Weather: [NOAA NCEI](https://www.ncei.noaa.gov/), Central Park station (hourly LCD + GHCN-Daily)
- Bike lanes and borough boundaries: [NYC Open Data](https://opendata.cityofnewyork.us/)

Nothing is committed; everything downloads into a gitignored `.cache/` on first run.

## Reproducing

```bash
pip install -r requirements.txt
```

Each script takes a period, `YYYY` (default `2024`) or `YYYYMM`. Run in this order (the first run
downloads ~7.5 GB of trip data; the forecast takes ~15 minutes and ~6 GB RAM on 4 cores):

```bash
python scripts/build_heatmap.py
```

```bash
python scripts/build_geomap.py
```

```bash
python scripts/build_seasonality.py
```

```bash
python scripts/build_rebalancing.py
```

```bash
python scripts/build_forecast.py
```

The notebook runs top to bottom from the repo root:

```bash
jupyter nbconvert --to notebook --execute --inplace CitiBike_Analysis.ipynb
```

The report and hero text in `docs/index.html` are hand-written; if you regenerate with different
data, update the figures cited there.

## Deployment

`.github/workflows/` deploys `docs/` to GitHub Pages on every push to `main`.
