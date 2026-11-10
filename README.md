# NYC Citi Bike 2024: where bikes are needed, and when

<!--fact:total_rides-->?<!--/fact--> trips, NOAA weather and NYC's bike lane network, used to answer one operating question:
where do bikes pile up and run out, and can tomorrow's demand be predicted station by station?

**[Live site](https://ancientabacus.github.io/NYC-CitiBike-Analysis/)** · **[Notebook](CitiBike_2024.ipynb)**

## Results

| Finding | Number |
|---|---|
| Bikes left in the wrong place each weekday morning (lower bound) | <!--fact:bikes_to_move-->?<!--/fact--> |
| Stations losing / gaining 2+ bikes each morning | <!--fact:drainer_count-->?<!--/fact--> / <!--fact:filler_count-->?<!--/fact--> |
| Morning vs. evening net flow, correlation (adjusted for station size) | <!--fact:am_pm_corr_normalized-->?<!--/fact--> |
| Rides on days with rain at commute hours, vs. a dry day (95% CI) | −<!--fact:commute_rain_effect-->?<!--/fact--> (<!--fact:commute_rain_ci-->?<!--/fact-->) |
| Rides on days with rain only at other hours | −<!--fact:offpeak_rain_effect-->?<!--/fact--> (<!--fact:offpeak_rain_ci-->?<!--/fact-->) |

Day-ahead forecast, tested on November–December after training on January–October:

| Forecast for | Model accuracy | 7-day average rule |
|---|---|---|
| Whole system, one day | <!--fact:acc_system_day-->?<!--/fact--> | <!--fact:acc_system_day_7d-->?<!--/fact--> |
| One borough, one hour | <!--fact:acc_borough_hour-->?<!--/fact--> | |
| One station, one day | <!--fact:acc_station_day-->?<!--/fact--> | |
| One station, one hour | <!--fact:acc_station_hour-->?<!--/fact--> (chance ceiling ≈ <!--fact:ceiling_station_hour-->?<!--/fact-->) | |

Accuracy = 100% − total absolute error ÷ total trips. At a single station-hour the count is so small that
chance alone caps accuracy; the ceiling shown assumes a perfect forecast with Poisson noise.

- Station-hour error is <!--fact:forecast_vs_7d-->?<!--/fact--> below the best simple rule (same hour's 7-day average) and
  <!--fact:forecast_vs_naive-->?<!--/fact--> below "same hour last week". It beat the best rule in <!--fact:backtest_wins-->?<!--/fact--> of
  <!--fact:backtest_folds-->?<!--/fact--> monthly backtests.
- Weekday-morning net flow per station: error <!--fact:net_mae_model-->?<!--/fact--> bikes vs. <!--fact:net_mae_7d-->?<!--/fact-->.
  At naming each morning's <!--fact:net_top_k-->?<!--/fact--> biggest losers the model finds <!--fact:net_precision_model-->?<!--/fact-->,
  the simple rule <!--fact:net_precision_7d-->?<!--/fact-->.

## Method

- **Cleaning** (`scripts/pipeline.py`): removes <!--fact:removed_share-->?<!--/fact--> of <!--fact:raw_rows-->?<!--/fact--> rows (out-of-month, missing
  station, non-positive duration, over 24 hours). Every analysis reads the same cleaned trips.
- **Weather:** regression of log daily rides with rain split by when it fell; 14-day block-bootstrap intervals.
  Associations in one year, not experiments.
- **Forecast:** Poisson gradient boosting on station × hour counts. Inputs are all known a day ahead. Split by
  date; settings chosen on September–October; trees added until validation error stopped improving
  (<!--fact:forecast_trees-->?<!--/fact--> trees). 80% prediction bands by split conformal calibration.
- **Limits:** observed, not forecast, weather. No dock capacity, so "bikes to move" is a lower bound. One year.

## Repo

| Path | Contents |
|---|---|
| `docs/` | The site (one HTML page, plain SVG charts, Leaflet maps) |
| `scripts/pipeline.py` | Loading, cleaning, caching, weather |
| `scripts/build_*.py` | One script per site section |
| `scripts/sync_facts.py` | Checks every number in the site text and this README against the data |
| `CitiBike_2024.ipynb` | Walkthrough of the analysis |
| `CitiBike_Analysis.ipynb` | January deep-dive with a member-vs-casual classifier (ROC-AUC 0.77) |
| `tests/` | Unit and data-consistency tests, run in CI |

## Run

```bash
pip install -r requirements-dev.txt
```

```bash
python scripts/build_heatmap.py && python scripts/build_geomap.py && python scripts/build_seasonality.py && python scripts/build_rebalancing.py && python scripts/build_forecast.py
```

```bash
python scripts/sync_facts.py --write && python -m pytest -q
```

The first run downloads about 7.5 GB into `.cache/`. Data: [Citi Bike](https://citibikenyc.com/system-data),
[NOAA NCEI](https://www.ncei.noaa.gov/), [NYC Open Data](https://opendata.cityofnewyork.us/).
