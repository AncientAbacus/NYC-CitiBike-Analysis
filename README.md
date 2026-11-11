# NYC Citi Bike 2024: where bikes are needed, and when

<!--fact:total_rides-->44,158,430<!--/fact--> trips, NOAA weather and NYC's bike lane network, used to answer one operating question:
where do bikes pile up and run out, and can tomorrow's demand be predicted station by station?

**[Live site](https://ancientabacus.github.io/NYC-CitiBike-Analysis/)** · **[Notebook](CitiBike_2024.ipynb)**

## Results

| Finding | Number |
|---|---|
| Bikes left in the wrong place each weekday morning (lower bound) | <!--fact:bikes_to_move-->5,927<!--/fact--> |
| Stations losing / gaining 2+ bikes each morning | <!--fact:drainer_count-->676<!--/fact--> / <!--fact:filler_count-->409<!--/fact--> |
| Morning vs. evening net flow, correlation (adjusted for station size) | <!--fact:am_pm_corr_normalized-->-0.80<!--/fact--> |
| Rides on days with rain at commute hours, vs. a dry day (95% CI) | −<!--fact:commute_rain_effect-->29%<!--/fact--> (<!--fact:commute_rain_ci-->22&ndash;32%<!--/fact-->) |
| Rides on days with rain only at other hours | −<!--fact:offpeak_rain_effect-->7%<!--/fact--> (<!--fact:offpeak_rain_ci-->2&ndash;10%<!--/fact-->) |

Day-ahead forecast, tested on June–August 2025 after training on 2024 and January–May 2025:

| Forecast for | Model, actual weather | Model, yesterday's weather | 7-day average rule |
|---|---|---|---|
| Whole system, one day | <!--fact:acc_system_day-->94%<!--/fact--> | <!--fact:acc_system_day_prev-->86%<!--/fact--> | <!--fact:acc_system_day_7d-->89%<!--/fact--> |
| One borough, one hour | <!--fact:acc_borough_hour-->91%<!--/fact--> | <!--fact:acc_borough_hour_prev-->83%<!--/fact--> | |
| One station, one day | <!--fact:acc_station_day-->85%<!--/fact--> | <!--fact:acc_station_day_prev-->80%<!--/fact--> | |
| One station, one hour | <!--fact:acc_station_hour-->56%<!--/fact--> (chance ceiling ≈ <!--fact:ceiling_station_hour-->65%<!--/fact-->) | <!--fact:acc_station_hour_prev-->54%<!--/fact--> | |

The model is given the day's actual weather. Retrained knowing only yesterday's weather, it scores the
second column; a real weather forecast sits between the two.

Accuracy = 100% − total absolute error ÷ total trips. At a single station-hour the count is so small that
chance alone caps accuracy; the ceiling shown assumes a perfect forecast with Poisson noise.

- Station-hour error is <!--fact:forecast_vs_7d_prev-->11%<!--/fact-->–<!--fact:forecast_vs_7d-->17%<!--/fact--> (yesterday's vs. actual weather) below the best simple rule (same hour's 7-day average) and
  <!--fact:forecast_vs_naive-->25%<!--/fact--> below "same hour last week". It beat the best rule in <!--fact:backtest_wins-->3<!--/fact--> of
  <!--fact:backtest_folds-->3<!--/fact--> monthly backtests.
- Weekday-morning net flow per station: error <!--fact:net_mae_model-->3.1<!--/fact--> bikes vs. <!--fact:net_mae_7d-->3.5<!--/fact-->.
  At naming each morning's <!--fact:net_top_k-->20<!--/fact--> biggest losers the model finds <!--fact:net_precision_model-->13.4<!--/fact-->,
  the simple rule <!--fact:net_precision_7d-->13.3<!--/fact-->.

## Method

- **Cleaning** (`scripts/pipeline.py`): removes <!--fact:removed_share-->0.33%<!--/fact--> of <!--fact:raw_rows-->44,303,209<!--/fact--> rows (out-of-month, missing
  station, non-positive duration, over 24 hours). Every analysis reads the same cleaned trips.
- **Weather:** regression of log daily rides with rain split by when it fell; 14-day block-bootstrap intervals.
  Associations in one year, not experiments.
- **Forecast:** Poisson gradient boosting on station × hour counts. Inputs are all known a day ahead. Split by
  date; settings chosen on April–May 2025; trees added until validation error stopped improving
  (<!--fact:forecast_trees-->300<!--/fact--> trees). 80% prediction bands by split conformal calibration.
- **Limits:** weather is bounded, not forecast (see table). No dock capacity, so "bikes to move" is a lower bound. One year.

## Repo

| Path | Contents |
|---|---|
| `docs/` | The site (one HTML page, plain SVG charts, Leaflet maps) |
| `scripts/pipeline.py` | Loading, cleaning, caching, weather |
| `scripts/build_*.py` | One script per site section |
| `scripts/sync_facts.py` | Checks every number in the site text and this README against the data |
| `CitiBike_2024.ipynb` | Walkthrough of the analysis |
| `CitiBike_Analysis.ipynb` | January deep-dive with a member-vs-casual classifier (ROC-AUC 0.77) |
| `tests/` | Unit, data-consistency and forecast leak tests, run in CI |

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
