"""
Forecast hourly trip departures for every station, and write the results
behind the site's forecast section (docs/data/forecast.json + .js).

Use case: at the end of each day, predict departures for every station and
hour of the next day, so rebalancing crews know where bikes will be needed.
Every feature is therefore available at least 24 hours ahead:

  calendar   hour, day of week, holiday (no day-of-year: with a date-based
             split the test dates are values the model never saw in training,
             and trees can't extrapolate; recent history carries the season)
  weather    temperature, precipitation, wind (NOAA Central Park, observed;
             a production system would use the weather *forecast*, so these
             results are an upper bound on what weather adds)
  history    same hour yesterday (lag 24h), same hour last week (lag 168h),
             mean of the same hour over the previous 7 days,
             station's average for this hour/day type in the training period

Split by date, never shuffled: train Jan 8 - Aug 31 (first week reserved as
lag warm-up), validate Sep 1 - Oct 31 (early stopping), test Nov 1 - Dec 31.

Baselines, scored on the same test hours:
  seasonal naive   same hour last week
  yesterday        same hour yesterday
  station profile  training-period mean for this station x hour x weekday/weekend

Model: HistGradientBoostingRegressor with Poisson loss (counts, many zeros).

Usage: python scripts/build_forecast.py [YYYY]   # defaults to 2024
"""

import json
import sys

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from pipeline import DEFAULT_PERIOD, REPO, load_clean_period, load_weather_hourly

FEATURES = [
    "hour", "dow", "is_holiday",
    "temp_c", "precip_mm", "wind_ms",
    "lag_24", "lag_168", "same_hour_7d_mean", "station_profile",
]
FEATURE_LABELS = {
    "hour": "Hour of day", "dow": "Day of week", "is_holiday": "Holiday",
    "temp_c": "Temperature", "precip_mm": "Precipitation", "wind_ms": "Wind speed",
    "lag_24": "Same hour yesterday", "lag_168": "Same hour last week",
    "same_hour_7d_mean": "Same hour, past-7-day mean", "station_profile": "Station's usual level",
}


def station_hour_grid(year: str) -> pd.DataFrame:
    """Departures per station per hour, zero-filled from each station's first ride on."""
    df = load_clean_period(year, columns=["started_at", "start_station_name"])
    hour = df["started_at"].dt.floor("h").rename("hour_start")
    counts = df.groupby([df["start_station_name"].astype(str).rename("station"), hour]).size()
    del df

    hours = pd.date_range(f"{year}-01-01", f"{int(year) + 1}-01-01", freq="h", inclusive="left")
    stations = counts.index.get_level_values(0).unique()
    first_seen = counts.reset_index().groupby("station")["hour_start"].min()
    print(f"{len(stations):,} stations x {len(hours):,} hours", flush=True)

    mat = counts.unstack(fill_value=0).reindex(columns=hours, fill_value=0).astype(np.float32)
    grid = mat.stack().rename("y").reset_index()
    grid.columns = ["station", "hour_start", "y"]
    grid = grid.sort_values(["station", "hour_start"], ignore_index=True)

    g = grid.groupby("station")["y"]
    grid["lag_24"] = g.shift(24)
    grid["lag_168"] = g.shift(168)
    grid["same_hour_7d_mean"] = sum(g.shift(24 * k) for k in range(1, 8)) / 7

    # drop hours before a station's first ride (not yet installed) and the lag warm-up
    grid = grid[grid["hour_start"] >= grid["station"].map(first_seen)]
    grid = grid.dropna(subset=["lag_168", "same_hour_7d_mean"])
    return grid.reset_index(drop=True)


def add_features(grid: pd.DataFrame, year: str, train_end: str) -> pd.DataFrame:
    ts = grid["hour_start"]
    holidays = USFederalHolidayCalendar().holidays(f"{year}-01-01", f"{year}-12-31")
    grid["hour"] = ts.dt.hour.astype(np.int8)
    grid["dow"] = ts.dt.dayofweek.astype(np.int8)
    grid["is_holiday"] = ts.dt.normalize().isin(holidays).astype(np.int8)

    weather = load_weather_hourly(year).set_index("hour_start")
    for col in ("temp_c", "precip_mm", "wind_ms"):
        grid[col] = ts.map(weather[col]).astype(np.float32)

    # station x hour x (weekday/weekend) mean, from the training period only
    daytype = (grid["dow"] >= 5).astype(np.int8)
    train = grid["hour_start"] < train_end
    profile = grid[train].groupby(["station", grid["hour"], daytype])["y"].mean()
    key = pd.MultiIndex.from_arrays([grid["station"], grid["hour"], daytype])
    grid["station_profile"] = profile.reindex(key).to_numpy(dtype=np.float32)
    grid["station_profile"] = grid["station_profile"].fillna(grid["same_hour_7d_mean"])  # stations opened after training
    return grid


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    return {
        "mae": round(float(np.abs(err).mean()), 4),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
        "wmape": round(float(np.abs(err).sum() / y.sum()), 4),
    }


def main() -> None:
    year = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERIOD
    train_end, valid_end = f"{year}-09-01", f"{year}-11-01"

    grid = add_features(station_hour_grid(year), year, train_end)
    ts = grid["hour_start"]
    tr, va, te = ts < train_end, (ts >= train_end) & (ts < valid_end), ts >= valid_end
    print(f"rows: train {tr.sum():,}  valid {va.sum():,}  test {te.sum():,}", flush=True)

    X = grid[FEATURES].to_numpy(dtype=np.float32)
    y = grid["y"].to_numpy(dtype=np.float32)

    params = dict(
        loss="poisson", learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False,
        categorical_features=[FEATURES.index("dow")], random_state=42,
    )
    trn, val = tr.to_numpy(), va.to_numpy()
    Xtr, ytr, Xva, yva = X[trn], y[trn], X[val], y[val]

    # Early stopping on the *later* validation months (sklearn's built-in early
    # stopping would carve a random, not time-ordered, holdout). warm_start
    # adds trees in steps of 25 until validation MAE stops improving.
    model = HistGradientBoostingRegressor(**params, warm_start=True, max_iter=0)
    best_iter, best_mae, patience = 0, np.inf, 0
    naive_va = float(np.abs(grid.loc[va, "lag_168"].to_numpy() - yva).mean())
    for n in range(25, 401, 25):
        model.set_params(max_iter=n)
        model.fit(Xtr, ytr)
        mae = float(np.abs(model.predict(Xva) - yva).mean())
        print(f"  {n:>3} trees  valid MAE {mae:.4f}  (same hour last week: {naive_va:.4f})", flush=True)
        if mae < best_mae - 1e-4:
            best_iter, best_mae, patience = n, mae, 0
        else:
            patience += 1
            if patience >= 2:
                break

    # refit on train + valid with the chosen size, then score the untouched test months
    model = HistGradientBoostingRegressor(**params, max_iter=best_iter)
    model.fit(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
    del Xtr, ytr, Xva, yva

    test = grid[te.to_numpy()].copy()
    test["pred"] = model.predict(X[te.to_numpy()])
    yt = test["y"].to_numpy()

    results = {
        "model": metrics(yt, test["pred"].to_numpy()),
        "seasonal_naive": metrics(yt, test["lag_168"].to_numpy()),
        "yesterday": metrics(yt, test["lag_24"].to_numpy()),
        "station_profile": metrics(yt, test["station_profile"].to_numpy()),
    }

    # planning grain: station x day totals (what a rebalancing crew plans on)
    day = test["hour_start"].dt.normalize()
    daily = test.groupby(["station", day])[["y", "pred", "lag_168", "station_profile"]].sum()
    results_daily = {
        "model": metrics(daily["y"].to_numpy(), daily["pred"].to_numpy()),
        "seasonal_naive": metrics(daily["y"].to_numpy(), daily["lag_168"].to_numpy()),
        "station_profile": metrics(daily["y"].to_numpy(), daily["station_profile"].to_numpy()),
    }

    by_hour = test.groupby("hour").apply(
        lambda d: pd.Series({
            "actual": d["y"].mean(),
            "model_mae": (d["pred"] - d["y"]).abs().mean(),
            "naive_mae": (d["lag_168"] - d["y"]).abs().mean(),
        }), include_groups=False,
    )

    # example: busiest station, first full test week starting Monday
    busiest = test.groupby("station")["y"].sum().idxmax()
    start = pd.Timestamp(f"{year}-12-02")
    ex = test[(test["station"] == busiest) & (test["hour_start"] >= start) & (test["hour_start"] < start + pd.Timedelta(days=7))]

    rng = np.random.default_rng(42)
    sample = rng.choice(np.flatnonzero(te.to_numpy()), size=min(300_000, int(te.sum())), replace=False)
    perm = permutation_importance(model, X[sample], y[sample], scoring="neg_mean_absolute_error",
                                  n_repeats=3, random_state=42, n_jobs=1)
    importances = sorted(
        ({"feature": f, "label": FEATURE_LABELS[f], "importance": round(float(m), 5)}
         for f, m in zip(FEATURES, perm.importances_mean)),
        key=lambda d: -d["importance"],
    )

    out = {
        "year": year,
        "split": {"train": f"{year}-01-08 to {year}-08-31", "valid": f"{year}-09-01 to {year}-10-31",
                  "test": f"{year}-11-01 to {year}-12-31"},
        "stations": int(grid["station"].nunique()),
        "rows": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
        "best_iter": best_iter,
        "test_mean_hourly_departures": round(float(yt.mean()), 4),
        "hourly": results,
        "daily": results_daily,
        "improvement_vs_naive_hourly": round(1 - results["model"]["mae"] / results["seasonal_naive"]["mae"], 4),
        "improvement_vs_profile_hourly": round(1 - results["model"]["mae"] / results["station_profile"]["mae"], 4),
        "improvement_vs_naive_daily": round(1 - results_daily["model"]["mae"] / results_daily["seasonal_naive"]["mae"], 4),
        "by_hour": [{"hour": int(h), **{k: round(float(v), 4) for k, v in r.items()}} for h, r in by_hour.iterrows()],
        "example": {
            "station": busiest, "week_start": start.strftime("%Y-%m-%d"),
            "series": [{"t": t.strftime("%Y-%m-%dT%H"), "actual": int(a), "pred": round(float(p), 2), "naive": int(n)}
                       for t, a, p, n in zip(ex["hour_start"], ex["y"], ex["pred"], ex["lag_168"])],
        },
        "feature_importances": importances,
    }
    data_dir = REPO / "docs" / "data"
    payload = json.dumps(out, separators=(",", ":"))
    (data_dir / "forecast.json").write_text(payload)
    (data_dir / "forecast.js").write_text("window.FORECAST_DATA = " + payload + ";\n")
    print("Wrote docs/data/forecast.{json,js}", flush=True)
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_hour", "example")}, indent=1))


if __name__ == "__main__":
    main()
