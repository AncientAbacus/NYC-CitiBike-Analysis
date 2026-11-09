"""
Forecast hourly trip departures *and* arrivals for every station, and write
the results behind the site's forecast section (docs/data/forecast.json + .js).

Use case: at the end of each day, predict the next day's departures and
arrivals for every station and hour, so rebalancing crews know which
stations will lose bikes tomorrow morning. Every feature is available at
least 24 hours ahead:

  calendar   hour, day of week, holiday (no day-of-year: with a date-based
             split the test dates are values the model never saw in training,
             and trees can't extrapolate; recent history carries the season)
  weather    temperature, precipitation, wind (NOAA Central Park, observed;
             a production system would use the weather *forecast*, so these
             results are an upper bound on what weather adds)
  history    same hour yesterday (lag 24h), same hour last week (lag 168h),
             mean of the same hour over the previous 7 days,
             station's average for this hour/day type in the training period

Evaluation (all splits by date, never shuffled):
  main       train Jan 8 - Aug 31 (first week = lag warm-up), early-stop on
             Sep 1 - Oct 31, refit on Jan 8 - Oct 31, test Nov 1 - Dec 31
  baselines  same-hour 7-day mean (the strongest simple rule, and the
             model's own top feature), same hour last week, same hour
             yesterday, station profile
  decision   weekday 7-10am net flow (arrivals - departures) per station:
             error, and how many of each morning's 20 biggest actual losers
             are among the 20 predicted to lose the most (precision@20)
  intervals  80% band for departures by split conformal calibration: the
             train-only model's predictions on the validation months are
             binned by predicted level, and each bin's 10th/90th percentile
             of actual departures becomes the band for test hours whose
             prediction falls in that bin; reports test coverage
  backtest   rolling origin, one fold per month Aug-Dec: train on all prior
             months, test on that month (smaller fixed-size model and a 5M-row
             training sample per fold, to keep the run time sane)

Model: HistGradientBoostingRegressor with Poisson loss (counts, many zeros).

Usage: python scripts/build_forecast.py [YYYY]   # defaults to 2024
"""

import json
import os
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
BASELINES = {
    "seven_day_mean": "same_hour_7d_mean",
    "seasonal_naive": "lag_168",
    "yesterday": "lag_24",
    "station_profile": "station_profile",
}
PARAMS = dict(
    loss="poisson", learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
    l2_regularization=1.0, early_stopping=False,
    categorical_features=[FEATURES.index("dow")], random_state=42,
)
MAX_TREES, TREE_STEP = 400, 25
AM = (7, 10)
TOP_K = 20
BACKTEST_SAMPLE, BACKTEST_TREES = 5_000_000, 200


def station_hour_grid(year: str, kind: str) -> pd.DataFrame:
    """Trips per station per hour (departures or arrivals), zero-filled from each station's first trip."""
    tcol, scol = ("started_at", "start_station_name") if kind == "departures" else ("ended_at", "end_station_name")
    df = load_clean_period(year, columns=[tcol, scol])
    hour = df[tcol].dt.floor("h")
    keep = (hour.dt.year == int(year)).to_numpy()  # arrivals can spill into next January
    counts = pd.Series(1, index=np.flatnonzero(keep)).groupby(
        [df[scol].astype(str).to_numpy()[keep], hour.to_numpy()[keep]]
    ).sum()
    counts.index.names = ["station", "hour_start"]
    del df, hour
    if os.environ.get("FORECAST_SMOKE_STATIONS"):  # quick end-to-end test on a few busy stations
        top = counts.groupby(level=0).sum().nlargest(int(os.environ["FORECAST_SMOKE_STATIONS"])).index
        counts = counts[counts.index.get_level_values(0).isin(top)]

    hours = pd.date_range(f"{year}-01-01", f"{int(year) + 1}-01-01", freq="h", inclusive="left")
    first_seen = counts.reset_index().groupby("station")["hour_start"].min()
    print(f"{kind}: {counts.index.get_level_values(0).nunique():,} stations x {len(hours):,} hours", flush=True)

    mat = counts.unstack(fill_value=0).reindex(columns=hours, fill_value=0).astype(np.float32)
    grid = mat.stack().rename("y").reset_index()
    grid.columns = ["station", "hour_start", "y"]
    grid = grid.sort_values(["station", "hour_start"], ignore_index=True)

    g = grid.groupby("station")["y"]
    grid["lag_24"] = g.shift(24)
    grid["lag_168"] = g.shift(168)
    grid["same_hour_7d_mean"] = sum(g.shift(24 * k) for k in range(1, 8)) / 7

    grid = grid[grid["hour_start"] >= grid["station"].map(first_seen)]
    grid = grid.dropna(subset=["lag_168", "same_hour_7d_mean"])
    return grid.reset_index(drop=True)


def station_profile(grid: pd.DataFrame, train_end: str) -> np.ndarray:
    """Station x hour x weekday/weekend mean over hours before `train_end` only."""
    daytype = (grid["dow"] >= 5).astype(np.int8)
    train = grid["hour_start"] < train_end
    profile = grid[train].groupby(["station", grid["hour"], daytype])["y"].mean()
    key = pd.MultiIndex.from_arrays([grid["station"], grid["hour"], daytype])
    out = profile.reindex(key).to_numpy(dtype=np.float32)
    # stations that opened after the training window fall back to their recent mean
    return np.where(np.isnan(out), grid["same_hour_7d_mean"].to_numpy(dtype=np.float32), out)


def add_features(grid: pd.DataFrame, year: str, train_end: str) -> pd.DataFrame:
    ts = grid["hour_start"]
    holidays = USFederalHolidayCalendar().holidays(f"{year}-01-01", f"{year}-12-31")
    grid["hour"] = ts.dt.hour.astype(np.int8)
    grid["dow"] = ts.dt.dayofweek.astype(np.int8)
    grid["is_holiday"] = ts.dt.normalize().isin(holidays).astype(np.int8)
    weather = load_weather_hourly(year).set_index("hour_start")
    for col in ("temp_c", "precip_mm", "wind_ms"):
        grid[col] = ts.map(weather[col]).astype(np.float32)
    grid["station_profile"] = station_profile(grid, train_end)
    return grid


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    return {
        "mae": round(float(np.abs(err).mean()), 4),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
        "wmape": round(float(np.abs(err).sum() / max(y.sum(), 1e-9)), 4),
    }


def fit_early_stopped(Xtr, ytr, Xva, yva, naive_va: float):
    """Add trees in steps on the training months; keep the count with the best
    validation MAE (sklearn's own early stopping uses a random, not
    time-ordered, holdout). Returns (best_iter, history, validation predictions
    of the train-only model at best_iter)."""
    model = HistGradientBoostingRegressor(**PARAMS, warm_start=True, max_iter=0)
    best_iter, best_mae, patience, history, best_pred = 0, np.inf, 0, [], None
    for n in range(TREE_STEP, MAX_TREES + 1, TREE_STEP):
        model.set_params(max_iter=n)
        model.fit(Xtr, ytr)
        pred_va = model.predict(Xva)
        mae = float(np.abs(pred_va - yva).mean())
        history.append({"trees": n, "valid_mae": round(mae, 4)})
        print(f"  {n:>3} trees  valid MAE {mae:.4f}  (same hour last week: {naive_va:.4f})", flush=True)
        if mae < best_mae - 1e-4:
            best_iter, best_mae, patience, best_pred = n, mae, 0, pred_va
        else:
            patience += 1
            if patience >= 2:
                break
    return best_iter, history, best_pred


def run_target(year, kind, train_end, valid_end, extras):
    grid = add_features(station_hour_grid(year, kind), year, train_end)
    ts = grid["hour_start"]
    tr = (ts < train_end).to_numpy()
    va = ((ts >= train_end) & (ts < valid_end)).to_numpy()
    te = (ts >= valid_end).to_numpy()
    print(f"{kind} rows: train {tr.sum():,}  valid {va.sum():,}  test {te.sum():,}", flush=True)

    X = grid[FEATURES].to_numpy(dtype=np.float32)
    y = grid["y"].to_numpy(dtype=np.float32)
    naive_va = float(np.abs(grid.loc[va, "lag_168"].to_numpy() - y[va]).mean())
    best_iter, history, pred_va = fit_early_stopped(X[tr], y[tr], X[va], y[va], naive_va)

    fit = tr | va
    model = HistGradientBoostingRegressor(**PARAMS, max_iter=best_iter)
    model.fit(X[fit], y[fit])

    test = grid.loc[te, ["station", "hour_start", "hour", "dow", "is_holiday", "y", *BASELINES.values()]].copy()
    test["pred"] = model.predict(X[te])
    yt = test["y"].to_numpy()

    hourly = {"model": metrics(yt, test["pred"].to_numpy())}
    hourly |= {name: metrics(yt, test[col].to_numpy()) for name, col in BASELINES.items()}
    day = test["hour_start"].dt.normalize()
    daily_sum = test.groupby(["station", day])[["y", "pred", "same_hour_7d_mean", "lag_168", "station_profile"]].sum()
    daily = {
        "model": metrics(daily_sum["y"].to_numpy(), daily_sum["pred"].to_numpy()),
        "seven_day_mean": metrics(daily_sum["y"].to_numpy(), daily_sum["same_hour_7d_mean"].to_numpy()),
        "seasonal_naive": metrics(daily_sum["y"].to_numpy(), daily_sum["lag_168"].to_numpy()),
        "station_profile": metrics(daily_sum["y"].to_numpy(), daily_sum["station_profile"].to_numpy()),
    }
    out = {"best_iter": best_iter, "early_stopping": history, "rows": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
           "stations": int(grid["station"].nunique()), "test_mean": round(float(yt.mean()), 4),
           "hourly": hourly, "daily": daily}

    if kind == "departures":
        extras.update(departures_extras(year, grid, X, y, te, test, model, pred_va, y[va]))
    return out, test[["station", "hour_start", "y", "pred", "same_hour_7d_mean"]]


def conformal_band(pred_cal, y_cal, pred_new, q=(0.1, 0.9)):
    """Empirical 10th/90th percentile of actuals, per bin of predicted level.

    Bins are quantiles of the calibration predictions, with extra resolution
    in the top 10% (the busy hours, where the spread changes fastest); with
    evenly spaced quantiles the top bin spanned ~15 to 60 trips and the band
    went flat across every rush hour."""
    levels = np.unique(np.concatenate([np.linspace(0, 0.9, 28), np.linspace(0.9, 0.99, 19), [0.995, 1.0]]))
    edges = np.unique(np.quantile(pred_cal, levels))
    cal_bin = np.clip(np.searchsorted(edges, pred_cal, side="right") - 1, 0, len(edges) - 2)
    lo_by_bin = np.array([np.quantile(y_cal[cal_bin == b], q[0]) if (cal_bin == b).any() else 0 for b in range(len(edges) - 1)])
    hi_by_bin = np.array([np.quantile(y_cal[cal_bin == b], q[1]) if (cal_bin == b).any() else 0 for b in range(len(edges) - 1)])
    new_bin = np.clip(np.searchsorted(edges, pred_new, side="right") - 1, 0, len(edges) - 2)
    return lo_by_bin[new_bin], hi_by_bin[new_bin]


def departures_extras(year, grid, X, y, te, test, model, pred_va, y_va) -> dict:
    rng = np.random.default_rng(42)

    # ---- 80% prediction interval, calibrated on the validation months ----
    yt = test["y"].to_numpy()
    lo, hi = conformal_band(pred_va, y_va, test["pred"].to_numpy())
    inside = (yt >= lo) & (yt <= hi)
    busy = test["same_hour_7d_mean"].to_numpy() >= 5
    intervals = {
        "nominal": 0.8,
        "coverage": round(float(inside.mean()), 4),
        "coverage_busy_hours": round(float(inside[busy].mean()), 4),
        "busy_hours_share": round(float(busy.mean()), 4),
        "mean_width": round(float((hi - lo).mean()), 3),
        "mean_width_busy_hours": round(float((hi - lo)[busy].mean()), 3),
    }
    test = test.assign(p10=lo, p90=hi)

    # ---- by hour of day ----
    by_hour = test.groupby("hour").apply(
        lambda d: pd.Series({
            "actual": d["y"].mean(),
            "model_mae": (d["pred"] - d["y"]).abs().mean(),
            "seven_day_mae": (d["same_hour_7d_mean"] - d["y"]).abs().mean(),
            "naive_mae": (d["lag_168"] - d["y"]).abs().mean(),
        }), include_groups=False,
    )

    # ---- example week at the busiest test station ----
    busiest = test.groupby("station")["y"].sum().idxmax()
    start = pd.Timestamp(f"{year}-12-02")
    ex = test[(test["station"] == busiest) & (test["hour_start"] >= start) & (test["hour_start"] < start + pd.Timedelta(days=7))]

    # ---- permutation importance on a test sample ----
    sample = rng.choice(np.flatnonzero(te), size=min(300_000, int(te.sum())), replace=False)
    perm = permutation_importance(model, X[sample], y[sample], scoring="neg_mean_absolute_error",
                                  n_repeats=3, random_state=42, n_jobs=1)
    importances = sorted(
        ({"feature": f, "label": FEATURE_LABELS[f], "importance": round(float(m), 5)}
         for f, m in zip(FEATURES, perm.importances_mean)),
        key=lambda d: -d["importance"],
    )

    # ---- rolling-origin backtest, one fold per month ----
    backtest = []
    ts = grid["hour_start"]
    sp_col = FEATURES.index("station_profile")
    for month in range(8, 13):
        fold_start = pd.Timestamp(f"{year}-{month:02d}-01")
        fold_end = fold_start + pd.offsets.MonthBegin(1)
        ftr = (ts < fold_start).to_numpy()
        fte = ((ts >= fold_start) & (ts < fold_end)).to_numpy()
        prof = station_profile(grid, str(fold_start.date()))
        idx = rng.choice(np.flatnonzero(ftr), size=min(BACKTEST_SAMPLE, int(ftr.sum())), replace=False)
        Xtr = X[idx].copy()
        Xtr[:, sp_col] = prof[idx]
        Xte = X[fte].copy()
        Xte[:, sp_col] = prof[fte]
        fm = HistGradientBoostingRegressor(**{**PARAMS, "max_iter": BACKTEST_TREES})
        fm.fit(Xtr, y[idx])
        yf = y[fte]
        fold = {
            "month": month,
            "model": round(float(np.abs(fm.predict(Xte) - yf).mean()), 4),
            "seven_day_mean": round(float(np.abs(grid.loc[fte, "same_hour_7d_mean"].to_numpy() - yf).mean()), 4),
            "seasonal_naive": round(float(np.abs(grid.loc[fte, "lag_168"].to_numpy() - yf).mean()), 4),
        }
        print("  backtest", fold, flush=True)
        backtest.append(fold)

    return {
        "intervals": intervals,
        "by_hour": [{"hour": int(h), **{k: round(float(v), 4) for k, v in r.items()}} for h, r in by_hour.iterrows()],
        "example": {
            "station": busiest, "week_start": start.strftime("%Y-%m-%d"),
            "series": [
                {"t": t.strftime("%Y-%m-%dT%H"), "actual": int(a), "pred": round(float(p), 2),
                 "p10": round(float(lo_), 2), "p90": round(float(hi_), 2), "naive": int(n)}
                for t, a, p, lo_, hi_, n in zip(ex["hour_start"], ex["y"], ex["pred"], ex["p10"], ex["p90"], ex["lag_168"])
            ],
        },
        "feature_importances": importances,
        "backtest": backtest,
    }


def net_flow_eval(dep: pd.DataFrame, arr: pd.DataFrame) -> dict:
    """Weekday 7-10am net flow per station-day: forecast vs. the 7-day-mean rule."""
    cols = ["y", "pred", "same_hour_7d_mean"]
    keys = ["station", "hour_start"]
    m = dep.merge(arr[keys + cols], on=keys, how="outer", suffixes=("_dep", "_arr"))
    for c in cols:
        for k in ("dep", "arr"):
            m[f"{c}_{k}"] = m[f"{c}_{k}"].fillna(0)
    h = m["hour_start"].dt.hour
    holidays = USFederalHolidayCalendar().holidays(m["hour_start"].min(), m["hour_start"].max())
    weekday = (m["hour_start"].dt.dayofweek < 5) & ~m["hour_start"].dt.normalize().isin(holidays)
    m = m[weekday & (h >= AM[0]) & (h < AM[1])]
    day = m["hour_start"].dt.normalize().rename("day")
    agg = m.groupby(["station", day])[[f"{c}_{k}" for c in cols for k in ("dep", "arr")]].sum()
    net = pd.DataFrame({
        "actual": agg["y_arr"] - agg["y_dep"],
        "model": agg["pred_arr"] - agg["pred_dep"],
        "seven_day_mean": agg["same_hour_7d_mean_arr"] - agg["same_hour_7d_mean_dep"],
    })

    def precision_at_k(col, losers=True):
        hits = []
        for _, d in net.groupby(level="day"):
            pick = d.nsmallest(TOP_K, col) if losers else d.nlargest(TOP_K, col)
            truth = d.nsmallest(TOP_K, "actual") if losers else d.nlargest(TOP_K, "actual")
            hits.append(len(pick.index.intersection(truth.index)) / TOP_K)
        return round(float(np.mean(hits)), 4)

    return {
        "window": list(AM), "station_days": int(len(net)), "mornings": int(net.index.get_level_values("day").nunique()),
        "mae": {k: round(float((net[k] - net["actual"]).abs().mean()), 3) for k in ("model", "seven_day_mean")},
        "top_k": TOP_K,
        "precision_losers": {k: precision_at_k(k, True) for k in ("model", "seven_day_mean")},
        "precision_gainers": {k: precision_at_k(k, False) for k in ("model", "seven_day_mean")},
    }


def main() -> None:
    year = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERIOD
    train_end, valid_end = f"{year}-09-01", f"{year}-11-01"

    extras = {}
    dep, dep_test = run_target(year, "departures", train_end, valid_end, extras)
    arr, arr_test = run_target(year, "arrivals", train_end, valid_end, extras)
    net = net_flow_eval(dep_test, arr_test)

    h = dep["hourly"]
    out = {
        "year": year,
        "split": {"train": f"{year}-01-08 to {year}-08-31", "valid": f"{year}-09-01 to {year}-10-31",
                  "test": f"{year}-11-01 to {year}-12-31"},
        "stations": dep["stations"],
        "departures": dep,
        "arrivals": arr,
        "improvement": {
            "vs_seven_day_mean_hourly": round(1 - h["model"]["mae"] / h["seven_day_mean"]["mae"], 4),
            "vs_seasonal_naive_hourly": round(1 - h["model"]["mae"] / h["seasonal_naive"]["mae"], 4),
            "vs_yesterday_hourly": round(1 - h["model"]["mae"] / h["yesterday"]["mae"], 4),
            "vs_profile_hourly": round(1 - h["model"]["mae"] / h["station_profile"]["mae"], 4),
            "vs_seven_day_mean_daily": round(1 - dep["daily"]["model"]["mae"] / dep["daily"]["seven_day_mean"]["mae"], 4),
            "arrivals_vs_seven_day_mean_hourly": round(1 - arr["hourly"]["model"]["mae"] / arr["hourly"]["seven_day_mean"]["mae"], 4),
        },
        "net_flow_am": net,
        **extras,
    }
    data_dir = REPO / "docs" / "data"
    if os.environ.get("FORECAST_SMOKE_STATIONS"):
        data_dir = REPO / ".cache"
    payload = json.dumps(out, separators=(",", ":"))
    (data_dir / "forecast.json").write_text(payload)
    (data_dir / "forecast.js").write_text("window.FORECAST_DATA = " + payload + ";\n")
    print("Wrote docs/data/forecast.{json,js}", flush=True)
    summary = {k: v for k, v in out.items() if k in ("improvement", "net_flow_am", "intervals", "backtest")}
    print(json.dumps(summary | {"dep": dep["hourly"], "arr": arr["hourly"]}, indent=1))


if __name__ == "__main__":
    main()
