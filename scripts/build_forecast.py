"""
Forecast hourly departures and arrivals for every station, one day ahead, and
write docs/data/forecast.{json,js}.

Use case: at the end of each day, predict tomorrow's trips per station and
hour so rebalancing crews know where bikes will be needed. Every input is
known a day ahead:

  calendar  hour, day of week, federal holiday
  weather   temperature, wind, rain that hour, rain over the past 3 hours,
            rain for the day (observed; production would use the forecast,
            so results are an upper bound on what weather adds)
  history   same hour yesterday / last week, same hour's 7-day mean, same
            weekday-hour's 4-week mean, yesterday's station total, station's
            7-day mean daily total, yesterday's system-wide demand relative
            to the week before, station's training-period profile

No day-of-year or month input: with a date split the test dates are values
the trees never saw, and validation error rose with it.

Evaluation, all by date:
  split      train Jan 8-Aug, validate Sep-Oct, refit Jan 8-Oct, test Nov-Dec
  tuning     small sweep on the validation months (TUNING), then trees are
             added until 50 more improve validation MAE by < 0.05%
  baselines  same hour's 7-day mean (strongest, and the model's top input),
             yesterday, last week, station profile
  levels     accuracy (1 - WMAPE) at station-hour, station-day, borough-hour,
             system-hour and system-day, next to a chance ceiling: the accuracy
             a perfect forecast of the *expected* count would get if trips
             were Poisson around it (real demand is noisier, so the true
             ceiling is lower)
  decision   weekday 7-10am net flow per station, and precision@20 for each
             morning's biggest losers
  intervals  shortest interval covering 80% of validation outcomes, per bin
             of predicted level (split conformal)
  backtest   rolling origin, one fold per month Aug-Dec

Model: HistGradientBoostingRegressor, Poisson loss.
FORECAST_SMOKE_STATIONS=40 runs end to end on 40 stations and writes to .cache/.

Usage: python scripts/build_forecast.py [YYYY]   # defaults to 2024
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from pipeline import DEFAULT_PERIOD, REPO, load_clean_period, load_weather_hourly

BASE_FEATURES = [
    "hour", "dow", "is_holiday", "temp_c", "precip_mm", "wind_ms",
    "lag_24", "lag_168", "same_hour_7d_mean", "station_profile",
]
FEATURES = BASE_FEATURES + [
    "precip_3h", "precip_day", "same_dow_4w_mean",
    "day_total_yesterday", "day_total_7d_mean", "system_ratio_yesterday",
]
FEATURE_LABELS = {
    "hour": "Hour of day", "dow": "Day of week", "is_holiday": "Holiday",
    "temp_c": "Temperature", "precip_mm": "Rain this hour", "wind_ms": "Wind speed",
    "precip_3h": "Rain, past 3 hours", "precip_day": "Rain, whole day",
    "lag_24": "Same hour yesterday", "lag_168": "Same hour last week",
    "same_hour_7d_mean": "Same hour, 7-day mean", "same_dow_4w_mean": "Same weekday-hour, 4-week mean",
    "day_total_yesterday": "Station total yesterday", "day_total_7d_mean": "Station daily total, 7-day mean",
    "system_ratio_yesterday": "System demand yesterday vs. prior week",
    "station_profile": "Station's usual level",
}
BASELINES = {
    "seven_day_mean": "same_hour_7d_mean",
    "seasonal_naive": "lag_168",
    "yesterday": "lag_24",
    "station_profile": "station_profile",
}
BASE_PARAMS = dict(loss="poisson", learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
                   l2_regularization=1.0, early_stopping=False, random_state=42)
# sweep on the validation months: (label, features, parameter overrides)
TUNING = [
    ("previous inputs, 63 leaves", BASE_FEATURES, {}),
    ("all inputs, 63 leaves", FEATURES, {}),
    ("all inputs, 127 leaves", FEATURES, {"max_leaf_nodes": 127}),
    ("all inputs, 255 leaves", FEATURES, {"max_leaf_nodes": 255}),
    ("all inputs, 127 leaves, min leaf 50", FEATURES, {"max_leaf_nodes": 127, "min_samples_leaf": 50}),
    ("all inputs, 127 leaves, min leaf 1000", FEATURES, {"max_leaf_nodes": 127, "min_samples_leaf": 1000}),
]
TUNING_TREES, TUNING_TRAIN, TUNING_VALID = 200, 3_000_000, 1_000_000
MAX_TREES, TREE_STEP, MIN_GAIN = 2000, 50, 0.0005
AM = (7, 10)
TOP_K = 20
BACKTEST_SAMPLE, BACKTEST_TREES = 5_000_000, 200
SMOKE = os.environ.get("FORECAST_SMOKE_STATIONS")


def params_for(features: list, overrides: dict) -> dict:
    return {**BASE_PARAMS, **overrides, "categorical_features": [features.index("dow")]}


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def station_hour_grid(year: str, kind: str) -> pd.DataFrame:
    """One row per station-hour from each station's first trip on (after a
    one-week warm-up): the trip count `y` and every history feature."""
    tcol, scol = ("started_at", "start_station_name") if kind == "departures" else ("ended_at", "end_station_name")
    df = load_clean_period(year, columns=[tcol, scol])
    hours = pd.date_range(f"{year}-01-01", f"{int(year) + 1}-01-01", freq="h", inclusive="left")
    n_hours, n_days = len(hours), len(hours) // 24
    h = ((df[tcol] - hours[0]) // pd.Timedelta(hours=1)).to_numpy()
    keep = (h >= 0) & (h < n_hours)  # arrivals can spill into next January
    names = df[scol].cat.categories.astype(str)
    codes = df[scol].cat.codes.to_numpy()[keep].astype(np.int64)
    y = np.bincount(codes * n_hours + h[keep], minlength=len(names) * n_hours).reshape(len(names), n_hours).astype(np.float32)
    del df

    used = y.sum(axis=1) > 0
    if SMOKE:  # quick end-to-end test on a few busy stations
        used &= y.sum(axis=1) >= np.sort(y.sum(axis=1))[-int(SMOKE)]
    y, names = y[used], names[used]
    n_st = len(names)
    print(f"{kind}: {n_st:,} stations x {n_hours:,} hours", flush=True)

    feats = {"lag_24": _lag(y, 24), "lag_168": _lag(y, 168)}
    feats["same_hour_7d_mean"] = sum(_lag(y, 24 * k) for k in range(1, 8)) / 7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN before the warm-up
        feats["same_dow_4w_mean"] = np.nanmean(np.stack([_lag(y, 168 * k) for k in range(1, 5)]), axis=0)

    daily = y.reshape(n_st, n_days, 24).sum(axis=2)  # station x day totals
    per_hour = lambda d: np.repeat(d, 24, axis=1)    # noqa: E731
    feats["day_total_yesterday"] = per_hour(_lag(daily, 1))
    feats["day_total_7d_mean"] = per_hour(sum(_lag(daily, k) for k in range(1, 8)) / 7)
    system = daily.sum(axis=0, keepdims=True)
    ratio = _lag(system, 1) / (sum(_lag(system, k) for k in range(2, 9)) / 7)
    feats["system_ratio_yesterday"] = np.broadcast_to(per_hour(ratio), y.shape)

    first = (y > 0).argmax(axis=1)
    mask = np.arange(n_hours)[None, :] >= np.maximum(first, 168)[:, None]
    s_idx, h_idx = np.nonzero(mask)
    grid = pd.DataFrame({
        "station": pd.Categorical.from_codes(s_idx, categories=names),
        "hour_start": hours[h_idx],
        "y": y[mask],
        **{k: v[mask].astype(np.float32) for k, v in feats.items()},
    })
    return grid


def station_profile(grid: pd.DataFrame, train_end: str) -> np.ndarray:
    """Station x hour x weekday/weekend mean of `y` over hours before `train_end` only."""
    key = (grid["station"].cat.codes.to_numpy().astype(np.int64) * 48
           + grid["hour"].to_numpy() * 2 + (grid["dow"].to_numpy() >= 5))
    train = (grid["hour_start"] < train_end).to_numpy()
    size = int(key.max()) + 1
    total = np.bincount(key[train], weights=grid["y"].to_numpy()[train], minlength=size)
    count = np.bincount(key[train], minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (total / count)[key].astype(np.float32)
    # stations that opened after the training window fall back to their recent mean
    return np.where(np.isnan(out), grid["same_hour_7d_mean"].to_numpy(), out)


def add_features(grid: pd.DataFrame, year: str, train_end: str) -> pd.DataFrame:
    ts = grid["hour_start"]
    holidays = USFederalHolidayCalendar().holidays(f"{year}-01-01", f"{year}-12-31")
    grid["hour"] = ts.dt.hour.astype(np.int8)
    grid["dow"] = ts.dt.dayofweek.astype(np.int8)
    grid["is_holiday"] = ts.dt.normalize().isin(holidays).astype(np.int8)
    weather = load_weather_hourly(year).set_index("hour_start")
    weather["precip_3h"] = weather["precip_mm"].rolling(3, min_periods=1).sum()
    weather["precip_day"] = weather.groupby(weather.index.normalize())["precip_mm"].transform("sum")
    for col in ("temp_c", "precip_mm", "wind_ms", "precip_3h", "precip_day"):
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


def tune(grid: pd.DataFrame, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> list:
    """Score each TUNING config on a fixed sample of the validation months."""
    rng = np.random.default_rng(7)
    itr = rng.choice(np.flatnonzero(tr), size=min(TUNING_TRAIN, int(tr.sum())), replace=False)
    iva = rng.choice(np.flatnonzero(va), size=min(TUNING_VALID, int(va.sum())), replace=False)
    out = []
    for label, feats, overrides in TUNING:
        X = grid[feats].to_numpy(dtype=np.float32)
        m = HistGradientBoostingRegressor(**params_for(feats, overrides), max_iter=TUNING_TREES)
        m.fit(X[itr], y[itr])
        mae = float(np.abs(m.predict(X[iva]) - y[iva]).mean())
        out.append({"config": label, "valid_mae": round(mae, 4)})
        print(f"  tuning  {label:40} valid MAE {mae:.4f}", flush=True)
    return out


def fit_early_stopped(params: dict, Xtr, ytr, Xva, yva):
    """Add trees in steps on the training months until a step improves
    validation MAE by less than MIN_GAIN (sklearn's own early stopping uses a
    random, not time-ordered, holdout). Returns (trees, history, validation
    predictions of the train-only model)."""
    model = HistGradientBoostingRegressor(**params, warm_start=True, max_iter=0)
    best_iter, best_mae, best_pred, history = 0, np.inf, None, []
    for n in range(TREE_STEP, MAX_TREES + 1, TREE_STEP):
        model.set_params(max_iter=n)
        model.fit(Xtr, ytr)
        pred_va = model.predict(Xva)
        mae = float(np.abs(pred_va - yva).mean())
        history.append({"trees": n, "valid_mae": round(mae, 4)})
        print(f"  {n:>4} trees  valid MAE {mae:.4f}", flush=True)
        gain = (best_mae - mae) / best_mae if np.isfinite(best_mae) else 1.0
        if mae < best_mae:
            best_iter, best_mae, best_pred = n, mae, pred_va
        if gain < MIN_GAIN:
            break
    return best_iter, history, best_pred


def conformal_band(pred_cal, y_cal, pred_new, coverage=0.8):
    """Per bin of predicted level, the shortest range of trip counts that
    held at least `coverage` of the calibration outcomes.

    Bins are quantiles of the calibration predictions, finer in the top 10%
    (busy hours, where the spread changes fastest). Counts are whole numbers,
    so quiet bins necessarily cover more than `coverage` (e.g. "0 or 1 trips"
    may already hold 95%); the shortest range keeps that excess to a minimum.
    """
    levels = np.unique(np.concatenate([np.linspace(0, 0.9, 28), np.linspace(0.9, 0.99, 19), [0.995, 1.0]]))
    edges = np.unique(np.quantile(pred_cal, levels))
    n_bins = len(edges) - 1
    cal_bin = np.clip(np.searchsorted(edges, pred_cal, side="right") - 1, 0, n_bins - 1)
    lo_by_bin, hi_by_bin = np.zeros(n_bins), np.zeros(n_bins)
    for b in range(n_bins):
        vals, counts = np.unique(y_cal[cal_bin == b], return_counts=True)
        if len(vals) == 0:
            continue
        cum = np.concatenate([[0], np.cumsum(counts)])
        need = coverage * cum[-1]
        best = (np.inf, 0, vals[0], vals[-1])
        for i in range(len(vals)):
            j = np.searchsorted(cum, cum[i] + need, side="left") - 1  # smallest j with mass(i..j) >= need
            if j >= len(vals):
                break
            cand = (vals[j] - vals[i], -(cum[j + 1] - cum[i]), vals[i], vals[j])
            if cand[:2] < best[:2]:
                best = cand
        lo_by_bin[b], hi_by_bin[b] = best[2], best[3]
    # a higher forecast never gets a lower band, and the band is interpolated
    # between bin centers so it doesn't jump when a forecast crosses a bin edge
    lo_by_bin, hi_by_bin = np.maximum.accumulate(lo_by_bin), np.maximum.accumulate(hi_by_bin)
    centers = np.array([np.median(pred_cal[cal_bin == b]) if (cal_bin == b).any() else edges[b] for b in range(n_bins)])
    # trip counts are whole numbers, so round the band outward
    return np.floor(np.interp(pred_new, centers, lo_by_bin)), np.ceil(np.interp(pred_new, centers, hi_by_bin))


def interval_report(y, pred, lo, hi) -> dict:
    inside = (y >= lo) & (y <= hi)

    def part(mask):
        return {"share_of_hours": round(float(mask.mean()), 4), "coverage": round(float(inside[mask].mean()), 4),
                "above": round(float((y > hi)[mask].mean()), 4), "below": round(float((y < lo)[mask].mean()), 4),
                "mean_width": round(float((hi - lo)[mask].mean()), 2)}

    return {
        "nominal": 0.8,
        "all": part(np.ones(len(y), bool)),
        "quiet": part(pred < 1),            # under 1 trip expected
        "medium": part((pred >= 1) & (pred < 5)),
        "busy": part(pred >= 5),
    }


def level_accuracy(test: pd.DataFrame) -> list:
    """Accuracy (1 - WMAPE) of the same forecasts summed to each planning level."""
    boro = {s["name"]: s["boro"] for s in json.loads((REPO / "docs" / "data" / "stations.json").read_text())["stations"]}
    t = test.assign(boro=test["station"].astype(str).map(boro).fillna("Other"), day=test["hour_start"].dt.normalize())
    levels = [
        ("station_hour", "One station, one hour", ["station", "hour_start"]),
        ("station_day", "One station, one day", ["station", "day"]),
        ("borough_hour", "One borough, one hour", ["boro", "hour_start"]),
        ("system_hour", "Whole system, one hour", ["hour_start"]),
        ("system_day", "Whole system, one day", ["day"]),
    ]
    out = []
    for key, label, by in levels:
        g = t.groupby(by, observed=True)[["y", "pred", "same_hour_7d_mean"]].sum()
        acc = lambda col: round(float(1 - (g[col] - g["y"]).abs().sum() / g["y"].sum()), 4)  # noqa: E731
        lam = g["pred"].to_numpy(dtype=np.float64)
        # mean absolute deviation of a Poisson(lam) count is 2 * lam * P(Y = floor(lam))
        noise = 2 * lam * poisson.pmf(np.floor(lam), lam)
        out.append({"level": key, "label": label, "avg_trips": round(float(g["y"].mean()), 1),
                    "model": acc("pred"), "seven_day_mean": acc("same_hour_7d_mean"),
                    "chance_ceiling": round(float(1 - noise.sum() / lam.sum()), 4)})
    return out


def run_target(year, kind, train_end, valid_end, shared):
    grid = add_features(station_hour_grid(year, kind), year, train_end)
    ts = grid["hour_start"]
    tr = (ts < train_end).to_numpy()
    va = ((ts >= train_end) & (ts < valid_end)).to_numpy()
    te = (ts >= valid_end).to_numpy()
    print(f"{kind} rows: train {tr.sum():,}  valid {va.sum():,}  test {te.sum():,}", flush=True)
    y = grid["y"].to_numpy(dtype=np.float32)

    if "params" not in shared:  # tune once, on departures; arrivals reuse the choice
        shared["tuning"] = tune(grid, y, tr, va)
        chosen = min((r for r in shared["tuning"][1:]), key=lambda r: r["valid_mae"])["config"]
        shared["chosen"] = chosen
        shared["params"] = params_for(FEATURES, next(o for label, _, o in TUNING if label == chosen))
        print(f"  chosen: {chosen}", flush=True)
    params = shared["params"]

    X = grid[FEATURES].to_numpy(dtype=np.float32)
    best_iter, history, pred_va = fit_early_stopped(params, X[tr], y[tr], X[va], y[va])
    model = HistGradientBoostingRegressor(**params, max_iter=best_iter)
    model.fit(X[tr | va], y[tr | va])  # refit on train + validation, then score the untouched test months

    test = grid.loc[te, ["station", "hour_start", "hour", "y", *BASELINES.values()]].copy()
    test["pred"] = model.predict(X[te])
    yt = test["y"].to_numpy()

    hourly = {"model": metrics(yt, test["pred"].to_numpy())}
    hourly |= {name: metrics(yt, test[col].to_numpy()) for name, col in BASELINES.items()}
    out = {"trees": best_iter, "hit_tree_cap": best_iter >= MAX_TREES, "early_stopping": history,
           "rows": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
           "stations": int(grid["station"].nunique()), "test_mean": round(float(yt.mean()), 4),
           "hourly": hourly, "levels": level_accuracy(test)}

    if not SMOKE:  # keep the test forecasts for follow-up analysis
        test.to_parquet(REPO / ".cache" / f"forecast_test_{kind}_{year}.parquet", index=False)
    if kind == "departures":
        shared.update(departures_extras(year, grid, X, y, te, test, model, params, pred_va, y[va]))
    return out, test[["station", "hour_start", "y", "pred", "same_hour_7d_mean"]]


def departures_extras(year, grid, X, y, te, test, model, params, pred_va, y_va) -> dict:
    rng = np.random.default_rng(42)
    yt, pt = test["y"].to_numpy(), test["pred"].to_numpy()

    lo, hi = conformal_band(pred_va, y_va, pt)
    test = test.assign(lo=lo, hi=hi)

    # example week at the busiest test station
    busiest = test.groupby("station", observed=True)["y"].sum().idxmax()
    start = pd.Timestamp(f"{year}-12-02")
    ex = test[(test["station"] == busiest) & (test["hour_start"] >= start) & (test["hour_start"] < start + pd.Timedelta(days=7))]

    sample = rng.choice(np.flatnonzero(te), size=min(300_000, int(te.sum())), replace=False)
    perm = permutation_importance(model, X[sample], y[sample], scoring="neg_mean_absolute_error",
                                  n_repeats=3, random_state=42, n_jobs=1)
    importances = sorted(
        ({"feature": f, "label": FEATURE_LABELS[f], "importance": round(float(m), 5)}
         for f, m in zip(FEATURES, perm.importances_mean)),
        key=lambda d: -d["importance"],
    )

    # rolling-origin backtest, one fold per month
    backtest = []
    ts = grid["hour_start"]
    sp_col = FEATURES.index("station_profile")
    for month in range(8, 13):
        fold_start = pd.Timestamp(f"{year}-{month:02d}-01")
        ftr = (ts < fold_start).to_numpy()
        fte = ((ts >= fold_start) & (ts < fold_start + pd.offsets.MonthBegin(1))).to_numpy()
        prof = station_profile(grid, str(fold_start.date()))
        idx = rng.choice(np.flatnonzero(ftr), size=min(BACKTEST_SAMPLE, int(ftr.sum())), replace=False)
        Xtr, Xte = X[idx].copy(), X[fte].copy()
        Xtr[:, sp_col], Xte[:, sp_col] = prof[idx], prof[fte]
        fm = HistGradientBoostingRegressor(**params, max_iter=BACKTEST_TREES).fit(Xtr, y[idx])
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
        "intervals": interval_report(yt, pt, lo, hi),
        "example": {
            "station": str(busiest), "week_start": start.strftime("%Y-%m-%d"),
            "series": [
                {"t": t.strftime("%Y-%m-%dT%H"), "actual": int(a), "pred": round(float(p), 2),
                 "lo": round(float(lo_), 1), "hi": round(float(hi_), 1), "naive": int(n)}
                for t, a, p, lo_, hi_, n in zip(ex["hour_start"], ex["y"], ex["pred"], ex["lo"], ex["hi"], ex["lag_168"])
            ],
        },
        "feature_importances": importances,
        "backtest": backtest,
    }


def net_flow_eval(dep: pd.DataFrame, arr: pd.DataFrame) -> dict:
    """Weekday 7-10am net flow per station-day: forecast vs. the 7-day-mean rule."""
    cols = ["y", "pred", "same_hour_7d_mean"]
    keys = ["station", "hour_start"]
    dep, arr = (d.assign(station=d["station"].astype(str)) for d in (dep, arr))
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

    def precision_at_k(col):
        hits = []
        for _, d in net.groupby(level="day"):
            hits.append(len(d.nsmallest(TOP_K, col).index.intersection(d.nsmallest(TOP_K, "actual").index)) / TOP_K)
        return round(float(np.mean(hits)), 4)

    return {
        "window": list(AM), "station_days": int(len(net)), "mornings": int(net.index.get_level_values("day").nunique()),
        "mae": {k: round(float((net[k] - net["actual"]).abs().mean()), 3) for k in ("model", "seven_day_mean")},
        "top_k": TOP_K,
        "precision_losers": {k: precision_at_k(k) for k in ("model", "seven_day_mean")},
    }


def main() -> None:
    year = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERIOD
    train_end, valid_end = f"{year}-09-01", f"{year}-11-01"

    shared = {}
    dep, dep_test = run_target(year, "departures", train_end, valid_end, shared)
    arr, arr_test = run_target(year, "arrivals", train_end, valid_end, shared)
    shared.pop("params")

    def gain(target, base):
        return round(1 - target["hourly"]["model"]["mae"] / target["hourly"][base]["mae"], 4)

    out = {
        "year": year,
        "split": {"train": f"{year}-01-08 to {year}-08-31", "valid": f"{year}-09-01 to {year}-10-31",
                  "test": f"{year}-11-01 to {year}-12-31"},
        "stations": dep["stations"],
        "features": [FEATURE_LABELS[f] for f in FEATURES],
        "stopping_rule": {"step": TREE_STEP, "min_gain": MIN_GAIN, "max_trees": MAX_TREES},
        "departures": dep,
        "arrivals": arr,
        "improvement": {
            "vs_seven_day_mean_hourly": gain(dep, "seven_day_mean"),
            "vs_seasonal_naive_hourly": gain(dep, "seasonal_naive"),
            "vs_yesterday_hourly": gain(dep, "yesterday"),
            "arrivals_vs_seven_day_mean_hourly": gain(arr, "seven_day_mean"),
        },
        "net_flow_am": net_flow_eval(dep_test, arr_test),
        **shared,
    }
    data_dir = REPO / (".cache" if SMOKE else "docs/data")
    payload = json.dumps(out, separators=(",", ":"))
    (data_dir / "forecast.json").write_text(payload)
    (data_dir / "forecast.js").write_text("window.FORECAST_DATA = " + payload + ";\n")
    print(f"Wrote {data_dir.relative_to(REPO)}/forecast.{{json,js}}", flush=True)
    show = {k: out[k] for k in ("tuning", "chosen", "improvement", "net_flow_am", "intervals", "backtest")}
    show |= {"levels": dep["levels"], "dep": dep["hourly"], "arr": arr["hourly"], "trees": [dep["trees"], arr["trees"]]}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
