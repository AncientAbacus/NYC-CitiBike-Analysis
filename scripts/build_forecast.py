"""
Forecast hourly departures and arrivals for every station, one day ahead.

Data: every Citi Bike trip in 2024 and 2025 on one hourly timeline. Each
station gets a row per hour (from its first trip, after a one-week warm-up).

Inputs, all known a day ahead:
  calendar  hour, day of week, federal holiday
  weather   temperature, wind, rain this hour / past 3 hours / whole day
            (observed; a live system would use the forecast, see below)
  history   same hour yesterday and last week, same hour's 7-day mean, same
            weekday-hour's 4-week mean, yesterday's station total, its 7-day
            mean, yesterday's system-wide demand vs. the week before, and
            yesterday's demand at the 5 nearest stations
  station   training-period profile (hour x weekday/weekend mean)

No day-of-year or month input: under a date split the test dates are values
the trees never saw, and validation error rose with them.

Hourly weather is only published to 2025-08-25, so the timeline ends there.
Protocols, all by date:
  headline   train on 2024 and 2025 up to 2025-04-01, early-stop on Apr-May
             2025, refit up to 2025-06-01, test Jun-Aug 2025.
  benchmark  the same model trained on 2024 only (early-stopped on Oct-Dec
             2024), scored on the same 2025 test months. The difference is
             what the five months of 2025 data are worth.
  weather    the headline model retrained with each weather input replaced
             by yesterday's value: a crude stand-in for a weather forecast.
             Real forecasts fall between this and the headline result.

Settings are chosen on the headline validation months (TUNING), trees are
added until 50 more improve validation MAE by < 0.05%, and every reported
number comes from the test months.

Model: HistGradientBoostingRegressor, Poisson loss.
FORECAST_SMOKE_STATIONS=40 runs end to end on 40 stations and writes to .cache/.

Usage: python scripts/build_forecast.py
"""

import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from pipeline import REPO, load_clean_period, load_weather_hourly, period_months

YEARS = [2024, 2025]
WEATHER_END = "2025-08-25"  # hourly weather is not published after this date
HEADLINE = {"train_end": "2025-04-01", "valid_end": "2025-06-01", "test_start": "2025-06-01"}
BENCHMARK = {"train_end": "2024-10-01", "valid_end": "2025-01-01", "test_start": "2025-06-01"}
TRIP_COLS = ["started_at", "ended_at", "start_station_name", "end_station_name",
             "start_lat", "start_lng", "end_lat", "end_lng"]

WEATHER = ["temp_c", "precip_mm", "wind_ms", "precip_3h", "precip_day"]
CALENDAR = ["hour", "dow", "is_holiday"]
HISTORY = ["lag_24", "lag_168", "same_hour_7d_mean", "same_dow_4w_mean", "day_total_yesterday",
           "day_total_7d_mean", "system_ratio_yesterday", "neighbor_lag_24"]
PREV = CALENDAR + ["temp_c", "precip_mm", "wind_ms", "lag_24", "lag_168", "same_hour_7d_mean", "station_profile"]
ALL = CALENDAR + WEATHER + HISTORY + ["station_profile"]
ALL_NO_NEIGHBOURS = [f for f in ALL if f != "neighbor_lag_24"]
FEATURE_LABELS = {
    "hour": "Hour of day", "dow": "Day of week", "is_holiday": "Holiday",
    "temp_c": "Temperature", "precip_mm": "Rain this hour", "wind_ms": "Wind speed",
    "precip_3h": "Rain, past 3 hours", "precip_day": "Rain, whole day",
    "lag_24": "Same hour yesterday", "lag_168": "Same hour last week",
    "same_hour_7d_mean": "Same hour, 7-day mean", "same_dow_4w_mean": "Same weekday-hour, 4-week mean",
    "day_total_yesterday": "Station total yesterday", "day_total_7d_mean": "Station daily total, 7-day mean",
    "system_ratio_yesterday": "System demand yesterday vs. prior week",
    "neighbor_lag_24": "Five nearest stations, same hour yesterday",
    "station_profile": "Station's usual level",
}
BASELINES = {"seven_day_mean": "same_hour_7d_mean", "seasonal_naive": "lag_168",
             "yesterday": "lag_24", "station_profile": "station_profile"}
BASE_PARAMS = dict(loss="poisson", learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
                   l2_regularization=1.0, early_stopping=False, random_state=42)
TUNING = [
    ("previous 10 inputs, 63 leaves", PREV, {}),
    ("all inputs, 127 leaves, min leaf 1000", ALL, {"max_leaf_nodes": 127, "min_samples_leaf": 1000}),
    ("all inputs but neighbours, 127 leaves, min leaf 1000", ALL_NO_NEIGHBOURS, {"max_leaf_nodes": 127, "min_samples_leaf": 1000}),
    ("all inputs, 255 leaves, min leaf 1000", ALL, {"max_leaf_nodes": 255, "min_samples_leaf": 1000}),
    ("all inputs, 127 leaves, min leaf 50", ALL, {"max_leaf_nodes": 127, "min_samples_leaf": 50}),
]
TUNING_TREES, TUNING_TRAIN, TUNING_VALID = 200, 3_000_000, 1_000_000
MAX_TREES, TREE_STEP, MIN_GAIN = 2000, 50, 0.0005
AM = (7, 10)
TOP_K = 20
BACKTEST_SAMPLE, BACKTEST_TREES = 5_000_000, 200
NEIGHBOURS = 5
SMOKE = os.environ.get("FORECAST_SMOKE_STATIONS")


# ---------------------------------------------------------------- timeline

@dataclass
class Timeline:
    t0: pd.Timestamp
    n_hours: int
    names: np.ndarray        # station names, index = station id
    lat: np.ndarray
    lng: np.ndarray
    counts: dict             # kind -> stations x hours float32 trip counts
    hourly: dict             # per-hour arrays: calendar and weather, plus "<weather>_prev"

    def hour_of(self, ts: str) -> int:
        return int((pd.Timestamp(ts) - self.t0) / pd.Timedelta(hours=1))

    def stamp(self, t) -> pd.DatetimeIndex:
        return self.t0 + pd.to_timedelta(np.asarray(t), unit="h")


def read_timeline(years, smoke=None) -> Timeline:
    """Trip counts and station coordinates up to WEATHER_END, read one month at a time."""
    t0 = pd.Timestamp(f"{years[0]}-01-01")
    t1 = pd.Timestamp(WEATHER_END)
    n_hours = int((t1 - t0) / pd.Timedelta(hours=1))
    idx_of, names = {}, []
    start_xy, end_xy = {}, {}
    keys = {"departures": [], "arrivals": []}
    cnts = {"departures": [], "arrivals": []}

    def index(vals):
        uniq, inv = np.unique(vals, return_inverse=True)
        out = np.empty(len(uniq), np.int64)
        for i, u in enumerate(uniq):
            j = idx_of.get(u)
            if j is None:
                j = idx_of[u] = len(names)
                names.append(u)
            out[i] = j
        return out[inv]

    def add_coords(store, vals, lat, lng):
        g = pd.DataFrame({"n": vals, "la": lat, "ln": lng}).groupby("n").agg(
            la=("la", "sum"), ln=("ln", "sum"), c=("la", "size"))
        for n, r in g.iterrows():
            acc = store.setdefault(n, [0.0, 0.0, 0])
            acc[0] += r.la
            acc[1] += r.ln
            acc[2] += r.c

    for year in years:
        for m in period_months(str(year)):
            df = load_clean_period(m, columns=TRIP_COLS)
            for kind, tc, sc, la, ln, store in [
                ("departures", "started_at", "start_station_name", "start_lat", "start_lng", start_xy),
                ("arrivals", "ended_at", "end_station_name", "end_lat", "end_lng", end_xy),
            ]:
                vals = df[sc].astype(str).to_numpy()
                h = ((df[tc] - t0) // pd.Timedelta(hours=1)).to_numpy()
                ok = (h >= 0) & (h < n_hours)
                k = index(vals[ok]) * n_hours + h[ok]
                u, c = np.unique(k, return_counts=True)
                keys[kind].append(u)
                cnts[kind].append(c)
                add_coords(store, vals, df[la].to_numpy(), df[ln].to_numpy())
            del df

    n_st = len(names)
    counts = {}
    for kind in keys:
        k, c = np.concatenate(keys[kind]), np.concatenate(cnts[kind])
        counts[kind] = np.bincount(k, weights=c, minlength=n_st * n_hours).reshape(n_st, n_hours).astype(np.float32)
    names = np.array(names)

    lat, lng = np.zeros(n_st), np.zeros(n_st)
    for i, n in enumerate(names):
        acc = start_xy.get(n) or end_xy.get(n)
        lat[i], lng[i] = acc[0] / acc[2], acc[1] / acc[2]

    if smoke:  # keep the busiest stations only, for a quick end-to-end run
        keep = np.argsort(counts["departures"].sum(axis=1))[-int(smoke):]
        counts = {k: v[keep] for k, v in counts.items()}
        names, lat, lng = names[keep], lat[keep], lng[keep]
        print(f"smoke test: {len(keep)} stations", flush=True)

    tl = Timeline(t0=t0, n_hours=n_hours, names=names, lat=lat, lng=lng, counts=counts, hourly={})
    tl.hourly = hourly_arrays(tl, years)
    print(f"timeline {t0:%Y-%m-%d} .. {t1:%Y-%m-%d}: {len(names):,} stations, "
          f"{counts['departures'].sum():,.0f} departures", flush=True)
    return tl


def hourly_arrays(tl: Timeline, years) -> dict:
    """Calendar and weather values for every hour of the timeline."""
    hours = np.arange(tl.n_hours)
    days = hours // 24
    day_dates = pd.date_range(tl.t0, periods=tl.n_hours // 24, freq="D")
    t1 = day_dates[-1] + pd.Timedelta(days=1)
    holidays = USFederalHolidayCalendar().holidays(tl.t0, t1)
    dow_day = day_dates.dayofweek.to_numpy()
    hol_day = day_dates.isin(holidays).astype(np.float32)
    out = {"hour": (hours % 24).astype(np.float32), "dow": dow_day[days].astype(np.float32),
           "is_holiday": hol_day[days]}

    w = pd.concat([load_weather_hourly(str(y)) for y in years]).set_index("hour_start")
    w = w[~w.index.duplicated()].reindex(pd.date_range(tl.t0, periods=tl.n_hours, freq="h"))
    gaps = int(w["temp_c"].isna().sum())
    w = w.interpolate(limit=24, limit_direction="both")
    if w["temp_c"].isna().any() or w["wind_ms"].isna().any():
        raise ValueError("hourly weather has a gap longer than 24 hours inside the timeline")
    print(f"weather: {gaps} missing hours filled by interpolation (gaps up to 24h)", flush=True)
    w["precip_mm"] = w["precip_mm"].fillna(0.0)
    w["precip_3h"] = w["precip_mm"].rolling(3, min_periods=1).sum()
    w["precip_day"] = pd.Series(w["precip_mm"].to_numpy()).groupby(days).transform("sum").to_numpy()
    for col in WEATHER:
        arr = w[col].to_numpy(dtype=np.float32)
        out[col] = arr
        out[col + "_prev"] = np.concatenate([arr[:24], arr[:-24]])  # yesterday's value at the same hour
    return out


def neighbour_matrix(lat, lng, k=NEIGHBOURS) -> csr_matrix:
    """Row i averages the k nearest other stations (projected km), weight 1/k each."""
    x = lng * np.cos(np.radians(40.75)) * 111.0
    pts = np.column_stack([x, lat * 111.0])
    n = len(pts)
    k = min(k, n - 1)
    _, nn = cKDTree(pts).query(pts, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = nn[:, 1:].ravel()
    return csr_matrix((np.full(rows.size, 1.0 / k), (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------- features

def _lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def history_features(y: np.ndarray, nbr=None) -> dict:
    """History inputs from a stations x hours count matrix (hours must be whole days).

    Invariant, enforced by tests/test_forecast_features.py: every value for a
    given calendar day depends only on counts from earlier days, so the whole
    day can be forecast at midnight.
    """
    n_st, n_hours = y.shape
    feats = {"lag_24": _lag(y, 24), "lag_168": _lag(y, 168)}
    feats["same_hour_7d_mean"] = sum(_lag(y, 24 * k) for k in range(1, 8)) / 7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN before the warm-up
        feats["same_dow_4w_mean"] = np.nanmean(np.stack([_lag(y, 168 * k) for k in range(1, 5)]), axis=0)

    daily = y.reshape(n_st, n_hours // 24, 24).sum(axis=2)
    per_hour = lambda d: np.repeat(d, 24, axis=1)  # noqa: E731
    feats["day_total_yesterday"] = per_hour(_lag(daily, 1))
    feats["day_total_7d_mean"] = per_hour(sum(_lag(daily, k) for k in range(1, 8)) / 7)
    system = daily.sum(axis=0, keepdims=True)
    ratio = _lag(system, 1) / (sum(_lag(system, k) for k in range(2, 9)) / 7)
    feats["system_ratio_yesterday"] = np.broadcast_to(per_hour(ratio), y.shape)
    if nbr is not None:
        feats["neighbor_lag_24"] = nbr @ np.nan_to_num(feats["lag_24"])
    return feats


@dataclass
class Panel:
    """One row per station-hour that has a full history, for one kind of trip."""
    s: np.ndarray       # station id
    t: np.ndarray       # hour index on the timeline
    y: np.ndarray       # trips in that hour
    hist: dict          # history features, aligned with s and t


def build_panel(tl: Timeline, kind: str, nbr) -> Panel:
    y = tl.counts[kind]
    feats = history_features(y, nbr)
    has = y.any(axis=1)
    first = np.where(has, (y > 0).argmax(axis=1), tl.n_hours)
    mask = (np.arange(tl.n_hours)[None, :] >= np.maximum(first, 168)[:, None]) & has[:, None]
    s, t = np.nonzero(mask)
    hist = {k: v[s, t].astype(np.float32) for k, v in feats.items()}
    del feats, mask
    print(f"{kind}: {len(s):,} station-hour rows", flush=True)
    return Panel(s=s.astype(np.int32), t=t.astype(np.int32), y=y[s, t], hist=hist)


def station_profile(p: Panel, n_st: int, hourly: dict, cut_t: int) -> np.ndarray:
    """Station x hour x weekday/weekend mean of y over hours before `cut_t` only."""
    weekend = (hourly["dow"][p.t] >= 5).astype(np.int64)
    key = p.s.astype(np.int64) * 48 + hourly["hour"][p.t].astype(np.int64) * 2 + weekend
    before = p.t < cut_t
    total = np.bincount(key[before], weights=p.y[before], minlength=n_st * 48)
    count = np.bincount(key[before], minlength=n_st * 48)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = (total / count)[key]
    # stations that opened after the training period fall back to their recent mean
    return np.where(np.isfinite(prof), prof, p.hist["same_hour_7d_mean"]).astype(np.float32)


def build_X(p: Panel, feats: list, profile: np.ndarray, hourly: dict, prev=False, rows=None) -> np.ndarray:
    cols = []
    idx = slice(None) if rows is None else rows
    for f in feats:
        if f == "station_profile":
            cols.append(profile[idx])
        elif f in hourly:
            src = hourly[f + "_prev"] if (prev and f in WEATHER) else hourly[f]
            cols.append(src[p.t[idx]])
        else:
            cols.append(p.hist[f][idx])
    return np.column_stack(cols).astype(np.float32)


def params_for(feats: list, overrides: dict) -> dict:
    return {**BASE_PARAMS, **overrides, "categorical_features": [feats.index("dow")]}


# ---------------------------------------------------------------- training

def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    return {
        "mae": round(float(np.abs(err).mean()), 4),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
        "wmape": round(float(np.abs(err).sum() / max(y.sum(), 1e-9)), 4),
    }


def tune(p: Panel, n_st: int, hourly: dict, cut: dict, rng) -> list:
    """Score each TUNING config on a sample of the headline validation months."""
    tr, va = p.t < cut["train_end"], (p.t >= cut["train_end"]) & (p.t < cut["valid_end"])
    itr = rng.choice(np.flatnonzero(tr), size=min(TUNING_TRAIN, int(tr.sum())), replace=False)
    iva = rng.choice(np.flatnonzero(va), size=min(TUNING_VALID, int(va.sum())), replace=False)
    profile = station_profile(p, n_st, hourly, cut["train_end"])
    out = []
    for label, feats, overrides in TUNING:
        X = build_X(p, feats, profile, hourly, rows=np.concatenate([itr, iva]))
        m = HistGradientBoostingRegressor(**params_for(feats, overrides), max_iter=TUNING_TREES)
        m.fit(X[: len(itr)], p.y[itr])
        mae = float(np.abs(m.predict(X[len(itr):]) - p.y[iva]).mean())
        out.append({"config": label, "valid_mae": round(mae, 4)})
        print(f"  tuning  {label:55} valid MAE {mae:.4f}", flush=True)
    return out


def fit_early_stopped(params: dict, Xtr, ytr, Xva, yva):
    """Add trees in steps on the training months until a step improves validation
    MAE by less than MIN_GAIN. Returns (trees, history, validation predictions)."""
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


TRAIN_CAP, VALID_CAP = 15_000_000, 3_000_000  # rows sampled for fitting, to keep memory under the limit


@dataclass
class Fit:
    model: object
    trees: int
    history: list
    profile: np.ndarray
    va: np.ndarray        # validation rows used for early stopping (sampled)
    pred_va: np.ndarray   # train-only model's predictions on va
    fit_rows: np.ndarray  # rows the final model was refit on
    te: np.ndarray        # test rows (all of them)
    pred_te: np.ndarray


def _cap(rows: np.ndarray, n: int, rng) -> np.ndarray:
    return rows if rows.size <= n else np.sort(rng.choice(rows, size=n, replace=False))


def fit_protocol(p: Panel, n_st: int, hourly: dict, feats: list, params: dict, cut: dict, rng) -> Fit:
    profile = station_profile(p, n_st, hourly, cut["train_end"])
    tr = _cap(np.flatnonzero(p.t < cut["train_end"]), TRAIN_CAP, rng)
    va = _cap(np.flatnonzero((p.t >= cut["train_end"]) & (p.t < cut["valid_end"])), VALID_CAP, rng)
    te = np.flatnonzero(p.t >= cut["test_start"])
    Xtr = build_X(p, feats, profile, hourly, rows=tr)
    Xva = build_X(p, feats, profile, hourly, rows=va)
    trees, history, pred_va = fit_early_stopped(params, Xtr, p.y[tr], Xva, p.y[va])
    Xfit = np.vstack([Xtr, Xva])
    del Xtr, Xva
    fit_rows = np.concatenate([tr, va])
    model = HistGradientBoostingRegressor(**params, max_iter=trees).fit(Xfit, p.y[fit_rows])
    del Xfit
    pred_te = model.predict(build_X(p, feats, profile, hourly, rows=te))
    return Fit(model, trees, history, profile, va, pred_va, fit_rows, te, pred_te)


def conformal_band(pred_cal, y_cal, pred_new, coverage=0.8):
    """Per bin of predicted level, the shortest range of trip counts holding
    `coverage` of calibration outcomes; bounds are interpolated between bin
    centres and rounded to the nearest whole trip. Bins are finer in the busy
    top 10%. Rounding to nearest (not outward) keeps the band from
    over-covering: outward rounding alone added about 5 points."""
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
            j = np.searchsorted(cum, cum[i] + need, side="left") - 1
            if j >= len(vals):
                break
            cand = (vals[j] - vals[i], -(cum[j + 1] - cum[i]), vals[i], vals[j])
            if cand[:2] < best[:2]:
                best = cand
        lo_by_bin[b], hi_by_bin[b] = best[2], best[3]
    lo_by_bin, hi_by_bin = np.maximum.accumulate(lo_by_bin), np.maximum.accumulate(hi_by_bin)
    centers = np.array([np.median(pred_cal[cal_bin == b]) if (cal_bin == b).any() else edges[b] for b in range(n_bins)])
    lo = np.round(np.interp(pred_new, centers, lo_by_bin))
    hi = np.round(np.interp(pred_new, centers, hi_by_bin))
    return lo, hi


SEGMENTS = [("quiet", lambda p: p < 1), ("medium", lambda p: (p >= 1) & (p < 5)), ("busy", lambda p: p >= 5)]


def calibrated_band(pred_va, y_va, pred_te, target=0.8):
    """One band level per segment of predicted demand (quiet, medium, busy), each
    chosen so that segment covers `target` of the VALIDATION months. The test
    months are never used to choose a level; they only receive the band."""
    lo_te, hi_te = np.zeros(len(pred_te)), np.zeros(len(pred_te))
    levels = {}
    for name, in_segment in SEGMENTS:
        mv, mt = in_segment(pred_va), in_segment(pred_te)
        lo_c, hi_c = 0.5, 0.95
        for _ in range(10):
            mid = (lo_c + hi_c) / 2
            lo, hi = conformal_band(pred_va[mv], y_va[mv], pred_va[mv], coverage=mid)
            if ((y_va[mv] >= lo) & (y_va[mv] <= hi)).mean() < target:
                lo_c = mid
            else:
                hi_c = mid
        level = (lo_c + hi_c) / 2
        levels[name] = round(float(level), 3)
        lo_te[mt], hi_te[mt] = conformal_band(pred_va[mv], y_va[mv], pred_te[mt], coverage=level)
    return lo_te, hi_te, levels


def interval_report(y, pred, lo, hi) -> dict:
    inside = (y >= lo) & (y <= hi)

    def part(mask):
        return {"share_of_hours": round(float(mask.mean()), 4), "coverage": round(float(inside[mask].mean()), 4),
                "above": round(float((y > hi)[mask].mean()), 4), "below": round(float((y < lo)[mask].mean()), 4),
                "mean_width": round(float((hi - lo)[mask].mean()), 2)}

    return {"nominal": 0.8, "all": part(np.ones(len(y), bool)), "quiet": part(pred < 1),
            "medium": part((pred >= 1) & (pred < 5)), "busy": part(pred >= 5)}


# ---------------------------------------------------------------- evaluation

def test_frame(tl: Timeline, p: Panel, te: np.ndarray, pred: np.ndarray, profile: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "station": tl.names[p.s[te]],
        "hour_start": tl.stamp(p.t[te]),
        "y": p.y[te].astype(np.float64),
        "pred": pred.astype(np.float64),
        "same_hour_7d_mean": p.hist["same_hour_7d_mean"][te],
        "lag_168": p.hist["lag_168"][te],
        "lag_24": p.hist["lag_24"][te],
        "station_profile": profile[te],
    })


def level_accuracy(f: pd.DataFrame, boro: dict) -> list:
    """Accuracy (1 - WMAPE) of the same forecasts summed to each planning level,
    next to a Poisson chance ceiling."""
    t = f.assign(boro=f["station"].map(boro).fillna("Other"), day=f["hour_start"].dt.normalize())
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
        noise = 2 * lam * poisson.pmf(np.floor(lam), lam)
        out.append({"level": key, "label": label, "avg_trips": round(float(g["y"].mean()), 1),
                    "model": acc("pred"), "seven_day_mean": acc("same_hour_7d_mean"),
                    "chance_ceiling": round(float(1 - noise.sum() / lam.sum()), 4)})
    return out


def hourly_block(f: pd.DataFrame) -> dict:
    out = {"model": metrics(f["y"].to_numpy(), f["pred"].to_numpy())}
    out |= {name: metrics(f["y"].to_numpy(), f[col].to_numpy()) for name, col in BASELINES.items()}
    return out


def net_flow_eval(dep: pd.DataFrame, arr: pd.DataFrame, holidays) -> dict:
    """Weekday 7-10am net flow per station-day, and how many of each morning's
    biggest losers each rule flags."""
    cols = ["y", "pred", "same_hour_7d_mean"]
    keys = ["station", "hour_start"]
    m = dep[keys + cols].merge(arr[keys + cols], on=keys, how="outer", suffixes=("_dep", "_arr"))
    for c in cols:
        for k in ("dep", "arr"):
            m[f"{c}_{k}"] = m[f"{c}_{k}"].fillna(0)
    h = m["hour_start"].dt.hour
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


def boroughs(tl: Timeline) -> dict:
    """Borough per station, by point-in-polygon against NYC's borough boundaries."""
    from build_geomap import assign_boroughs
    stations = [{"name": str(n), "lat": float(la), "lng": float(ln)} for n, la, ln in zip(tl.names, tl.lat, tl.lng)]
    assign_boroughs(stations)
    return {s["name"]: s["boro"] for s in stations}


# ---------------------------------------------------------------- main

def main() -> None:
    tl = read_timeline(YEARS, SMOKE)
    n_st = len(tl.names)
    nbr = neighbour_matrix(tl.lat, tl.lng)
    rng = np.random.default_rng(42)
    H, B = HEADLINE, BENCHMARK
    cut_H = {k: tl.hour_of(v) for k, v in H.items()}
    cut_B = {k: tl.hour_of(v) for k, v in B.items()}
    hourly = tl.hourly

    # ---- departures: tune, fit the headline and the benchmark, weather test, backtest
    dep = build_panel(tl, "departures", nbr)
    tuning = tune(dep, n_st, hourly, cut_H, rng)
    chosen_label = min(tuning, key=lambda r: r["valid_mae"])["config"]
    feats, overrides = next((f, o) for lbl, f, o in TUNING if lbl == chosen_label)
    params = params_for(feats, overrides)
    print(f"chosen: {chosen_label}", flush=True)

    fH = fit_protocol(dep, n_st, hourly, feats, params, cut_H, rng)
    print(f"headline departures: {fH.trees} trees", flush=True)
    dep_frame = test_frame(tl, dep, fH.te, fH.pred_te, fH.profile)
    y_te = dep_frame["y"].to_numpy()

    fB = fit_protocol(dep, n_st, hourly, feats, params, cut_B, rng)
    print(f"benchmark (2024 only) departures: {fB.trees} trees", flush=True)
    bench_pred = fB.pred_te

    # weather test: retrain with yesterday's weather in place of the day's
    prev_model = HistGradientBoostingRegressor(**params, max_iter=fH.trees).fit(
        build_X(dep, feats, fH.profile, hourly, prev=True, rows=fH.fit_rows), dep.y[fH.fit_rows])
    prev_pred = prev_model.predict(build_X(dep, feats, fH.profile, hourly, prev=True, rows=fH.te))
    del prev_model

    # rolling backtest, one fold per month of the test year
    backtest = []
    for month in range(6, 9):
        start = tl.hour_of(f"2025-{month:02d}-01")
        end = tl.hour_of(f"2025-{month + 1:02d}-01") if month < 8 else tl.n_hours
        ftr = np.flatnonzero(dep.t < start)
        fte = np.flatnonzero((dep.t >= start) & (dep.t < end))
        sample = rng.choice(ftr, size=min(BACKTEST_SAMPLE, ftr.size), replace=False)
        profile_f = station_profile(dep, n_st, hourly, start)
        Xf = build_X(dep, feats, profile_f, hourly, rows=np.concatenate([sample, fte]))
        fm = HistGradientBoostingRegressor(**params, max_iter=BACKTEST_TREES).fit(Xf[: len(sample)], dep.y[sample])
        yf = dep.y[fte]
        fold = {
            "month": month,
            "model": round(float(np.abs(fm.predict(Xf[len(sample):]) - yf).mean()), 4),
            "seven_day_mean": round(float(np.abs(dep.hist["same_hour_7d_mean"][fte] - yf).mean()), 4),
            "seasonal_naive": round(float(np.abs(dep.hist["lag_168"][fte] - yf).mean()), 4),
        }
        print("  backtest", fold, flush=True)
        backtest.append(fold)
        del Xf

    # permutation importance on a sample of test hours
    sample = rng.choice(fH.te, size=min(300_000, fH.te.size), replace=False)
    Xs = build_X(dep, feats, fH.profile, hourly, rows=sample)
    perm = permutation_importance(fH.model, Xs, dep.y[sample], scoring="neg_mean_absolute_error",
                                  n_repeats=3, random_state=42, n_jobs=1)
    importances = sorted(
        ({"feature": f, "label": FEATURE_LABELS[f], "importance": round(float(m), 5)}
         for f, m in zip(feats, perm.importances_mean)),
        key=lambda d: -d["importance"],
    )
    del Xs

    y_te = dep.y[fH.te]
    np.savez(REPO / ".cache" / ("forecast_preds_smoke.npz" if SMOKE else "forecast_preds.npz"),
             pred_va=fH.pred_va, y_va=dep.y[fH.va], pred_te=fH.pred_te, y_te=y_te,
             t_te=dep.t[fH.te], s_te=dep.s[fH.te])  # for diagnosing the prediction band
    lo, hi, band_levels = calibrated_band(fH.pred_va, dep.y[fH.va], fH.pred_te)
    intervals = interval_report(y_te, fH.pred_te, lo, hi)
    intervals["band_levels_chosen_on_validation"] = band_levels
    dep_frame = dep_frame.assign(lo=lo, hi=hi)

    # example week at the busiest test station
    busiest = dep_frame.groupby("station")["y"].sum().idxmax()
    ws = pd.Timestamp("2025-07-07")
    ex = dep_frame[(dep_frame["station"] == busiest) & (dep_frame["hour_start"] >= ws)
                   & (dep_frame["hour_start"] < ws + pd.Timedelta(days=7))]
    example = {"station": str(busiest), "week_start": ws.strftime("%Y-%m-%d"),
               "series": [{"t": t.strftime("%Y-%m-%dT%H"), "actual": int(a), "pred": round(float(p), 2),
                           "lo": float(l_), "hi": float(h_), "naive": int(n)}
                          for t, a, p, l_, h_, n in zip(ex["hour_start"], ex["y"], ex["pred"], ex["lo"], ex["hi"], ex["lag_168"])]}

    # ---- arrivals: headline only, for net flow
    arr = build_panel(tl, "arrivals", nbr)
    fA = fit_protocol(arr, n_st, hourly, feats, params, cut_H, rng)
    print(f"headline arrivals: {fA.trees} trees", flush=True)
    arr_frame = test_frame(tl, arr, fA.te, fA.pred_te, fA.profile)

    boro = boroughs(tl)
    holidays = USFederalHolidayCalendar().holidays(tl.t0, tl.t0 + pd.Timedelta(hours=tl.n_hours))
    dep_hourly = hourly_block(dep_frame)
    bench_frame = dep_frame.assign(pred=bench_pred)
    prev_frame = dep_frame.assign(pred=prev_pred)

    def gain(block, base):
        return round(1 - block["model"]["mae"] / block[base]["mae"], 4)

    arr_hourly = hourly_block(arr_frame)
    out = {
        "split": {"headline_train": f"{tl.t0:%Y-%m-%d} to {H['valid_end']}",
                  "validation": f"{H['train_end']} to {H['valid_end']}",
                  "test": f"{H['test_start']} to {WEATHER_END}",
                  "benchmark_train": "2024-01-08 to 2024-12-31", "benchmark_test": f"{B['test_start']} to {WEATHER_END}"},
        "stations": int(len(np.unique(dep.s[fH.te]))),
        "chosen": chosen_label,
        "features": [FEATURE_LABELS[f] for f in feats],
        "tuning": tuning,
        "stopping_rule": {"step": TREE_STEP, "min_gain": MIN_GAIN, "max_trees": MAX_TREES},
        "departures": {"trees": fH.trees, "early_stopping": fH.history, "hourly": dep_hourly,
                       "levels": level_accuracy(dep_frame, boro),
                       "rows": {"test": int(fH.te.size)}},
        "arrivals": {"trees": fA.trees, "hourly": arr_hourly},
        "benchmark_2024_only": {"trees": fB.trees, "hourly": hourly_block(bench_frame),
                                "levels": level_accuracy(bench_frame, boro)},
        "yesterdays_weather": {"hourly_mae": metrics(y_te, prev_pred)["mae"],
                               "levels": level_accuracy(prev_frame, boro)},
        "improvement": {
            "vs_seven_day_mean_hourly": gain(dep_hourly, "seven_day_mean"),
            "vs_seasonal_naive_hourly": gain(dep_hourly, "seasonal_naive"),
            "vs_yesterday_hourly": gain(dep_hourly, "yesterday"),
            "arrivals_vs_seven_day_mean_hourly": gain(arr_hourly, "seven_day_mean"),
            "benchmark_vs_headline_hourly": round(1 - dep_hourly["model"]["mae"] / hourly_block(bench_frame)["model"]["mae"], 4),
        },
        "net_flow_am": net_flow_eval(dep_frame, arr_frame, holidays),
        "intervals": intervals,
        "example": example,
        "feature_importances": importances,
        "backtest": backtest,
    }
    data_dir = REPO / (".cache" if SMOKE else "docs/data")
    payload = json.dumps(out, separators=(",", ":"))
    (data_dir / "forecast.json").write_text(payload)
    (data_dir / "forecast.js").write_text("window.FORECAST_DATA = " + payload + ";\n")
    print(f"Wrote {data_dir.relative_to(REPO)}/forecast.{{json,js}}", flush=True)
    print(json.dumps({k: out[k] for k in ("chosen", "improvement", "benchmark_2024_only", "yesterdays_weather",
                                          "net_flow_am", "intervals")}, indent=1))


if __name__ == "__main__":
    main()
