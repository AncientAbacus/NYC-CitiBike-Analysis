"""
Build the "seasons and weather" data behind the site (docs/data/seasonality.json + .js):
daily ride counts for a year (members vs. casual) joined to NOAA's Central Park
daily weather, plus a simple log-linear model that separates the effect of
temperature and rain from the day of week.

Model: log(daily rides) ~ day of week + federal holiday + tmax + tmax^2
                          + rain type + snow day
                          + linear trend (system growth over the year)

Rain type uses NOAA's *hourly* observations, so a 3am shower isn't counted
the same as a wet rush hour. Mutually exclusive, vs. a dry day:
  off-peak rain   >= 1 mm that day, but < 1 mm during the commute windows
  commute rain    >= 1 mm during 7-10am / 4-7pm, < 10 mm for the day
  heavy rain      >= 10 mm for the day

Rain and temperature are correlated with season, so each effect is read as
"vs. a dry day with the same temperature and weekday", not a raw average.

Uncertainty: 95% intervals from a moving-block bootstrap (14-day blocks of
days, 2,000 resamples). Daily ridership is autocorrelated, so resampling
single days would understate the uncertainty.
The temperature terms are controls only: the quadratic can't capture the
plateau-then-dip above ~30C, so the site charts binned averages of dry
weekdays for the temperature relationship instead.

Usage: python scripts/build_seasonality.py [YYYY]   # defaults to 2024
"""

import json
import sys

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from pipeline import DEFAULT_PERIOD, REPO, load_clean_period, load_weather_daily, load_weather_hourly

LIGHT_RAIN_MM = 1.0   # >= 1 mm counts as a rainy day / a wet commute
HEAVY_RAIN_MM = 10.0  # >= 10 mm counts as heavy rain
COMMUTE_HOURS = [7, 8, 9, 16, 17, 18]
BLOCK_DAYS, N_BOOT = 14, 2000
EFFECTS = ["offpeak_rain", "commute_rain", "heavy_rain", "snow", "holiday", "trend"]


def commute_precip(year: str) -> pd.DataFrame:
    h = load_weather_hourly(year)
    wet = h[h["hour_start"].dt.hour.isin(COMMUTE_HOURS)]
    return wet.groupby(wet["hour_start"].dt.normalize().rename("date"))["precip_mm"].sum().rename("commute_precip_mm").reset_index()


def daily_rides(year: str) -> pd.DataFrame:
    df = load_clean_period(year, columns=["started_at", "member_casual"])
    day = df["started_at"].dt.normalize().rename("date")
    counts = df.groupby([day, "member_casual"], observed=True).size().unstack(fill_value=0)
    counts["rides"] = counts.sum(axis=1)
    return counts.reset_index()


def design(d: pd.DataFrame) -> tuple:
    dow = pd.get_dummies(d["date"].dt.dayofweek, prefix="dow", drop_first=True, dtype=float)
    holidays = USFederalHolidayCalendar().holidays(d["date"].min(), d["date"].max())
    t = (d["tmax_c"] - 20) / 10  # centered/scaled so the quadratic is well-conditioned
    heavy = d["precip_mm"] >= HEAVY_RAIN_MM
    commute = (d["commute_precip_mm"] >= LIGHT_RAIN_MM) & ~heavy
    offpeak = (d["precip_mm"] >= LIGHT_RAIN_MM) & ~heavy & ~commute
    X = pd.concat(
        [
            pd.Series(1.0, index=d.index, name="const"),
            dow,
            d["date"].isin(holidays).astype(float).rename("holiday"),
            t.rename("t"),
            (t ** 2).rename("t2"),
            offpeak.astype(float).rename("offpeak_rain"),
            commute.astype(float).rename("commute_rain"),
            heavy.astype(float).rename("heavy_rain"),
            (d["snow_mm"] > 0).astype(float).rename("snow"),
            # the system kept growing through the year (October beat July), which
            # would otherwise be credited to temperature
            ((d["date"] - d["date"].min()).dt.days / 365).rename("trend"),
        ],
        axis=1,
    )
    return X, np.log(d["rides"].to_numpy())


def fit_weather_model(daily: pd.DataFrame) -> dict:
    X, y = design(daily)
    cols = list(X.columns)
    Xv = X.to_numpy()
    beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
    pred = Xv @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    # moving-block bootstrap over days
    rng = np.random.default_rng(42)
    n = len(y)
    starts = np.arange(n - BLOCK_DAYS + 1)
    n_blocks = int(np.ceil(n / BLOCK_DAYS))
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([np.arange(s, s + BLOCK_DAYS) for s in rng.choice(starts, n_blocks)])[:n]
        b, *_ = np.linalg.lstsq(Xv[idx], y[idx], rcond=None)
        draws.append(b)
    draws = np.array(draws)

    def effect(c):
        i = cols.index(c)
        lo, hi = np.percentile(draws[:, i], [2.5, 97.5])
        return {
            "effect": round(float(np.exp(beta[i]) - 1), 4),
            "ci95": [round(float(np.exp(lo) - 1), 4), round(float(np.exp(hi) - 1), 4)],
        }

    flags = {c: int(X[c].sum()) for c in ("offpeak_rain", "commute_rain", "heavy_rain", "snow", "holiday")}
    return {
        "r2": round(float(r2), 3),
        "residual_sd_log": round(float(np.std(y - pred)), 3),
        "bootstrap": {"block_days": BLOCK_DAYS, "resamples": N_BOOT},
        "days_flagged": flags,
        "effects": {c: effect(c) for c in EFFECTS},
    }


def main() -> None:
    year = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERIOD
    daily = (daily_rides(year).merge(load_weather_daily(year), on="date", how="left")
             .merge(commute_precip(year), on="date", how="left").fillna({"commute_precip_mm": 0.0}))

    model = fit_weather_model(daily)
    dry_weekday = (daily["date"].dt.dayofweek < 5) & (daily["precip_mm"] < LIGHT_RAIN_MM)
    bins = (daily["tmax_c"] // 5 * 5).astype(int)
    binned = daily[dry_weekday].groupby(bins[dry_weekday])["rides"].agg(["mean", "size"])
    monthly = daily.groupby(daily["date"].dt.month).agg(rides=("rides", "sum"), casual=("casual", "sum"))
    weekday = daily["date"].dt.dayofweek < 5
    busiest = daily.loc[daily["rides"].idxmax()]
    quietest = daily.loc[daily["rides"].idxmin()]

    out = {
        "year": year,
        "total_rides": int(daily["rides"].sum()),
        "thresholds_mm": {"light_rain": LIGHT_RAIN_MM, "heavy_rain": HEAVY_RAIN_MM},
        "commute_hours": COMMUTE_HOURS,
        "daily": [
            {
                "date": r.date.strftime("%Y-%m-%d"),
                "rides": int(r.rides),
                "member": int(r.member),
                "casual": int(r.casual),
                "tmax_c": float(r.tmax_c),
                "precip_mm": float(r.precip_mm),
                "snow_mm": float(r.snow_mm),
                "commute_precip_mm": round(float(r.commute_precip_mm), 1),
            }
            for r in daily.itertuples()
        ],
        "monthly": [
            {"month": int(m), "rides": int(r.rides), "casual_share": round(float(r.casual / r.rides), 4)}
            for m, r in monthly.iterrows()
        ],
        "busiest_day": {"date": busiest["date"].strftime("%Y-%m-%d"), "rides": int(busiest["rides"]),
                        "tmax_c": float(busiest["tmax_c"]), "precip_mm": float(busiest["precip_mm"])},
        "quietest_day": {"date": quietest["date"].strftime("%Y-%m-%d"), "rides": int(quietest["rides"]),
                         "tmax_c": float(quietest["tmax_c"]), "precip_mm": float(quietest["precip_mm"]),
                         "snow_mm": float(quietest["snow_mm"])},
        "dry_weekday_by_tmax_5c": [
            {"tmax_from_c": int(b), "avg_rides": round(float(r["mean"])), "days": int(r["size"])}
            for b, r in binned.iterrows()
        ],
        "avg_weekday_rides": round(float(daily.loc[weekday, "rides"].mean())),
        "avg_weekend_rides": round(float(daily.loc[~weekday, "rides"].mean())),
        "model": model,
    }

    data_dir = REPO / "docs" / "data"
    (data_dir / "seasonality.json").write_text(json.dumps(out, separators=(",", ":")))
    (data_dir / "seasonality.js").write_text("window.SEASONALITY_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print("Wrote docs/data/seasonality.{json,js}", flush=True)
    print(json.dumps({k: v for k, v in out.items() if k != "daily"}, indent=1))


if __name__ == "__main__":
    main()
