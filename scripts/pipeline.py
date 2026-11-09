"""
Shared load / clean / feature-engineering steps for every consumer of the
Citi Bike trip data: CitiBike_Analysis.ipynb, build_heatmap.py,
build_geomap.py, and build_model.py.

This module is the single source of truth for what counts as a "ride". Every
number on the site and in the notebook comes from a DataFrame produced by
`load_clean_month()`, so the ride totals agree everywhere.

Downloads (monthly zips, cleaned per-month parquet, NOAA weather) are cached
in `.cache/` (gitignored) so running several build scripts back to back only
downloads and cleans each month once.

A "period" is either a month (`YYYYMM`, e.g. 202401) or a full year (`YYYY`,
e.g. 2024); every build script accepts either.
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / ".cache"
DEFAULT_MONTH = "202401"
DEFAULT_PERIOD = "2024"

STATION_COLS = ["start_station_name", "end_station_name", "start_lat", "start_lng", "end_lat", "end_lng"]
RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}

FEATURES = [
    "hour",
    "is_weekend",
    "is_rush_hour",
    "is_holiday",
    "trip_minutes",
    "distance_km",
    "speed_kmh",
    "is_electric",
    "is_roundtrip",
    "start_station_rides_log",
    "end_station_rides_log",
]
FEATURE_LABELS = {
    "hour": "Hour of day",
    "is_weekend": "Weekend",
    "is_rush_hour": "Rush hour",
    "is_holiday": "Federal holiday",
    "trip_minutes": "Trip duration (min)",
    "distance_km": "Distance (km)",
    "speed_kmh": "Speed (km/h)",
    "is_electric": "Electric bike",
    "is_roundtrip": "Round trip",
    "start_station_rides_log": "Start station popularity",
    "end_station_rides_log": "End station popularity",
}


def month_label(yyyymm: str) -> str:
    return pd.Period(yyyymm, freq="M").strftime("%B %Y")


def period_months(period: str) -> list:
    """`2024` -> ['202401', ..., '202412']; `202401` -> ['202401']."""
    if len(period) == 4:
        return [f"{period}{m:02d}" for m in range(1, 13)]
    if len(period) == 6:
        return [period]
    raise ValueError(f"period must be YYYY or YYYYMM, got {period!r}")


def period_label(period: str) -> str:
    return period if len(period) == 4 else month_label(period)


def _month_zip(yyyymm: str) -> bytes:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{yyyymm}-citibike-tripdata.zip"
    if path.exists():
        print(f"Using cached {path.name}", flush=True)
        return path.read_bytes()
    url = f"https://s3.amazonaws.com/tripdata/{yyyymm}-citibike-tripdata.zip"
    print(f"Downloading {url} ...", flush=True)
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return resp.content


def load_month(yyyymm: str = DEFAULT_MONTH) -> pd.DataFrame:
    """Read every trip CSV in the month's zip into one DataFrame."""
    with zipfile.ZipFile(io.BytesIO(_month_zip(yyyymm))) as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n]
        print("Reading CSVs:", names, flush=True)
        dfs = [
            pd.read_csv(z.open(n), dtype={"start_station_id": str, "end_station_id": str})
            for n in names
        ]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} rows", flush=True)
    return df


def clean(df: pd.DataFrame, yyyymm: str = DEFAULT_MONTH, verbose: bool = True) -> pd.DataFrame:
    """Apply the cleaning rules, in order, logging how many rows each removes.

    1. drop duplicate ride_ids
    2. drop trips starting outside the month (spillover at the file's edges)
    3. drop trips missing station names or coordinates
    4. drop non-positive durations
    5. drop trips over 24h (Citi Bike treats these as lost/stolen bikes)
    """
    month_start = pd.Period(yyyymm, freq="M").start_time
    month_end = month_start + pd.offsets.MonthBegin(1)

    df = df.copy()
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["trip_minutes"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

    steps = [
        ("duplicate ride_id", lambda d: d.drop_duplicates(subset="ride_id")),
        ("outside the month", lambda d: d[(d["started_at"] >= month_start) & (d["started_at"] < month_end)]),
        ("missing station/coordinates", lambda d: d.dropna(subset=STATION_COLS)),
        ("non-positive duration", lambda d: d[d["trip_minutes"] > 0]),
        ("over 24 hours", lambda d: d[d["trip_minutes"] <= 24 * 60]),
    ]
    start = len(df)
    removed = {}
    for label, step in steps:
        before = len(df)
        df = step(df)
        removed[label] = before - len(df)
        if verbose:
            print(f"  removed {removed[label]:>7,}  {label}", flush=True)
    if verbose:
        print(f"Removed {start - len(df):,} rows ({(start - len(df)) / start:.2%}); remaining {len(df):,}", flush=True)
    df = df.reset_index(drop=True)
    df.attrs["cleaning"] = {"raw_rows": start, "clean_rows": len(df), "removed": removed}
    return df


def load_clean_month(yyyymm: str = DEFAULT_MONTH) -> pd.DataFrame:
    return clean(load_month(yyyymm), yyyymm)


# Columns kept in the per-month cleaned cache: everything any analysis reads.
CLEAN_COLS = [
    "started_at", "ended_at", "trip_minutes", "member_casual", "rideable_type",
    "start_station_name", "end_station_name", "start_lat", "start_lng", "end_lat", "end_lng",
]


def clean_month_path(yyyymm: str) -> Path:
    """Clean one month and cache it as parquet (only once), alongside a small
    JSON of how many rows each cleaning rule removed; return the parquet path."""
    out = CACHE_DIR / "clean" / f"{yyyymm}.parquet"
    stats = out.with_suffix(".stats.json")
    if not (out.exists() and stats.exists()):
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"== {yyyymm}", flush=True)
        df = load_clean_month(yyyymm)
        stats.write_text(json.dumps(df.attrs["cleaning"]))
        df = df[CLEAN_COLS]
        for col in ("member_casual", "rideable_type", "start_station_name", "end_station_name"):
            df[col] = df[col].astype("category")
        df.to_parquet(out, index=False)
    return out


def cleaning_stats(period: str = DEFAULT_PERIOD) -> dict:
    """Raw rows, clean rows and rows removed per rule, summed over the period."""
    total = {"raw_rows": 0, "clean_rows": 0, "removed": {}}
    for m in period_months(period):
        clean_month_path(m)
        st = json.loads((CACHE_DIR / "clean" / f"{m}.stats.json").read_text())
        total["raw_rows"] += st["raw_rows"]
        total["clean_rows"] += st["clean_rows"]
        for k, v in st["removed"].items():
            total["removed"][k] = total["removed"].get(k, 0) + v
    return total


def load_clean_period(period: str = DEFAULT_PERIOD, columns=None) -> pd.DataFrame:
    """Cleaned trips for a month or a whole year, optionally only some columns.

    Each month is cleaned on its own (trips are assigned to the month they
    start in), then concatenated.
    """
    paths = [clean_month_path(m) for m in period_months(period)]
    dfs = [pd.read_parquet(p, columns=columns) for p in paths]
    # re-unify categoricals across months so concat doesn't fall back to object
    for col in ("member_casual", "rideable_type", "start_station_name", "end_station_name"):
        if all(col in d for d in dfs):
            cats = sorted(set().union(*(d[col].cat.categories for d in dfs)))
            for d in dfs:
                d[col] = d[col].cat.set_categories(cats)
    df = pd.concat(dfs, ignore_index=True)
    print(f"{period_label(period)}: {len(df):,} cleaned trips", flush=True)
    return df


# ---- weather: NOAA observations at the Central Park station ----
NOAA_API = "https://www.ncei.noaa.gov/access/services/data/v1"
CENTRAL_PARK_LCD = "72505394728"
CENTRAL_PARK_GHCND = "USW00094728"


def _noaa_csv(name: str, params: dict) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / name
    if not path.exists():
        print(f"Downloading NOAA {params['dataset']} ...", flush=True)
        resp = requests.get(NOAA_API, params={**params, "format": "csv", "units": "metric"}, timeout=300)
        resp.raise_for_status()
        path.write_text(resp.text)
    return pd.read_csv(path, low_memory=False)


def _noaa_number(series: pd.Series) -> pd.Series:
    # LCD values can carry an "s" (suspect) suffix; "T" is a trace of precipitation
    s = series.astype(str).str.strip().str.rstrip("s").replace({"T": "0.01", "": None, "nan": None})
    return pd.to_numeric(s, errors="coerce")


def load_weather_hourly(year: str) -> pd.DataFrame:
    """One row per hour of `year`: temp_c, precip_mm, wind_ms (Central Park, NOAA LCD).

    Uses the routine hourly reports (FM-15) only. An observation at hh:51
    is assigned to hour hh. Gaps are filled by interpolation (temp/wind) or 0 (precip).
    """
    raw = _noaa_csv(f"noaa_lcd_{year}.csv", {
        "dataset": "local-climatological-data", "stations": CENTRAL_PARK_LCD,
        "startDate": f"{year}-01-01", "endDate": f"{year}-12-31",
        "dataTypes": "HourlyDryBulbTemperature,HourlyPrecipitation,HourlyWindSpeed",
    })
    raw = raw[raw["REPORT_TYPE"].astype(str).str.strip() == "FM-15"]
    df = pd.DataFrame({
        "hour_start": pd.to_datetime(raw["DATE"]).dt.floor("h"),
        "temp_c": _noaa_number(raw["HourlyDryBulbTemperature"]),
        "precip_mm": _noaa_number(raw["HourlyPrecipitation"]),
        "wind_ms": _noaa_number(raw["HourlyWindSpeed"]),
    }).groupby("hour_start").mean()
    hours = pd.date_range(f"{year}-01-01", f"{int(year) + 1}-01-01", freq="h", inclusive="left")
    df = df.reindex(hours)
    df["temp_c"] = df["temp_c"].interpolate(limit_direction="both")
    df["wind_ms"] = df["wind_ms"].interpolate(limit_direction="both")
    df["precip_mm"] = df["precip_mm"].fillna(0.0)
    df.index.name = "hour_start"
    return df.reset_index()


def load_weather_daily(year: str) -> pd.DataFrame:
    """One row per day: tmax_c, tmin_c, precip_mm, snow_mm (Central Park, NOAA GHCN-Daily)."""
    raw = _noaa_csv(f"noaa_daily_{year}.csv", {
        "dataset": "daily-summaries", "stations": CENTRAL_PARK_GHCND,
        "startDate": f"{year}-01-01", "endDate": f"{year}-12-31",
        "dataTypes": "TMAX,TMIN,PRCP,SNOW",
    })
    return pd.DataFrame({
        "date": pd.to_datetime(raw["DATE"]),
        "tmax_c": raw["TMAX"], "tmin_c": raw["TMIN"],
        "precip_mm": raw["PRCP"].fillna(0.0), "snow_mm": raw["SNOW"].fillna(0.0),
    })


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    r = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame, yyyymm: str = DEFAULT_MONTH) -> pd.DataFrame:
    """Add the model's features (see FEATURES) and the `is_member` target."""
    df = df.copy()
    month_start = pd.Period(yyyymm, freq="M").start_time
    month_end = month_start + pd.offsets.MonthBegin(1)

    df["hour"] = df["started_at"].dt.hour
    df["is_weekend"] = (df["started_at"].dt.dayofweek >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype(int)

    holidays = USFederalHolidayCalendar().holidays(month_start, month_end)
    df["is_holiday"] = df["started_at"].dt.normalize().isin(holidays).astype(int)

    df["distance_km"] = haversine_km(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"])
    df["speed_kmh"] = df["distance_km"] / (df["trip_minutes"] / 60)
    df["is_electric"] = (df["rideable_type"] == "electric_bike").astype(int)
    df["is_roundtrip"] = (df["start_station_name"] == df["end_station_name"]).astype(int)

    # Station popularity counts *all* rides through a station and never looks
    # at member_casual, so it can't leak the label.
    df["start_station_rides_log"] = np.log1p(df.groupby("start_station_name")["ride_id"].transform("count"))
    df["end_station_rides_log"] = np.log1p(df.groupby("end_station_name")["ride_id"].transform("count"))

    df["is_member"] = (df["member_casual"] == "member").astype(int)
    return df
