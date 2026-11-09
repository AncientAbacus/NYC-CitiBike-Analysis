"""
Shared load / clean / feature-engineering steps for every consumer of the
Citi Bike trip data: CitiBike_Analysis.ipynb, build_heatmap.py,
build_geomap.py, and build_model.py.

This module is the single source of truth for what counts as a "ride". Every
number on the site and in the notebook comes from a DataFrame produced by
`load_clean_month()`, so the ride totals agree everywhere.

The downloaded zip is cached in `.cache/` (gitignored) so running several
build scripts back to back only downloads the month once.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / ".cache"
DEFAULT_MONTH = "202401"

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
    for label, step in steps:
        before = len(df)
        df = step(df)
        if verbose:
            print(f"  removed {before - len(df):>7,}  {label}", flush=True)
    if verbose:
        print(f"Removed {start - len(df):,} rows ({(start - len(df)) / start:.2%}); remaining {len(df):,}", flush=True)
    return df.reset_index(drop=True)


def load_clean_month(yyyymm: str = DEFAULT_MONTH) -> pd.DataFrame:
    return clean(load_month(yyyymm), yyyymm)


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
