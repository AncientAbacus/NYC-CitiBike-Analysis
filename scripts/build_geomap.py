"""
Build the station-level ridership data behind docs/map.html: per-station
location, total rides, and an hourly ride-count profile (for the time-of-day
slider), plus a trimmed copy of NYC DOT's bike route network.

Cleaning mirrors CitiBike_Analysis.ipynb / scripts/build_heatmap.py so all
three stay consistent.

Usage: python scripts/build_geomap.py [YYYYMM]
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
BIKE_ROUTES_URL = (
    "https://data.cityofnewyork.us/resource/mzxg-pwib.geojson"
    "?$select=the_geom,facilitycl,onoffst,boro,street"
    "&$where=status='Current'"
    "&$limit=50000"
)
BORO_NAMES = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens", "5": "Staten Island"}


def load_month(yyyymm: str) -> pd.DataFrame:
    url = f"https://s3.amazonaws.com/tripdata/{yyyymm}-citibike-tripdata.zip"
    print(f"Downloading {yyyymm} Citi Bike trip data...", flush=True)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n]
        print("Reading CSVs:", names, flush=True)
        cols = ["ride_id", "started_at", "ended_at", "start_station_name", "start_lat", "start_lng"]
        dfs = [pd.read_csv(z.open(n), usecols=cols, dtype={"start_lat": float, "start_lng": float}) for n in names]
    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame, yyyymm: str) -> pd.DataFrame:
    month_start = pd.Period(yyyymm, freq="M").start_time
    month_end = month_start + pd.offsets.MonthBegin(1)

    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    trip_minutes = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

    before = len(df)
    df = df.drop_duplicates(subset="ride_id")
    df = df[(df["started_at"] >= month_start) & (df["started_at"] < month_end)]
    df = df[(trip_minutes.loc[df.index] > 0) & (trip_minutes.loc[df.index] <= 24 * 60)]
    df = df.dropna(subset=["start_station_name", "start_lat", "start_lng"])
    after = len(df)
    print(f"Removed {before - after:,} rows ({(before - after) / before:.2%}); remaining {after:,}", flush=True)
    return df


def build_stations(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["hour"] = df["started_at"].dt.hour

    grouped = df.groupby("start_station_name")
    stations = []
    for name, g in grouped:
        hourly = g.groupby("hour").size().reindex(range(24), fill_value=0).tolist()
        stations.append(
            {
                "name": name,
                "lat": round(float(g["start_lat"].median()), 6),
                "lng": round(float(g["start_lng"].median()), 6),
                "total": int(len(g)),
                "hourly": hourly,
            }
        )
    stations.sort(key=lambda s: -s["total"])
    return stations


def write_stations(stations: list, label: str) -> None:
    out = {
        "title": "Citi Bike Ridership by Station",
        "subtitle": f"{label}, station-level trip starts by hour",
        "stations": stations,
    }
    data_dir = REPO / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "stations.json").write_text(json.dumps(out, separators=(",", ":")))
    (data_dir / "stations.js").write_text("window.STATIONS_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"Wrote docs/data/stations.{{json,js}} ({len(stations)} stations)", flush=True)


def write_bike_routes() -> None:
    print("Downloading NYC DOT bike route network...", flush=True)
    resp = requests.get(BIKE_ROUTES_URL, timeout=120)
    resp.raise_for_status()
    geo = resp.json()

    def round_coords(coords):
        if isinstance(coords[0], (int, float)):
            return [round(c, 5) for c in coords]
        return [round_coords(c) for c in coords]

    for feat in geo["features"]:
        feat["geometry"]["coordinates"] = round_coords(feat["geometry"]["coordinates"])
        props = feat["properties"]
        props["boro_name"] = BORO_NAMES.get(props.get("boro"), "")

    data_dir = REPO / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(geo, separators=(",", ":"))
    (data_dir / "bike_paths.geojson").write_text(payload)
    (data_dir / "bike_paths.js").write_text("window.BIKE_PATHS_DATA = " + payload + ";\n")
    print(f"Wrote docs/data/bike_paths.{{geojson,js}} ({len(geo['features'])} segments)", flush=True)


def main() -> None:
    yyyymm = sys.argv[1] if len(sys.argv) > 1 else "202401"
    label = pd.Period(yyyymm, freq="M").strftime("%B %Y")

    df = load_month(yyyymm)
    print(f"Loaded {len(df):,} rows", flush=True)
    df = clean(df, yyyymm)

    stations = build_stations(df)
    write_stations(stations, label=label)
    write_bike_routes()


if __name__ == "__main__":
    main()
