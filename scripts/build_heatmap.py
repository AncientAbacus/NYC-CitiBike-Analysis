"""
Rebuild the ride-volume heatmap (docs/data/heatmap.json + .js, docs/assets/heatmap.png)
from a month of Citi Bike trip data. Mirrors the cleaning steps in
CitiBike_Analysis.ipynb so the site and notebook stay consistent.

Usage: python scripts/build_heatmap.py [YYYYMM]
       python scripts/build_heatmap.py 202401   # defaults to this if omitted
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_month(yyyymm: str) -> pd.DataFrame:
    url = f"https://s3.amazonaws.com/tripdata/{yyyymm}-citibike-tripdata.zip"
    print(f"Downloading {yyyymm} Citi Bike trip data...", flush=True)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n]
        print("Reading CSVs:", names, flush=True)
        dfs = [
            pd.read_csv(z.open(n), usecols=["ride_id", "started_at", "ended_at", "member_casual"])
            for n in names
        ]
    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame, yyyymm: str) -> pd.DataFrame:
    month_start = pd.Period(yyyymm, freq="M").start_time
    month_end = month_start + pd.offsets.MonthBegin(1)

    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["trip_minutes"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

    before = len(df)
    df = df.drop_duplicates(subset="ride_id")
    df = df[(df["started_at"] >= month_start) & (df["started_at"] < month_end)]
    df = df[df["trip_minutes"] > 0]
    df = df[df["trip_minutes"] <= 24 * 60]
    after = len(df)
    print(f"Removed {before - after:,} rows ({(before - after) / before:.2%}); remaining {after:,}", flush=True)
    return df


def build_heat_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day"] = df["started_at"].dt.day_name()
    df["hour"] = df["started_at"].dt.hour
    return df.groupby(["day", "hour"]).size().unstack(fill_value=0).reindex(DAY_ORDER)


def write_site_data(heat: pd.DataFrame, total_rides: int, label: str) -> None:
    records = [
        {"day": day, "hour": hour, "rides": int(heat.loc[day, hour]) if hour in heat.columns else 0}
        for day in DAY_ORDER
        for hour in range(24)
    ]
    out = {
        "title": "Citi Bike Ride Volume by Hour and Day of Week",
        "subtitle": f"{label}, NYC Citi Bike system data",
        "days": DAY_ORDER,
        "hours": list(range(24)),
        "data": records,
        "total_rides": total_rides,
    }

    data_dir = REPO / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "heatmap.json").write_text(json.dumps(out))
    (data_dir / "heatmap.js").write_text("window.HEATMAP_DATA = " + json.dumps(out) + ";\n")
    print("Wrote docs/data/heatmap.json and heatmap.js", flush=True)


def write_png(heat: pd.DataFrame, label: str) -> None:
    blue_seq = LinearSegmentedColormap.from_list(
        "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    )
    assets_dir = REPO / "docs" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 4.5))
    sns.heatmap(heat, cmap=blue_seq, cbar_kws={"label": "Rides"})
    plt.title(f"Citi Bike Ride Volume by Hour and Day of Week ({label})")
    plt.xlabel("Hour of day")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(assets_dir / "heatmap.png", dpi=150)
    print("Wrote docs/assets/heatmap.png", flush=True)


def main() -> None:
    yyyymm = sys.argv[1] if len(sys.argv) > 1 else "202401"
    label = pd.Period(yyyymm, freq="M").strftime("%B %Y")

    df = load_month(yyyymm)
    print(f"Loaded {len(df):,} rows", flush=True)
    df = clean(df, yyyymm)

    heat = build_heat_matrix(df)
    write_site_data(heat, total_rides=len(df), label=label)
    write_png(heat, label=label)


if __name__ == "__main__":
    main()
