"""
Rebuild the ride-volume heatmap (docs/data/heatmap.json + .js, docs/assets/heatmap.png)
from a month of Citi Bike trip data. Loading and cleaning come from
scripts/pipeline.py, shared with the notebook and the other build scripts.

Usage: python scripts/build_heatmap.py [YYYY | YYYYMM]   # defaults to 2024
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from pipeline import DEFAULT_PERIOD, cleaning_stats, load_clean_period, period_label

REPO = Path(__file__).resolve().parent.parent
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_heat_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day"] = df["started_at"].dt.day_name()
    df["hour"] = df["started_at"].dt.hour
    return df.groupby(["day", "hour"]).size().unstack(fill_value=0).reindex(DAY_ORDER)


def rider_mix(df: pd.DataFrame) -> dict:
    """Who rides: member/casual and bike-type shares, trip lengths, weekend mix."""
    casual = df["member_casual"] == "casual"
    weekend = df["started_at"].dt.dayofweek >= 5
    minutes = df.groupby("member_casual", observed=True)["trip_minutes"]
    return {
        "casual_share": round(float(casual.mean()), 4),
        "casual_share_weekend": round(float(casual[weekend].mean()), 4),
        "casual_share_weekday": round(float(casual[~weekend].mean()), 4),
        "electric_share": round(float((df["rideable_type"] == "electric_bike").mean()), 4),
        "median_minutes": {k: round(float(v), 1) for k, v in minutes.median().items()},
        "mean_minutes": {k: round(float(v), 1) for k, v in minutes.mean().items()},
    }


def write_site_data(heat: pd.DataFrame, total_rides: int, label: str, extra: dict) -> None:
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
        **extra,
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
    period = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERIOD
    label = period_label(period)
    df = load_clean_period(period, columns=["started_at", "member_casual", "rideable_type", "trip_minutes"])

    heat = build_heat_matrix(df)
    write_site_data(heat, total_rides=len(df), label=label,
                    extra={"riders": rider_mix(df), "cleaning": cleaning_stats(period)})
    write_png(heat, label=label)


if __name__ == "__main__":
    main()
