"""
Train a member-vs-casual rider classifier and write the results behind the
site's "Predict" section (docs/data/model.json + .js). Mirrors the cleaning
steps in CitiBike_Analysis.ipynb / scripts/build_heatmap.py / build_geomap.py
so the notebook, site charts, and this model all agree on the same month of
trips.

The features are deliberately the same signals the rest of the site already
tells a story about: duration/distance ("who's riding" -- casual trips run
longer), hour/weekend ("when they ride" -- commute vs. leisure), and start
station popularity ("where they ride" -- Manhattan concentration). The model
asks whether those three threads, taken together, can predict which type of
rider took a given trip.

Usage: python scripts/build_model.py [YYYYMM]
       python scripts/build_model.py 202401   # defaults to this if omitted
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent

FEATURES = [
    "hour",
    "is_weekend",
    "trip_minutes",
    "distance_km",
    "speed_kmh",
    "is_electric",
    "start_station_rides_log",
]
FEATURE_LABELS = {
    "hour": "Hour of day",
    "is_weekend": "Weekend",
    "trip_minutes": "Trip duration (min)",
    "distance_km": "Distance (km)",
    "speed_kmh": "Speed (km/h)",
    "is_electric": "Electric bike",
    "start_station_rides_log": "Start station popularity",
}


def load_month(yyyymm: str) -> pd.DataFrame:
    url = f"https://s3.amazonaws.com/tripdata/{yyyymm}-citibike-tripdata.zip"
    print(f"Downloading {yyyymm} Citi Bike trip data...", flush=True)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    cols = [
        "ride_id", "started_at", "ended_at", "member_casual", "rideable_type",
        "start_station_name", "start_lat", "start_lng", "end_lat", "end_lng",
    ]
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n]
        print("Reading CSVs:", names, flush=True)
        dfs = [pd.read_csv(z.open(n), usecols=cols) for n in names]
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
    df = df.dropna(subset=["start_station_name", "start_lat", "start_lng", "end_lat", "end_lng"])
    df = df[df["trip_minutes"] > 0]
    df = df[df["trip_minutes"] <= 24 * 60]
    after = len(df)
    print(f"Removed {before - after:,} rows ({(before - after) / before:.2%}); remaining {after:,}", flush=True)
    return df


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    r = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["started_at"].dt.hour
    df["is_weekend"] = (df["started_at"].dt.dayofweek >= 5).astype(int)
    df["distance_km"] = haversine_km(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"])
    df["speed_kmh"] = df["distance_km"] / (df["trip_minutes"] / 60)
    df["is_electric"] = (df["rideable_type"] == "electric_bike").astype(int)

    # Station popularity computed independently of member_casual -- an
    # aggregate of *all* rides through a station, so it can't leak the label.
    station_rides = df.groupby("start_station_name")["ride_id"].transform("count")
    df["start_station_rides_log"] = np.log1p(station_rides)

    df["is_member"] = (df["member_casual"] == "member").astype(int)
    return df


def train_and_evaluate(df: pd.DataFrame) -> dict:
    X = df[FEATURES]
    y = df["is_member"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # With an ~89/11 class split, always guessing "member" already scores
    # ~89% accuracy while identifying zero casual riders -- report that trap
    # explicitly rather than letting a headline accuracy number imply skill
    # it doesn't have.
    majority_label = int(y_train.mode()[0])
    majority_preds = np.full(len(y_test), majority_label)
    majority_accuracy = float((majority_preds == y_test).mean())

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    logistic.fit(X_train, y_train)
    logistic_proba = logistic.predict_proba(X_test)[:, 1]
    logistic_preds = logistic.predict(X_test)

    forest = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42
    )
    forest.fit(X_train, y_train)
    forest_proba = forest.predict_proba(X_test)[:, 1]
    forest_preds = forest.predict(X_test)

    def model_summary(preds, proba) -> dict:
        accuracy = float((preds == y_test).mean())
        auc = float(roc_auc_score(y_test, proba))
        # casual = label 0, the minority class the majority baseline misses entirely
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, labels=[0], zero_division=0
        )
        fpr, tpr, _ = roc_curve(y_test, proba)
        idx = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)
        return {
            "accuracy": accuracy,
            "roc_auc": auc,
            "casual_precision": float(precision[0]),
            "casual_recall": float(recall[0]),
            "casual_f1": float(f1[0]),
            "roc_curve": [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx],
        }

    cm = confusion_matrix(y_test, forest_preds, labels=[0, 1]).tolist()

    importances = sorted(
        (
            {"feature": f, "label": FEATURE_LABELS[f], "importance": float(imp)}
            for f, imp in zip(FEATURES, forest.feature_importances_)
        ),
        key=lambda d: -d["importance"],
    )

    return {
        "sample_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "majority_baseline_accuracy": majority_accuracy,
        "logistic_regression": model_summary(logistic_preds, logistic_proba),
        "random_forest": model_summary(forest_preds, forest_proba),
        "confusion_matrix": {
            "labels": ["casual", "member"],
            "matrix": cm,
        },
        "feature_importances": importances,
    }


def write_model_data(results: dict, label: str) -> None:
    out = {
        "title": "Predicting Member vs. Casual Riders",
        "subtitle": f"{label}, random forest vs. logistic regression vs. naive baseline",
        **results,
    }
    data_dir = REPO / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "model.json").write_text(json.dumps(out))
    (data_dir / "model.js").write_text("window.MODEL_DATA = " + json.dumps(out) + ";\n")
    print("Wrote docs/data/model.json and model.js", flush=True)
    summary = {
        "sample_size": results["sample_size"],
        "majority_baseline_accuracy": results["majority_baseline_accuracy"],
        "logistic_regression": {k: v for k, v in results["logistic_regression"].items() if k != "roc_curve"},
        "random_forest": {k: v for k, v in results["random_forest"].items() if k != "roc_curve"},
        "confusion_matrix": results["confusion_matrix"],
        "top_features": results["feature_importances"][:3],
    }
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    yyyymm = sys.argv[1] if len(sys.argv) > 1 else "202401"
    label = pd.Period(yyyymm, freq="M").strftime("%B %Y")

    df = load_month(yyyymm)
    print(f"Loaded {len(df):,} rows", flush=True)
    df = clean(df, yyyymm)
    df = engineer_features(df)

    results = train_and_evaluate(df)
    write_model_data(results, label=label)


if __name__ == "__main__":
    main()
