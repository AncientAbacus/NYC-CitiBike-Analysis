"""Consistency checks on the committed site data: the numbers every section
shows must agree with each other, and every number in the prose must match
the data (via scripts/sync_facts.py)."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "data"


def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def test_ride_totals_agree_across_sections():
    total = load("summary")["total_rides"]
    assert load("heatmap")["total_rides"] == total
    assert load("seasonality")["total_rides"] == total
    assert sum(s["total"] for s in load("stations")["stations"]) == total
    assert load("heatmap")["cleaning"]["clean_rows"] == total


def test_js_wrappers_match_json():
    for js in DATA.glob("*.js"):
        payload = js.read_text().split("=", 1)[1].strip().rstrip(";")
        assert json.loads(payload) == json.loads(js.with_suffix(".json").read_text() if js.stem != "bike_paths"
                                                 else (DATA / "bike_paths.geojson").read_text()), js.name


def test_forecast_is_scored_against_its_strongest_baseline():
    f = load("forecast")
    hourly = f["departures"]["hourly"]
    best_baseline = min(v["mae"] for k, v in hourly.items() if k != "model")
    assert hourly["seven_day_mean"]["mae"] == best_baseline
    assert abs(f["improvement"]["vs_seven_day_mean_hourly"] - (1 - hourly["model"]["mae"] / best_baseline)) < 1e-3


def test_bike_lanes_stay_small():
    assert (DATA / "bike_paths.js").stat().st_size < 1_000_000


def test_prose_numbers_match_data():
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "sync_facts.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
