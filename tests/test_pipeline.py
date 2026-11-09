"""Unit tests for the pure logic in scripts/: no network, no cached data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_forecast  # noqa: E402
import build_geomap  # noqa: E402
import build_rebalancing  # noqa: E402
import pipeline  # noqa: E402


def trip(ride_id, start, minutes, **kw):
    row = {
        "ride_id": ride_id,
        "started_at": start,
        "ended_at": str(pd.Timestamp(start) + pd.Timedelta(minutes=minutes)),
        "member_casual": "member", "rideable_type": "classic_bike",
        "start_station_name": "A", "end_station_name": "B",
        "start_lat": 40.7, "start_lng": -74.0, "end_lat": 40.71, "end_lng": -74.01,
    }
    row.update(kw)
    return row


def test_clean_applies_every_rule_and_records_stats():
    raw = pd.DataFrame([
        trip("ok", "2024-01-10 08:00", 12),
        trip("ok", "2024-01-10 08:00", 12),                    # duplicate ride_id
        trip("dec", "2023-12-31 23:50", 15),                   # spills in from December
        trip("nostation", "2024-01-11 09:00", 5, end_station_name=None),
        trip("negative", "2024-01-12 10:00", -3),
        trip("lost", "2024-01-13 10:00", 25 * 60),             # > 24h: lost/stolen
        trip("ok2", "2024-01-31 23:59", 30),                   # starts in January: kept
    ])
    out = pipeline.clean(raw, "202401", verbose=False)
    assert list(out["ride_id"]) == ["ok", "ok2"]
    stats = out.attrs["cleaning"]
    assert stats["raw_rows"] == 7 and stats["clean_rows"] == 2
    assert stats["removed"] == {
        "duplicate ride_id": 1, "outside the month": 1, "missing station/coordinates": 1,
        "non-positive duration": 1, "over 24 hours": 1,
    }


def test_period_months_and_labels():
    assert pipeline.period_months("2024") == [f"2024{m:02d}" for m in range(1, 13)]
    assert pipeline.period_months("202403") == ["202403"]
    assert pipeline.period_label("202403") == "March 2024"
    with pytest.raises(ValueError):
        pipeline.period_months("24")


def test_haversine_known_distance():
    # Times Square to Grand Central is ~0.95 km
    d = pipeline.haversine_km(40.7580, -73.9855, 40.7527, -73.9772)
    assert 0.85 < float(d) < 1.05


def test_noaa_number_parsing():
    s = pd.Series(["1.5", "2.0s", "T", "", None, "abc"])
    out = pipeline._noaa_number(s)
    assert out.iloc[0] == 1.5 and out.iloc[1] == 2.0 and out.iloc[2] == 0.01
    assert out.iloc[3:].isna().all()


def test_rebalancing_window_counts_align_start_and_end_categories():
    ts = pd.Series(pd.to_datetime(["2024-03-04 07:30", "2024-03-04 08:10", "2024-03-04 12:00", "2024-03-09 08:00"]))
    st = pd.Series(["A", "A", "A", "A"], dtype="category")
    am = build_rebalancing.window_counts(ts, st, (7, 10))
    # Monday 07:30 and 08:10 count; noon is outside the window; Saturday is excluded
    assert am.to_dict() == {"A": 2}
    assert am.index.name == "station"


def test_conformal_band_covers_its_calibration_quantiles():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 20, 20_000)
    y = rng.poisson(pred).astype(float)
    lo, hi = build_forecast.conformal_band(pred, y, pred)
    coverage = ((y >= lo) & (y <= hi)).mean()
    assert 0.78 <= coverage <= 0.92  # nominal 80%, counts are discrete so it can run a little high
    assert (lo <= hi).all()


def test_compact_bike_routes_merges_and_keeps_needed_properties():
    feats = [
        {"type": "Feature", "properties": {"street": "1 AV", "facilitycl": "I", "boro": "1", "onoffst": "ON"},
         "geometry": {"type": "MultiLineString", "coordinates": [[[-73.98, 40.73], [-73.981, 40.731]]]}},
        {"type": "Feature", "properties": {"street": "1 AV", "facilitycl": "I", "boro": "1", "onoffst": "ON"},
         "geometry": {"type": "MultiLineString", "coordinates": [[[-73.981, 40.731], [-73.982, 40.732]]]}},
        {"type": "Feature", "properties": {"street": "2 AV", "facilitycl": "II", "boro": "1", "onoffst": "ON"},
         "geometry": {"type": "LineString", "coordinates": [[-73.97, 40.73], [-73.971, 40.731]]}},
    ]
    out = build_geomap.compact_bike_routes(feats)
    assert len(out["features"]) == 2
    first = next(f for f in out["features"] if f["properties"]["street"] == "1 AV")
    assert first["properties"] == {"street": "1 AV", "facilitycl": "I", "boro_name": "Manhattan"}
    assert len(first["geometry"]["coordinates"]) == 1  # the two touching pieces were joined


def test_target_encode_oof_never_uses_a_rows_own_label():
    # one station, two folds; each training row's encoding must come only from the other fold
    df = pd.DataFrame({"station": ["A"] * 4 + ["A"], "y": [1, 1, 0, 0, 1]})
    train = pd.Series([True, True, True, True, False])
    folds = pd.Series([0, 0, 1, 1, -1])
    enc = pipeline.target_encode_oof(df, ["station"], "y", train, folds, smoothing=0.0)
    assert list(enc[:4]) == [0.0, 0.0, 1.0, 1.0]   # fold 0 sees only fold 1's zeros, and vice versa
    assert enc[4] == 0.5                           # held-out row sees all training rows
