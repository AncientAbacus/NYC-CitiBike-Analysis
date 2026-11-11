"""Leak tests for the forecast's inputs: nothing may use data from the day being forecast."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_forecast  # noqa: E402

N_STATIONS, N_DAYS = 6, 60


def counts(seed=0):
    return np.random.default_rng(seed).poisson(5, size=(N_STATIONS, N_DAYS * 24)).astype(np.float32)


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


def test_history_inputs_never_see_the_day_they_forecast():
    y = counts()
    base = build_forecast.history_features(y)
    for day in (30, 45, N_DAYS - 1):
        start = day * 24
        changed = y.copy()
        changed[:, start:] += 100  # rewrite everything from this day on
        new = build_forecast.history_features(changed)
        for name in base:
            # inputs for this day (and every earlier one) must not move
            assert same(base[name][:, : start + 24], new[name][:, : start + 24]), f"{name} leaks day {day}"


def test_history_inputs_do_use_the_previous_day():
    # guards against the leak test passing only because an input is constant
    y = counts()
    base = build_forecast.history_features(y)
    changed = y.copy()
    changed[:, 30 * 24: 31 * 24] += 100  # rewrite day 30 only
    new = build_forecast.history_features(changed)
    day31 = slice(31 * 24, 32 * 24)
    for name in ("lag_24", "same_hour_7d_mean", "day_total_yesterday", "day_total_7d_mean", "system_ratio_yesterday"):
        assert not same(base[name][:, day31], new[name][:, day31]), f"{name} ignores yesterday"


def test_every_history_input_is_covered():
    history = set(build_forecast.history_features(counts()))
    not_history = {"hour", "dow", "is_holiday", "station_profile", *build_forecast.WEATHER}
    assert history | not_history == set(build_forecast.FEATURES)


def grid_for_profile(y):
    hours = pd.date_range("2024-01-01", periods=y.shape[1], freq="h")
    s_idx, h_idx = np.nonzero(np.ones_like(y, dtype=bool))
    ts = hours[h_idx]
    return pd.DataFrame({
        "station": pd.Categorical.from_codes(s_idx, categories=[f"s{i}" for i in range(y.shape[0])]),
        "hour_start": ts, "y": y.ravel(), "hour": ts.hour, "dow": ts.dayofweek,
        "same_hour_7d_mean": np.zeros(y.size, dtype=np.float32),
    })


def test_station_profile_uses_training_period_only():
    y = counts()
    train_end = "2024-02-10"  # day 40
    changed = y.copy()
    changed[:, 40 * 24:] += 100  # rewrite everything after the training period
    base = build_forecast.station_profile(grid_for_profile(y), train_end)
    new = build_forecast.station_profile(grid_for_profile(changed), train_end)
    assert same(base, new)
