"""Leak tests for the forecast inputs: nothing may use data from the day being forecast."""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_forecast as bf  # noqa: E402

N_STATIONS, N_DAYS = 6, 60


def counts(seed=0):
    return np.random.default_rng(seed).poisson(5, size=(N_STATIONS, N_DAYS * 24)).astype(np.float32)


def neighbours(seed=1):
    rng = np.random.default_rng(seed)
    dense = (rng.random((N_STATIONS, N_STATIONS)) < 0.5).astype(float)
    np.fill_diagonal(dense, 0)
    dense /= np.maximum(dense.sum(axis=1, keepdims=True), 1)
    return csr_matrix(dense)


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


def test_history_inputs_never_see_the_day_they_forecast():
    y, nbr = counts(), neighbours()
    base = bf.history_features(y, nbr)
    for day in (30, 45, N_DAYS - 1):
        start = day * 24
        changed = y.copy()
        changed[:, start:] += 100  # rewrite everything from this day on
        new = bf.history_features(changed, nbr)
        for name in base:
            assert same(base[name][:, : start + 24], new[name][:, : start + 24]), f"{name} leaks day {day}"


def test_history_inputs_do_use_the_previous_day():
    # guards against the leak test passing only because an input never changes
    y, nbr = counts(), neighbours()
    base = bf.history_features(y, nbr)
    changed = y.copy()
    changed[:, 30 * 24: 31 * 24] += 100  # rewrite day 30 only
    new = bf.history_features(changed, nbr)
    day31 = slice(31 * 24, 32 * 24)
    for name in ("lag_24", "same_hour_7d_mean", "day_total_yesterday", "day_total_7d_mean",
                 "system_ratio_yesterday", "neighbor_lag_24"):
        assert not same(base[name][:, day31], new[name][:, day31]), f"{name} ignores yesterday"


def test_every_history_input_is_covered():
    history = set(bf.history_features(counts(), neighbours()))
    assert history == set(bf.HISTORY)
    assert set(bf.HISTORY) | set(bf.CALENDAR) | set(bf.WEATHER) | {"station_profile"} == set(bf.ALL)


def test_station_profile_uses_training_period_only():
    rng = np.random.default_rng(3)
    n_hours = N_DAYS * 24
    s = np.repeat(np.arange(N_STATIONS), n_hours).astype(np.int32)
    t = np.tile(np.arange(n_hours), N_STATIONS).astype(np.int32)
    y = rng.poisson(4, size=s.size).astype(np.float32)
    hourly = {"hour": (t % 24).astype(np.float32), "dow": ((t // 24) % 7).astype(np.float32)}
    hist = {"same_hour_7d_mean": np.zeros(s.size, np.float32)}
    cut = 40 * 24

    def profile(yy):
        p = bf.Panel(s=s, t=t, y=yy, hist=hist)
        return bf.station_profile(p, N_STATIONS, hourly, cut)

    changed = y.copy()
    changed[t >= cut] += 100  # rewrite everything after the training cut
    assert same(profile(y), profile(changed))


def test_calibrated_band_hits_target_on_its_own_calibration_data():
    rng = np.random.default_rng(11)
    pred = rng.uniform(0, 20, 40_000)
    y = rng.poisson(pred).astype(float)
    lo, hi, levels = bf.calibrated_band(pred, y, pred, target=0.8)
    for name, in_segment in bf.SEGMENTS:
        m = in_segment(pred)
        coverage = ((y[m] >= lo[m]) & (y[m] <= hi[m])).mean()
        assert abs(coverage - 0.8) < 0.03, (name, coverage)  # chosen on these same rows
    assert (lo <= hi).all() and all(0.5 <= v <= 0.95 for v in levels.values())
