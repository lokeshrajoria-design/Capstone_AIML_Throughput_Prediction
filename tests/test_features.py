"""Timing and leakage rules checked on a hand-built trace.  Run: python -m pytest tests"""
import numpy as np
import pandas as pd
import pytest

from tput import config as C
from tput.features import build_features
from tput.splits import make_splits
from tput.checks import check_splits


def trace(seconds, values, tid="A/Driving/x.csv"):
    t0 = pd.Timestamp("2020-01-01 00:00:00")
    n = len(seconds)
    return pd.DataFrame({
        "trace_id": tid, "t": [t0 + pd.Timedelta(seconds=s) for s in seconds],
        "throughput_mbps": values, "application_type": "Netflix", "mobility_condition": "Driving",
        "recording_date": "2020-01-01", "NetworkMode": "5G", "Speed": 0.0,
        "RSRP": -90.0, "RSRQ": -10.0, "SNR": 5.0, "CQI": 10.0}).iloc[:n]


def row(f, s):
    return f[f.t == pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=s)].iloc[0]


def test_target_is_exactly_five_seconds_ahead():
    s = list(range(30))
    f = build_features(trace(s, [float(x) for x in s]))
    r = row(f, 12)
    assert r[C.TARGET] == 17 and r.throughput_lag_0 == 12 and r.throughput_lag_5 == 7


def test_gap_gives_missing_target_not_a_neighbour():
    s = [x for x in range(30) if x != 17]            # second 17 missing from the log
    f = build_features(trace(s, [float(x) for x in s]))
    assert np.isnan(row(f, 12)[C.TARGET]) and not row(f, 12).eligible


def test_window_uses_only_past_and_origin():
    s = list(range(30))
    v = [0.0] * 20 + [1000.0] * 10                   # a jump that starts after the origin at t=19
    f = build_features(trace(s, v))
    r = row(f, 19)
    assert r.throughput_mean_10s == 0 and r.throughput_sd_10s == 0 and r[C.TARGET] == 1000


def test_window_minimum_observations():
    s = [x for x in range(30) if x not in (8, 9, 10, 11)]     # window 3..12 has only 6 of 10 seconds
    f = build_features(trace(s, [1.0] * len(s)))
    assert row(f, 12).window_n_obs == 6 and not row(f, 12).eligible
    assert row(f, 18).window_n_obs == 7 and row(f, 18).eligible == (C.WINDOW_MIN_OBS <= 7)
    assert row(f, 20).window_n_obs == 9 and row(f, 20).eligible == (C.WINDOW_MIN_OBS <= 9)


def test_features_never_cross_traces():
    a = trace(range(10), [1.0] * 10, "A/Driving/a.csv")
    b = trace(range(10), [9.0] * 10, "A/Driving/b.csv")       # same clock times, different trace
    f = build_features(pd.concat([a, b]))
    assert set(f[f.trace_id == "A/Driving/a.csv"][C.TARGET].dropna()) == {1.0}


def test_splits_keep_traces_and_dates_whole():
    rng = np.random.default_rng(0)
    m = pd.DataFrame({"trace_id": [f"t{i}" for i in range(40)],
                      "application_type": rng.choice(["Netflix", "Download"], 40),
                      "mobility_condition": rng.choice(["Static", "Driving"], 40),
                      "recording_date": rng.choice([f"2020-01-{d:02d}" for d in range(1, 12)], 40)})
    s = make_splits(m, m.trace_id)
    check_splits(s)
    assert len(s) == 40


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
