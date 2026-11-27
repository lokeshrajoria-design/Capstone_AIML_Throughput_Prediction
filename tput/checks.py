"""Integrity checks from synopsis 6.3. Each raises AssertionError on failure."""
import numpy as np
import pandas as pd

from . import config as C


def check_no_leakage(clean, feats, n=2000, seed=C.SEED):
    """Recompute a random sample of rows straight from the cleaned data and compare."""
    rng = np.random.default_rng(seed)
    y = clean.set_index(["trace_id", "t"]).throughput_mbps
    sample = feats.iloc[rng.choice(len(feats), size=min(n, len(feats)), replace=False)]
    for r in sample.itertuples():
        def at(dt):
            return y.get((r.trace_id, r.t + pd.Timedelta(seconds=dt)), np.nan)
        assert np.allclose(getattr(r, C.TARGET), at(C.HORIZON_S), equal_nan=True), "target mismatch"
        for k in C.LAGS_S:
            assert np.allclose(getattr(r, f"throughput_lag_{k}"), at(-k), equal_nan=True), f"lag_{k} mismatch"


def check_splits(splits):
    assert splits.trace_id.is_unique, "a trace appears in more than one outer fold"
    assert set(splits.outer_fold) == set(range(C.OUTER_FOLDS)), "missing outer fold"
    for k in range(C.OUTER_FOLDS):
        test = set(splits.trace_id[splits.outer_fold == k])
        train = set(splits.trace_id[splits.outer_fold != k])
        assert not test & train, f"fold {k}: evaluation trace also in training"
    for col in ("date_fold",):
        per_date = splits.groupby("recording_date")[col].nunique()
        assert (per_date == 1).all(), f"{col}: a recording date is split across folds"
