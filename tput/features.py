"""Step 3: exact 5 s-ahead target and history features, using only data at or before the origin.

Every lookup is by exact timestamp inside the same trace, so a gap in the log gives a missing value
instead of silently pairing readings that are further apart than intended. Which missing values make an
origin ineligible is set by WINDOW_MIN_OBS / REQUIRE_ALL_LAGS in config.
"""
import warnings

import numpy as np
import pandas as pd

from . import config as C


def _at(series, times):
    """Value of a time-indexed series at each requested timestamp (NaN where that second is absent)."""
    return series.reindex(times).to_numpy()


def build_features(clean):
    out = []
    for _, d in clean.groupby("trace_id", sort=False):
        d = d.sort_values("t").reset_index(drop=True)
        y = d.set_index("t").throughput_mbps
        t = d.t
        f = d[["trace_id", "t", "application_type", "mobility_condition", "recording_date",
               "NetworkMode", "Speed", *C.RADIO]].copy()

        for k in C.LAGS_S:
            f[f"throughput_lag_{k}"] = _at(y, t - pd.Timedelta(seconds=k))
        win = np.column_stack([_at(y, t - pd.Timedelta(seconds=k)) for k in range(C.WINDOW_S)])
        n_obs = (~np.isnan(win)).sum(axis=1)
        enough = n_obs >= C.WINDOW_MIN_OBS
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN rows are masked below
            f["throughput_mean_10s"] = np.where(enough, np.nanmean(win, axis=1), np.nan)
            f["throughput_sd_10s"] = np.where(enough, np.nanstd(win, axis=1), np.nan)  # divisor = seconds observed
        f["window_n_obs"] = n_obs
        f[C.TARGET] = _at(y, t + pd.Timedelta(seconds=C.HORIZON_S))
        out.append(f)
    feats = pd.concat(out, ignore_index=True)

    required = [C.TARGET, "throughput_lag_0", "throughput_mean_10s", "throughput_sd_10s"]
    if C.REQUIRE_ALL_LAGS:
        required += [f"throughput_lag_{k}" for k in C.LAGS_S]
    feats["eligible"] = feats[required].notna().all(axis=1)
    for c in C.RADIO + ["Speed"] + [f"throughput_lag_{k}" for k in C.LAGS_S if k]:
        feats[f"{c}_missing"] = feats[c].isna().astype(int)
    return feats


def coverage(feats):
    """How many origins survive the exact-timing rules, by group (synopsis step 3 check)."""
    g = feats.groupby(["application_type", "mobility_condition"])
    return pd.DataFrame({
        "traces": g.trace_id.nunique(),
        "origins": g.size(),
        "with_target": g[C.TARGET].apply(lambda s: s.notna().sum()),
        "eligible": g.eligible.sum(),
    }).assign(eligible_pct=lambda x: (100 * x.eligible / x.origins).round(1)).reset_index()
