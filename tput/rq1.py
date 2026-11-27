"""RQ1: within-trace Spearman correlation between each radio measurement and contemporaneous throughput.

Eligibility per trace and measurement: >= 30 complete pairs and variation in both variables; no imputation.
Inference treats traces as the units (one-sample t-test on trace correlations, Holm across the four
measurements). The date sensitivity averages traces within each recording date first.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .stats import holm


def trace_correlations(clean):
    rows = []
    for tid, d in clean.groupby("trace_id"):
        meta = d.iloc[0]
        for m in C.RADIO:
            pair = d[[m, "throughput_mbps"]].dropna()
            ok = len(pair) >= C.MIN_PAIRS_RQ1 and pair[m].nunique() > 1 and pair.throughput_mbps.nunique() > 1
            r = stats.spearmanr(pair[m], pair.throughput_mbps).statistic if ok else np.nan
            rows.append({"trace_id": tid, "measurement": m, "n_pairs": len(pair), "eligible": ok, "rho": r,
                         "application_type": meta.application_type, "mobility_condition": meta.mobility_condition,
                         "recording_date": meta.recording_date})
    return pd.DataFrame(rows)


def _ttest(x):
    x = np.asarray(x, float)
    t = stats.ttest_1samp(x, 0.0)
    ci = t.confidence_interval(0.95)
    return x.mean(), ci.low, ci.high, t.pvalue, len(x)


def summarise(corr):
    out = []
    for m in C.RADIO:
        c = corr[(corr.measurement == m) & corr.eligible]
        mean, lo, hi, p, n = _ttest(c.rho)
        by_date = c.groupby("recording_date").rho.mean()
        _, dlo, dhi, dp, dn = _ttest(by_date)
        out.append({"measurement": m, "eligible_traces": n,
                    "excluded_traces": int((corr.measurement == m).sum() - n),
                    "mean_rho": mean, "ci95_low": lo, "ci95_high": hi, "p": p,
                    "share_positive": (c.rho > 0).mean(),
                    "dates": dn, "date_mean_rho": by_date.mean(), "date_ci95_low": dlo, "date_ci95_high": dhi,
                    "date_p": dp,
                    "mean_rho_static": c[c.mobility_condition == "Static"].rho.mean(),
                    "mean_rho_driving": c[c.mobility_condition == "Driving"].rho.mean()})
    s = pd.DataFrame(out)
    s["p_holm"] = holm(s.p)
    s["date_p_holm"] = holm(s.date_p)
    return s
