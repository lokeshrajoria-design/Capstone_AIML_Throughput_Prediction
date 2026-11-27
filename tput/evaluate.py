"""Trace-level error metrics and the persistence baseline (synopsis 5.2 and 5.5)."""
import numpy as np
import pandas as pd

from . import config as C


def persistence(feats):
    """Predicted throughput at t + 5 s = observed throughput at t."""
    return feats["throughput_lag_0"].to_numpy()


def trace_errors(feats, pred, name):
    e = feats[["trace_id", "application_type", "mobility_condition", C.TARGET]].assign(pred=pred)
    e["abs_err"] = (e.pred - e[C.TARGET]).abs()
    e["sq_err"] = (e.pred - e[C.TARGET]) ** 2
    g = e.groupby(["trace_id", "application_type", "mobility_condition"])
    return pd.DataFrame({"n": g.size(), "mae": g.abs_err.mean(), "rmse": np.sqrt(g.sq_err.mean())}) \
        .reset_index().assign(model=name)


def summarise(per_trace, by=("mobility_condition",)):
    """Trace-weighted means (each trace counts once), the primary aggregate in the synopsis."""
    rows = [per_trace.assign(group="all")] + [per_trace.assign(group=per_trace[b]) for b in by]
    s = pd.concat(rows).groupby(["model", "group"]).agg(
        traces=("trace_id", "nunique"), origins=("n", "sum"), mean_trace_mae=("mae", "mean"),
        mean_trace_rmse=("rmse", "mean"))
    return s.round(3).reset_index()
