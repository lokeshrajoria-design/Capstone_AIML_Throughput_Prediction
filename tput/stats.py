"""Uncertainty helpers. Traces (or recording dates) are the resampling units, never rows."""
import numpy as np
import pandas as pd

from . import config as C


def holm(pvals):
    """Holm step-down adjusted p-values, in the original order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_mean(values, clusters=None, n_boot=5000, level=0.95, seed=C.SEED):
    """Mean with a percentile bootstrap CI. If clusters are given (e.g. recording dates),
    whole clusters are resampled so traces sharing a date move together."""
    v = pd.Series(np.asarray(values, float))
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        boots = v.to_numpy()[idx].mean(axis=1)
    else:
        groups = [g.to_numpy() for _, g in v.groupby(np.asarray(clusters))]
        k = len(groups)
        boots = np.array([np.concatenate([groups[j] for j in rng.integers(0, k, k)]).mean() for _ in range(n_boot)])
    a = (1 - level) / 2
    return {"mean": v.mean(), "ci_low": np.quantile(boots, a), "ci_high": np.quantile(boots, 1 - a),
            "share_boot_le_0": float((boots <= 0).mean()), "n": len(v)}


def bootstrap_diff(a, b, n_boot=5000, level=0.95, seed=C.SEED):
    """Difference in means (a - b) between two independent groups of traces."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = (a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
             - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1))
    q = (1 - level) / 2
    return {"diff": a.mean() - b.mean(), "ci_low": np.quantile(boots, q), "ci_high": np.quantile(boots, 1 - q),
            "n_a": len(a), "n_b": len(b)}
