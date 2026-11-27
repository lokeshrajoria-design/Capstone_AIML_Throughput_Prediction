"""PoC: predict 5G downlink throughput 5 seconds ahead and compare with a persistence baseline.

Maps to the synopsis: RQ2 (AI/ML vs persistence) and RQ3 (value of context), evaluated per trace.
It is an offline proof of concept with fixed model settings, not the final analysis.
AI-assisted: written with Claude on 2026-10-18 (see the AI-use statement in the report).

Usage:  python run_poc.py [path/to/5G-production-dataset.zip]
Needs:  pandas, numpy, scipy, scikit-learn
"""
import json, os, sys, zipfile
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ZIP = os.path.join(HERE, "..", "data", "raw", "5G-production-dataset.zip")
H = 5                                   # prediction horizon in seconds
LAGS = [0, 1, 2, 3, 4]                  # recent throughput history (seconds back)
NUM = ["Speed", "RSRP", "RSRQ", "SNR", "CQI", "DL_bitrate"]
MODES = ["5G", "LTE", "HSPA+", "UMTS", "HSUPA", "HSDPA", "GPRS", "EDGE"]
APPS = ["Amazon_Prime", "Netflix", "Download"]
MOBS = ["Static", "Driving"]
SEED = 0


def load_traces(zpath):
    out = []
    with zipfile.ZipFile(zpath) as z:
        for name in sorted(z.namelist()):
            if name.startswith("__MACOSX") or not name.lower().endswith(".csv"):
                continue
            parts = name.split("/")
            app = next((p for p in parts if p in APPS), None)
            mob = next((p for p in parts if p in MOBS), None)
            if app is None or mob is None:
                continue
            with z.open(name) as f:
                out.append((name, app, mob, pd.read_csv(f)))
    return out


def to_1hz(df):
    d = df.copy()
    d["t"] = pd.to_datetime(d["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    d = d.dropna(subset=["t"])
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")           # "-" placeholders become missing
    g = d.groupby("t").agg({**{c: "mean" for c in NUM}, "NetworkMode": "last"})
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="1s"))   # no interpolation
    g["NetworkMode"] = g["NetworkMode"].ffill()
    return g


def build(name, app, mob, g):
    X = pd.DataFrame(index=g.index)
    for c in ["RSRP", "RSRQ", "SNR", "CQI"]:
        X[c] = g[c]
    for k in LAGS:
        X[f"dl_lag{k}"] = g["DL_bitrate"].shift(k)
    X["dl_mean5"] = g["DL_bitrate"].rolling(5, min_periods=3).mean()
    base_cols = list(X.columns)
    X["Speed"] = g["Speed"]
    for m in MODES:
        X[f"mode_{m}"] = (g["NetworkMode"] == m).astype(float)
    for a in APPS:
        X[f"app_{a}"] = float(app == a)
    y = g["DL_bitrate"].shift(-H)
    persist = g["DL_bitrate"]
    keep = y.notna() & persist.notna()
    X = X[keep]
    out = X.copy()
    out["y"] = y[keep]; out["persist"] = persist[keep]
    out["trace"] = name; out["app"] = app; out["mob"] = mob
    out["sec"] = (X.index - g.index.min()).total_seconds().astype(int) + H   # time of the predicted value
    return out.reset_index(drop=True), base_cols


def model():
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.08,
                                         max_leaf_nodes=31, min_samples_leaf=50, random_state=SEED)


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return float(m), float(m - t * se), float(m + t * se)


def summarize(v):
    v = np.asarray(v, float)
    m, lo, hi = ci95(v)
    p_t = float(stats.ttest_1samp(v, 0, alternative="greater").pvalue)
    try:
        p_w = float(stats.wilcoxon(v, alternative="greater").pvalue)
    except ValueError:
        p_w = None
    return {"n": int(len(v)), "mean": m, "ci95": [lo, hi], "median": float(np.median(v)),
            "share_positive": float((v > 0).mean()), "p_ttest_one_sided": p_t, "p_wilcoxon_one_sided": p_w}


def main():
    zpath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZIP
    raw = load_traces(zpath)
    frames, base_cols = [], None
    audit = {"source_files": len(raw), "raw_rows": int(sum(len(r[3]) for r in raw))}
    for name, app, mob, df in raw:
        d, base_cols = build(name, app, mob, to_1hz(df))
        if len(d) >= 30:
            frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    ctx_cols = [c for c in data.columns if c not in ("y", "persist", "trace", "app", "mob", "sec")]
    audit.update({"traces_used": int(data["trace"].nunique()), "usable_rows": int(len(data)),
                  "zero_throughput_share_by_app": {a: float((data.loc[data.app == a, "persist"] == 0).mean()) for a in APPS},
                  "median_throughput_kbps_by_group": {f"{a}|{m}": float(data[(data.app == a) & (data.mob == m)]["y"].median())
                                                      for a in APPS for m in MOBS if len(data[(data.app == a) & (data.mob == m)])}})

    traces = data["trace"].values
    hist_cols = [c for c in base_cols if c.startswith("dl_")]          # diagnostic: throughput history only
    oof_base = np.zeros(len(data)); oof_ctx = np.zeros(len(data)); oof_hist = np.zeros(len(data))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(data, groups=traces):          # whole traces are held out together
        for cols, store in ((base_cols, oof_base), (ctx_cols, oof_ctx), (hist_cols, oof_hist)):
            m = model().fit(data.iloc[tr][cols], data.iloc[tr]["y"])
            store[te] = np.clip(m.predict(data.iloc[te][cols]), 0, None)
    data["p_base"] = oof_base; data["p_ctx"] = oof_ctx; data["p_hist"] = oof_hist
    data["roll"] = data["dl_mean5"]                                      # diagnostic: 5 s rolling-mean baseline

    rows = []
    for tname, g in data.groupby("trace"):
        e = lambda p: float(np.mean(np.abs(g["y"] - g[p])))
        rows.append({"trace": tname, "app": g["app"].iloc[0], "mobility": g["mob"].iloc[0], "rows": len(g),
                     "mae_persistence": e("persist"), "mae_ml_base": e("p_base"), "mae_ml_context": e("p_ctx"),
                     "mae_rolling_mean5": e("roll"), "mae_ml_history_only": e("p_hist")})
    pt = pd.DataFrame(rows)
    pt["improvement_ml_base"] = pt["mae_persistence"] - pt["mae_ml_base"]
    pt["context_benefit"] = pt["mae_ml_base"] - pt["mae_ml_context"]
    pt["improvement_ml_context"] = pt["mae_persistence"] - pt["mae_ml_context"]
    pt["diag_rolling_mean_vs_persistence"] = pt["mae_persistence"] - pt["mae_rolling_mean5"]
    pt["diag_history_only_vs_persistence"] = pt["mae_persistence"] - pt["mae_ml_history_only"]
    pt["diag_radio_added_value"] = pt["mae_ml_history_only"] - pt["mae_ml_base"]
    pt["rel_improvement_ml_base_pct"] = 100 * pt["improvement_ml_base"] / pt["mae_persistence"].replace(0, np.nan)
    pt.to_csv(os.path.join(HERE, "results", "per_trace_results.csv"), index=False)

    res = {"settings": {"horizon_seconds": H, "model": "HistGradientBoostingRegressor(loss=absolute_error, max_iter=200, lr=0.08, leaves=31, min_leaf=50)",
                        "cv": "GroupKFold(5) by trace, fixed settings (no tuning in this PoC)", "base_features": base_cols, "context_features": [c for c in ctx_cols if c not in base_cols]},
           "audit": audit,
           "overall": {"mae_persistence_mean_of_traces": float(pt["mae_persistence"].mean()),
                       "mae_ml_base_mean_of_traces": float(pt["mae_ml_base"].mean()),
                       "mae_ml_context_mean_of_traces": float(pt["mae_ml_context"].mean())},
           "rq2_improvement_ml_base_vs_persistence": summarize(pt["improvement_ml_base"]),
           "rq3_context_benefit": summarize(pt["context_benefit"]),
           "diagnostics_not_in_synopsis": {
               "rolling_mean5_vs_persistence": summarize(pt["diag_rolling_mean_vs_persistence"]),
               "ml_history_only_vs_persistence": summarize(pt["diag_history_only_vs_persistence"]),
               "radio_metrics_added_value_over_history_only": summarize(pt["diag_radio_added_value"]),
               "mean_mae": {"rolling_mean5": float(pt["mae_rolling_mean5"].mean()), "ml_history_only": float(pt["mae_ml_history_only"].mean())}},
           "rq4_descriptive": {}}
    for key, col in (("by_mobility", "mobility"), ("by_application", "app")):
        res["rq4_descriptive"][key] = {k: {"n": int(len(g)), "mean_improvement_kbps": float(g["improvement_ml_base"].mean()),
                                           "mean_rel_improvement_pct": float(g["rel_improvement_ml_base_pct"].mean()),
                                           "share_traces_improved": float((g["improvement_ml_base"] > 0).mean())}
                                       for k, g in pt.groupby(col)}
    res["rq4_descriptive"]["by_group"] = {f"{a}|{m}": {"n": int(len(g)), "mean_improvement_kbps": float(g["improvement_ml_base"].mean())}
                                          for (a, m), g in pt.groupby(["app", "mobility"])}
    with open(os.path.join(HERE, "results", "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    # compact out-of-fold series for the report explorer
    series = {}
    for tname, g in data.groupby("trace"):
        series[tname] = {"app": g["app"].iloc[0], "mob": g["mob"].iloc[0], "sec": g["sec"].astype(int).tolist(),
                         "actual": [round(v) for v in g["y"]], "persist": [round(v) for v in g["persist"]],
                         "ml": [round(v) for v in g["p_base"]], "ctx": [round(v) for v in g["p_ctx"]]}
    with open(os.path.join(HERE, "results", "series.json"), "w") as f:
        json.dump(series, f, separators=(",", ":"))
    print(json.dumps({k: res[k] for k in ("overall", "rq2_improvement_ml_base_vs_persistence", "rq3_context_benefit", "diagnostics_not_in_synopsis")}, indent=1))
    print(json.dumps(res["rq4_descriptive"]["by_mobility"], indent=2))


if __name__ == "__main__":
    main()
