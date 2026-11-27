"""RQ2-RQ4 results from saved held-out predictions (run run_models.py first).

Usage:  python analyze_models.py [--split outer_fold|date_fold|chrono_test]
Intervals are approximate: they resample saved trace errors and do not cover model-retraining
variability (synopsis 5.5). The date-cluster interval resamples whole recording dates.
"""
import argparse

import numpy as np
import pandas as pd

from tput import config as C
from tput.stats import bootstrap_diff, bootstrap_mean

MODELS = ["persistence", "base_ridge", "base_rf", "base_hgb", "base_selected", "context_selected"]
# Reference baselines added after the primary results were seen (not in the synopsis plan): video traces are
# mostly idle, so "always 0" and the 10 s mean show how much of the gain over persistence is that effect.
REFERENCE = ["ref_zero", "ref_mean_10s"]
KEYS = ["trace_id", "application_type", "mobility_condition", "recording_date"]


def per_trace(p):
    rows = []
    for m in MODELS + REFERENCE:
        e = p[KEYS].assign(ae=(p[m] - p[C.TARGET]).abs(), se=(p[m] - p[C.TARGET]) ** 2)
        g = e.groupby(KEYS)
        rows.append(pd.DataFrame({"n": g.size(), "mae": g.ae.mean(), "rmse": np.sqrt(g.se.mean())})
                    .reset_index().assign(model=m))
    return pd.concat(rows, ignore_index=True)


def paired(errs, better, worse):
    """Per-trace MAE(worse) - MAE(better); positive = 'better' has lower error."""
    w = errs.pivot_table(index=KEYS, columns="model", values="mae").reset_index()
    return w.assign(delta=w[worse] - w[better], rel=(w[worse] - w[better]) / w[worse].replace(0, np.nan))


def effect_row(name, d):
    b = bootstrap_mean(d.delta)
    bd = bootstrap_mean(d.delta, clusters=d.recording_date)
    return {"comparison": name, "traces": b["n"], "mean_delta_mbps": b["mean"], "ci95_low": b["ci_low"],
            "ci95_high": b["ci_high"], "date_ci95_low": bd["ci_low"], "date_ci95_high": bd["ci_high"],
            "boot_share_le_0": b["share_boot_le_0"], "traces_improved_pct": 100 * (d.delta > 0).mean(),
            "median_rel_improvement_pct": 100 * d.rel.median()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="outer_fold")
    a = ap.parse_args()
    out = C.TABLES / a.split
    out.mkdir(parents=True, exist_ok=True)

    p = pd.read_parquet(C.ROOT / "outputs" / "models" / f"predictions_{a.split}.parquet")
    hist = pd.read_parquet(C.PROCESSED / "features.parquet", columns=["trace_id", "t", "throughput_mean_10s"])
    p = p.merge(hist, on=["trace_id", "t"], how="left", validate="one_to_one")
    p["ref_zero"] = 0.0
    p["ref_mean_10s"] = p.throughput_mean_10s
    errs = per_trace(p)
    errs.to_csv(out / "trace_errors.csv", index=False)

    # Overall error table: trace-weighted (primary) and row-weighted (supplementary)
    tw = errs.groupby("model").agg(traces=("trace_id", "nunique"), mean_trace_mae=("mae", "mean"),
                                   mean_trace_rmse=("rmse", "mean"))
    rw = pd.Series({m: (p[m] - p[C.TARGET]).abs().mean() for m in MODELS + REFERENCE}, name="row_weighted_mae")
    overall = tw.join(rw).loc[MODELS + REFERENCE].round(3)
    overall.to_csv(out / "errors_overall.csv")
    by_app = errs.pivot_table(index="application_type", columns="model", values="mae")[MODELS + REFERENCE].round(3)
    by_app.to_csv(out / "errors_by_application.csv")

    # Robustness: the selected approach against the reference baselines (exploratory, added post hoc)
    ref = pd.DataFrame([effect_row(f"{r} - base_selected", paired(errs, "base_selected", r)) for r in REFERENCE]
                       + [effect_row(f"{r} - context_selected", paired(errs, "context_selected", r))
                          for r in REFERENCE])
    ref.round(3).to_csv(out / "robustness_vs_reference_baselines.csv", index=False)

    # RQ2: every family and the fold-selected approach vs persistence (base features)
    rq2 = pd.DataFrame([effect_row(f"persistence - {m}", paired(errs, m, "persistence"))
                        for m in ["base_ridge", "base_rf", "base_hgb", "base_selected"]])
    rq2.round(3).to_csv(out / "rq2_improvement_over_persistence.csv", index=False)

    # RQ3: context benefit within the selected family
    rq3 = pd.DataFrame([effect_row("base_selected - context_selected",
                                   paired(errs, "context_selected", "base_selected"))])
    rq3.round(3).to_csv(out / "rq3_context_benefit.csv", index=False)

    # RQ4: improvement of the context-augmented selected approach over persistence, by group
    d = paired(errs, "context_selected", "persistence")
    groups = []
    for col in ("mobility_condition", "application_type"):
        for g, x in d.groupby(col):
            r = effect_row(f"{col}={g}", x)
            r["mean_mae_persistence"] = x.persistence.mean()
            r["mean_mae_model"] = x.context_selected.mean()
            groups.append(r)
    pd.DataFrame(groups).round(3).to_csv(out / "rq4_by_group.csv", index=False)

    comps = []
    s, dr = d[d.mobility_condition == "Static"].delta, d[d.mobility_condition == "Driving"].delta
    comps.append({"comparison": "Driving - Static", "level": 0.95, **bootstrap_diff(dr, s)})
    apps = sorted(d.application_type.unique())
    lvl = 1 - 0.05 / 3                                                   # Bonferroni, three pairs
    for i in range(len(apps)):
        for j in range(i + 1, len(apps)):
            comps.append({"comparison": f"{apps[i]} - {apps[j]}", "level": round(lvl, 4),
                          **bootstrap_diff(d[d.application_type == apps[i]].delta,
                                           d[d.application_type == apps[j]].delta, level=lvl)})
    pd.DataFrame(comps).round(3).to_csv(out / "rq4_group_differences.csv", index=False)

    sel = pd.read_csv(C.ROOT / "outputs" / "models" / f"selections_{a.split}.csv")
    chosen = p.groupby("fold").selected_family.first()

    pd.set_option("display.width", 200)
    print(f"== {a.split} ==\nSelected family per fold: {chosen.to_dict()}")
    print("\nErrors (Mbps):\n", overall.to_string())
    print("\nRQ2 improvement over persistence (Mbps, + = AI/ML better):\n", rq2.round(3).to_string(index=False))
    print("\nErrors by application (trace-mean MAE):\n", by_app.to_string())
    print("\nRobustness vs reference baselines (post hoc):\n", ref.round(3).to_string(index=False))
    print("\nRQ3 context benefit:\n", rq3.round(3).to_string(index=False))
    print("\nRQ4 by group:\n", pd.DataFrame(groups).round(3).to_string(index=False))
    print("\nRQ4 differences:\n", pd.DataFrame(comps).round(3).to_string(index=False))
    print("\nTuned params:\n", sel[["fold", "feature_set", "family", "params", "inner_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
