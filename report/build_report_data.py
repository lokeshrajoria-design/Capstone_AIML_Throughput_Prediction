"""Collect every number the interim report quotes into report_data.json (run from the analysis folder).

Usage:  python report/build_report_data.py
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tput import config as C  # noqa: E402

T = C.TABLES
out = {}


def f(x, n=2):
    return f"{x:.{n}f}".replace("-", "−")


def nz(x, n=3):
    """APA: no leading zero for statistics that cannot exceed 1 (correlations, p values)."""
    return f(x, n).replace("0.", ".", 1) if abs(x) < 1 else f(x, n)


def ci(lo, hi, n=2):
    return f"[{f(lo, n)}, {f(hi, n)}]"


# Overview by group
ov = pd.read_csv(T / "eda_overview_by_group.csv")
out["overview"] = [[r.application_type, r.mobility_condition, str(r.traces), f"{r.seconds:,}",
                    f(r.median_trace_minutes, 1), f(r.mean_mbps), f(r.median_mbps), f(r.p90_mbps),
                    f(r.zero_share_pct, 1), f(r.share_5g_pct, 1), f(r.mean_speed_kmh, 1)] for r in ov.itertuples()]

# Descriptive statistics
ds = pd.read_csv(T / "eda_descriptive_stats.csv", index_col=0)
out["descriptive"] = [[idx, f"{int(r['count']):,}", f(r.missing_pct, 1), f(r['mean']), f(r['std']), f(r['min'], 1),
                       f(r['50%'], 1), f(r['max'], 1)] for idx, r in ds.iterrows()]

# Data quality, missingness, network mode
q = pd.read_csv(T / "eda_data_quality.csv", index_col=0)["value"]
out["quality"] = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in q.items()}
miss = pd.read_csv(T / "eda_missingness.csv", index_col=0).iloc[:, 0]
out["missing"] = {k: f(v, 1) for k, v in miss.items()}
comp = pd.read_csv(T / "eda_network_mode_by_mobility.csv", index_col=0)
out["mode"] = {mob: {c: f(comp.loc[mob, c], 1) for c in comp.columns} for mob in comp.index}
cov = pd.read_csv(T / "coverage_by_group.csv")
out["eligible_total"] = int(cov.eligible.sum())

# Achieved precision (half-width in SD units) versus the synopsis plan
n_app = ov.groupby("application_type").traces.sum()
n_mob = ov.groupby("mobility_condition").traces.sum()
worst = max(math.sqrt(1 / n_app[a] + 1 / n_app[b]) for a in n_app.index for b in n_app.index if a < b)


def prec(rq, required, available, achieved, target):
    """One Table 2 row; 'Target met?' says Yes, or No with the achieved and target half-widths."""
    a, t = f(achieved) + "σ", f(target) + "σ"
    return [rq, required, available, a, t, "Yes" if round(achieved, 2) <= target else f"No ({a} vs. {t})"]


out["precision"] = [
    prec("RQ1", "100 traces", "82–83 traces", 2.4977 / math.sqrt(82), 0.25),
    prec("RQ2", "62 traces", "83 traces", 1.96 / math.sqrt(83), 0.25),
    prec("RQ3", "62 traces", "83 traces", 1.96 / math.sqrt(83), 0.25),
    prec("RQ4a", "22 static, 58 driving", f"{n_mob['Static']} static, {n_mob['Driving']} driving",
         1.96 * math.sqrt(1 / n_mob['Static'] + 1 / n_mob['Driving']), 0.50),
    prec("RQ4b", "46 per application", ", ".join(f"{n_app[a]} {a}" for a in ["Amazon Prime", "Netflix", "Download"]),
         2.3940 * worst, 0.50),
]

# Feature importance
imp = pd.read_csv(T / "feature_importance.csv")
names = {"throughput_mean_10s": "10 s mean throughput", "application_type": "Application type",
         "throughput_lag_0": "Throughput now", "throughput_lag_5": "Throughput 5 s ago",
         "throughput_lag_2": "Throughput 2 s ago", "throughput_sd_10s": "10 s throughput SD",
         "throughput_lag_1": "Throughput 1 s ago", "Speed": "Speed", "NetworkMode": "Network mode"}
out["importance"] = [[names.get(r.feature, r.feature), f(r.mean_mae_increase, 3), f(r.sd, 3)] for r in imp.itertuples()]

# RQ1
r1 = pd.read_csv(T / "rq1_summary.csv")
out["rq1"] = [[r.measurement, str(r.eligible_traces), nz(r.mean_rho), f"[{nz(r.ci95_low)}, {nz(r.ci95_high)}]",
               nz(r.p_holm), nz(r.date_p_holm)] for r in r1.itertuples()]

# RQ2 errors and improvements (primary split)
err = pd.read_csv(T / "outer_fold" / "errors_overall.csv", index_col=0)
imp2 = pd.read_csv(T / "outer_fold" / "rq2_improvement_over_persistence.csv").set_index("comparison")
rob = pd.read_csv(T / "outer_fold" / "robustness_vs_reference_baselines.csv").set_index("comparison")
lab = {"persistence": "Persistence", "base_ridge": "Ridge regression", "base_rf": "Random forest",
       "base_hgb": "Gradient boosting (selected)", "ref_mean_10s": "Reference: 10 s mean",
       "ref_zero": "Reference: always 0"}
rows = []
for m, name in lab.items():
    e = err.loc[m]
    if m == "persistence":
        imp_s, share = "–", "–"
    elif m.startswith("ref_"):
        imp_s, share = f(err.loc["persistence", "mean_trace_mae"] - e.mean_trace_mae), "–"
    else:
        r = imp2.loc[f"persistence - {m}"]
        imp_s, share = f"{f(r.mean_delta_mbps)} {ci(r.ci95_low, r.ci95_high)}", f"{r.traces_improved_pct:.0f}%"
    rows.append([name, f(e.mean_trace_mae), f(e.mean_trace_rmse), imp_s, share])
out["rq2"] = rows
# Mean trace MAE and RMSE of every predictor, for the model performance comparison table
out["err"] = {m: [f(r.mean_trace_mae), f(r.mean_trace_rmse)] for m, r in err.iterrows()}
_p = pd.read_parquet(C.ROOT / "outputs" / "models" / "predictions_outer_fold.parquet", columns=["trace_id"])
out["err"].update({"rows": f"{len(_p):,}", "row_pers": f(err.loc["persistence", "row_weighted_mae"]),
                   "row_sel": f(err.loc["base_selected", "row_weighted_mae"])})
# Context model versus persistence, per trace, with the same trace bootstrap as the planned comparisons
from tput.stats import bootstrap_mean  # noqa: E402
te = pd.read_csv(T / "outer_fold" / "trace_errors.csv").pivot(index="trace_id", columns="model", values="mae")
dctx = te["persistence"] - te["context_selected"]
b = bootstrap_mean(dctx)
out["rq2_ctx"] = {"delta": f(b["mean"]), "ci": ci(b["ci_low"], b["ci_high"]), "share": f"{100 * (dctx > 0).mean():.0f}%"}


def pval(p, n_boot=5000):
    """APA p value; a bootstrap share of 0 is reported as an upper bound."""
    return "< .001" if p < 0.001 else "> .999" if p > 0.999 else "= " + nz(p)


# One-sided bootstrap p values (share of resampled means at or below zero) for RQ2 and RQ3, per scheme
out["pvals"] = {}
for split in ["outer_fold", "date_fold", "chrono_test"]:
    a2 = pd.read_csv(T / split / "rq2_improvement_over_persistence.csv").set_index("comparison")
    a3 = pd.read_csv(T / split / "rq3_context_benefit.csv").iloc[0]
    out["pvals"][split] = {"rq2": pval(a2.loc["persistence - base_selected", "boot_share_le_0"]),
                           "rq3": pval(a3.boot_share_le_0)}
# Training-fold versus test-fold error (run_train_errors.py)
tt_path = T / "outer_fold" / "train_test_errors.csv"
if tt_path.exists():
    tt = pd.read_csv(tt_path, index_col=0)
    out["traintest"] = {m: [f(r.train_mae), f(r.test_mae), f(r.train_rmse), f(r.test_rmse), f(r.test_mae - r.train_mae)]
                        for m, r in tt.iterrows()}
sel = imp2.loc["persistence - base_selected"]
out["rq2_sel"] = {"delta": f(sel.mean_delta_mbps), "ci": ci(sel.ci95_low, sel.ci95_high),
                  "date_ci": ci(sel.date_ci95_low, sel.date_ci95_high), "rel": f(sel.median_rel_improvement_pct, 1),
                  "pers": f(err.loc["persistence", "mean_trace_mae"]), "model": f(err.loc["base_selected", "mean_trace_mae"])}
r = rob.loc["ref_mean_10s - base_selected"]
out["rq2_vs_mean"] = {"delta": f(r.mean_delta_mbps), "ci": ci(r.ci95_low, r.ci95_high), "share": f"{r.traces_improved_pct:.0f}"}

# RQ3
r3 = pd.read_csv(T / "outer_fold" / "rq3_context_benefit.csv").iloc[0]
out["rq3"] = {"delta": f(r3.mean_delta_mbps), "ci": ci(r3.ci95_low, r3.ci95_high),
              "date_ci": ci(r3.date_ci95_low, r3.date_ci95_high), "share": f(r3.traces_improved_pct, 1),
              "rel": f(r3.median_rel_improvement_pct, 1), "context_mae": f(err.loc["context_selected", "mean_trace_mae"])}

# RQ4
g = pd.read_csv(T / "outer_fold" / "rq4_by_group.csv")
order = ["mobility_condition=Static", "mobility_condition=Driving", "application_type=Amazon Prime",
         "application_type=Netflix", "application_type=Download"]
g = g.set_index("comparison").loc[order]
out["rq4"] = [[k.split("=")[1], str(int(r.traces)), f(r.mean_mae_persistence), f(r.mean_mae_model),
               f"{f(r.mean_delta_mbps)} {ci(r.ci95_low, r.ci95_high)}", f"{r.median_rel_improvement_pct:.0f}%"]
              for k, r in g.iterrows()]
d4 = pd.read_csv(T / "outer_fold" / "rq4_group_differences.csv").set_index("comparison")
out["rq4_diff"] = {k: f"{f(r['diff'])} {ci(r.ci_low, r.ci_high)}" for k, r in d4.iterrows()}

# Sensitivity
sens = []
for split, name in [("outer_fold", "Trace folds (primary)"), ("date_fold", "Recording dates held out"),
                    ("chrono_test", "Latest dates held out")]:
    a = pd.read_csv(T / split / "rq2_improvement_over_persistence.csv").set_index("comparison").loc[
        "persistence - base_selected"]
    b = pd.read_csv(T / split / "rq3_context_benefit.csv").iloc[0]
    c = pd.read_csv(T / split / "rq4_group_differences.csv").set_index("comparison").loc["Driving - Static"]
    sens.append([name, str(int(a.traces)), f"{f(a.mean_delta_mbps)} {ci(a.ci95_low, a.ci95_high)}",
                 f"{f(b.mean_delta_mbps)} {ci(b.ci95_low, b.ci95_high)}", f"{f(c['diff'])} {ci(c.ci_low, c.ci_high)}"])
out["sensitivity"] = sens

Path(__file__).with_name("report_data.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
print(json.dumps({k: out[k] for k in ["precision", "rq2_sel", "rq3", "rq4_diff", "sensitivity"]}, indent=1,
                 ensure_ascii=False))
