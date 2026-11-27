"""Final analysis for the QM640 capstone: 5G downlink throughput prediction (Lokesh Rajoria).

Implements the analytic plan of the final synopsis (13 Oct 2026):
  RQ1  within-trace Spearman correlations of RSRP, RSRQ, SNR, CQI with contemporaneous throughput,
       mean across traces, Holm adjustment over the four measurements.
  RQ2  ridge, random forest and gradient boosting (radio + recent throughput history) versus persistence,
       5 s ahead, whole traces withheld, model selection inside the training traces only.
  RQ3  same algorithm family with and without context (speed, network mode, application type).
  RQ4  improvement over persistence by mobility (primary) and by application (secondary).
  Plus: a history-only diagnostic and a recording-date sensitivity check.

AI-assisted: written with Claude (Sonnet 5.5) on 2026-11-10 under the student's stated instructor approval
(see the AI-use statement in the report). Reuses the data preparation functions of run_poc.py in this folder.

Usage:  python final_analysis.py [path/to/5G-production-dataset.zip]
Needs:  pandas, numpy, scipy, scikit-learn, matplotlib
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_poc as P  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
SEED = 0
ZIP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "raw", "5G-production-dataset.zip")

GRIDS = {
    "ridge": [{"alpha": 1.0}, {"alpha": 100.0}, {"alpha": 1000.0}],
    "rf": [{"max_depth": 8}, {"max_depth": 14}],
    "hgb": [{"lr": 0.05, "leaves": 15}, {"lr": 0.10, "leaves": 31}],
}
FAMILY_NAME = {"ridge": "Ridge regression", "rf": "Random forest", "hgb": "Gradient boosting"}


def make(family, p):
    if family == "ridge":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=p["alpha"]))])
    if family == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", RandomForestRegressor(n_estimators=100, max_depth=p["max_depth"], min_samples_leaf=50,
                                                     n_jobs=-1, random_state=SEED))])
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=p["lr"], max_leaf_nodes=p["leaves"],
                                         min_samples_leaf=50, random_state=SEED)


def fit_predict(family, p, Xtr, ytr, Xte):
    m = make(family, p).fit(Xtr, ytr)
    return np.clip(m.predict(Xte), 0, None)


def tune(family, tr_df, cols, step=2):
    """Pick the best setting of one family by 3-fold grouped CV inside the training traces (MAE)."""
    gkf = GroupKFold(n_splits=3)
    best, best_mae, table = None, np.inf, []
    for p in GRIDS[family]:
        errs = []
        for a, b in gkf.split(tr_df, groups=tr_df["trace"].values):
            A = tr_df.iloc[a].iloc[::step]
            B = tr_df.iloc[b]
            pred = fit_predict(family, p, A[cols], A["y"], B[cols])
            errs.append(np.mean(np.abs(B["y"].values - pred)))
        mae = float(np.mean(errs))
        table.append({"params": p, "inner_mae": mae})
        if mae < best_mae:
            best, best_mae = p, mae
    return best, best_mae, table


def ci_t(x, level=0.95):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return float(m), float(m - t * se), float(m + t * se)


def one_sided(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    m, lo, hi = ci_t(v)
    try:
        pw = float(stats.wilcoxon(v, alternative="greater").pvalue)
    except ValueError:
        pw = None
    return {"n": int(len(v)), "mean": m, "ci95": [lo, hi], "median": float(np.median(v)), "sd": float(v.std(ddof=1)),
            "share_positive": float((v > 0).mean()), "dz": float(m / v.std(ddof=1)),
            "p_ttest_one_sided": float(stats.ttest_1samp(v, 0, alternative="greater").pvalue), "p_wilcoxon_one_sided": pw}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.zeros(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    t0 = time.time()
    raw = P.load_traces(ZIP)
    dates = {r[0]: str(pd.to_datetime(r[3]["Timestamp"].iloc[0], format="%Y.%m.%d_%H.%M.%S", errors="coerce").date()) for r in raw}
    res = {"settings": {"horizon_s": P.H, "outer_cv": "GroupKFold(5) by trace", "inner_cv": "GroupKFold(3) by trace inside training traces",
                        "tuning_row_step": 2, "grids": GRIDS, "seed": SEED}}

    # ---------------- RQ1: within-trace Spearman correlations ----------------
    rq1_rows, frames, base_cols = [], [], None
    for name, app, mob, df in raw:
        g = P.to_1hz(df)
        for c in ["RSRP", "RSRQ", "SNR", "CQI"]:
            pair = g[[c, "DL_bitrate"]].dropna()
            rho = np.nan
            if len(pair) >= 30 and pair[c].nunique() > 1 and pair["DL_bitrate"].nunique() > 1:
                rho = float(stats.spearmanr(pair[c], pair["DL_bitrate"]).statistic)
            rq1_rows.append({"trace": name, "app": app, "mobility": mob, "measure": c, "n_pairs": len(pair), "rho": rho})
        d, base_cols = P.build(name, app, mob, g)
        if len(d) >= 30:
            frames.append(d)
    rq1 = pd.DataFrame(rq1_rows); rq1["date"] = rq1["trace"].map(dates)
    rq1.to_csv(os.path.join(OUT, "rq1_trace_correlations.csv"), index=False)
    r1 = {}
    for c in ["RSRP", "RSRQ", "SNR", "CQI"]:
        v = rq1.loc[(rq1.measure == c) & rq1.rho.notna(), "rho"].values
        m, lo, hi = ci_t(v)
        r1[c] = {"eligible_traces": int(len(v)), "excluded_traces": int(rq1[rq1.measure == c].shape[0] - len(v)), "mean_rho": m,
                 "ci95": [lo, hi], "median_rho": float(np.median(v)), "sd": float(v.std(ddof=1)),
                 "share_positive": float((v > 0).mean()), "p_ttest_two_sided": float(stats.ttest_1samp(v, 0).pvalue),
                 "p_wilcoxon_two_sided": float(stats.wilcoxon(v).pvalue)}
    adj = holm([r1[c]["p_ttest_two_sided"] for c in r1])
    for c, a in zip(r1, adj):
        r1[c]["p_holm"] = a
    # date-level sensitivity: average rho within each recording date first
    for c in r1:
        sub = rq1[(rq1.measure == c) & rq1.rho.notna()].groupby("date")["rho"].mean()
        r1[c]["date_level"] = {"n_dates": int(len(sub)), "mean_rho": float(sub.mean()),
                               "p_ttest_two_sided": float(stats.ttest_1samp(sub.values, 0).pvalue)}
    res["rq1"] = r1

    # ---------------- Data for RQ2 to RQ4 ----------------
    data = pd.concat(frames, ignore_index=True)
    ctx_cols = [c for c in data.columns if c not in ("y", "persist", "trace", "app", "mob", "sec")]
    hist_cols = [c for c in base_cols if c.startswith("dl_")]
    data["date"] = data["trace"].map(dates)
    res["audit"] = {"source_files": len(raw), "raw_rows": int(sum(len(r[3]) for r in raw)), "traces_used": int(data.trace.nunique()),
                    "usable_rows": int(len(data)), "recording_dates": int(len(set(dates.values()))),
                    "traces_by_group": {f"{a}|{m}": int(data[(data.app == a) & (data.mob == m)].trace.nunique()) for a in P.APPS for m in P.MOBS},
                    "zero_share_by_app": {a: float((data.loc[data.app == a, "persist"] == 0).mean()) for a in P.APPS},
                    "base_features": base_cols, "context_features": [c for c in ctx_cols if c not in base_cols]}

    # ---------------- Nested evaluation ----------------
    preds = {k: np.zeros(len(data)) for k in ["ridge", "rf", "hgb", "base", "ctx", "hist"]}
    folds = []
    outer = GroupKFold(n_splits=5)
    for k, (tr, te) in enumerate(outer.split(data, groups=data["trace"].values)):
        trd, ted = data.iloc[tr], data.iloc[te]
        info = {"fold": k + 1, "train_traces": int(trd.trace.nunique()), "test_traces": int(ted.trace.nunique()), "inner": {}}
        best = {}
        for fam in GRIDS:
            p, mae, table = tune(fam, trd, base_cols)
            best[fam] = (p, mae)
            info["inner"][fam] = {"best_params": p, "inner_mae": mae}
            preds[fam][te] = fit_predict(fam, p, trd[base_cols], trd["y"], ted[base_cols])
        fam_star = min(best, key=lambda f: best[f][1])
        info["selected_family"] = fam_star
        preds["base"][te] = preds[fam_star][te]
        p_ctx, _, _ = tune(fam_star, trd, ctx_cols)
        preds["ctx"][te] = fit_predict(fam_star, p_ctx, trd[ctx_cols], trd["y"], ted[ctx_cols])
        p_hist, _, _ = tune(fam_star, trd, hist_cols)
        preds["hist"][te] = fit_predict(fam_star, p_hist, trd[hist_cols], trd["y"], ted[hist_cols])
        info["ctx_params"] = p_ctx; info["hist_params"] = p_hist
        folds.append(info)
        print(f"fold {k+1}/5 done, selected {fam_star}, {time.time()-t0:.0f}s", flush=True)
    res["folds"] = folds
    for k, v in preds.items():
        data["p_" + k] = v

    # ---------------- Trace-level metrics ----------------
    rows = []
    for tname, g in data.groupby("trace"):
        r = {"trace": tname, "app": g.app.iloc[0], "mobility": g.mob.iloc[0], "date": g.date.iloc[0], "rows": len(g)}
        for lab, col in [("persistence", "persist"), ("ridge", "p_ridge"), ("rf", "p_rf"), ("hgb", "p_hgb"), ("ml_base", "p_base"),
                         ("ml_context", "p_ctx"), ("ml_history_only", "p_hist")]:
            e = g["y"].values - g[col].values
            r["mae_" + lab] = float(np.mean(np.abs(e))); r["rmse_" + lab] = float(np.sqrt(np.mean(e ** 2)))
        rows.append(r)
    pt = pd.DataFrame(rows)
    pt["improve_base"] = pt.mae_persistence - pt.mae_ml_base
    pt["context_benefit"] = pt.mae_ml_base - pt.mae_ml_context
    pt["improve_context"] = pt.mae_persistence - pt.mae_ml_context
    pt["radio_added_value"] = pt.mae_ml_history_only - pt.mae_ml_base
    pt["rel_improve_context"] = np.where(pt.mae_persistence > 0, 1 - pt.mae_ml_context / pt.mae_persistence, np.nan)
    pt.to_csv(os.path.join(OUT, "per_trace_results.csv"), index=False)
    data[["trace", "app", "mob", "sec", "y", "persist", "p_base", "p_ctx"]].to_csv(os.path.join(OUT, "predictions_long.csv.gz"), index=False)

    res["overall_mae_mean_of_traces"] = {k: float(pt["mae_" + k].mean()) for k in
                                         ["persistence", "ridge", "rf", "hgb", "ml_base", "ml_context", "ml_history_only"]}
    res["overall_rmse_mean_of_traces"] = {k: float(pt["rmse_" + k].mean()) for k in
                                          ["persistence", "ridge", "rf", "hgb", "ml_base", "ml_context", "ml_history_only"]}
    res["rq2"] = one_sided(pt.improve_base.values)
    res["rq2"]["relative_of_persistence_mae"] = float(pt.improve_base.mean() / pt.mae_persistence.mean())
    res["rq2_by_family"] = {f: one_sided((pt.mae_persistence - pt["mae_" + f]).values) for f in ["ridge", "rf", "hgb"]}
    res["rq3"] = one_sided(pt.context_benefit.values)
    res["rq3"]["relative_of_base_mae"] = float(pt.context_benefit.mean() / pt.mae_ml_base.mean())
    res["diag_history_only_vs_persistence"] = one_sided((pt.mae_persistence - pt.mae_ml_history_only).values)
    res["diag_radio_added_value"] = one_sided(pt.radio_added_value.values)
    # recording-date sensitivity for RQ2 and RQ3
    for key, col in (("rq2", "improve_base"), ("rq3", "context_benefit")):
        d = pt.groupby("date")[col].mean()
        res[key]["date_level"] = {"n_dates": int(len(d)), "mean": float(d.mean()), "p_ttest_one_sided": float(stats.ttest_1samp(d.values, 0, alternative="greater").pvalue)}

    # ---------------- RQ4 ----------------
    s = pt[pt.mobility == "Static"].improve_context.values; d_ = pt[pt.mobility == "Driving"].improve_context.values
    tt = stats.ttest_ind(s, d_, equal_var=False)
    nS, nD = len(s), len(d_); vS, vD = s.var(ddof=1), d_.var(ddof=1)
    se = np.sqrt(vS / nS + vD / nD); dfw = se ** 4 / ((vS / nS) ** 2 / (nS - 1) + (vD / nD) ** 2 / (nD - 1))
    diff = s.mean() - d_.mean(); tcrit = stats.t.ppf(0.975, dfw)
    pooled = np.sqrt(((nS - 1) * vS + (nD - 1) * vD) / (nS + nD - 2))
    rel = pt.dropna(subset=["rel_improve_context"])
    rs = rel[rel.mobility == "Static"].rel_improve_context.values; rd = rel[rel.mobility == "Driving"].rel_improve_context.values
    res["rq4_mobility"] = {"static": {"n": nS, "mean": float(s.mean()), "median": float(np.median(s))},
                           "driving": {"n": nD, "mean": float(d_.mean()), "median": float(np.median(d_))},
                           "mean_diff_static_minus_driving": float(diff), "ci95_diff": [float(diff - tcrit * se), float(diff + tcrit * se)],
                           "welch_t": float(tt.statistic), "welch_df": float(dfw), "p_welch": float(tt.pvalue),
                           "p_mannwhitney": float(stats.mannwhitneyu(s, d_, alternative="two-sided").pvalue),
                           "cohens_d": float(diff / pooled),
                           "relative_improvement": {"static_mean": float(rs.mean()), "driving_mean": float(rd.mean()),
                                                    "n_static": int(len(rs)), "n_driving": int(len(rd)),
                                                    "p_welch": float(stats.ttest_ind(rs, rd, equal_var=False).pvalue),
                                                    "p_mannwhitney": float(stats.mannwhitneyu(rs, rd, alternative="two-sided").pvalue)}}
    groups = {a: pt[pt.app == a].improve_context.values for a in P.APPS}
    relg = {a: rel[rel.app == a].rel_improve_context.values for a in P.APPS}
    pair = []
    names = list(P.APPS)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = groups[names[i]], groups[names[j]]
            pair.append({"pair": f"{names[i]} vs {names[j]}", "p_welch": float(stats.ttest_ind(a, b, equal_var=False).pvalue)})
    for q, a in zip(pair, holm([q["p_welch"] for q in pair])):
        q["p_holm"] = a
    res["rq4_application"] = {a: {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
                                  "mean_relative": float(np.nanmean(relg[a])) if len(relg[a]) else None, "n_relative": int(len(relg[a]))} for a, v in groups.items()}
    res["rq4_application"]["anova_p"] = float(stats.f_oneway(*groups.values()).pvalue)
    res["rq4_application"]["kruskal_p"] = float(stats.kruskal(*groups.values()).pvalue)
    res["rq4_application"]["kruskal_p_relative"] = float(stats.kruskal(*relg.values()).pvalue)
    res["rq4_application"]["pairwise_welch_exploratory"] = pair
    # descriptive group table
    tab = []
    for a in P.APPS:
        for m in P.MOBS:
            g = pt[(pt.app == a) & (pt.mobility == m)]
            if len(g):
                tab.append({"app": a, "mobility": m, "traces": int(len(g)), "mae_persistence": float(g.mae_persistence.mean()),
                            "mae_ml_context": float(g.mae_ml_context.mean()), "improve_context": float(g.improve_context.mean())})
    res["group_table"] = tab
    res["runtime_seconds"] = round(time.time() - t0)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("DONE in", res["runtime_seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
