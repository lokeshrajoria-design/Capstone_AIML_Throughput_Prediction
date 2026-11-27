"""Extra analyses for the final report: training versus test error of the selected model, missing-value rates,
throughput distribution (outlier check) and group zero shares. Uses the folds and settings in results/results.json.
AI-assisted: written with Claude (Sonnet 5.5) on 2026-11-10 (see the AI-use statement in the report)."""
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_poc as P            # noqa: E402
import final_analysis as FA    # noqa: E402

res = json.load(open(os.path.join(HERE, "results", "results.json")))
raw = P.load_traces(FA.ZIP)
miss = {c: [0, 0] for c in ["Speed", "RSRP", "RSRQ", "SNR", "CQI", "DL_bitrate"]}
frames, base_cols = [], None
for name, app, mob, df in raw:
    for c in miss:
        x = pd.to_numeric(df[c], errors="coerce"); miss[c][0] += int(x.isna().sum()); miss[c][1] += len(x)
    d, base_cols = P.build(name, app, mob, P.to_1hz(df))
    if len(d) >= 30:
        frames.append(d)
data = pd.concat(frames, ignore_index=True)
out = {"missing_share_raw_records": {c: v[0] / v[1] for c, v in miss.items()}, "raw_records": int(sum(v[1] for v in [miss["RSRP"]]))}

# distribution of the target (throughput five seconds ahead) by application
out["target_percentiles_kbps"] = {}
for a in P.APPS:
    y = data.loc[data.app == a, "y"]
    out["target_percentiles_kbps"][a] = {"n": int(len(y)), "zero_share": float((y == 0).mean()), "median": float(y.median()),
                                         "p95": float(y.quantile(0.95)), "p99": float(y.quantile(0.99)), "max": float(y.max()),
                                         "share_of_total_above_p99": float(y[y > y.quantile(0.99)].sum() / y.sum())}
out["zero_share_by_group"] = {f"{a}|{m}": float((data[(data.app == a) & (data.mob == m)]["y"] == 0).mean()) for a in P.APPS for m in P.MOBS}
out["rows_by_group"] = {f"{a}|{m}": int(len(data[(data.app == a) & (data.mob == m)])) for a in P.APPS for m in P.MOBS}

# training versus test error of the selected (gradient boosting) model, same folds as final_analysis.py
rows = []
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(data, groups=data["trace"].values)):
    f = res["folds"][k]
    assert f["train_traces"] == data.iloc[tr].trace.nunique() and f["test_traces"] == data.iloc[te].trace.nunique()
    p = f["inner"]["hgb"]["best_params"]
    trd, ted = data.iloc[tr], data.iloc[te]
    pred_tr = FA.fit_predict("hgb", p, trd[base_cols], trd["y"], trd[base_cols])
    pred_te = FA.fit_predict("hgb", p, trd[base_cols], trd["y"], ted[base_cols])
    for part, d, pred in (("train", trd, pred_tr), ("test", ted, pred_te)):
        t = d.assign(pred=pred)
        for tname, g in t.groupby("trace"):
            rows.append({"fold": k + 1, "part": part, "trace": tname, "mae_persistence": float(np.mean(np.abs(g.y - g.persist))),
                         "mae_model": float(np.mean(np.abs(g.y - g.pred))),
                         "rmse_persistence": float(np.sqrt(np.mean((g.y - g.persist) ** 2))), "rmse_model": float(np.sqrt(np.mean((g.y - g.pred) ** 2)))})
    print("fold", k + 1, "done", flush=True)
tt = pd.DataFrame(rows); tt.to_csv(os.path.join(HERE, "results", "train_vs_test_per_trace.csv"), index=False)
out["train_vs_test"] = {part: {c: float(tt[tt.part == part][c].mean()) for c in ["mae_persistence", "mae_model", "rmse_persistence", "rmse_model"]} for part in ("train", "test")}
out["train_vs_test"]["note"] = "Mean of trace-level errors. Training figures are in-sample: each trace appears in the training set of four of the five folds, so the train means average four fits per trace; test figures are one withheld fit per trace."
json.dump(out, open(os.path.join(HERE, "results", "extra_results.json"), "w"), indent=1)
print(json.dumps(out["train_vs_test"], indent=1)[:700])
