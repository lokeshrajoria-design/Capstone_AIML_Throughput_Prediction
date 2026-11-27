"""Permutation feature importance of the selected context-augmented model, on held-out traces.

For each outer fold the model is refitted with the fold's tuned settings (selections_outer_fold.csv) and the
increase in MAE when each input is shuffled is measured on that fold's test traces. Results are averaged
over the five folds. Usage (after run_models.py):  python run_importance.py
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from make_figures import BLUE, INK2, MUTED, save
from tput import config as C
from tput.models import FEATURE_SETS, make_model

LABELS = {"throughput_lag_0": "Throughput now", "throughput_lag_1": "Throughput 1 s ago",
          "throughput_lag_2": "Throughput 2 s ago", "throughput_lag_5": "Throughput 5 s ago",
          "throughput_mean_10s": "10 s mean throughput", "throughput_sd_10s": "10 s throughput SD",
          "Speed": "Speed", "NetworkMode": "Network mode", "application_type": "Application type"}
MAX_TEST_ROWS = 15000


def main():
    feats = pd.read_parquet(C.PROCESSED / "features.parquet")
    splits = pd.read_csv(C.PROCESSED / "splits.csv")
    sel = pd.read_csv(C.ROOT / "outputs" / "models" / "selections_outer_fold.csv")
    data = feats[feats.eligible].merge(splits[["trace_id", "outer_fold"]], on="trace_id")
    cols = FEATURE_SETS["context"]["num"] + FEATURE_SETS["context"]["cat"]
    rng = np.random.default_rng(C.SEED)

    rows = []
    for k in sorted(data.outer_fold.unique()):
        s = sel[(sel.fold == k) & (sel.feature_set == "context")].iloc[0]
        train, test = data[data.outer_fold != k], data[data.outer_fold == k]
        if len(test) > MAX_TEST_ROWS:
            test = test.iloc[rng.choice(len(test), MAX_TEST_ROWS, replace=False)]
        model = make_model(s.family, json.loads(s.params), "context").fit(train, train[C.TARGET])
        r = permutation_importance(model, test[cols], test[C.TARGET], scoring="neg_mean_absolute_error",
                                   n_repeats=5, random_state=C.SEED, n_jobs=-1)
        rows.append(pd.DataFrame({"feature": cols, "fold": k, "mae_increase": r.importances_mean}))
        print(f"fold {k}: {s.family} done")

    imp = pd.concat(rows)
    imp["feature"] = imp.feature.where(~imp.feature.str.endswith("_missing"), "Missing-value flags")
    summ = (imp.groupby(["feature", "fold"]).mae_increase.sum().groupby("feature")
            .agg(mean_mae_increase="mean", sd="std").sort_values("mean_mae_increase", ascending=False))
    summ.round(4).to_csv(C.TABLES / "feature_importance.csv")
    print(summ.round(4).to_string())
    plot(summ)


def plot(summ):
    s = summ.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    y = np.arange(len(s))
    ax.barh(y, s.mean_mae_increase, xerr=s.sd, color=BLUE, height=0.6, error_kw=dict(ecolor=MUTED, lw=0.8))
    ax.set_yticks(y, [LABELS.get(f, f) for f in s.index], fontsize=9)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Increase in MAE when the input is shuffled (Mbps); bars = SD across folds")
    ax.grid(axis="y", visible=False)
    save(fig, "fig8_feature_importance.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:   # redraw the figure from the saved table without refitting
        plot(pd.read_csv(C.TABLES / "feature_importance.csv", index_col=0))
    else:
        main()
