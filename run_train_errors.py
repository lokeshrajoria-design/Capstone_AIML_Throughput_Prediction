"""Training-fold versus test-fold error for every model of the primary split (final-report table).

Usage (from the analysis folder, after run_models.py):  python run_train_errors.py
Refits each model of each outer fold with the settings saved in selections_outer_fold.csv, scores it on the
training traces it was fitted on, and pairs that with its held-out error from predictions_outer_fold.parquet.
Writes outputs/tables/outer_fold/train_test_errors.csv; no existing output is changed.
"""
import json
import time

import numpy as np
import pandas as pd

from tput import config as C
from tput.models import make_model

MODELS = ["base_ridge", "base_rf", "base_hgb", "context_selected"]


def trace_errors(trace_ids, y, pred):
    """Mean over traces of the per-trace MAE and RMSE (each trace counts once)."""
    d = pd.DataFrame({"trace": np.asarray(trace_ids), "e": np.asarray(y) - np.asarray(pred)})
    g = d.groupby("trace").e
    return g.apply(lambda e: e.abs().mean()).mean(), g.apply(lambda e: np.sqrt((e ** 2).mean())).mean()


def main():
    feats = pd.read_parquet(C.PROCESSED / "features.parquet")
    splits = pd.read_csv(C.PROCESSED / "splits.csv")
    sel = pd.read_csv(C.ROOT / "outputs" / "models" / "selections_outer_fold.csv")
    preds = pd.read_parquet(C.ROOT / "outputs" / "models" / "predictions_outer_fold.parquet")
    data = feats[feats.eligible].merge(splits[["trace_id", "outer_fold"]], on="trace_id").reset_index(drop=True)

    rows = []
    for k in sorted(data.outer_fold.unique()):
        t0 = time.time()
        train = data[data.outer_fold != k].reset_index(drop=True)
        test = preds[preds.fold == k]
        y = train[C.TARGET]
        mae, rmse = trace_errors(train.trace_id, y, train.throughput_lag_0)
        tmae, trmse = trace_errors(test.trace_id, test[C.TARGET], test.persistence)
        rows.append({"fold": k, "model": "persistence", "train_mae": mae, "train_rmse": rmse,
                     "test_mae": tmae, "test_rmse": trmse})
        chosen = test.selected_family.iloc[0]
        for m in MODELS:
            fset, fam = ("context", chosen) if m == "context_selected" else ("base", m.split("_", 1)[1])
            s = sel[(sel.fold == k) & (sel.feature_set == fset) & (sel.family == fam)].iloc[0]
            model = make_model(fam, json.loads(s.params), fset).fit(train, y)
            mae, rmse = trace_errors(train.trace_id, y, model.predict(train))
            tmae, trmse = trace_errors(test.trace_id, test[C.TARGET], test[m])
            rows.append({"fold": k, "model": m, "train_mae": mae, "train_rmse": rmse, "test_mae": tmae, "test_rmse": trmse})
        print(f"fold {k} done in {time.time() - t0:.0f} s", flush=True)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(C.TABLES / "outer_fold" / "train_test_errors_by_fold.csv", index=False)
    summ = res.groupby("model", sort=False)[["train_mae", "train_rmse", "test_mae", "test_rmse"]].mean()
    summ.round(4).to_csv(C.TABLES / "outer_fold" / "train_test_errors.csv")
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
