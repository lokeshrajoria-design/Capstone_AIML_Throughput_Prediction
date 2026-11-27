"""RQ2/RQ3 models and nested grouped cross-validation (synopsis 5.2, 5.3, 5.5).

Per outer fold:
  1. Inner grouped CV on the outer training traces tunes every candidate of every family (base features).
  2. The family with the lowest inner trace-mean MAE is the fold's selected AI/ML approach (RQ2).
  3. That same family is re-tuned on the context features with the same inner folds and grid (RQ3).
  4. Tuned models are refitted on the full outer training data and predict the held-out traces.
All preprocessing lives inside the sklearn pipeline, so it is fitted on training rows only.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .splits import inner_folds

FLAGS = [f"{c}_missing" for c in C.RADIO] + [f"throughput_lag_{k}_missing" for k in C.LAGS_S if k]
FEATURE_SETS = {
    "base": {"num": C.RADIO + C.HISTORY + FLAGS, "cat": []},
    "context": {"num": C.RADIO + C.HISTORY + FLAGS + C.CONTEXT_NUM, "cat": C.CONTEXT_CAT},
}

GRIDS = {
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    "rf": [{"min_samples_leaf": l, "max_features": m} for l in (5, 20) for m in (0.5, 1.0)],
    "hgb": [{"loss": l, "learning_rate": r} for l in ("squared_error", "absolute_error") for r in (0.05, 0.1)],
}
if os.environ.get("TPUT_QUICK") == "1":
    # Demonstration mode (Colab notebook): search only the settings most often selected by the full grid.
    GRIDS = {"ridge": [{"alpha": 0.1}], "rf": [{"min_samples_leaf": 20, "max_features": 1.0}],
             "hgb": [{"loss": "absolute_error", "learning_rate": 0.1}]}


def make_model(family, params, fset):
    cols = FEATURE_SETS[fset]
    num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if family == "ridge" else [])
    steps = [("num", make_pipeline(*num), cols["num"])]
    if cols["cat"]:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cols["cat"]))
    pre = ColumnTransformer(steps)
    if family == "ridge":
        est = Ridge(**params)
    elif family == "rf":
        est = RandomForestRegressor(n_estimators=200, max_samples=0.3, n_jobs=-1, random_state=C.SEED, **params)
    else:
        est = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=31, early_stopping=False,
                                            random_state=C.SEED, **params)
    return Pipeline([("pre", pre), ("model", est)])


def trace_mean_mae(trace_ids, y, pred):
    e = pd.DataFrame({"t": np.asarray(trace_ids), "ae": np.abs(np.asarray(y) - pred)})
    return e.groupby("t").ae.mean().mean()


def _inner_scores(train, folds, family, fset, grid):
    scores = []
    for params in grid:
        s = []
        for tr_idx, va_idx in folds:
            tr, va = train.iloc[tr_idx], train.iloc[va_idx]
            m = make_model(family, params, fset).fit(tr, tr[C.TARGET])
            s.append(trace_mean_mae(va.trace_id, va[C.TARGET], m.predict(va)))
        scores.append(float(np.mean(s)))
    best = int(np.argmin(scores))
    return grid[best], scores[best], scores


def _row_folds(train, group_col):
    """Inner folds built on whole groups (traces, or dates for the date split), mapped to row indices."""
    units = train.drop_duplicates(group_col)[[group_col, "application_type", "mobility_condition"]]
    strata = (units.application_type + "|" + units.mobility_condition).to_numpy()
    unit_ids = units[group_col].to_numpy()
    out = []
    for tr_u, va_u in inner_folds(unit_ids, strata):
        tr_set, va_set = set(unit_ids[tr_u]), set(unit_ids[va_u])
        out.append((np.flatnonzero(train[group_col].isin(tr_set)), np.flatnonzero(train[group_col].isin(va_set))))
    return out


def run_nested_cv(feats, splits, split_col="outer_fold", log=print):
    data = feats[feats.eligible].merge(splits[["trace_id", split_col]], on="trace_id", validate="many_to_one")
    data = data.reset_index(drop=True)
    group_col = "recording_date" if split_col == "date_fold" else "trace_id"
    test_values = [1] if split_col == "chrono_test" else sorted(data[split_col].unique())

    preds, selections = [], []
    for k in test_values:
        t0 = time.time()
        train = data[data[split_col] != k].reset_index(drop=True)
        test = data[data[split_col] == k].reset_index(drop=True)
        folds = _row_folds(train, group_col)

        out = test[["trace_id", "t", "application_type", "mobility_condition", "recording_date", C.TARGET]].copy()
        out["fold"] = k
        out["persistence"] = test["throughput_lag_0"].to_numpy()

        tuned = {}
        for fam, grid in GRIDS.items():
            params, score, all_scores = _inner_scores(train, folds, fam, "base", grid)
            tuned[fam] = (params, score)
            selections.append({"fold": k, "feature_set": "base", "family": fam, "params": params,
                               "inner_mae": score, "grid_scores": all_scores})
            out[f"base_{fam}"] = make_model(fam, params, "base").fit(train, train[C.TARGET]).predict(test)

        chosen = min(tuned, key=lambda f: tuned[f][1])
        out["base_selected"] = out[f"base_{chosen}"]
        params, score, all_scores = _inner_scores(train, folds, chosen, "context", GRIDS[chosen])
        selections.append({"fold": k, "feature_set": "context", "family": chosen, "params": params,
                           "inner_mae": score, "grid_scores": all_scores})
        out["context_selected"] = make_model(chosen, params, "context").fit(train, train[C.TARGET]).predict(test)
        out["selected_family"] = chosen
        preds.append(out)
        log(f"  fold {k}: selected {chosen}, {len(train)} train / {len(test)} test rows, {time.time() - t0:.0f} s")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(selections)
