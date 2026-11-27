"""Weeks 5-6: nested grouped CV for RQ2/RQ3, saving held-out predictions.

Usage (from the analysis folder, after run_pipeline.py):
  python run_models.py                    # primary: 5 outer folds of whole traces
  python run_models.py --split date_fold  # sensitivity: whole recording dates held out
  python run_models.py --split chrono_test  # sensitivity: latest dates held out
"""
import argparse
import json
import time

import pandas as pd

from tput import config as C
from tput.models import GRIDS, run_nested_cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="outer_fold", choices=["outer_fold", "date_fold", "chrono_test"])
    a = ap.parse_args()

    feats = pd.read_parquet(C.PROCESSED / "features.parquet")
    splits = pd.read_csv(C.PROCESSED / "splits.csv")
    out_dir = C.ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"nested CV on {a.split}")
    preds, sel = run_nested_cv(feats, splits, a.split)
    preds.to_parquet(out_dir / f"predictions_{a.split}.parquet", index=False)
    sel.assign(params=sel.params.map(json.dumps), grid_scores=sel.grid_scores.map(json.dumps)) \
        .to_csv(out_dir / f"selections_{a.split}.csv", index=False)
    (out_dir / f"run_{a.split}.json").write_text(json.dumps(
        {"split": a.split, "grids": GRIDS, "seed": C.SEED, "minutes": round((time.time() - t0) / 60, 1),
         "rows": len(preds), "traces": int(preds.trace_id.nunique())}, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
