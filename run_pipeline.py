"""Run steps 1-4 of the synopsis pipeline plus the persistence baseline.

Usage (from the analysis folder):  python run_pipeline.py
Outputs: data/interim, data/processed, outputs/tables, outputs/run_info.json
"""
import json
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

from tput import config as C
from tput.checks import check_no_leakage, check_splits
from tput.clean import clean_all
from tput.evaluate import persistence, summarise, trace_errors
from tput.features import build_features, coverage
from tput.manifest import build_manifest


def main():
    for p in (C.INTERIM, C.PROCESSED, C.TABLES, C.FIGURES):
        p.mkdir(parents=True, exist_ok=True)

    print("1/5 manifest")
    manifest = build_manifest()
    manifest.to_csv(C.PROCESSED / "trace_manifest.csv", index=False)

    print("2/5 clean")
    clean, log = clean_all(manifest)
    clean.to_parquet(C.INTERIM / "clean_1s.parquet", index=False)
    log.to_csv(C.TABLES / "cleaning_log.csv", index=False)

    print("3/5 features")
    feats = build_features(clean)
    check_no_leakage(clean, feats)
    feats.to_parquet(C.PROCESSED / "features.parquet", index=False)
    cov = coverage(feats)
    cov.to_csv(C.TABLES / "coverage_by_group.csv", index=False)

    print("4/5 splits")
    from tput.splits import make_splits
    elig = feats[feats.eligible]
    splits = make_splits(manifest, elig.trace_id.unique())
    check_splits(splits)
    splits.to_csv(C.PROCESSED / "splits.csv", index=False)
    fold_mix = pd.crosstab([splits.application_type, splits.mobility_condition], splits.outer_fold)
    fold_mix.to_csv(C.TABLES / "outer_fold_composition.csv")

    print("5/5 persistence baseline")
    per_trace = trace_errors(elig, persistence(elig), "persistence")
    per_trace.to_csv(C.TABLES / "persistence_by_trace.csv", index=False)
    summary = summarise(per_trace, by=("mobility_condition", "application_type"))
    summary.to_csv(C.TABLES / "persistence_summary.csv", index=False)

    info = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "input_sha256": manifest.attrs["sha256"],
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "seed": C.SEED,
        "traces": len(manifest), "raw_rows": int(manifest.n_raw_rows.sum()),
        "seconds_after_cleaning": len(clean), "origins": len(feats), "eligible_origins": int(feats.eligible.sum()),
        "eligible_traces": int(elig.trace_id.nunique()),
    }
    (C.ROOT / "outputs" / "run_info.json").write_text(json.dumps(info, indent=2))

    pd.set_option("display.width", 160)
    print("\n", json.dumps(info, indent=2))
    print("\nCleaning totals:\n", log.drop(columns="trace_id").sum().to_string())
    print("\nCoverage:\n", cov.to_string(index=False))
    print("\nOuter folds:\n", fold_mix.to_string())
    print("\nPersistence baseline (Mbps, trace-weighted):\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
