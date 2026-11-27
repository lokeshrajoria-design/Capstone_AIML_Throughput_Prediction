"""RQ1 tables: within-trace Spearman correlations (primary) and the active-seconds check (exploratory).

Usage (after run_pipeline.py):  python run_rq1.py
"""
import pandas as pd

from tput import config as C
from tput.rq1 import summarise, trace_correlations


def main():
    clean = pd.read_parquet(C.INTERIM / "clean_1s.parquet")
    runs = {"": clean,
            "_exploratory_active_seconds": clean[(clean.State == "D") & (clean.throughput_mbps > 0)]}
    for suffix, data in runs.items():
        corr = trace_correlations(data)
        summ = summarise(corr)
        corr.to_csv(C.TABLES / f"rq1{suffix}_trace_correlations.csv", index=False)
        summ.to_csv(C.TABLES / f"rq1{suffix}_summary.csv", index=False)
        print(f"rq1{suffix}:\n", summ[["measurement", "eligible_traces", "mean_rho", "ci95_low", "ci95_high",
                                       "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
