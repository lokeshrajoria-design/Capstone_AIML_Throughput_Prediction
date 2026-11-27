"""Load the project data into a SQLite database (data/capstone.sqlite) and check it with sample queries.

Usage (from the analysis folder, after run_pipeline.py):  python build_database.py
Tables: traces (one row per recording session), records (cleaned one-second records),
features (model inputs and target per prediction origin), splits (fold assignment per trace).
The modeling scripts keep reading the Parquet files; the database is the queryable copy of the same data.
"""
import sqlite3
import time

import pandas as pd

from tput import config as C

DB = C.ROOT / "data" / "capstone.sqlite"
EXCLUDED = ["Longitude", "Latitude", "CellID", "RAWCELLID", "CELLHEX", "NODEHEX", "LACHEX"]  # privacy: not stored


def main():
    t0 = time.time()
    tables = {
        "traces": pd.read_csv(C.PROCESSED / "trace_manifest.csv"),
        "records": pd.read_parquet(C.INTERIM / "clean_1s.parquet"),
        "features": pd.read_parquet(C.PROCESSED / "features.parquet"),
        "splits": pd.read_csv(C.PROCESSED / "splits.csv"),
    }
    tables["records"] = tables["records"].drop(columns=[c for c in EXCLUDED if c in tables["records"].columns])
    for name, df in tables.items():  # SQLite has no datetime type: store timestamps as ISO text
        for c in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
            df[c] = df[c].astype(str)
        for c in df.select_dtypes(include=["category", "bool"]).columns:
            df[c] = df[c].astype(str) if str(df[c].dtype) == "category" else df[c].astype(int)
    DB.unlink(missing_ok=True)
    with sqlite3.connect(DB) as con:
        for name, df in tables.items():
            df.to_sql(name, con, index=False)
        con.execute("CREATE UNIQUE INDEX ix_traces ON traces(trace_id)")
        con.execute("CREATE INDEX ix_records ON records(trace_id)")
        con.execute("CREATE INDEX ix_features ON features(trace_id)")
        con.execute("CREATE UNIQUE INDEX ix_splits ON splits(trace_id)")
        con.execute("""CREATE VIEW v_group_summary AS
            SELECT t.application_type, t.mobility_condition, COUNT(DISTINCT t.trace_id) AS traces,
                   COUNT(*) AS seconds, ROUND(AVG(r.throughput_mbps), 2) AS mean_mbps
            FROM records r JOIN traces t USING (trace_id)
            GROUP BY t.application_type, t.mobility_condition""")
        counts = {n: con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in tables}
        summary = pd.read_sql("SELECT * FROM v_group_summary ORDER BY application_type, mobility_condition", con)
        schema = {n: [r[1] for r in con.execute(f"PRAGMA table_info({n})")] for n in tables}
    print("row counts:", counts)
    for n, cols in schema.items():
        print(f"{n}: {len(cols)} columns: {', '.join(cols[:12])}{' ...' if len(cols) > 12 else ''}")
    print(summary.to_string(index=False))
    print(f"wrote {DB} ({DB.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
