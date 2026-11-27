"""Step 1: verify the archive and build the trace manifest (one row per CSV trace)."""
import hashlib

import pandas as pd

from . import config as C


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest():
    digest = sha256(C.RAW_ZIP)
    if digest != C.RAW_ZIP_SHA256:
        raise RuntimeError(f"Archive checksum {digest} does not match the recorded input version")

    rows = []
    for path in sorted(C.RAW_DIR.rglob("*.csv")):
        rel = path.relative_to(C.RAW_DIR)
        parts = rel.parts                       # App / Static|Driving / [content /] file.csv
        ts = pd.read_csv(path, usecols=["Timestamp"]).Timestamp
        t = pd.to_datetime(ts, format="%Y.%m.%d_%H.%M.%S")
        rows.append({
            "trace_id": rel.as_posix(),
            "application_type": parts[0].replace("_", " "),
            "mobility_condition": parts[1],
            "content": parts[2] if len(parts) == 4 else "",
            "recording_date": t.min().date().isoformat(),
            "start_time": t.min(),
            "n_raw_rows": len(ts),
            "duration_s": int((t.max() - t.min()).total_seconds()),
        })
    m = pd.DataFrame(rows)
    if len(m) != C.EXPECTED_TRACES or m.n_raw_rows.sum() != C.EXPECTED_ROWS:
        raise RuntimeError(f"Expected {C.EXPECTED_TRACES} traces / {C.EXPECTED_ROWS} rows, "
                           f"found {len(m)} / {m.n_raw_rows.sum()}")
    m.attrs["sha256"] = digest
    return m
