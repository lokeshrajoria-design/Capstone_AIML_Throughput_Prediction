"""Step 2: parse, mark invalid values, and consolidate repeated timestamps to one row per second.

Rules (each one is counted in the exclusion log):
  - literal "-" -> missing
  - RSRP below the 3GPP reporting floor (the logger's -200 marker) -> missing
  - rows sharing a timestamp within a trace -> one row: numeric fields averaged,
    categorical fields take the last value, n_merged records how many rows were combined
  - zero throughput is kept: it can mean the application was idle, not that the link failed
"""
import pandas as pd

from . import config as C

NUMERIC = ["Longitude", "Latitude", "Speed", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "DL_bitrate", "UL_bitrate",
           "PINGAVG", "PINGMIN", "PINGMAX", "PINGSTDEV", "PINGLOSS", "NRxRSRP", "NRxRSRQ"]
CATEGORICAL = ["Operatorname", "CellID", "NetworkMode", "State", "CELLHEX", "NODEHEX", "LACHEX", "RAWCELLID"]


def load_trace(path):
    d = pd.read_csv(path, na_values=["-"], keep_default_na=False, dtype=str)
    d["t"] = pd.to_datetime(d.pop("Timestamp"), format="%Y.%m.%d_%H.%M.%S")
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="raise")
    return d


def clean_all(manifest):
    log, frames = [], []
    for tr in manifest.itertuples():
        d = load_trace(C.RAW_DIR / tr.trace_id)
        bad_rsrp = d.RSRP < C.RSRP_MIN_VALID
        d.loc[bad_rsrp, "RSRP"] = float("nan")

        agg = {c: "mean" for c in NUMERIC} | {c: "last" for c in CATEGORICAL}
        g = d.groupby("t", sort=True)
        s = g.agg(agg)
        s["n_merged"] = g.size()
        s = s.reset_index()
        s.insert(0, "trace_id", tr.trace_id)
        s["throughput_mbps"] = s.DL_bitrate / 1000.0
        frames.append(s)

        steps = s.t.diff().dt.total_seconds()
        log.append({"trace_id": tr.trace_id, "raw_rows": len(d), "rsrp_invalid": int(bad_rsrp.sum()),
                    "rows_merged_away": len(d) - len(s), "seconds_kept": len(s),
                    "gaps_over_1s": int((steps > 1).sum()), "missing_seconds": int((steps - 1).clip(lower=0).sum())})
    clean = pd.concat(frames, ignore_index=True)
    clean = clean.merge(manifest[["trace_id", "application_type", "mobility_condition", "recording_date"]],
                        on="trace_id", how="left", validate="many_to_one")
    return clean, pd.DataFrame(log)
