"""Paths and fixed study settings (synopsis sections 4-6). Change these only with a documented reason."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "5G-production-dataset.zip"
RAW_DIR = ROOT / "data" / "raw" / "5G-production-dataset"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

# Expected input (verified in the preliminary audit). The pipeline stops if the archive differs.
RAW_ZIP_SHA256 = "abc729d696b5e0ba34a9c6ed35a851f2868d2b5071be5efb8a49691c23bb1a9b"
EXPECTED_TRACES = 83
EXPECTED_ROWS = 188_711

SEED = 640
HORIZON_S = 5                 # predict throughput exactly 5 s after the origin
LAGS_S = (0, 1, 2, 5)         # throughput_lag_k = throughput exactly k s before the origin
WINDOW_S = 10                 # throughput_mean_10s / throughput_sd_10s over seconds t-9 .. t
# Eligibility rule. The target and lag_0 are always exact. ~10 % of seconds are missing from the logs,
# so demanding all 10 window seconds and every lag keeps only 19 % of origins (4-31 % by group).
# Default: window needs >= 7 observed seconds; missing lag_1/2/5 are imputed with indicators.
# Strict rule (sensitivity analysis): WINDOW_MIN_OBS = 10 and REQUIRE_ALL_LAGS = True.
WINDOW_MIN_OBS = 7
REQUIRE_ALL_LAGS = False
OUTER_FOLDS = 5
INNER_FOLDS = 3
MIN_PAIRS_RQ1 = 30

# 3GPP TS 38.133 / 36.133 RSRP reporting floor; the logger writes -200 as a "no value" marker.
RSRP_MIN_VALID = -156

RADIO = ["RSRP", "RSRQ", "SNR", "CQI"]
HISTORY = [f"throughput_lag_{k}" for k in LAGS_S] + ["throughput_mean_10s", "throughput_sd_10s"]
CONTEXT_NUM = ["Speed"]
CONTEXT_CAT = ["NetworkMode", "application_type"]
BASE_FEATURES = RADIO + HISTORY
CONTEXT_FEATURES = BASE_FEATURES + CONTEXT_NUM + CONTEXT_CAT
TARGET = "target_throughput_5s"
