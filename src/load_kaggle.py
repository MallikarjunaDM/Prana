"""
load_kaggle.py — Smartphone IMU Road Accident Detection Dataset loader
───────────────────────────────────────────────────────────────────────
Schema: Timestamp, Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z,
        Speed_kmh, Latitude, Longitude, Motion_Intensity, Crash_Label

Limitations (documented):
 - Timestamps are synthetic (start 2026-01-01, 1-second intervals)
 - Appears to be a generated/simulated dataset, not real crash recordings
 - Sensor is smartphone, not motorcycle-mounted sensor
 - Only CRASH and NORMAL classes available (no POTHOLE/BUMP)
Session grouping: 100-row blocks → 80 synthetic session IDs
"""

import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import KAGGLE_SOURCE, DATA_PROCESSED, SCHEMA_COLS, RANDOM_SEED

logger = logging.getLogger(__name__)

KAGGLE_CSV = KAGGLE_SOURCE / "road_accident_imu_dataset_8000.csv"

# Session grouping: rows per artificial session block
ROWS_PER_SESSION = 100   # 8000 / 100 = 80 session groups

# Column mapping
COL_MAP = {
    "Acc_X":  "acc_x",
    "Acc_Y":  "acc_y",
    "Acc_Z":  "acc_z",
    "Gyro_X": "gyro_x",
    "Gyro_Y": "gyro_y",
    "Gyro_Z": "gyro_z",
}
LABEL_COL     = "Crash_Label"
TIMESTAMP_COL = "Timestamp"


def load_kaggle() -> pd.DataFrame:
    """
    Load the Kaggle smartphone IMU crash dataset.

    Returns
    -------
    pd.DataFrame with SCHEMA_COLS
    """
    if not KAGGLE_CSV.exists():
        logger.warning(f"Kaggle CSV not found: {KAGGLE_CSV}")
        return pd.DataFrame(columns=SCHEMA_COLS)

    logger.info(f"Loading Kaggle IMU dataset: {KAGGLE_CSV.name} ...")
    df = pd.read_csv(KAGGLE_CSV, on_bad_lines="skip")
    logger.info(f"  Raw rows: {len(df):,} | Columns: {list(df.columns)}")

    # Rename signal columns
    df.rename(columns=COL_MAP, inplace=True)

    # Label mapping: 0→NORMAL, 1→CRASH
    df["label"] = df[LABEL_COL].map({0: "NORMAL", 1: "CRASH"})
    df.dropna(subset=["label"], inplace=True)

    # Compute acc_mag
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)

    # Convert timestamp to seconds-from-start
    try:
        ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        t0 = ts.iloc[0]
        df["timestamp"] = (ts - t0).dt.total_seconds()
    except Exception:
        df["timestamp"] = np.arange(len(df), dtype=float)

    # Artificial session grouping: every ROWS_PER_SESSION rows = one session
    df = df.reset_index(drop=True)
    df["session_id"] = [f"kaggle_s{i // ROWS_PER_SESSION:03d}" for i in range(len(df))]

    df["source"]      = "kaggle"
    df["source_file"] = KAGGLE_CSV.name

    out = df[SCHEMA_COLS].copy()
    out.dropna(subset=["acc_x", "acc_y", "acc_z"], inplace=True)
    logger.info(f"Kaggle total: {len(out):,} rows | Labels: {out['label'].value_counts().to_dict()}")
    logger.info("⚠ Limitation: Kaggle data uses synthetic timestamps and appears generated/simulated.")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_kaggle()
    out = DATA_PROCESSED / "kaggle_raw.parquet"
    df.to_parquet(out, index=False)
    print(f"\n✓ Saved {len(df):,} rows → {out}")
    print(df["label"].value_counts())
    print(df.dtypes)
