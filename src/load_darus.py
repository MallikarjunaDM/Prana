"""
load_darus.py — DaRUS motorcycle simulation dataset loader
──────────────────────────────────────────────────────────
Reads trainingData.csv, testData.csv, and individual control scenario CSVs.
Maps each scenario to one of: NORMAL, POTHOLE, BUMP, CRASH.
Produces the common internal schema.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DARUS_SOURCE, DARUS_DIR, DATA_PROCESSED,
    SCHEMA_COLS, SIGNAL_COLS, DARUS_DEV_FRAC, RANDOM_SEED
)

logger = logging.getLogger(__name__)

# ── Column mapping: DaRUS → common schema ──────────────────────────────────────
DARUS_COL_MAP = {
    "Time":               "timestamp",
    "MotoBody_linaccX":   "acc_x",
    "MotoBody_linaccY":   "acc_y",
    "MotoBody_linaccZ":   "acc_z",
    "MotoBody_angvelX":   "gyro_x",
    "MotoBody_angvelY":   "gyro_y",
    "MotoBody_angvelZ":   "gyro_z",
    "MotoBody_linaccRes": "acc_mag",   # pre-computed resultant — verified ≈ sqrt(x²+y²+z²)
    "CrashLabelML":       "_crash_flag",
}

# Control scenario filenames → PRANA label
# Curbstone → BUMP (user confirmed)
CONTROL_LABEL_MAP = {
    "Curbstone_5_Out.csv":         "BUMP",
    "ReverseCurbstone_5_Out.csv":  "BUMP",
    "Pothole_5_Out.csv":           "POTHOLE",
    "Speedbump_5_Out.csv":         "BUMP",
}
# Randomstate_* → NORMAL
RANDOMSTATE_LABEL = "NORMAL"
# ISO crash scenarios → CRASH
ISO_PREFIXES = ("ISO13232-", "Iso_")


def _infer_control_label(filename: str) -> str:
    """Return PRANA label for a control scenario CSV filename."""
    fname = Path(filename).name
    if fname in CONTROL_LABEL_MAP:
        return CONTROL_LABEL_MAP[fname]
    if fname.startswith("Randomstate_"):
        return RANDOMSTATE_LABEL
    for prefix in ISO_PREFIXES:
        if fname.startswith(prefix):
            return "CRASH"
    raise ValueError(f"Unknown control scenario file: {fname}")


def _load_csv_minimal(filepath: Path, session_id: str, label: str, source_file: str) -> pd.DataFrame:
    """Load one DaRUS CSV and return it in the common schema."""
    df = pd.read_csv(filepath, usecols=list(DARUS_COL_MAP.keys()),
                     dtype=float, on_bad_lines="skip")
    df.rename(columns=DARUS_COL_MAP, inplace=True)

    # If label is derived from CrashLabelML (main train/test files)
    if label is None:
        df["label"] = df["_crash_flag"].map({0.0: "NORMAL", 1.0: "CRASH"})
    else:
        df["label"] = label

    df.drop(columns=["_crash_flag"], errors="ignore", inplace=True)

    # Recompute acc_mag to ensure it matches schema (fallback if column missing)
    if "acc_mag" not in df.columns or df["acc_mag"].isna().all():
        df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)

    df["source"]      = "darus"
    df["session_id"]  = session_id
    df["source_file"] = source_file

    # Gyroscope units: DaRUS angular velocity is in rad/s — keep as-is
    # DaRUS linear acceleration in m/s² — keep as-is

    return df[SCHEMA_COLS]


def load_darus(dev_mode: bool = True) -> pd.DataFrame:
    """
    Load DaRUS dataset.

    Parameters
    ----------
    dev_mode : bool
        If True, subsample trainingData.csv to DARUS_DEV_FRAC.
        Control scenarios are always loaded in full (they're small).

    Returns
    -------
    pd.DataFrame  with columns matching SCHEMA_COLS
    """
    frames = []

    # ── 1. Training data ────────────────────────────────────────────────────────
    train_path = DARUS_SOURCE / "trainingData.csv"
    if train_path.exists():
        logger.info("Loading DaRUS trainingData.csv ...")
        df_train = pd.read_csv(train_path, usecols=list(DARUS_COL_MAP.keys()),
                                dtype=float, on_bad_lines="skip")
        if dev_mode and DARUS_DEV_FRAC < 1.0:
            df_train = df_train.sample(frac=DARUS_DEV_FRAC, random_state=RANDOM_SEED)
            logger.info(f"  Subsampled to {len(df_train):,} rows ({DARUS_DEV_FRAC*100:.0f}%).")

        df_train.rename(columns=DARUS_COL_MAP, inplace=True)
        df_train["label"]       = df_train["_crash_flag"].map({0.0: "NORMAL", 1.0: "CRASH"})
        df_train["source"]      = "darus"
        df_train["session_id"]  = "darus_train"
        df_train["source_file"] = "trainingData.csv"
        df_train.drop(columns=["_crash_flag"], errors="ignore", inplace=True)
        if "acc_mag" not in df_train.columns or df_train["acc_mag"].isna().all():
            df_train["acc_mag"] = np.sqrt(df_train["acc_x"]**2 + df_train["acc_y"]**2 + df_train["acc_z"]**2)
        frames.append(df_train[SCHEMA_COLS])
        logger.info(f"  trainingData: {len(df_train):,} rows, label dist: {df_train['label'].value_counts().to_dict()}")
    else:
        logger.warning(f"trainingData.csv not found at {train_path}")

    # ── 2. Test data ────────────────────────────────────────────────────────────
    test_path = DARUS_SOURCE / "testData.csv"
    if test_path.exists():
        logger.info("Loading DaRUS testData.csv ...")
        df_test = _load_csv_minimal(test_path, "darus_test", None, "testData.csv")
        frames.append(df_test)
        logger.info(f"  testData: {len(df_test):,} rows")
    else:
        logger.warning(f"testData.csv not found at {test_path}")

    # ── 3. Control scenarios ────────────────────────────────────────────────────
    control_files = list(DARUS_SOURCE.glob("*.csv"))
    skip_files = {"trainingData.csv", "testData.csv"}
    scenario_count = 0
    for csv_path in sorted(control_files):
        if csv_path.name in skip_files:
            continue
        try:
            header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        except Exception:
            continue
        if not any(c in header_cols for c in ["MotoBody_linaccX", "Time"]):
            continue
        try:
            label = _infer_control_label(csv_path.name)
            session_id = f"darus_ctrl_{csv_path.stem}"
            df_ctrl = _load_csv_minimal(csv_path, session_id, label, csv_path.name)
            frames.append(df_ctrl)
            scenario_count += 1
        except ValueError as e:
            logger.warning(str(e))

    logger.info(f"Loaded {scenario_count} DaRUS control scenario files.")

    if not frames:
        raise RuntimeError("No DaRUS data loaded. Check DARUS_SOURCE path.")

    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["acc_x", "acc_y", "acc_z", "label"], inplace=True)
    logger.info(f"DaRUS total: {len(combined):,} rows | Labels: {combined['label'].value_counts().to_dict()}")
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_darus(dev_mode=True)
    out = DATA_PROCESSED / "darus_raw.parquet"
    df.to_parquet(out, index=False)
    print(f"\n✓ Saved {len(df):,} rows → {out}")
    print(df["label"].value_counts())
    print(df.dtypes)
