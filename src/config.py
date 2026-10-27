# PRANA — Explainable AI for Motorcycle Crash Detection
# Common configuration shared across the pipeline

import os
from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # prana/
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# ── Raw dataset roots ───────────────────────────────────────────────────────────
DARUS_DIR = DATA_RAW / "darus"
ROADSENS_DIR = DATA_RAW / "roadsens"
KAGGLE_DIR = DATA_RAW / "kaggle_crash"

# ── Source dataset root (original downloads, never overwritten) ─────────────────
_HERE = Path(__file__).parent.parent.parent   # Downloads/PRANA/
DARUS_SOURCE = _HERE / "DaRUS motorcycle simulation dataset"
ROADSENS_SOURCE = _HERE / "RoadSens-4M"
KAGGLE_SOURCE = _HERE / "Smartphone IMU Road Accident Detection Dataset"

# ── Processed artifacts ─────────────────────────────────────────────────────────
WINDOWS_PARQUET = DATA_PROCESSED / "windows.parquet"
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"
HARMONISED_PARQUET = DATA_PROCESSED / "harmonised.parquet"

# ── Four class labels ───────────────────────────────────────────────────────────
LABELS = ["NORMAL", "POTHOLE", "BUMP", "CRASH"]
LABEL2INT = {l: i for i, l in enumerate(LABELS)}
INT2LABEL = {i: l for i, l in enumerate(LABELS)}

# ── Common internal schema columns ──────────────────────────────────────────────
SCHEMA_COLS = [
    "source", "session_id", "source_file", "timestamp",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "acc_mag",
    "label",
]
SIGNAL_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "acc_mag"]

# ── Window parameters ───────────────────────────────────────────────────────────
WINDOW_SIZE_S = 1.0        # 1 second
HOP_SIZE_S    = 0.5        # 50 % overlap
MIN_SAMPLES   = 20         # skip windows with fewer samples

# ── Split parameters ────────────────────────────────────────────────────────────
N_FOLDS     = 5
RANDOM_SEED = 42

# ── DaRUS development subsample fraction ────────────────────────────────────────
DARUS_DEV_FRAC = 0.20      # 20 % during development; set to 1.0 for final run

# ── Action policy defaults (tuned later on validation) ─────────────────────────
TAU_A  = 0.70   # ALERT threshold on P(CRASH)
TAU_C  = 0.40   # CONFIRM lower bound
DELTA  = 0.20   # minimum class-margin for confident CRASH

# ── Ensure output directories exist ────────────────────────────────────────────
for _d in [DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, DARUS_DIR, ROADSENS_DIR, KAGGLE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
