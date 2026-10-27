"""
make_windows.py — Sliding-window segmentation
──────────────────────────────────────────────
Reads harmonised.parquet; segments each session into fixed-length
windows with 50% overlap; assigns a majority-vote label; writes
data/processed/windows.parquet.

Window parameters (from config):
  WINDOW_SIZE_S = 1.0 s
  HOP_SIZE_S    = 0.5 s (50% overlap)
  MIN_SAMPLES   = 20
"""

import logging
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    HARMONISED_PARQUET, WINDOWS_PARQUET,
    WINDOW_SIZE_S, HOP_SIZE_S, MIN_SAMPLES,
    SIGNAL_COLS, LABELS, LABEL2INT
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _majority_label(labels: pd.Series) -> str:
    """Return the most common label in a window."""
    return Counter(labels).most_common(1)[0][0]


def _estimate_sr(timestamps: np.ndarray) -> float:
    """Estimate sampling rate from timestamps array."""
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 100.0   # fallback
    return 1.0 / np.median(diffs)


TARGET_SAMPLES = 100   # 1.0 second at 100 Hz normalized uniform length


def segment_session(session_df: pd.DataFrame) -> list[dict]:
    """
    Slide a window over one session and return a list of window records.
    Each record contains resampled uniform signal values (TARGET_SAMPLES) and metadata.
    """
    session_df = session_df.sort_values("timestamp").reset_index(drop=True)
    ts = session_df["timestamp"].values
    sr = _estimate_sr(ts)

    win_samples = max(int(round(WINDOW_SIZE_S * sr)), MIN_SAMPLES)
    hop_samples = max(int(round(HOP_SIZE_S  * sr)), 1)

    records = []
    n = len(session_df)
    start = 0
    t_target = np.linspace(0, 1, TARGET_SAMPLES)

    while start + win_samples <= n:
        end = start + win_samples
        window = session_df.iloc[start:end]

        if len(window) < MIN_SAMPLES:
            start += hop_samples
            continue

        label = _majority_label(window["label"])

        rec = {
            "session_id":  session_df["session_id"].iloc[0],
            "source":      session_df["source"].iloc[0],
            "source_file": session_df["source_file"].iloc[0],
            "label":       label,
            "label_int":   LABEL2INT[label],
            "start_time":  float(ts[start]),
            "end_time":    float(ts[end - 1]),
            "n_samples":   int(len(window)),
            "sampling_hz": round(sr, 1),
        }

        # Resample each signal channel uniformly to TARGET_SAMPLES
        t_orig = np.linspace(0, 1, len(window))
        for col in SIGNAL_COLS:
            vals = window[col].fillna(0.0).values
            if len(vals) == TARGET_SAMPLES:
                resampled = vals
            else:
                resampled = np.interp(t_target, t_orig, vals)
            for i, v in enumerate(resampled):
                rec[f"{col}_{i}"] = float(v)

        records.append(rec)
        start += hop_samples

    return records


def make_windows() -> pd.DataFrame:
    """
    Load harmonised data and produce the windows Parquet.

    Returns
    -------
    pd.DataFrame where each row is a labelled window.
    """
    if not HARMONISED_PARQUET.exists():
        raise FileNotFoundError(
            f"{HARMONISED_PARQUET} not found. Run harmonise.py first."
        )

    logger.info(f"Loading {HARMONISED_PARQUET} ...")
    df = pd.read_parquet(HARMONISED_PARQUET)
    logger.info(f"  {len(df):,} rows, {df['session_id'].nunique()} sessions.")

    all_records = []
    sessions = df["session_id"].unique()
    logger.info(f"Segmenting {len(sessions)} sessions ...")

    for i, sid in enumerate(sessions):
        sdf = df[df["session_id"] == sid]
        recs = segment_session(sdf)
        all_records.extend(recs)
        if (i + 1) % 40 == 0 or (i + 1) == len(sessions):
            logger.info(f"  {i+1}/{len(sessions)} sessions done → {len(all_records):,} windows so far")

    if not all_records:
        raise RuntimeError("No windows were generated. Check harmonised data.")

    windows_df = pd.DataFrame(all_records)

    # Quick label-distribution summary
    label_dist = windows_df["label"].value_counts().to_dict()
    logger.info(f"\nWindow label distribution: {label_dist}")
    logger.info(f"Total windows: {len(windows_df):,}")

    windows_df.to_parquet(WINDOWS_PARQUET, index=False)
    logger.info(f"✓ windows.parquet saved → {WINDOWS_PARQUET}")
    logger.info(f"  Shape: {windows_df.shape}")

    return windows_df


if __name__ == "__main__":
    df = make_windows()
    print(f"\nShape: {df.shape}")
    print("Labels:\n", df["label"].value_counts())
    print("Samples per source:\n", df.groupby("source")["label"].value_counts())
