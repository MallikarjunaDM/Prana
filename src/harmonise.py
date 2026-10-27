"""
harmonise.py — Merge all sources into one harmonised Parquet
────────────────────────────────────────────────────────────
Loads DaRUS, RoadSens (if available), and Kaggle; concatenates them
in the common schema; applies a final data-quality pass; and writes
data/processed/harmonised.parquet.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED, HARMONISED_PARQUET, SCHEMA_COLS,
    SIGNAL_COLS, LABELS, RANDOM_SEED
)
from load_darus    import load_darus
from load_roadsens import load_roadsens
from load_kaggle   import load_kaggle

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def harmonise(dev_mode: bool = True) -> pd.DataFrame:
    """
    Load and merge all three datasets.

    Parameters
    ----------
    dev_mode : bool
        Passed to load_darus for 20% subsampling.

    Returns
    -------
    pd.DataFrame with SCHEMA_COLS + 'label_int' column.
    """
    frames = []

    # ── DaRUS ──
    logger.info("Loading DaRUS ...")
    df_darus = load_darus(dev_mode=dev_mode)
    frames.append(df_darus)

    # ── RoadSens ──
    logger.info("Loading RoadSens ...")
    df_rs = load_roadsens()
    if len(df_rs) > 0:
        frames.append(df_rs)
    else:
        logger.warning("RoadSens empty — proceeding without it.")

    # ── Kaggle ──
    logger.info("Loading Kaggle ...")
    df_kg = load_kaggle()
    if len(df_kg) > 0:
        frames.append(df_kg)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined raw: {len(combined):,} rows from {combined['source'].nunique()} sources.")

    # ── Quality pass ──
    # 1. Drop rows missing key acc channels
    before = len(combined)
    combined.dropna(subset=["acc_x", "acc_y", "acc_z", "label"], inplace=True)
    logger.info(f"  Dropped {before - len(combined):,} rows missing acc or label.")

    # 2. Clip extreme acceleration outliers (beyond ±200 m/s²) — simulation artefacts
    for c in ["acc_x", "acc_y", "acc_z", "acc_mag"]:
        combined[c] = combined[c].clip(-200, 200)

    # 3. Ensure label is valid
    combined = combined[combined["label"].isin(LABELS)].copy()

    # 4. Reset index
    combined.reset_index(drop=True, inplace=True)

    # 5. Add integer label column for ML
    label2int = {l: i for i, l in enumerate(LABELS)}
    combined["label_int"] = combined["label"].map(label2int)

    logger.info(f"Harmonised: {len(combined):,} rows | "
                f"Labels: {combined['label'].value_counts().to_dict()}")
    logger.info(f"Sessions: {combined['session_id'].nunique()}")
    logger.info(f"Sources: {combined['source'].value_counts().to_dict()}")

    # Save
    combined.to_parquet(HARMONISED_PARQUET, index=False)
    logger.info(f"✓ Saved → {HARMONISED_PARQUET}")

    return combined


if __name__ == "__main__":
    df = harmonise(dev_mode=True)
    print(f"\nShape: {df.shape}")
    print(df.groupby(["source", "label"]).size().to_frame("count"))
