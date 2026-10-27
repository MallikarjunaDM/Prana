"""
splits.py — Leakage-free grouped train/test/calibration splits
───────────────────────────────────────────────────────────────
Primary: StratifiedGroupKFold (5 folds, groups=session_id, seed=42)
Also generates LOSO (Leave-One-Source-Out) fold assignments.
Validates that no session appears in both train and test.
Saves fold assignments to results/fold_assignments.csv.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURES_PARQUET, RESULTS_DIR, N_FOLDS, RANDOM_SEED, LABELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


META_COLS = ["session_id", "source", "label", "label_int", "start_time", "source_file"]


def _validate_no_leakage(fold_df: pd.DataFrame, fold_col: str = "fold"):
    """Assert no session_id appears in both train and test of any fold."""
    n_folds = fold_df[fold_col].max() + 1
    for f in range(n_folds):
        test_sessions  = set(fold_df[fold_df[fold_col] == f]["session_id"])
        train_sessions = set(fold_df[fold_df[fold_col] != f]["session_id"])
        overlap = test_sessions & train_sessions
        if overlap:
            raise AssertionError(
                f"Fold {f}: {len(overlap)} session(s) in both train and test! "
                f"Sample: {list(overlap)[:5]}"
            )
    logger.info(f"  ✓ Leakage check passed: no session overlap across {n_folds} folds.")


def make_splits(features_df: pd.DataFrame = None) -> dict:
    """
    Load features and produce grouped CV + LOSO fold assignments.

    Returns
    -------
    dict with keys:
      'features_df'     — full feature DataFrame
      'X_cols'          — list of feature column names
      'y'               — integer labels
      'groups'          — session_id array
      'cv_folds'        — list of (train_idx, test_idx) for primary CV
      'loso_folds'      — list of (train_idx, test_idx, held_out_source)
      'fold_df'         — DataFrame with fold assignments saved
    """
    if features_df is None:
        if not FEATURES_PARQUET.exists():
            raise FileNotFoundError(f"{FEATURES_PARQUET} not found. Run features.py first.")
        logger.info(f"Loading {FEATURES_PARQUET} ...")
        features_df = pd.read_parquet(FEATURES_PARQUET)

    logger.info(f"Features: {features_df.shape}")
    X_cols = [c for c in features_df.columns if c not in set(META_COLS)]
    logger.info(f"  {len(X_cols)} feature columns.")

    X      = features_df[X_cols].values
    y      = features_df["label_int"].values
    groups = features_df["session_id"].values

    # ── Primary: StratifiedGroupKFold ─────────────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Check we have enough unique groups per class
    min_groups = features_df.groupby(["label", "session_id"]).ngroups
    logger.info(f"  Unique label-session combinations: {min_groups}")

    cv_folds = list(sgkf.split(X, y, groups))
    logger.info(f"Primary CV: {len(cv_folds)} folds.")
    for i, (tr, te) in enumerate(cv_folds):
        logger.info(f"  Fold {i}: train={len(tr):,}  test={len(te):,}")

    # Build fold assignment column
    fold_assignment = np.full(len(features_df), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(cv_folds):
        fold_assignment[test_idx] = fold_idx

    fold_df = features_df[META_COLS].copy()
    fold_df["fold"] = fold_assignment

    # Validate leakage
    _validate_no_leakage(fold_df, fold_col="fold")

    # ── LOSO: Leave-One-Source-Out ────────────────────────────────────────────
    sources = features_df["source"].unique()
    loso_folds = []
    for src in sources:
        test_mask  = features_df["source"] == src
        train_mask = ~test_mask
        test_idx   = np.where(test_mask)[0]
        train_idx  = np.where(train_mask)[0]
        loso_folds.append((train_idx, test_idx, src))
        logger.info(f"  LOSO hold-out [{src}]: train={len(train_idx):,}  test={len(test_idx):,}")

    fold_df["loso_source"] = features_df["source"].values

    # Save
    out_path = RESULTS_DIR / "fold_assignments.csv"
    fold_df.to_csv(out_path, index=False)
    logger.info(f"✓ fold_assignments.csv → {out_path}")

    # Calibration set: 20% of each training fold (carved before test exposure)
    calib_assignment = np.full(len(features_df), -1, dtype=int)
    rng = np.random.default_rng(RANDOM_SEED)
    for fold_idx, (train_idx, _) in enumerate(cv_folds):
        n_calib = max(1, int(len(train_idx) * 0.2))
        calib_idx = rng.choice(train_idx, size=n_calib, replace=False)
        calib_assignment[calib_idx] = fold_idx
    fold_df["calib_fold"] = calib_assignment

    return {
        "features_df": features_df,
        "X_cols":      X_cols,
        "y":           y,
        "groups":      groups,
        "cv_folds":    cv_folds,
        "loso_folds":  loso_folds,
        "fold_df":     fold_df,
    }


if __name__ == "__main__":
    result = make_splits()
    fd = result["fold_df"]
    print(f"\nFold distribution:")
    print(fd["fold"].value_counts().sort_index())
    print(f"\nFeature count: {len(result['X_cols'])}")
