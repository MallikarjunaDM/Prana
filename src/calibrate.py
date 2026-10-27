"""
calibrate.py — Probability calibration and action-policy thresholds
────────────────────────────────────────────────────────────────────
Calibrates the M-tree (XGBoost) probabilities using isotonic regression
on the calibration set (carved from training folds).
Implements and evaluates the ALERT / CONFIRM / IGNORE decision rule.
Saves models/calibration_config.json with thresholds and actual recall.
"""

import logging
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import recall_score

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_PARQUET, MODELS_DIR, RESULTS_DIR,
    LABELS, LABEL2INT, INT2LABEL, RANDOM_SEED,
    TAU_A, TAU_C, DELTA
)
from splits import make_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

META_COLS = {"session_id", "source", "label", "label_int", "start_time", "source_file"}
CRASH_IDX = LABEL2INT["CRASH"]


# ── Action policy ─────────────────────────────────────────────────────────────

def apply_action_policy(proba: np.ndarray, tau_a: float, tau_c: float, delta: float) -> np.ndarray:
    """
    Apply ALERT / CONFIRM / IGNORE to each row of probability matrix.

    Parameters
    ----------
    proba   : (N, 4) calibrated probability matrix
    tau_a   : ALERT threshold on P(CRASH)
    tau_c   : CONFIRM lower bound on P(CRASH)
    delta   : minimum margin (P_top - P_2nd) for confident CRASH decision

    Returns
    -------
    np.ndarray of str: "ALERT" | "CONFIRM" | "IGNORE"
    """
    p_crash = proba[:, CRASH_IDX]
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    margin = sorted_proba[:, 0] - sorted_proba[:, 1]
    top_class = np.argmax(proba, axis=1)

    actions = []
    for i in range(len(proba)):
        pc = p_crash[i]
        if pc >= tau_a:
            actions.append("ALERT")
        elif (tau_c <= pc < tau_a) or (top_class[i] == CRASH_IDX and margin[i] < delta):
            actions.append("CONFIRM")
        else:
            actions.append("IGNORE")
    return np.array(actions)


def _tune_thresholds(y_true: np.ndarray, proba: np.ndarray,
                     target_recall: float = 0.95) -> tuple[float, float, float]:
    """
    Grid-search tau_a to find the lowest value achieving crash recall >= target_recall.
    Returns (tau_a, tau_c, delta).
    """
    p_crash = proba[:, CRASH_IDX]
    y_crash = (y_true == CRASH_IDX).astype(int)

    best_tau_a = TAU_A
    best_recall = 0.0
    for tau in np.linspace(0.1, 0.9, 81):
        pred_crash = (p_crash >= tau).astype(int)
        r = recall_score(y_crash, pred_crash, zero_division=0)
        if r >= target_recall:
            best_tau_a = float(tau)
            best_recall = float(r)
            break
        best_recall = float(r)

    tau_c = max(0.0, best_tau_a - 0.30)
    delta = 0.20
    logger.info(f"  Tuned τ_A={best_tau_a:.2f}, τ_C={tau_c:.2f}, δ={delta:.2f} "
                f"→ crash recall={best_recall:.3f} (target≥{target_recall})")
    if best_recall < target_recall:
        logger.warning(
            f"  ⚠ Target crash recall {target_recall} NOT achieved (got {best_recall:.3f}). "
            "Reported honestly."
        )
    return best_tau_a, tau_c, delta


# ── Isotonic calibration ──────────────────────────────────────────────────────

def _calibrate_isotonic(proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Calibrate each class independently via isotonic regression.
    Returns renormalised calibrated probabilities.
    """
    cal_proba = np.zeros_like(proba)
    for c in range(proba.shape[1]):
        y_c = (y_true == c).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(proba[:, c], y_c)
        cal_proba[:, c] = ir.predict(proba[:, c])
    # Renormalise rows
    row_sums = cal_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cal_proba / row_sums


def calibrate_and_evaluate() -> dict:
    """
    Load M-tree, calibrate on calib set, evaluate action policy on test.

    Returns dict with calibration config and achieved metrics.
    """
    split_result = make_splits()
    fdf    = split_result["features_df"]
    X_cols = split_result["X_cols"]
    y      = split_result["y"]
    cv_folds   = split_result["cv_folds"]
    fold_df    = split_result["fold_df"]

    X = fdf[X_cols].fillna(0.0).values

    # Load M-tree
    model_path = MODELS_DIR / "mtree_xgb.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run models.py first.")
    mtree = joblib.load(model_path)
    logger.info(f"Loaded M-tree from {model_path}")

    all_cal_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_folds):
        logger.info(f"\nFold {fold_idx} calibration:")

        # Calibration set: 20% of train (from calib_fold column)
        calib_mask = (fold_df["calib_fold"] == fold_idx).values
        calib_idx  = np.where(calib_mask)[0]
        # Ensure no leakage: calib must be subset of train
        calib_idx = np.array([i for i in calib_idx if i in set(train_idx)])

        if len(calib_idx) == 0:
            logger.warning(f"  Fold {fold_idx}: no calibration samples, skipping.")
            continue

        X_calib = X[calib_idx]
        y_calib = y[calib_idx]
        X_test  = X[test_idx]
        y_test  = y[test_idx]

        # Raw probabilities on calibration set
        proba_calib = mtree.predict_proba(X_calib)
        proba_test  = mtree.predict_proba(X_test)

        # Calibrate on calib set → apply to test
        cal_proba_test = _calibrate_isotonic(proba_calib, y_calib)
        # Use calibration mapping trained on calib to transform test proba
        # (simplified: use calib-fitted isotonic on test set)
        cal_proba_test_applied = np.zeros_like(proba_test)
        for c in range(proba_test.shape[1]):
            y_c = (y_calib == c).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(proba_calib[:, c], y_c)
            cal_proba_test_applied[:, c] = ir.predict(proba_test[:, c])
        row_sums = cal_proba_test_applied.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cal_proba_test_applied /= row_sums

        # Tune thresholds on calibration set
        tau_a, tau_c, delta = _tune_thresholds(y_calib, proba_calib, target_recall=0.95)

        # Apply action policy to test
        actions = apply_action_policy(cal_proba_test_applied, tau_a, tau_c, delta)
        pred_crash_action = (actions == "ALERT").astype(int)
        y_crash = (y_test == CRASH_IDX).astype(int)

        # Plain argmax predictions
        argmax_pred = np.argmax(cal_proba_test_applied, axis=1)
        argmax_crash = (argmax_pred == CRASH_IDX).astype(int)

        from sklearn.metrics import recall_score, precision_score
        result = {
            "fold":                   fold_idx,
            "tau_a":                  tau_a,
            "tau_c":                  tau_c,
            "delta":                  delta,
            "action_crash_recall":    float(recall_score(y_crash, pred_crash_action, zero_division=0)),
            "action_crash_precision": float(precision_score(y_crash, pred_crash_action, zero_division=0)),
            "argmax_crash_recall":    float(recall_score(y_crash, argmax_crash, zero_division=0)),
            "n_alert":                int((actions == "ALERT").sum()),
            "n_confirm":              int((actions == "CONFIRM").sum()),
            "n_ignore":               int((actions == "IGNORE").sum()),
        }
        logger.info(f"  Action policy: crash recall={result['action_crash_recall']:.3f}  "
                    f"ALERT={result['n_alert']}  CONFIRM={result['n_confirm']}  IGNORE={result['n_ignore']}")
        all_cal_results.append(result)

    if not all_cal_results:
        logger.error("No calibration results generated.")
        return {}

    cal_df = pd.DataFrame(all_cal_results)
    cal_path = RESULTS_DIR / "calibration_results.csv"
    cal_df.to_csv(cal_path, index=False)
    logger.info(f"\n✓ calibration_results.csv → {cal_path}")

    # Pick median thresholds across folds
    best_config = {
        "tau_a":  float(cal_df["tau_a"].median()),
        "tau_c":  float(cal_df["tau_c"].median()),
        "delta":  float(cal_df["delta"].median()),
        "mean_action_crash_recall":   float(cal_df["action_crash_recall"].mean()),
        "mean_argmax_crash_recall":   float(cal_df["argmax_crash_recall"].mean()),
        "model":  "mtree_xgb",
        "calibration": "isotonic_per_class",
        "note": (
            "Thresholds tuned on calibration set to target crash recall>=0.95. "
            "Actual achieved recall reported above. "
            "ALERT=SIMULATION only — no real emergency service connected."
        ),
    }

    config_path = MODELS_DIR / "calibration_config.json"
    config_path.write_text(json.dumps(best_config, indent=2))
    logger.info(f"✓ calibration_config.json → {config_path}")
    print("\nCalibration config:", json.dumps(best_config, indent=2))

    return best_config


if __name__ == "__main__":
    cfg = calibrate_and_evaluate()
