"""
evaluate.py — Full evaluation of all models
────────────────────────────────────────────
Loads all trained models; evaluates on the held-out fold;
produces comprehensive metrics including FCAR, matched-recall comparisons,
per-class breakdown, confusion matrices, and figures.
Saves results/evaluation_results.csv and figures/confusion_matrix.png.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    recall_score, precision_score, f1_score
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_PARQUET, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    LABELS, LABEL2INT, INT2LABEL, RANDOM_SEED
)
from splits    import make_splits
from calibrate import apply_action_policy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

META_COLS = {"session_id", "source", "label", "label_int", "start_time", "source_file"}
CRASH_IDX = LABEL2INT["CRASH"]


def _fcar_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """False Crash Alarm Rate overall and per non-crash class."""
    result = {}
    non_crash = (y_true != CRASH_IDX)
    if non_crash.sum() > 0:
        result["fcar_overall"] = float((y_pred[non_crash] == CRASH_IDX).mean())
    for lab in ["NORMAL", "POTHOLE", "BUMP"]:
        idx = LABEL2INT[lab]
        mask = (y_true == idx)
        if mask.sum() > 0:
            result[f"fcar_{lab.lower()}"] = float((y_pred[mask] == CRASH_IDX).mean())
        else:
            result[f"fcar_{lab.lower()}"] = None
    return result


def _plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"  ✓ Saved confusion matrix → {save_path}")


def evaluate_all() -> pd.DataFrame:
    """Evaluate all saved models and save results."""
    split_result = make_splits()
    fdf    = split_result["features_df"]
    X_cols = split_result["X_cols"]
    y      = split_result["y"]
    cv_folds   = split_result["cv_folds"]
    loso_folds = split_result["loso_folds"]

    X = fdf[X_cols].fillna(0.0).values

    # Load calibration config for action policy
    import json
    calib_cfg_path = MODELS_DIR / "calibration_config.json"
    if calib_cfg_path.exists():
        calib_cfg = json.loads(calib_cfg_path.read_text())
        tau_a = calib_cfg.get("tau_a", 0.70)
        tau_c = calib_cfg.get("tau_c", 0.40)
        delta = calib_cfg.get("delta", 0.20)
    else:
        logger.warning("calibration_config.json not found; using defaults.")
        tau_a, tau_c, delta = 0.70, 0.40, 0.20

    # Use fold 0 test for primary evaluation
    train_idx, test_idx = cv_folds[0]
    X_test  = X[test_idx]
    y_test  = y[test_idx]

    all_results = []

    # ── M-tree XGBoost ────────────────────────────────────────────────────────
    mtree_path = MODELS_DIR / "mtree_xgb.joblib"
    if mtree_path.exists():
        mtree = joblib.load(mtree_path)
        proba = mtree.predict_proba(X_test)
        pred  = mtree.predict(X_test)

        report = classification_report(y_test, pred, labels=list(range(len(LABELS))),
                                       target_names=LABELS,
                                       output_dict=True, zero_division=0)
        fcar = _fcar_breakdown(y_test, pred)

        # Crash recall at different thresholds (matched-recall table)
        crash_recalls, precisions, thresholds = [], [], []
        for t in np.linspace(0.1, 0.95, 50):
            pred_t = (proba[:, CRASH_IDX] >= t).astype(int)
            y_bin  = (y_test == CRASH_IDX).astype(int)
            crash_recalls.append(recall_score(y_bin, pred_t, zero_division=0))
            precisions.append(precision_score(y_bin, pred_t, zero_division=0))
            thresholds.append(t)

        matched_df = pd.DataFrame({
            "threshold": thresholds,
            "crash_recall": crash_recalls,
            "crash_precision": precisions,
        })
        matched_df.to_csv(RESULTS_DIR / "matched_recall_mtree.csv", index=False)

        # Action policy evaluation
        actions = apply_action_policy(proba, tau_a, tau_c, delta)
        alert_pred  = (actions == "ALERT").astype(int)
        y_crash_bin = (y_test == CRASH_IDX).astype(int)

        r = {
            "model":           "mtree_xgb",
            "accuracy":        report["accuracy"],
            "macro_f1":        report["macro avg"]["f1-score"],
            "crash_recall":    report.get("CRASH", {}).get("recall", None),
            "crash_precision": report.get("CRASH", {}).get("precision", None),
            "alert_recall":    float(recall_score(y_crash_bin, alert_pred, zero_division=0)),
            **fcar,
        }
        try:
            y_bin_mat = label_binarize(y_test, classes=list(range(len(LABELS))))
            r["roc_auc_macro"] = float(roc_auc_score(y_bin_mat, proba, multi_class="ovr", average="macro"))
        except Exception:
            r["roc_auc_macro"] = None
        all_results.append(r)

        _plot_confusion(y_test, pred, "M-tree XGBoost — Fold 0",
                        FIGURES_DIR / "confusion_mtree.png")
        logger.info(f"M-tree: crash_recall={r['crash_recall']}, FCAR={r.get('fcar_overall')}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_path = MODELS_DIR / "rf_model.joblib"
    if rf_path.exists():
        rf = joblib.load(rf_path)
        rf_pred  = rf.predict(X_test)
        rf_proba = rf.predict_proba(X_test)
        if rf_proba.shape[1] < len(LABELS):
            pad = np.zeros((len(rf_proba), len(LABELS)))
            for i, cls in enumerate(rf.classes_):
                pad[:, cls] = rf_proba[:, i]
            rf_proba = pad

        rf_report = classification_report(y_test, rf_pred, labels=list(range(len(LABELS))),
                                          target_names=LABELS,
                                          output_dict=True, zero_division=0)
        rf_fcar = _fcar_breakdown(y_test, rf_pred)
        r_rf = {
            "model":           "random_forest",
            "accuracy":        rf_report["accuracy"],
            "macro_f1":        rf_report["macro avg"]["f1-score"],
            "crash_recall":    rf_report.get("CRASH", {}).get("recall", None),
            "crash_precision": rf_report.get("CRASH", {}).get("precision", None),
            "alert_recall":    None,
            **rf_fcar,
        }
        all_results.append(r_rf)
        _plot_confusion(y_test, rf_pred, "Random Forest — Fold 0",
                        FIGURES_DIR / "confusion_rf.png")

    # ── LOSO evaluation (M-tree) ─────────────────────────────────────────────
    if mtree_path.exists():
        for loso_train, loso_test, held_src in loso_folds:
            if len(loso_test) == 0:
                continue
            loso_proba = mtree.predict_proba(X[loso_test])
            loso_pred  = mtree.predict(X[loso_test])
            y_loso     = y[loso_test]
            loso_report = classification_report(y_loso, loso_pred, labels=list(range(len(LABELS))),
                                                target_names=LABELS,
                                                output_dict=True, zero_division=0)
            loso_fcar = _fcar_breakdown(y_loso, loso_pred)
            r_loso = {
                "model":           f"mtree_loso_{held_src}",
                "accuracy":        loso_report["accuracy"],
                "macro_f1":        loso_report["macro avg"]["f1-score"],
                "crash_recall":    loso_report.get("CRASH", {}).get("recall", None),
                "crash_precision": loso_report.get("CRASH", {}).get("precision", None),
                "alert_recall":    None,
                **loso_fcar,
            }
            all_results.append(r_loso)
            logger.info(f"LOSO [{held_src}]: crash_recall={r_loso['crash_recall']}, "
                        f"FCAR={r_loso.get('fcar_overall')}")

    # Save
    eval_df = pd.DataFrame(all_results)
    eval_path = RESULTS_DIR / "evaluation_results.csv"
    eval_df.to_csv(eval_path, index=False)
    logger.info(f"\n✓ evaluation_results.csv → {eval_path}")
    print("\n" + eval_df[["model", "crash_recall", "macro_f1", "fcar_overall"]].to_string())
    return eval_df


if __name__ == "__main__":
    evaluate_all()
