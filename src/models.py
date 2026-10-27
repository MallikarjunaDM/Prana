"""
models.py — Model training and persistence
──────────────────────────────────────────
Trains in order:
  1. Threshold baseline (acc_mag peak)
  2. Binary XGBoost (CRASH vs NOT-CRASH)
  3. Four-class XGBoost (M-tree)
  4. Random Forest (additional baseline)
  (5. 1D CNN — deferred, see note below)

All models saved to models/  as versioned .joblib files.
Results saved to results/model_metrics.csv.
"""

import logging
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_PARQUET, MODELS_DIR, RESULTS_DIR,
    LABELS, LABEL2INT, INT2LABEL, RANDOM_SEED, N_FOLDS
)
from splits import make_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

META_COLS = {"session_id", "source", "label", "label_int", "start_time", "source_file"}


# ── Threshold Baseline ─────────────────────────────────────────────────────────

class ThresholdClassifier:
    """Simple rule: if peak acc_mag > threshold → CRASH else NORMAL."""
    def __init__(self, threshold: float = 25.0):
        self.threshold = threshold

    def predict(self, X_acc_mag_peaks: np.ndarray) -> np.ndarray:
        return np.where(X_acc_mag_peaks > self.threshold,
                        LABEL2INT["CRASH"], LABEL2INT["NORMAL"])

    def predict_proba(self, X_acc_mag_peaks: np.ndarray) -> np.ndarray:
        """Returns [P(NORMAL), P(POTHOLE), P(BUMP), P(CRASH)] synthetic probs."""
        crash = (X_acc_mag_peaks > self.threshold).astype(float)
        out = np.zeros((len(crash), len(LABELS)))
        out[:, LABEL2INT["CRASH"]]  = crash
        out[:, LABEL2INT["NORMAL"]] = 1 - crash
        return out


def _get_acc_mag_peak(features_df: pd.DataFrame) -> np.ndarray:
    """Extract acc_mag_max as threshold feature."""
    if "acc_mag_max" in features_df.columns:
        return features_df["acc_mag_max"].values
    # Fallback: compute from raw max
    if "acc_mag_rms" in features_df.columns:
        return features_df["acc_mag_rms"].values
    return np.zeros(len(features_df))


# ── XGBoost helper ─────────────────────────────────────────────────────────────

def _xgb_params_4class():
    return {
        "objective":        "multi:softprob",
        "num_class":        len(LABELS),
        "max_depth":        5,
        "n_estimators":     300,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     RANDOM_SEED,
        "use_label_encoder": False,
        "eval_metric":      "mlogloss",
        "n_jobs":           -1,
        "verbosity":        0,
    }


def _xgb_params_binary():
    return {
        "objective":        "binary:logistic",
        "max_depth":        5,
        "n_estimators":     300,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     RANDOM_SEED,
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "n_jobs":           -1,
        "verbosity":        0,
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _false_crash_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """FCAR overall and per non-crash class."""
    crash_idx = LABEL2INT["CRASH"]
    non_crash_mask = (y_true != crash_idx)
    if non_crash_mask.sum() == 0:
        return {"fcar_overall": None}

    fcar = float((y_pred[non_crash_mask] == crash_idx).mean())
    result = {"fcar_overall": fcar}
    for lab in ["NORMAL", "POTHOLE", "BUMP"]:
        mask = (y_true == LABEL2INT[lab])
        if mask.sum() > 0:
            result[f"fcar_{lab.lower()}"] = float((y_pred[mask] == crash_idx).mean())
        else:
            result[f"fcar_{lab.lower()}"] = None
    return result


def _eval_model(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: np.ndarray = None, classes=None) -> dict:
    """Compute and log metrics. Returns metrics dict."""
    if classes is None:
        classes = LABELS
    report = classification_report(y_true, y_pred,
                                   labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    fcar = _false_crash_alarm_rate(y_true, y_pred)
    crash_recall = report.get("CRASH", {}).get("recall", None)

    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            y_bin = label_binarize(y_true, classes=list(range(len(classes))))
            if y_bin.shape[1] == 2:
                y_bin = y_bin[:, 1]
                roc_auc = float(roc_auc_score(y_bin, y_proba[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
        except Exception as e:
            logger.warning(f"ROC-AUC failed: {e}")

    metrics = {
        "model":          name,
        "crash_recall":   crash_recall,
        "macro_f1":       report.get("macro avg", {}).get("f1-score", None),
        "accuracy":       report.get("accuracy", None),
        "roc_auc":        roc_auc,
        **fcar,
    }
    logger.info(f"\n── {name} ──")
    logger.info(f"  Crash Recall: {crash_recall}")
    logger.info(f"  FCAR overall: {fcar.get('fcar_overall')}")
    logger.info(f"  Macro F1: {metrics['macro_f1']}")
    return metrics


# ── Main training function ─────────────────────────────────────────────────────

def train_all(dev_mode: bool = True) -> dict:
    """
    Train all models and return trained artifacts.

    Returns dict with keys: 'threshold', 'binary_xgb', 'mtree_xgb', 'rf'
    """
    # Load features and splits
    split_result = make_splits()
    fdf   = split_result["features_df"]
    X_cols = split_result["X_cols"]
    y      = split_result["y"]
    cv_folds = split_result["cv_folds"]
    loso_folds = split_result["loso_folds"]

    X = fdf[X_cols].fillna(0.0).values
    acc_mag_peak = _get_acc_mag_peak(fdf)

    all_metrics = []

    # Use fold 0 for quick single-fold evaluation during dev
    fold_idx = 0
    train_idx, test_idx = cv_folds[fold_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    peak_test = acc_mag_peak[test_idx]

    logger.info(f"Training on fold {fold_idx}: {len(train_idx):,} train, {len(test_idx):,} test")
    logger.info(f"Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"Test  label dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ── 1. Threshold baseline ──────────────────────────────────────────────────
    logger.info("\n[1/4] Threshold Baseline ...")
    thresh_clf = ThresholdClassifier(threshold=25.0)
    thresh_pred = thresh_clf.predict(peak_test)
    thresh_proba = thresh_clf.predict_proba(peak_test)
    # Map to 4-class by treating everything non-crash as NORMAL
    thresh_pred_4 = np.where(thresh_pred == LABEL2INT["CRASH"],
                             LABEL2INT["CRASH"], LABEL2INT["NORMAL"])
    metrics_thresh = _eval_model("threshold_baseline", y_test, thresh_pred_4, thresh_proba)
    all_metrics.append(metrics_thresh)
    joblib.dump(thresh_clf, MODELS_DIR / "threshold_clf.joblib")

    # ── 2. Binary XGBoost ─────────────────────────────────────────────────────
    logger.info("\n[2/4] Binary XGBoost (CRASH vs NOT-CRASH) ...")
    y_train_bin = (y_train == LABEL2INT["CRASH"]).astype(int)
    y_test_bin  = (y_test  == LABEL2INT["CRASH"]).astype(int)

    bin_xgb = xgb.XGBClassifier(**_xgb_params_binary())
    bin_xgb.fit(X_train, y_train_bin,
                eval_set=[(X_test, y_test_bin)],
                verbose=False)
    bin_pred  = bin_xgb.predict(X_test)
    bin_proba = bin_xgb.predict_proba(X_test)

    # Map back to 4-class int for unified reporting
    bin_pred_4 = np.where(bin_pred == 1, LABEL2INT["CRASH"], LABEL2INT["NORMAL"])
    bin_proba_4 = np.zeros((len(bin_proba), 4))
    bin_proba_4[:, LABEL2INT["CRASH"]]  = bin_proba[:, 1]
    bin_proba_4[:, LABEL2INT["NORMAL"]] = bin_proba[:, 0]

    metrics_bin = _eval_model("binary_xgb", y_test, bin_pred_4, bin_proba_4)
    all_metrics.append(metrics_bin)
    joblib.dump(bin_xgb, MODELS_DIR / "binary_xgb.joblib")
    logger.info(f"  Saved → models/binary_xgb.joblib")

    # ── 3. Four-class XGBoost (M-tree) ────────────────────────────────────────
    logger.info("\n[3/4] Four-Class XGBoost M-tree ...")
    mtree = xgb.XGBClassifier(**_xgb_params_4class())
    mtree.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    mtree_pred  = mtree.predict(X_test)
    mtree_proba = mtree.predict_proba(X_test)

    metrics_mtree = _eval_model("mtree_xgb_4class", y_test, mtree_pred, mtree_proba)
    all_metrics.append(metrics_mtree)
    joblib.dump(mtree, MODELS_DIR / "mtree_xgb.joblib")
    logger.info(f"  Saved → models/mtree_xgb.joblib")

    # ── 4. Random Forest ──────────────────────────────────────────────────────
    logger.info("\n[4/4] Random Forest (4-class baseline) ...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    # RF may have fewer classes — pad if needed
    if rf_proba.shape[1] < len(LABELS):
        pad = np.zeros((len(rf_proba), len(LABELS)))
        for i, cls in enumerate(rf.classes_):
            pad[:, cls] = rf_proba[:, i]
        rf_proba = pad

    metrics_rf = _eval_model("random_forest_4class", y_test, rf_pred, rf_proba)
    all_metrics.append(metrics_rf)
    joblib.dump(rf, MODELS_DIR / "rf_model.joblib")
    logger.info(f"  Saved → models/rf_model.joblib")

    # ── LOSO evaluation (M-tree) ──────────────────────────────────────────────
    logger.info("\nLOSO evaluation (M-tree) ...")
    for loso_train_idx, loso_test_idx, held_src in loso_folds:
        if len(loso_test_idx) == 0:
            continue
        loso_model = xgb.XGBClassifier(**_xgb_params_4class())
        loso_model.fit(X[loso_train_idx], y[loso_train_idx], verbose=False)
        loso_pred  = loso_model.predict(X[loso_test_idx])
        loso_proba = loso_model.predict_proba(X[loso_test_idx])
        m = _eval_model(f"mtree_loso_{held_src}", y[loso_test_idx], loso_pred, loso_proba)
        m["loso_held_source"] = held_src
        all_metrics.append(m)

    # ── Save results ─────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    out_path   = RESULTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(out_path, index=False)
    logger.info(f"\n✓ model_metrics.csv → {out_path}")
    print("\n" + metrics_df[["model", "crash_recall", "macro_f1", "fcar_overall"]].to_string())

    # Save feature column list for dashboard
    feat_config = {"X_cols": list(X_cols), "n_features": len(X_cols)}
    (MODELS_DIR / "feature_config.json").write_text(json.dumps(feat_config, indent=2))

    return {
        "threshold": thresh_clf,
        "binary_xgb": bin_xgb,
        "mtree_xgb": mtree,
        "rf": rf,
        "X_cols": X_cols,
    }


if __name__ == "__main__":
    artifacts = train_all(dev_mode=True)
    print("\nDone. Models saved to models/")
