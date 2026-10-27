"""
probe.py — Source probe (shortcut / dataset-artefact detection)
───────────────────────────────────────────────────────────────
Trains a classifier to predict dataset SOURCE from features.
High accuracy indicates source-specific artefacts that could cause
the main model to "shortcut" on dataset identity rather than true physics.
Saves results/source_probe_results.csv.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURES_PARQUET, RESULTS_DIR, RANDOM_SEED, N_FOLDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

META_COLS = {"session_id", "source", "label", "label_int", "start_time", "source_file"}


def run_source_probe():
    """
    Predict dataset source from features using grouped CV.
    High source prediction accuracy = shortcut risk.
    """
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"{FEATURES_PARQUET} not found. Run features.py first.")

    logger.info(f"Loading {FEATURES_PARQUET} ...")
    fdf = pd.read_parquet(FEATURES_PARQUET)
    X_cols = [c for c in fdf.columns if c not in META_COLS]

    X      = fdf[X_cols].fillna(0.0).values
    le     = LabelEncoder()
    y_src  = le.fit_transform(fdf["source"].values)
    groups = fdf["session_id"].values
    sources = list(le.classes_)
    logger.info(f"Sources: {sources}")
    logger.info(f"Source distribution: {dict(zip(*np.unique(y_src, return_counts=True)))}")

    # StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))),
                                 shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y_src, groups)):
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(sources),
            n_estimators=200, max_depth=4,
            random_state=RANDOM_SEED, verbosity=0, n_jobs=-1,
        )
        model.fit(X[train_idx], y_src[train_idx], verbose=False)
        pred = model.predict(X[test_idx])
        acc  = float(accuracy_score(y_src[test_idx], pred))

        report = classification_report(y_src[test_idx], pred,
                                       target_names=sources, output_dict=True,
                                       zero_division=0)
        fold_results.append({
            "fold": fold_idx,
            "source_accuracy": acc,
            **{f"{src}_f1": report.get(src, {}).get("f1-score", None) for src in sources},
        })
        logger.info(f"  Fold {fold_idx}: source accuracy = {acc:.3f}")

    results_df = pd.DataFrame(fold_results)
    mean_acc = float(results_df["source_accuracy"].mean())

    out_path = RESULTS_DIR / "source_probe_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"\n✓ source_probe_results.csv → {out_path}")
    logger.info(f"Mean source classification accuracy: {mean_acc:.3f}")

    # Interpretation guidance
    if mean_acc > 0.80:
        logger.warning(
            f"⚠ HIGH source accuracy ({mean_acc:.3f}). "
            "The feature set can strongly distinguish dataset origins. "
            "This is a SHORTCUT RISK — investigate which features drive source separation "
            "and whether they represent true physics or recording artefacts."
        )
    elif mean_acc > 0.60:
        logger.warning(
            f"MODERATE source accuracy ({mean_acc:.3f}). "
            "Some source-specific signal present. Investigate top differentiating features."
        )
    else:
        logger.info(f"Source accuracy {mean_acc:.3f} — low source-specific signal (good).")

    # Top differentiating features (last fold's model)
    fi = model.feature_importances_
    top_idx = np.argsort(fi)[::-1][:10]
    top_feat = [(X_cols[i], float(fi[i])) for i in top_idx]
    logger.info(f"Top features differentiating sources: {top_feat}")

    probe_summary = {
        "mean_source_accuracy": mean_acc,
        "interpretation": (
            "High accuracy (>0.8) = shortcut risk. "
            "Moderate (0.6-0.8) = some artefacts. Low (<0.6) = good."
        ),
        "top_differentiating_features": top_feat,
    }
    import json
    (RESULTS_DIR / "source_probe_summary.json").write_text(
        json.dumps(probe_summary, indent=2, default=str)
    )
    return results_df


if __name__ == "__main__":
    run_source_probe()
