"""
xai.py — Explainability analysis using SHAP
────────────────────────────────────────────
1. Global SHAP feature importance (M-tree)
2. Local SHAP explanations for 1 CRASH, 1 POTHOLE, 1 BUMP example
3. Faithfulness check: mask top-k SHAP features, compare probability drop
4. Sanity check: shuffled-label model vs. real-label SHAP importance
5. Stability check: 5-seed M-tree, pairwise Spearman correlation of importances

Saves figures/ and results/
"""

import logging
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import scipy.stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_PARQUET, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    LABELS, LABEL2INT, INT2LABEL, RANDOM_SEED
)
from splits import make_splits
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

META_COLS = {"session_id", "source", "label", "label_int", "start_time", "source_file"}
CRASH_IDX = LABEL2INT["CRASH"]
N_SHAP_SAMPLE = 500   # subsample for SHAP (speed)


def _xgb_params():
    return {
        "objective": "multi:softprob", "num_class": len(LABELS),
        "max_depth": 5, "n_estimators": 300, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED, "eval_metric": "mlogloss",
        "n_jobs": -1, "verbosity": 0,
    }


def run_xai():
    """Run all XAI checks and save outputs."""
    split_result = make_splits()
    fdf    = split_result["features_df"]
    X_cols = split_result["X_cols"]
    y      = split_result["y"]
    cv_folds = split_result["cv_folds"]

    X = fdf[X_cols].fillna(0.0).values
    train_idx, test_idx = cv_folds[0]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Load or retrain M-tree
    model_path = MODELS_DIR / "mtree_xgb.joblib"
    if model_path.exists():
        mtree = joblib.load(model_path)
        logger.info("Loaded M-tree for SHAP analysis.")
    else:
        logger.info("Training M-tree for SHAP ...")
        mtree = xgb.XGBClassifier(**_xgb_params())
        mtree.fit(X_train, y_train, verbose=False)

    # Subsample test set for speed
    rng = np.random.default_rng(RANDOM_SEED)
    n_sample = min(N_SHAP_SAMPLE, len(X_test))
    sample_idx = rng.choice(len(X_test), size=n_sample, replace=False)
    X_sample  = X_test[sample_idx]
    y_sample  = y_test[sample_idx]

    # ── 1. Global SHAP (TreeExplainer) ────────────────────────────────────────
    logger.info("Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(mtree)
    shap_values = explainer.shap_values(X_sample)

    def _reduce_shap(sv):
        if isinstance(sv, list):
            arr = np.stack(sv, axis=0)  # (n_classes, n_samples, n_features)
            return np.mean(np.abs(arr), axis=(0, 1))
        elif sv.ndim == 3:
            # (n_samples, n_features, n_classes)
            return np.mean(np.abs(sv), axis=(0, 2))
        return np.mean(np.abs(sv), axis=0)

    mean_abs_shap = _reduce_shap(shap_values)
    global_imp_df = pd.DataFrame({
        "feature":         X_cols,
        "shap_importance": mean_abs_shap,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    global_imp_df.to_csv(RESULTS_DIR / "shap_global_importance.csv", index=False)
    logger.info(f"  Top 10 features: {global_imp_df['feature'].head(10).tolist()}")

    # Plot global importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top = global_imp_df.head(20)
    ax.barh(top["feature"][::-1], top["shap_importance"][::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("PRANA — Global SHAP Feature Importance (M-tree, top 20)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_global.png", dpi=150)
    plt.close()
    logger.info(f"  ✓ shap_global.png saved.")

    # ── 2. Local SHAP examples ────────────────────────────────────────────────
    logger.info("Local SHAP explanations ...")
    for target_label in ["CRASH", "POTHOLE", "BUMP"]:
        tl_idx = LABEL2INT[target_label]
        mask   = (y_sample == tl_idx)
        if mask.sum() == 0:
            logger.warning(f"  No {target_label} examples in test sample, skipping local SHAP.")
            continue
        example_pos = np.where(mask)[0][0]

        # Class SHAP values for this example
        if isinstance(shap_values, list):
            sv = shap_values[tl_idx][example_pos]  # (n_features,)
        elif shap_values.ndim == 3:
            sv = shap_values[example_pos, :, tl_idx]  # (n_features,)
        else:
            sv = shap_values[example_pos]

        top_n = 15
        order = np.argsort(np.abs(sv))[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#e74c3c" if v > 0 else "#2980b9" for v in sv[order]]
        ax.barh([X_cols[i] for i in order][::-1], sv[order][::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"Local SHAP — {target_label} example (class={target_label})")
        plt.tight_layout()
        out = FIGURES_DIR / f"shap_local_{target_label.lower()}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"  ✓ {out.name} saved.")

    # ── 3. Faithfulness check ─────────────────────────────────────────────────
    logger.info("Faithfulness check ...")
    faith_results = []
    baseline_proba = mtree.predict_proba(X_sample)
    top_shap_order = np.argsort(mean_abs_shap)[::-1]
    training_means = X_train.mean(axis=0)

    for pct in [0.10, 0.30, 0.50]:
        k = max(1, int(len(X_cols) * pct))
        # SHAP-guided masking
        mask_idx = top_shap_order[:k]
        X_masked = X_sample.copy()
        X_masked[:, mask_idx] = training_means[mask_idx]
        masked_proba = mtree.predict_proba(X_masked)
        prob_drop_shap = float(np.mean(np.abs(baseline_proba - masked_proba)))

        # Random masking (5 repeats)
        random_drops = []
        for seed in range(5):
            rng2 = np.random.default_rng(seed)
            rand_idx = rng2.choice(len(X_cols), size=k, replace=False)
            X_rand = X_sample.copy()
            X_rand[:, rand_idx] = training_means[rand_idx]
            rand_proba = mtree.predict_proba(X_rand)
            random_drops.append(float(np.mean(np.abs(baseline_proba - rand_proba))))

        faith_results.append({
            "pct_masked": pct,
            "k_features": k,
            "prob_drop_shap_guided":   prob_drop_shap,
            "prob_drop_random_mean":   float(np.mean(random_drops)),
            "prob_drop_random_std":    float(np.std(random_drops)),
            "faithfulness_ratio":      prob_drop_shap / (np.mean(random_drops) + 1e-9),
        })
        logger.info(f"  {int(pct*100)}% masked: SHAP drop={prob_drop_shap:.4f}, "
                    f"Random drop={np.mean(random_drops):.4f}")

    faith_df = pd.DataFrame(faith_results)
    faith_df.to_csv(RESULTS_DIR / "faithfulness_check.csv", index=False)
    logger.info(f"  ✓ faithfulness_check.csv saved.")

    # ── 4. Sanity check (shuffled labels) ─────────────────────────────────────
    logger.info("Sanity check (shuffled labels) ...")
    y_shuffled = y_train.copy()
    rng3 = np.random.default_rng(RANDOM_SEED + 99)
    rng3.shuffle(y_shuffled)
    shuffle_model = xgb.XGBClassifier(**_xgb_params())
    shuffle_model.fit(X_train, y_shuffled, verbose=False)
    shuf_explainer = shap.TreeExplainer(shuffle_model)
    shuf_shap = shuf_explainer.shap_values(X_sample[:100])  # small sample
    shuf_mean_abs = _reduce_shap(shuf_shap)
    spearman_r, spearman_p = scipy_stats.spearmanr(mean_abs_shap, shuf_mean_abs)
    sanity_result = {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "interpretation": (
            "Low |r| (< 0.4) indicates real model importance differs from shuffled — good. "
            "High |r| would indicate model may be learning dataset artefacts."
        ),
    }
    pd.DataFrame([sanity_result]).to_csv(RESULTS_DIR / "shap_sanity_check.csv", index=False)
    logger.info(f"  Sanity Spearman r={spearman_r:.3f} (p={spearman_p:.4f})")

    # ── 5. Stability check (5 seeds) ──────────────────────────────────────────
    logger.info("Stability check (5 seeds) ...")
    seed_importances = {}
    for seed in [42, 7, 13, 99, 2024]:
        p = _xgb_params()
        p["random_state"] = seed
        m = xgb.XGBClassifier(**p)
        m.fit(X_train, y_train, verbose=False)
        exp = shap.TreeExplainer(m)
        sv  = exp.shap_values(X_sample[:200])
        seed_importances[seed] = _reduce_shap(sv)

    seeds = list(seed_importances.keys())
    corr_rows = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            r, _ = scipy_stats.spearmanr(seed_importances[seeds[i]],
                                          seed_importances[seeds[j]])
            corr_rows.append({"seed_a": seeds[i], "seed_b": seeds[j], "spearman_r": float(r)})
    stab_df = pd.DataFrame(corr_rows)
    stab_df.to_csv(RESULTS_DIR / "shap_stability.csv", index=False)
    mean_r = float(stab_df["spearman_r"].mean())
    logger.info(f"  Stability mean Spearman r={mean_r:.3f} (higher = more stable)")
    logger.info(f"  ✓ shap_stability.csv saved.")

    logger.info("\n✓ XAI analysis complete. All outputs saved.")


if __name__ == "__main__":
    run_xai()
