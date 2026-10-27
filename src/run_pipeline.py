"""
run_pipeline.py — End-to-end PRANA pipeline runner
────────────────────────────────────────────────────
Runs all stages in order:
  1. Data audit
  2. Harmonisation → windows → features
  3. Model training
  4. Calibration
  5. Evaluation
  6. XAI
  7. Source probe

Run with:
  python src/run_pipeline.py [--full]   # --full uses 100% DaRUS data
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prana")


def main():
    parser = argparse.ArgumentParser(description="PRANA pipeline runner")
    parser.add_argument("--full", action="store_true",
                        help="Use 100%% DaRUS data (dev mode = 20%% by default)")
    parser.add_argument("--skip-audit", action="store_true", help="Skip data audit")
    parser.add_argument("--skip-harmonise", action="store_true", help="Skip harmonisation")
    parser.add_argument("--skip-windows", action="store_true", help="Skip windowing")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-calibrate", action="store_true", help="Skip calibration")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-xai", action="store_true", help="Skip XAI")
    parser.add_argument("--skip-probe", action="store_true", help="Skip source probe")
    args = parser.parse_args()

    dev_mode = not args.full
    if dev_mode:
        logger.info("Running in DEV MODE (20%% DaRUS subsample). Use --full for complete run.")
    else:
        logger.info("Running in FULL MODE (100%% DaRUS data). This may take a while.")

    from config import HARMONISED_PARQUET, WINDOWS_PARQUET, FEATURES_PARQUET

    # ── Stage 1: Audit ────────────────────────────────────────────────────────
    if not args.skip_audit:
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: DATA AUDIT")
        from audit_data import run_audit
        run_audit(dev_mode=dev_mode)

    # ── Stage 2: Harmonise ────────────────────────────────────────────────────
    if not args.skip_harmonise:
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: HARMONISATION")
        from harmonise import harmonise
        harmonise(dev_mode=dev_mode)

    # ── Stage 3: Windows ─────────────────────────────────────────────────────
    if not args.skip_windows:
        if not HARMONISED_PARQUET.exists():
            logger.error("harmonised.parquet missing — run harmonise first.")
            sys.exit(1)
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: WINDOWING")
        from make_windows import make_windows
        make_windows()

    # ── Stage 4: Features ─────────────────────────────────────────────────────
    if not args.skip_features:
        if not WINDOWS_PARQUET.exists():
            logger.error("windows.parquet missing — run make_windows first.")
            sys.exit(1)
        logger.info("\n" + "="*60)
        logger.info("STAGE 4: FEATURE ENGINEERING")
        from features import make_features
        make_features()

    # ── Stage 5: Models ───────────────────────────────────────────────────────
    if not args.skip_train:
        if not FEATURES_PARQUET.exists():
            logger.error("features.parquet missing — run features.py first.")
            sys.exit(1)
        logger.info("\n" + "="*60)
        logger.info("STAGE 5: MODEL TRAINING")
        from models import train_all
        train_all(dev_mode=dev_mode)

    # ── Stage 6: Calibration ──────────────────────────────────────────────────
    if not args.skip_calibrate:
        logger.info("\n" + "="*60)
        logger.info("STAGE 6: CALIBRATION & ACTION POLICY")
        from calibrate import calibrate_and_evaluate
        calibrate_and_evaluate()

    # ── Stage 7: Evaluation ───────────────────────────────────────────────────
    if not args.skip_evaluate:
        logger.info("\n" + "="*60)
        logger.info("STAGE 7: EVALUATION")
        from evaluate import evaluate_all
        evaluate_all()

    # ── Stage 8: XAI ─────────────────────────────────────────────────────────
    if not args.skip_xai:
        logger.info("\n" + "="*60)
        logger.info("STAGE 8: EXPLAINABILITY (SHAP)")
        from xai import run_xai
        run_xai()

    # ── Stage 9: Source Probe ─────────────────────────────────────────────────
    if not args.skip_probe:
        logger.info("\n" + "="*60)
        logger.info("STAGE 9: SOURCE PROBE")
        from probe import run_source_probe
        run_source_probe()

    logger.info("\n" + "="*60)
    logger.info("✓ PRANA pipeline complete!")
    logger.info("  Next: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
