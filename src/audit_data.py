"""
audit_data.py — Pre-modelling dataset audit
────────────────────────────────────────────
Runs all three loaders, computes quality statistics per source,
and saves:
  results/dataset_summary.csv   — quantitative per-source stats
  dataset_notes.md              — human-readable findings and limitations
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, ROOT, SCHEMA_COLS, SIGNAL_COLS
from load_darus   import load_darus
from load_roadsens import load_roadsens
from load_kaggle  import load_kaggle

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _sample_rate(df: pd.DataFrame) -> float | None:
    """Estimate median sampling rate (Hz) per session."""
    rates = []
    for _, grp in df.groupby("session_id"):
        ts = grp["timestamp"].dropna().sort_values()
        if len(ts) < 2:
            continue
        diffs = ts.diff().dropna()
        median_dt = diffs.median()
        if median_dt > 0:
            rates.append(1.0 / median_dt)
    return float(np.median(rates)) if rates else None


def _audit_source(df: pd.DataFrame, name: str) -> dict:
    """Compute quality metrics for one data source."""
    n_rows     = len(df)
    n_sessions = df["session_id"].nunique()
    label_dist = df["label"].value_counts().to_dict()
    n_missing  = df[SIGNAL_COLS].isna().sum().to_dict()

    # Constant channels (std == 0 over entire source)
    constant_cols = [c for c in SIGNAL_COLS if df[c].std() == 0]

    # Duplicate rows
    n_dupes = df.duplicated(subset=SIGNAL_COLS + ["session_id", "timestamp"]).sum()

    # Timestamp ordering check (per session)
    ts_issues = 0
    for _, grp in df.groupby("session_id"):
        ts = grp["timestamp"].dropna()
        if (ts.diff().dropna() < 0).any():
            ts_issues += 1

    # Clipping: values at 99.99th vs 99.9th percentile
    clip_suspects = []
    for c in SIGNAL_COLS:
        col = df[c].dropna()
        if len(col) < 100:
            continue
        p999 = col.quantile(0.999)
        p9999 = col.quantile(0.9999)
        if p9999 == p999 and p999 != 0:
            clip_suspects.append(c)

    # Estimated sample rate
    sr = _sample_rate(df)

    # Sessions with too few samples (< 50 rows)
    session_sizes = df.groupby("session_id").size()
    tiny_sessions = int((session_sizes < 50).sum())

    return {
        "source":             name,
        "n_rows":             n_rows,
        "n_sessions":         n_sessions,
        "label_distribution": json.dumps(label_dist),
        "missing_per_col":    json.dumps({k: int(v) for k, v in n_missing.items()}),
        "constant_channels":  ", ".join(constant_cols) if constant_cols else "none",
        "duplicate_rows":     int(n_dupes),
        "ts_ordering_issues": int(ts_issues),
        "clipping_suspects":  ", ".join(clip_suspects) if clip_suspects else "none",
        "estimated_hz":       round(sr, 2) if sr else "unknown",
        "tiny_sessions_lt50": tiny_sessions,
    }


def run_audit(dev_mode: bool = True):
    """Load all sources, audit, and save outputs."""
    logger.info("=== PRANA Data Audit ===\n")
    summaries = []

    # DaRUS
    logger.info("── DaRUS ──")
    df_darus = load_darus(dev_mode=dev_mode)
    summaries.append(_audit_source(df_darus, "darus"))

    # RoadSens
    logger.info("\n── RoadSens ──")
    df_roadsens = load_roadsens()
    if len(df_roadsens) > 0:
        summaries.append(_audit_source(df_roadsens, "roadsens"))
    else:
        logger.warning("RoadSens produced no data — skipping audit entry.")

    # Kaggle
    logger.info("\n── Kaggle IMU ──")
    df_kaggle = load_kaggle()
    summaries.append(_audit_source(df_kaggle, "kaggle"))

    # Save summary CSV
    summary_df = pd.DataFrame(summaries)
    out_csv = RESULTS_DIR / "dataset_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"\n✓ dataset_summary.csv → {out_csv}")
    print("\n" + summary_df.to_string())

    # Save dataset_notes.md
    _write_notes(summary_df, df_darus, df_roadsens, df_kaggle)

    return df_darus, df_roadsens, df_kaggle, summary_df


def _write_notes(summary_df, df_darus, df_roadsens, df_kaggle):
    """Write human-readable dataset notes."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# PRANA Dataset Notes",
        f"Generated: {ts}\n",
        "## Summary\n",
        "```\n" + summary_df.to_string(index=False) + "\n```",
        "",
        "## Source-Specific Notes\n",
        "### DaRUS Motorcycle Simulation Dataset",
        "- **Type**: Simulation (multi-body dynamics), NOT real crash recordings.",
        "- **Label source**: `CrashLabelML` (0=NORMAL, 1=CRASH) for main files;",
        "  scenario filename for control files.",
        "- **Curbstone policy**: Mapped to BUMP (user decision, 12 Oct 2026).",
        "- **Units**: Linear acceleration m/s², angular velocity rad/s.",
        "- **Session grouping**: Each CSV file = one session.",
        "- **Dev mode**: 20% random subsample used during development.",
        "",
        "### RoadSens-4M",
        "- **Type**: Real road-anomaly recordings (NOT crash data).",
        "- **Usage**: NORMAL, POTHOLE, BUMP only — no CRASH labels assigned from peaks.",
        "- **Gyroscope**: May be absent in some files — left as NaN, NOT zero-filled.",
        "- **Licence**: Confirm Figshare attribution before publication.",
        "- **RAR extracted via Python rarfile.**",
        "",
        "### Kaggle Smartphone IMU Dataset",
        "- **Type**: GENERATED/SIMULATED data (synthetic timestamps from 2026-01-01).",
        "- **⚠ Limitation**: This is NOT verified real-world motorcycle crash data.",
        "  Sensor is a smartphone, not a motorcycle-mounted IMU.",
        "  Should NOT be described as real crash recordings.",
        "- **Classes available**: CRASH and NORMAL only (no POTHOLE/BUMP).",
        "- **Session grouping**: Artificial — every 100 rows = one session (80 groups).",
        "- **Units**: Assumed m/s² (Z-axis ≈9.8 at rest confirms g-scale).",
        "",
        "## Known Risks & Shortcut Concerns",
        "- DaRUS is simulation data; generalisation to real motorcycle crashes is unverified.",
        "- Source probe (probe.py) should be run to detect dataset-specific artefacts.",
        "- If source probe achieves high accuracy, investigate responsible features.",
        "- LOSO evaluation should be the primary measure of cross-source generalisation.",
        "",
        "## Label Policy",
        "| PRANA Label | DaRUS | RoadSens | Kaggle |",
        "|---|---|---|---|",
        "| NORMAL | CrashLabelML=0, Randomstate_* | Normal Road CSVs | Crash_Label=0 |",
        "| POTHOLE | Pothole_5_Out.csv | Pothole-annotated CSVs | — |",
        "| BUMP | Speedbump/Curbstone CSVs | Bump-annotated CSVs | — |",
        "| CRASH | CrashLabelML=1, ISO*/Iso_* | — | Crash_Label=1 |",
    ]
    notes_path = ROOT / "dataset_notes.md"
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"✓ dataset_notes.md → {notes_path}")


if __name__ == "__main__":
    run_audit(dev_mode=True)
