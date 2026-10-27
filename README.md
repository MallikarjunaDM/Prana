# PRANA — Explainable AI for Motorcycle Crash Detection and Safety Response

> **Proactive Road Accident Neural Analyzer (PRANA)**  
> Version 1.0 · October 2026

---

## Overview

PRANA is a software-only, explainable IMU analysis pipeline for motorcycles. It consumes short windows of acceleration and angular-rate data and predicts one of four classes: **NORMAL**, **POTHOLE**, **BUMP**, or **CRASH**. Every prediction includes calibrated class probabilities, SHAP-based feature explanations, and an action decision: **ALERT**, **CONFIRM**, or **IGNORE**.

A Streamlit dashboard demonstrates the pipeline including a **simulated** emergency countdown (clearly labelled as simulation — no real services contacted).

---

## ⚠️ Research Integrity Notice

- **DaRUS dataset** is multi-body dynamics *simulation* data, NOT real crash recordings.
- **Kaggle IMU dataset** appears to be *generated/synthetic* data with synthetic timestamps. A smartphone sensor is NOT equivalent to a motorcycle-mounted IMU.
- **RoadSens-4M** contains real road-anomaly recordings, used for NORMAL/POTHOLE/BUMP only.
- The dashboard emergency flow is **demonstrative only** and does not contact any real service.
- If one source lacks a gyroscope, it is NOT zero-filled — gyro features are excluded for that source.
- The source probe (shortcut detection) result is reported honestly, including any shortcut risk.

---

## Project Structure

```
prana/
├── data/
│   ├── raw/           # pointers to original source data (never overwritten)
│   └── processed/     # windows.parquet, features.parquet, harmonised.parquet
├── src/               # all production pipeline code
│   ├── config.py          # central paths, labels, hyperparameters
│   ├── audit_data.py      # dataset quality audit (run first)
│   ├── load_darus.py      # DaRUS loader
│   ├── load_roadsens.py   # RoadSens-4M loader (auto-extracts RAR)
│   ├── load_kaggle.py     # Kaggle IMU loader
│   ├── harmonise.py       # merge all sources → common schema
│   ├── make_windows.py    # sliding-window segmentation (1s, 50% hop)
│   ├── features.py        # time/freq/wavelet feature extraction
│   ├── splits.py          # StratifiedGroupKFold + LOSO, leakage checks
│   ├── models.py          # threshold → binary XGB → 4-class XGB → RF
│   ├── calibrate.py       # isotonic calibration + ALERT/CONFIRM/IGNORE
│   ├── evaluate.py        # FCAR, matched-recall, confusion matrices
│   ├── xai.py             # SHAP global+local, faithfulness, stability, sanity
│   ├── probe.py           # source probe (shortcut detection)
│   └── run_pipeline.py    # master runner (all stages)
├── app/
│   └── dashboard.py   # Streamlit dashboard
├── models/            # saved .joblib / .json artifacts
├── results/           # CSVs and JSON outputs
├── figures/           # paper figures and dashboard screenshots
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── dataset_notes.md
└── README.md
```

---

## Datasets

| Dataset | Source | Labels Used |
|---|---|---|
| DaRUS Motorcycle Simulation | Stuttgart U. (simulation) | NORMAL, CRASH, POTHOLE, BUMP |
| RoadSens-4M | Road anomaly recordings | NORMAL, POTHOLE, BUMP |
| Smartphone IMU (Kaggle) | Synthetic/generated | NORMAL, CRASH |

---

## Quick Start

### 1. Install dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run the full pipeline (dev mode — 20% DaRUS subsample)
```powershell
python src/run_pipeline.py
```

### 3. Run full pipeline (100% DaRUS data for final results)
```powershell
python src/run_pipeline.py --full
```

### 4. Run individual stages (with skip flags)
```powershell
python src/run_pipeline.py --skip-audit --skip-harmonise --skip-windows --skip-features
```

### 5. Launch the dashboard
```powershell
streamlit run app/dashboard.py
```

### 6. Run tests
```powershell
python -m pytest tests/ -v
```

---

## Pipeline Stages

| Stage | Script | Output |
|---|---|---|
| 1. Data Audit | `audit_data.py` | `results/dataset_summary.csv`, `dataset_notes.md` |
| 2. Harmonise | `harmonise.py` | `data/processed/harmonised.parquet` |
| 3. Windowing | `make_windows.py` | `data/processed/windows.parquet` |
| 4. Features | `features.py` | `data/processed/features.parquet`, `results/feature_list.csv` |
| 5. Models | `models.py` | `models/*.joblib`, `results/model_metrics.csv` |
| 6. Calibration | `calibrate.py` | `models/calibration_config.json`, `results/calibration_results.csv` |
| 7. Evaluation | `evaluate.py` | `results/evaluation_results.csv`, `figures/confusion_*.png` |
| 8. XAI | `xai.py` | `figures/shap_*.png`, `results/shap_*.csv`, `results/faithfulness_check.csv` |
| 9. Source Probe | `probe.py` | `results/source_probe_results.csv` |
| 10. Dashboard | `app/dashboard.py` | Interactive Streamlit app |

---

## Four-Class Label Policy

| PRANA Label | DaRUS | RoadSens | Kaggle |
|---|---|---|---|
| NORMAL | `CrashLabelML=0`, `Randomstate_*` | Normal Road CSVs | `Crash_Label=0` |
| POTHOLE | `Pothole_5_Out.csv` | Pothole-annotated CSVs | — |
| BUMP | `Speedbump_*`, `Curbstone_*` (→BUMP) | Bump-annotated CSVs | — |
| CRASH | `CrashLabelML=1`, `ISO13232-*`, `Iso_*` | — | `Crash_Label=1` |

---

## Action Policy (ALERT / CONFIRM / IGNORE)

```
p_c = calibrated P(CRASH)
m   = P(top class) - P(2nd class)

ALERT   if p_c >= τ_A
CONFIRM if τ_C <= p_c < τ_A  OR  (top = CRASH AND m < δ)
IGNORE  otherwise
```

Thresholds (τ_A, τ_C, δ) tuned on validation to target crash recall ≥ 0.95. Actual achieved recall is always reported.

---

## Key Configuration (src/config.py)

| Parameter | Value | Description |
|---|---|---|
| `WINDOW_SIZE_S` | 1.0 | Window length in seconds |
| `HOP_SIZE_S` | 0.5 | Hop size (50% overlap) |
| `N_FOLDS` | 5 | StratifiedGroupKFold folds |
| `RANDOM_SEED` | 42 | Global random seed |
| `DARUS_DEV_FRAC` | 0.20 | DaRUS subsample in dev mode |

---

## Explainability Checks

| Check | Method | Output |
|---|---|---|
| Global importance | SHAP TreeExplainer | `figures/shap_global.png` |
| Local examples | SHAP per class | `figures/shap_local_*.png` |
| Faithfulness | Mask top-k features, compare prob drop | `results/faithfulness_check.csv` |
| Sanity | Shuffled-label model, Spearman r | `results/shap_sanity_check.csv` |
| Stability | 5-seed pairwise Spearman | `results/shap_stability.csv` |
| Source probe | Predict dataset origin | `results/source_probe_results.csv` |

---

## Citation

If you use PRANA in research, please cite the original dataset sources:

- **DaRUS**: Rodegast, Philipp et al. "Simulation Data from Motorcycle Sensors in Operational and Crash Scenarios." Stuttgart University.
- **RoadSens-4M**: Please confirm Figshare attribution terms before publication.
- **Kaggle IMU**: Confirm licence at time of download.
