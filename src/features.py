"""
features.py — Feature engineering from windowed IMU data
─────────────────────────────────────────────────────────
Computes time-domain, frequency-domain, and wavelet features
for each signal channel in each window.

Saves:
  data/processed/features.parquet
  results/feature_list.csv
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import welch
import pywt

sys.path.insert(0, str(Path(__file__).parent))
from config import WINDOWS_PARQUET, FEATURES_PARQUET, RESULTS_DIR, SIGNAL_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ── Feature computation functions ─────────────────────────────────────────────

def _time_features(x: np.ndarray, prefix: str) -> dict:
    """Statistical time-domain features for signal x."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        x = np.zeros(1)
    n = len(x)
    energy = float(np.sum(x**2))
    rms = float(np.sqrt(energy / n))
    zcr = float(np.mean(np.diff(np.sign(x)) != 0)) if n > 1 else 0.0

    return {
        f"{prefix}_mean":     float(np.mean(x)),
        f"{prefix}_std":      float(np.std(x)),
        f"{prefix}_min":      float(np.min(x)),
        f"{prefix}_max":      float(np.max(x)),
        f"{prefix}_range":    float(np.ptp(x)),
        f"{prefix}_rms":      rms,
        f"{prefix}_p2p":      float(np.max(x) - np.min(x)),
        f"{prefix}_skew":     float(stats.skew(x)) if n >= 3 else 0.0,
        f"{prefix}_kurtosis": float(stats.kurtosis(x)) if n >= 4 else 0.0,
        f"{prefix}_zcr":      zcr,
        f"{prefix}_energy":   energy,
        f"{prefix}_iqr":      float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{prefix}_median":   float(np.median(x)),
        f"{prefix}_mad":      float(np.median(np.abs(x - np.median(x)))),
    }


def _freq_features(x: np.ndarray, fs: float, prefix: str) -> dict:
    """Frequency-domain features using Welch PSD."""
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return {
            f"{prefix}_dom_freq": 0.0,
            f"{prefix}_band_low": 0.0, f"{prefix}_band_mid": 0.0,
            f"{prefix}_band_high": 0.0, f"{prefix}_band_vhigh": 0.0,
            f"{prefix}_spec_entropy": 0.0,
        }
    nperseg = min(len(x), 32)
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)

    # Dominant frequency
    dom_freq = float(freqs[np.argmax(psd)])

    # Band power (0-5, 5-15, 15-30, 30+ Hz) — adapted to available bandwidth
    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
        return float(_trapz(psd[mask], freqs[mask])) if mask.any() else 0.0

    f_max = float(freqs[-1])
    b1 = band_power(0, min(5, f_max * 0.25))
    b2 = band_power(min(5, f_max * 0.25), min(15, f_max * 0.5))
    b3 = band_power(min(15, f_max * 0.5), min(30, f_max * 0.75))
    b4 = band_power(min(30, f_max * 0.75), f_max)

    # Spectral entropy
    psd_norm = psd / (psd.sum() + 1e-12)
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    return {
        f"{prefix}_dom_freq":     dom_freq,
        f"{prefix}_band_low":     b1,
        f"{prefix}_band_mid":     b2,
        f"{prefix}_band_high":    b3,
        f"{prefix}_band_vhigh":   b4,
        f"{prefix}_spec_entropy": spec_entropy,
    }


def _wavelet_features(x: np.ndarray, prefix: str, wavelet: str = "db4", level: int = 3) -> dict:
    """Wavelet decomposition energy per level."""
    x = x[~np.isnan(x)]
    result = {}
    if len(x) < 2 ** level:
        for lv in range(level + 1):
            result[f"{prefix}_wav_energy_L{lv}"] = 0.0
        return result
    try:
        coeffs = pywt.wavedec(x, wavelet, level=level)
        for lv, c in enumerate(coeffs):
            result[f"{prefix}_wav_energy_L{lv}"] = float(np.sum(c**2))
    except Exception:
        for lv in range(level + 1):
            result[f"{prefix}_wav_energy_L{lv}"] = 0.0
    return result


def compute_window_features(signals: dict[str, np.ndarray], fs: float) -> dict:
    """Compute all feature groups for one window given dict of 1D signal arrays."""
    feats = {}

    # Per-channel features
    has_gyro = not np.all(np.isnan(signals.get("gyro_x", np.array([np.nan]))))

    for col in SIGNAL_COLS:
        if col.startswith("gyro") and not has_gyro:
            # Document missing gyro — don't compute garbage features
            continue
        x = signals[col]
        feats.update(_time_features(x, col))
        feats.update(_freq_features(x, fs, col))
        feats.update(_wavelet_features(x, col))

    # Cross-channel correlation features (only where both channels valid)
    def safe_corr(a, b):
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        n = min(len(a), len(b))
        if n < 3:
            return 0.0
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    feats["corr_acc_xy"]  = safe_corr(signals["acc_x"], signals["acc_y"])
    feats["corr_acc_xz"]  = safe_corr(signals["acc_x"], signals["acc_z"])
    feats["corr_acc_yz"]  = safe_corr(signals["acc_y"], signals["acc_z"])
    if has_gyro:
        feats["corr_gyr_xy"] = safe_corr(signals["gyro_x"], signals["gyro_y"])

    # Replace NaN/Inf in features
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats


def make_features() -> pd.DataFrame:
    """
    Load windows.parquet, compute features for every window,
    save features.parquet and feature_list.csv.
    """
    if not WINDOWS_PARQUET.exists():
        raise FileNotFoundError(f"{WINDOWS_PARQUET} not found. Run make_windows.py first.")

    logger.info(f"Loading {WINDOWS_PARQUET} ...")
    windows = pd.read_parquet(WINDOWS_PARQUET)
    logger.info(f"  {len(windows):,} windows.")

    # Detect signal length from column names
    acc_cols = [c for c in windows.columns if c.startswith("acc_x_")]
    if not acc_cols:
        raise ValueError("No signal columns found in windows.parquet")
    signal_length = len(acc_cols)
    logger.info(f"  Signal length per window: {signal_length} samples")

    # Estimate typical sampling rate
    median_hz = float(windows["sampling_hz"].median())
    logger.info(f"  Median sampling rate: {median_hz:.1f} Hz")

    # Pre-extract numpy matrices for fast zero-overhead slicing
    logger.info("Extracting signal matrices ...")
    signal_matrices = {
        col: windows[[f"{col}_{j}" for j in range(signal_length)]].to_numpy(dtype=float)
        for col in SIGNAL_COLS
    }
    session_ids  = windows["session_id"].values
    sources      = windows["source"].values
    labels       = windows["label"].values
    label_ints   = windows["label_int"].values
    start_times  = windows["start_time"].values
    source_files = windows["source_file"].values
    n_windows    = len(windows)

    # Compute features window by window
    logger.info("Computing features ...")
    feature_records = []
    for i in range(n_windows):
        window_signals = {col: signal_matrices[col][i] for col in SIGNAL_COLS}
        feats = compute_window_features(window_signals, median_hz)
        feats["session_id"]  = session_ids[i]
        feats["source"]      = sources[i]
        feats["label"]       = labels[i]
        feats["label_int"]   = int(label_ints[i])
        feats["start_time"]  = float(start_times[i])
        feats["source_file"] = source_files[i]
        feature_records.append(feats)
        if (i + 1) % 2500 == 0 or (i + 1) == n_windows:
            logger.info(f"  {i+1}/{n_windows} windows processed")

    features_df = pd.DataFrame(feature_records)
    logger.info(f"Feature matrix shape: {features_df.shape}")

    # Save
    features_df.to_parquet(FEATURES_PARQUET, index=False)
    logger.info(f"✓ features.parquet → {FEATURES_PARQUET}")

    # Feature list CSV
    meta_cols = {"session_id", "source", "label", "label_int", "start_time", "source_file"}
    feat_names = [c for c in features_df.columns if c not in meta_cols]
    feat_list  = []
    for f in feat_names:
        # Determine which signal it's derived from
        signal = next((s for s in SIGNAL_COLS if f.startswith(s)), "cross_channel")
        feat_list.append({
            "feature":              f,
            "signal":               signal,
            "applicable_sources":   "all",
            "missing_data_handling": "0.0 substituted for NaN/Inf; gyro features excluded if gyro all-NaN",
        })
    fl_df = pd.DataFrame(feat_list)
    fl_path = RESULTS_DIR / "feature_list.csv"
    fl_df.to_csv(fl_path, index=False)
    logger.info(f"✓ feature_list.csv → {fl_path}  ({len(feat_names)} features)")

    return features_df


if __name__ == "__main__":
    df = make_features()
    meta_cols = {"session_id", "source", "label", "label_int", "start_time", "source_file"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    print(f"\nFeature matrix: {df.shape}")
    print(f"Number of features: {len(feat_cols)}")
    print("Label distribution:\n", df["label"].value_counts())
