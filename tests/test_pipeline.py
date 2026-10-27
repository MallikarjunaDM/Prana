"""
Tests for PRANA pipeline components.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from config import SCHEMA_COLS, SIGNAL_COLS, LABELS, LABEL2INT


# ── Loader schema tests ───────────────────────────────────────────────────────

def _check_schema(df: pd.DataFrame, source_name: str):
    """Assert DataFrame matches common schema."""
    assert all(c in df.columns for c in SCHEMA_COLS), \
        f"{source_name}: missing columns {set(SCHEMA_COLS) - set(df.columns)}"
    assert df["source"].nunique() >= 1, f"{source_name}: source column empty"
    assert df["label"].isin(LABELS).all(), \
        f"{source_name}: invalid labels {df['label'].unique()}"
    assert df["acc_x"].notna().any(), f"{source_name}: acc_x all NaN"
    assert df["acc_y"].notna().any(), f"{source_name}: acc_y all NaN"


def test_kaggle_loader_schema():
    """Kaggle loader produces correct schema with valid labels."""
    from load_kaggle import load_kaggle
    df = load_kaggle()
    if len(df) == 0:
        pytest.skip("Kaggle CSV not available")
    _check_schema(df, "kaggle")
    assert set(df["label"].unique()).issubset({"NORMAL", "CRASH"}), \
        "Kaggle should only have NORMAL and CRASH"
    # Check session grouping: 80 groups from 8000 rows
    assert df["session_id"].nunique() <= 80, "Expected at most 80 session groups"


def test_kaggle_no_gyro_fabrication():
    """Kaggle gyro data should be present (not fabricated as zero)."""
    from load_kaggle import load_kaggle
    df = load_kaggle()
    if len(df) == 0:
        pytest.skip("Kaggle CSV not available")
    # Gyro columns must exist and have non-trivial variance
    for col in ["gyro_x", "gyro_y", "gyro_z"]:
        assert col in df.columns, f"Missing {col}"
        assert df[col].std() > 0.0 or df[col].isna().all(), \
            f"{col} has zero variance — may be fabricated"


def test_darus_label_mapping():
    """DaRUS control scenarios map correctly (spot-check)."""
    from load_darus import _infer_control_label
    assert _infer_control_label("Pothole_5_Out.csv") == "POTHOLE"
    assert _infer_control_label("Speedbump_5_Out.csv") == "BUMP"
    assert _infer_control_label("Curbstone_5_Out.csv") == "BUMP"
    assert _infer_control_label("ReverseCurbstone_5_Out.csv") == "BUMP"
    assert _infer_control_label("Randomstate_10_Out.csv") == "NORMAL"
    assert _infer_control_label("ISO13232-1_Out.csv") == "CRASH"
    assert _infer_control_label("Iso_115_Out.csv") == "CRASH"


# ── Window tests ─────────────────────────────────────────────────────────────

def _make_dummy_session(n=200, label="NORMAL", session_id="s0", hz=100):
    t = np.arange(n) / hz
    return pd.DataFrame({
        "source": "test", "session_id": session_id, "source_file": "dummy.csv",
        "timestamp": t,
        "acc_x": np.random.randn(n), "acc_y": np.random.randn(n), "acc_z": np.random.randn(n) + 9.8,
        "gyro_x": np.random.randn(n) * 0.1, "gyro_y": np.random.randn(n) * 0.1, "gyro_z": np.random.randn(n) * 0.1,
        "acc_mag": np.random.randn(n) + 9.8,
        "label": label, "label_int": LABEL2INT[label],
    })


def test_window_sizes():
    """Windows should have the expected sample count."""
    from make_windows import segment_session
    from config import WINDOW_SIZE_S, HOP_SIZE_S
    hz = 100
    df = _make_dummy_session(n=300, hz=hz)
    recs = segment_session(df)
    assert len(recs) > 0, "No windows generated"
    expected_samples = int(round(WINDOW_SIZE_S * hz))
    # All windows should have consistent n_samples (±1 tolerance for float SR)
    for r in recs:
        assert abs(r["n_samples"] - expected_samples) <= 1, \
            f"Window has {r['n_samples']} samples, expected ~{expected_samples}"


def test_window_label_majority():
    """Majority label assignment: a window mostly CRASH should be labelled CRASH."""
    from make_windows import segment_session
    hz = 100
    n = 200
    labels = ["CRASH"] * 160 + ["NORMAL"] * 40
    df = _make_dummy_session(n=n, hz=hz)
    df["label"] = labels[:n]
    df["label_int"] = [LABEL2INT[l] for l in labels[:n]]
    recs = segment_session(df)
    assert recs[0]["label"] == "CRASH", "First window should be CRASH (majority)"


def test_no_cross_session_windows():
    """Windows should never span two sessions."""
    from make_windows import segment_session
    df1 = _make_dummy_session(n=200, label="NORMAL", session_id="sess_A")
    df2 = _make_dummy_session(n=200, label="CRASH",  session_id="sess_B")
    # Process separately (as the pipeline does)
    recs1 = segment_session(df1)
    recs2 = segment_session(df2)
    for r in recs1:
        assert r["session_id"] == "sess_A"
    for r in recs2:
        assert r["session_id"] == "sess_B"


# ── Split leakage tests ───────────────────────────────────────────────────────

def test_no_session_leakage():
    """No session_id should appear in both train and test of the same fold."""
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    from config import N_FOLDS, RANDOM_SEED

    # Simulate 100 sessions × 20 rows each
    n_sessions = 100
    rows_per   = 20
    sessions   = np.repeat([f"sess_{i}" for i in range(n_sessions)], rows_per)
    labels     = np.tile([0, 1, 2, 3], n_sessions * rows_per // 4)[:len(sessions)]
    X_dummy    = np.random.randn(len(sessions), 5)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold_idx, (tr, te) in enumerate(sgkf.split(X_dummy, labels, sessions)):
        train_sessions = set(sessions[tr])
        test_sessions  = set(sessions[te])
        overlap = train_sessions & test_sessions
        assert len(overlap) == 0, \
            f"Fold {fold_idx}: session leakage! {overlap}"


# ── Feature tests ─────────────────────────────────────────────────────────────

def test_feature_no_all_nan_column():
    """Feature computation should produce no all-NaN columns."""
    from features import compute_window_features
    from config import SIGNAL_COLS
    hz = 100.0
    n = 100
    signals = {col: np.random.randn(n) for col in SIGNAL_COLS}
    feats = compute_window_features(signals, hz)
    nan_feats = [k for k, v in feats.items() if isinstance(v, float) and np.isnan(v)]
    assert len(nan_feats) == 0, f"NaN features: {nan_feats}"


def test_feature_finite_values():
    """All feature values must be finite (no Inf)."""
    from features import compute_window_features
    from config import SIGNAL_COLS
    hz, n = 100.0, 100
    signals = {col: np.random.randn(n) for col in SIGNAL_COLS}
    feats = compute_window_features(signals, hz)
    inf_feats = [k for k, v in feats.items() if isinstance(v, float) and not np.isfinite(v)]
    assert len(inf_feats) == 0, f"Non-finite features: {inf_feats}"
