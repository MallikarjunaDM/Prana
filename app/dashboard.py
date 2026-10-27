"""
dashboard.py — PRANA Streamlit Dashboard
──────────────────────────────────────────────────────────────────────────
Explainable AI for Motorcycle Crash Detection and Safety Response

⚠ SIMULATION ONLY — No real emergency services are contacted.
"""

import sys
import json
import time
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Path setup ─────────────────────────────────────────────────────────────────
APP_DIR  = Path(__file__).parent
SRC_DIR  = APP_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    FEATURES_PARQUET, WINDOWS_PARQUET, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    LABELS, LABEL2INT, INT2LABEL, SIGNAL_COLS
)
from calibrate import apply_action_policy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRANA — Motorcycle Crash Analyzer",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0d1117; color: #e6edf3; }

.metric-card {
    background: linear-gradient(135deg, #1c2333 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 6px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

.alert-banner {
    background: linear-gradient(135deg, #da3633 0%, #b91c1c 100%);
    border: 2px solid #ff4444;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    animation: pulse 1s infinite;
    box-shadow: 0 0 30px rgba(218,54,51,0.5);
}

.confirm-banner {
    background: linear-gradient(135deg, #d29922 0%, #b45309 100%);
    border: 2px solid #fbbf24;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 0 20px rgba(210,153,34,0.4);
}

.ignore-banner {
    background: linear-gradient(135deg, #238636 0%, #166534 100%);
    border: 1px solid #3fb950;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 0 10px rgba(35,134,54,0.3);
}

.simulation-warning {
    background: rgba(136,77,0,0.25);
    border: 1px solid #d29922;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: center;
    font-weight: 600;
    font-size: 0.85rem;
    color: #fbbf24;
}

.label-badge-CRASH   { background: #da3633; color: white; }
.label-badge-POTHOLE { background: #d29922; color: black; }
.label-badge-BUMP    { background: #388bfd; color: white; }
.label-badge-NORMAL  { background: #238636; color: white; }
.label-badge {
    padding: 4px 14px; border-radius: 20px; font-weight: 600;
    font-size: 1.0rem; display: inline-block;
}

@keyframes pulse {
    0%   { box-shadow: 0 0 20px rgba(218,54,51,0.5); }
    50%  { box-shadow: 0 0 50px rgba(218,54,51,0.9); }
    100% { box-shadow: 0 0 20px rgba(218,54,51,0.5); }
}

.countdown-ring {
    font-size: 3.5rem;
    font-weight: 700;
    color: #ff4444;
    text-align: center;
    font-family: 'Inter', monospace;
}

div[data-testid="stSidebar"] { background: #161b22; }
.stSlider > div > div > div > div { background: #388bfd; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model loading (cached) ─────────────────────────────────────────────

@st.cache_data(show_spinner="Loading feature data ...")
def load_features():
    if not FEATURES_PARQUET.exists():
        return None
    return pd.read_parquet(FEATURES_PARQUET)


@st.cache_data(show_spinner="Loading window signals ...")
def load_windows():
    if not WINDOWS_PARQUET.exists():
        return None
    return pd.read_parquet(WINDOWS_PARQUET)


@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    p = MODELS_DIR / "mtree_xgb.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource(show_spinner="Loading calibration config ...")
def load_calib_config():
    p = MODELS_DIR / "calibration_config.json"
    if not p.exists():
        return {"tau_a": 0.70, "tau_c": 0.40, "delta": 0.20}
    return json.loads(p.read_text())


@st.cache_resource(show_spinner="Loading feature config ...")
def load_feature_config():
    p = MODELS_DIR / "feature_config.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ── Helper: predict one window ────────────────────────────────────────────────

def predict_window(model, X_row: np.ndarray, calib_cfg: dict):
    """Run prediction and action policy for one window feature vector."""
    proba = model.predict_proba(X_row.reshape(1, -1))[0]
    pred_int = int(np.argmax(proba))
    pred_label = INT2LABEL[pred_int]

    actions = apply_action_policy(
        proba.reshape(1, -1),
        calib_cfg["tau_a"], calib_cfg["tau_c"], calib_cfg["delta"]
    )
    action = actions[0]
    return pred_label, proba, action


# ── Helper: reconstruct signal from window row ────────────────────────────────

def get_window_signals(windows_df: pd.DataFrame, win_idx: int) -> dict:
    """Extract signal arrays from one window row."""
    row = windows_df.iloc[win_idx]
    sig_cols = [c for c in windows_df.columns if c.startswith("acc_x_")]
    n = len(sig_cols)
    signals = {}
    for col in SIGNAL_COLS:
        arr = [row.get(f"{col}_{i}", np.nan) for i in range(n)]
        signals[col] = np.array(arr, dtype=float)
    return signals, n


# ── Signal plot ───────────────────────────────────────────────────────────────

def plot_signals(signals: dict, n: int, hz: float, label: str):
    t = np.arange(n) / hz
    fig = go.Figure()
    colors = {"acc_x": "#388bfd", "acc_y": "#3fb950", "acc_z": "#ffa657",
              "acc_mag": "#f0883e", "gyro_x": "#bc8cff", "gyro_y": "#ff7b72", "gyro_z": "#79c0ff"}
    show_cols = ["acc_x", "acc_y", "acc_z", "acc_mag"]
    for col in show_cols:
        arr = signals.get(col, np.full(n, np.nan))
        if not np.all(np.isnan(arr)):
            fig.add_trace(go.Scatter(
                x=t, y=arr, name=col, mode="lines",
                line=dict(color=colors.get(col, "#ccc"), width=1.5),
            ))
    fig.update_layout(
        title=dict(text=f"IMU Signal Window  |  True label: <b>{label}</b>",
                   font=dict(size=14, color="#e6edf3")),
        xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)",
        height=280, paper_bgcolor="#1c2333", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"), legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"),
    )
    return fig


# ── Probability bar chart ─────────────────────────────────────────────────────

def plot_proba(proba: np.ndarray):
    colors = {"NORMAL": "#238636", "POTHOLE": "#d29922", "BUMP": "#388bfd", "CRASH": "#da3633"}
    fig = go.Figure(go.Bar(
        x=LABELS, y=proba,
        marker_color=[colors[l] for l in LABELS],
        text=[f"{p*100:.1f}%" for p in proba],
        textposition="outside",
        textfont=dict(color="#e6edf3"),
    ))
    fig.update_layout(
        title=dict(text="Calibrated Class Probabilities", font=dict(size=13, color="#e6edf3")),
        yaxis=dict(range=[0, 1.15], gridcolor="#30363d", tickformat=".0%"),
        xaxis=dict(gridcolor="#30363d"),
        height=260, paper_bgcolor="#1c2333", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"), showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


# ── SHAP chart (from global importance) ──────────────────────────────────────

def plot_shap_global():
    p = RESULTS_DIR / "shap_global_importance.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).head(15)
    fig = go.Figure(go.Bar(
        x=df["shap_importance"][::-1],
        y=df["feature"][::-1],
        orientation="h",
        marker_color="#388bfd",
        text=[f"{v:.4f}" for v in df["shap_importance"][::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="SHAP Global Feature Importance (Top 15)", font=dict(size=13, color="#e6edf3")),
        height=360, paper_bgcolor="#1c2333", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        xaxis=dict(title="Mean |SHAP|", gridcolor="#30363d"),
        yaxis=dict(gridcolor="#30363d"),
        margin=dict(l=160, r=60, t=40, b=30),
    )
    return fig


# ── Alert log ────────────────────────────────────────────────────────────────

def write_alert_log(session_id: str, window_idx: int, proba: np.ndarray, action: str):
    log_path = RESULTS_DIR / "alert_log.csv"
    entry = {
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "session_id":  session_id,
        "window_idx":  window_idx,
        "action":      action,
        "p_crash":     float(proba[LABEL2INT["CRASH"]]),
        "simulation":  True,
        "note":        "SIMULATION ONLY — no real emergency service contacted.",
    }
    entry_df = pd.DataFrame([entry])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        entry_df = pd.concat([existing, entry_df], ignore_index=True)
    entry_df.to_csv(log_path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a2744 0%, #0d1117 100%);
                padding: 24px 32px; border-radius: 14px; margin-bottom: 16px;
                border: 1px solid #30363d;">
        <div style="display:flex; align-items:center; gap:16px;">
            <span style="font-size:2.5rem;">🏍️</span>
            <div>
                <h1 style="margin:0; font-size:1.9rem; color:#e6edf3; font-weight:700;">
                    PRANA Analyzer
                </h1>
                <p style="margin:0; color:#8b949e; font-size:0.9rem;">
                    Proactive Road Accident Neural Analyzer &nbsp;|&nbsp;
                    Explainable AI for Motorcycle Safety
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Simulation warning (always visible) ───────────────────────────────────
    st.markdown("""
    <div class="simulation-warning">
        ⚠️ SIMULATION — No message sent. Not connected to emergency services.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── Load artifacts ─────────────────────────────────────────────────────────
    features_df  = load_features()
    windows_df   = load_windows()
    model        = load_model()
    calib_cfg    = load_calib_config()
    feat_cfg     = load_feature_config()

    pipeline_ready = (features_df is not None and model is not None)

    if not pipeline_ready:
        st.error("""
        ### ⚙️ Pipeline not yet run

        No model or features found. Run the pipeline first:
        ```powershell
        cd prana
        python src/run_pipeline.py
        ```
        Then restart this dashboard.
        """)
        st.info("**Quick start:** The pipeline will train models and save them to `models/`. "
                "This takes a few minutes on the first run.")
        _show_pipeline_status()
        return

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔧 Controls")

        sources = sorted(features_df["source"].unique())
        sel_src = st.selectbox("Dataset source", sources)

        src_sessions = sorted(features_df[features_df["source"] == sel_src]["session_id"].unique())
        sel_session  = st.selectbox("Recording session", src_sessions)

        sess_windows = features_df[features_df["session_id"] == sel_session].reset_index(drop=True)
        n_windows = len(sess_windows)
        win_idx = st.slider("Window", 0, max(0, n_windows - 1), 0)

        st.markdown("---")
        st.markdown("### ⚙️ Action Policy")
        st.caption(f"τ_A (ALERT): **{calib_cfg['tau_a']:.2f}**")
        st.caption(f"τ_C (CONFIRM): **{calib_cfg['tau_c']:.2f}**")
        st.caption(f"δ (margin): **{calib_cfg['delta']:.2f}**")

        st.markdown("---")
        st.markdown("### 📊 Pipeline Status")
        _show_pipeline_status_sidebar()

        st.markdown("---")
        st.markdown(
            "<div style='color:#8b949e; font-size:0.78rem;'>"
            "⚠️ SIMULATION ONLY<br>No real emergency services connected.<br>"
            "Crash data is from simulation/lab recordings."
            "</div>",
            unsafe_allow_html=True
        )

    # ── Main panel ─────────────────────────────────────────────────────────────
    if n_windows == 0:
        st.warning("No windows found for this session.")
        return

    # Get current window data
    win_row = sess_windows.iloc[win_idx]
    true_label = win_row["label"]

    # Feature vector
    X_cols = feat_cfg["X_cols"] if feat_cfg else [c for c in features_df.columns
                                                    if c not in {"session_id","source","label",
                                                                 "label_int","start_time","source_file"}]
    X_cols = [c for c in X_cols if c in win_row.index]
    X_vec  = win_row[X_cols].fillna(0.0).values

    # Predict
    pred_label, proba, action = predict_window(model, X_vec, calib_cfg)

    # ── Action badge ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if action == "ALERT":
            st.markdown(f"""
            <div class="alert-banner">
                <div style="font-size:2rem; font-weight:700; color:white;">🚨 ALERT</div>
                <div style="color:#ffcccc; margin-top:4px;">Crash detected — high confidence</div>
            </div>
            """, unsafe_allow_html=True)
        elif action == "CONFIRM":
            st.markdown(f"""
            <div class="confirm-banner">
                <div style="font-size:1.8rem; font-weight:700; color:white;">⚠️ CONFIRM</div>
                <div style="color:#ffe0a0; margin-top:4px;">Uncertain — monitoring</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ignore-banner">
                <div style="font-size:1.8rem; font-weight:700; color:white;">✅ IGNORE</div>
                <div style="color:#c0ffd4; margin-top:4px;">Normal operation</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#8b949e; font-size:0.78rem; text-transform:uppercase;">Prediction</div>
            <div style="margin-top:6px;">
                <span class="label-badge label-badge-{pred_label}">{pred_label}</span>
            </div>
        </div>
        <div class="metric-card" style="margin-top:8px;">
            <div style="color:#8b949e; font-size:0.78rem; text-transform:uppercase;">True Label</div>
            <div style="margin-top:6px;">
                <span class="label-badge label-badge-{true_label}">{true_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        p_crash = float(proba[LABEL2INT["CRASH"]])
        correct = "✓ Correct" if pred_label == true_label else "✗ Wrong"
        correct_color = "#3fb950" if pred_label == true_label else "#f85149"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#8b949e; font-size:0.78rem;">P(CRASH)</div>
            <div style="font-size:1.6rem; font-weight:700; color:#da3633;">{p_crash:.1%}</div>
        </div>
        <div class="metric-card" style="margin-top:8px;">
            <div style="color:#8b949e; font-size:0.78rem;">Match</div>
            <div style="font-size:1.1rem; font-weight:600; color:{correct_color};">{correct}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Alert simulation ──────────────────────────────────────────────────────
    if action == "ALERT":
        write_alert_log(sel_session, win_idx, proba, action)
        _render_alert_simulation(sel_session, win_idx, proba)

    # ── Signal + Proba row ─────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 IMU Signal", "📊 Probabilities", "🔍 SHAP Explanations"])

    with tab1:
        if windows_df is not None:
            win_rows = windows_df[windows_df["session_id"] == sel_session].reset_index(drop=True)
            if win_idx < len(win_rows):
                hz = float(win_rows.iloc[win_idx].get("sampling_hz", 100))
                signals, n = get_window_signals(win_rows, win_idx)
                st.plotly_chart(plot_signals(signals, n, hz, true_label),
                                use_container_width=True)
            else:
                st.info("Window signal data not available for this window index.")
        else:
            st.info("windows.parquet not found — run make_windows.py to enable signal plots.")

    with tab2:
        st.plotly_chart(plot_proba(proba), use_container_width=True)
        st.markdown(f"""
        | Class | Probability | Rank |
        |-------|------------|------|
        {"".join(f"| **{l}** | {proba[i]*100:.1f}% | {sorted(proba, reverse=True).index(proba[i])+1} |" + chr(10) for i, l in enumerate(LABELS))}
        """)

    with tab3:
        shap_fig = plot_shap_global()
        if shap_fig:
            st.plotly_chart(shap_fig, use_container_width=True)
            st.caption("Global SHAP importance averaged across all test windows. "
                       "Run xai.py to generate local explanations for this specific window.")
        else:
            st.info("SHAP data not found. Run `python src/xai.py` to generate explanations.")

    # ── Footer stats ──────────────────────────────────────────────────────────
    with st.expander("📋 Session Info & Model Details"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            **Session:** `{sel_session}`  
            **Source:** `{sel_src}`  
            **Window:** {win_idx + 1} / {n_windows}  
            **Start time:** {win_row.get('start_time', 'N/A')}  
            **Source file:** `{win_row.get('source_file', 'N/A')}`
            """)
        with col_b:
            st.markdown(f"""
            **Model:** M-tree XGBoost (4-class)  
            **Calibration:** Isotonic regression  
            **τ_A (ALERT):** {calib_cfg['tau_a']:.2f}  
            **τ_C (CONFIRM):** {calib_cfg['tau_c']:.2f}  
            **δ (margin):** {calib_cfg['delta']:.2f}
            """)


def _render_alert_simulation(session_id: str, win_idx: int, proba: np.ndarray):
    """Render the 30-second emergency countdown simulation."""
    st.markdown("---")
    st.markdown("""
    <div class="alert-banner" style="padding:24px;">
        <div style="font-size:1.4rem; font-weight:700; color:white; margin-bottom:8px;">
            🚨 EMERGENCY RESPONSE SIMULATION
        </div>
        <div style="color:#ffcccc; font-size:0.9rem;">
            ⚠️ SIMULATION ONLY — Not connected to emergency services
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:#da3633;">
            <div style="color:#8b949e; font-size:0.78rem;">P(CRASH)</div>
            <div style="font-size:2.2rem; font-weight:700; color:#da3633;">{proba[LABEL2INT['CRASH']]:.1%}</div>
            <div style="color:#ff7b72; font-size:0.8rem; margin-top:4px;">Above ALERT threshold</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        now = datetime.datetime.now()
        st.markdown(f"""
        **📍 Sample Alert Message** (SIMULATION)
        > *"⚠️ Possible motorcycle crash detected at approximately*  
        > *{now.strftime('%H:%M:%S')}, near session `{session_id}`.*  
        > *Automated safety system activated. This is a SIMULATION —*  
        > *no real services have been contacted."*
        """)

    col_count, col_cancel = st.columns([2, 1])
    with col_count:
        countdown_placeholder = st.empty()
    with col_cancel:
        st.markdown("<br>", unsafe_allow_html=True)
        cancelled = st.button("🛑 Cancel Alert", key=f"cancel_{win_idx}", type="primary")

    if not cancelled:
        if f"alert_start_{win_idx}" not in st.session_state:
            st.session_state[f"alert_start_{win_idx}"] = time.time()
        elapsed = time.time() - st.session_state[f"alert_start_{win_idx}"]
        remaining = max(0, 30 - int(elapsed))
        countdown_placeholder.markdown(f"""
        <div class="countdown-ring">{remaining:02d}s</div>
        <div style="text-align:center; color:#8b949e; font-size:0.82rem;">
            Simulated emergency countdown
        </div>
        """, unsafe_allow_html=True)
        if remaining > 0:
            time.sleep(1)
            st.rerun()
        else:
            countdown_placeholder.markdown("""
            <div style="text-align:center; color:#da3633; font-weight:700; font-size:1.2rem;">
                ✓ Simulated alert sent (SIMULATION ONLY)
            </div>
            """, unsafe_allow_html=True)
    else:
        st.session_state.pop(f"alert_start_{win_idx}", None)
        countdown_placeholder.markdown("""
        <div style="text-align:center; color:#3fb950; font-weight:600; font-size:1.1rem;">
            ✓ Alert cancelled by user
        </div>
        """, unsafe_allow_html=True)

    st.caption("ℹ️ Log entry written to results/alert_log.csv")


def _show_pipeline_status():
    """Show pipeline artifact status."""
    st.markdown("### Pipeline Artifact Status")
    items = [
        ("harmonised.parquet",     (Path(__file__).parent.parent / "data/processed/harmonised.parquet").exists()),
        ("windows.parquet",        (Path(__file__).parent.parent / "data/processed/windows.parquet").exists()),
        ("features.parquet",       (Path(__file__).parent.parent / "data/processed/features.parquet").exists()),
        ("mtree_xgb.joblib",       (MODELS_DIR / "mtree_xgb.joblib").exists()),
        ("calibration_config.json",(MODELS_DIR / "calibration_config.json").exists()),
        ("shap_global.png",        (FIGURES_DIR / "shap_global.png").exists()),
    ]
    for name, ok in items:
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} `{name}`")


def _show_pipeline_status_sidebar():
    items = [
        ("Features",  FEATURES_PARQUET.exists()),
        ("Model",     (MODELS_DIR / "mtree_xgb.joblib").exists()),
        ("Calibration",(MODELS_DIR / "calibration_config.json").exists()),
        ("SHAP",      (RESULTS_DIR / "shap_global_importance.csv").exists()),
    ]
    for name, ok in items:
        color = "#3fb950" if ok else "#f85149"
        icon  = "●" if ok else "○"
        st.markdown(
            f"<span style='color:{color}; font-size:0.85rem;'>{icon} {name}</span>",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
