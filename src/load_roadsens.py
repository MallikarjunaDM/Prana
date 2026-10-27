"""
load_roadsens.py — RoadSens-4M road-anomaly dataset loader
────────────────────────────────────────────────────────────
Loads the pre-extracted CSVs from RoadSens-4M.
If RAR archives haven't been extracted yet, attempts extraction using
7z.exe (bundled in prana/7zip_files/), rarfile, or patoolib.

RoadSens column mapping (confirmed from real data):
  accelerometer_x/y/z  → acc_x/y/z  (m/s²)
  gyroscope_x/y/z      → gyro_x/y/z (rad/s)
  seconds_elapsed       → timestamp

Label logic:
  Normal Road CSVs (no annotation) → label = NORMAL
  Road Anomaly CSVs:
    - rows with annotation_text == 'Pothole' → POTHOLE
    - rows with annotation_text == 'Bump'    → BUMP
    - rows with NaN annotation_text          → NORMAL (road between events)

RoadSens is used only for NORMAL, POTHOLE, and BUMP — NOT for CRASH.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ROADSENS_SOURCE, ROADSENS_DIR, DATA_PROCESSED,
    SCHEMA_COLS, RANDOM_SEED
)

logger = logging.getLogger(__name__)

# ── RAR archive paths (inside the downloaded RoadSens-4M folder) ───────────────
_COMBINED_INNER = (
    ROADSENS_SOURCE
    / "Combined CSV with GIS and Weather Data"
    / "Combined CSV with GIS and Weather Data"
)
_NORMAL_RAR  = _COMBINED_INNER / "Normal Road (No Annotation)" / "Normal Road (No Annotation).rar"
_ANOMALY_RAR = _COMBINED_INNER / "Road Anomalies" / "Road Anomalies.rar"

# ── Destination extraction directories ────────────────────────────────────────
_NORMAL_EXTRACT  = ROADSENS_DIR / "normal_road"
_ANOMALY_EXTRACT = ROADSENS_DIR / "road_anomalies"

# ── Bundled 7z.exe (extracted from 7-Zip installer in previous setup step) ────
_BUNDLED_7Z = Path(__file__).parent.parent / "7zip_files" / "7z.exe"


# ── RAR extraction ─────────────────────────────────────────────────────────────
def _extract_rar(rar_path: Path, dest_dir: Path) -> bool:
    """
    Extract a RAR archive using multiple strategies (whichever works):
    1. Bundled 7z.exe (prana/7zip_files/7z.exe)
    2. 7-Zip from PATH or standard install location
    3. rarfile (requires unrar binary on PATH)
    4. patoolib (uses any available system tool)
    Returns True on success.
    """
    if not rar_path.exists():
        logger.warning(f"RAR not found: {rar_path}")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    already = list(dest_dir.glob("*.csv"))
    if already:
        logger.info(f"  {len(already)} CSV(s) already in {dest_dir} — skipping extraction.")
        return True

    logger.info(f"Extracting {rar_path.name} to {dest_dir} ...")

    import subprocess, shutil

    # Strategy 1: Bundled 7z.exe
    if _BUNDLED_7Z.exists():
        try:
            result = subprocess.run(
                [str(_BUNDLED_7Z), "x", str(rar_path), f"-o{dest_dir}", "-y"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                logger.info("  Extracted via bundled 7z.exe.")
                return True
            else:
                logger.debug(f"  bundled 7z.exe failed: {result.stderr[:200]}")
        except Exception as e:
            logger.debug(f"  bundled 7z.exe exception: {e}")

    # Strategy 2: System 7-Zip
    for sevenzip in [
        shutil.which("7z"),
        shutil.which("7za"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]:
        if sevenzip and Path(sevenzip).exists():
            try:
                result = subprocess.run(
                    [sevenzip, "x", str(rar_path), f"-o{dest_dir}", "-y"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    logger.info(f"  Extracted via {sevenzip}.")
                    return True
            except Exception as e:
                logger.debug(f"  {sevenzip} failed: {e}")

    # Strategy 3: rarfile
    try:
        import rarfile
        with rarfile.RarFile(str(rar_path)) as rf:
            rf.extractall(str(dest_dir))
        logger.info("  Extracted via rarfile.")
        return True
    except Exception as e:
        logger.debug(f"  rarfile failed: {e}")

    # Strategy 4: patoolib
    try:
        import patoolib
        patoolib.extract_archive(str(rar_path), outdir=str(dest_dir), verbosity=-1)
        logger.info("  Extracted via patoolib.")
        return True
    except Exception as e:
        logger.debug(f"  patoolib failed: {e}")

    logger.error(
        f"Could not extract {rar_path.name}. "
        "Ensure 7-Zip is available (prana/7zip_files/7z.exe or system PATH). "
        "RoadSens data from this archive will be skipped."
    )
    return False


def extract_all():
    """Extract both Normal Road and Road Anomalies RARs if not already done."""
    ok1 = _extract_rar(_NORMAL_RAR,  _NORMAL_EXTRACT)
    ok2 = _extract_rar(_ANOMALY_RAR, _ANOMALY_EXTRACT)
    return ok1, ok2


# ── Anomaly label mapping ──────────────────────────────────────────────────────
def _map_annotation(raw) -> str:
    """Map RoadSens annotation_text values to PRANA labels."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "NORMAL"
    s = str(raw).strip().lower()
    if "pothole" in s or "hole" in s or "pit" in s:
        return "POTHOLE"
    if "bump" in s or "speed" in s or "hump" in s or "ridge" in s:
        return "BUMP"
    if "normal" in s or "smooth" in s or "none" in s:
        return "NORMAL"
    logger.debug(f"Unknown RoadSens annotation '{raw}' → NORMAL")
    return "NORMAL"


# ── Single CSV loader ──────────────────────────────────────────────────────────
def _load_csv(csv_path: Path, default_label: str | None, session_prefix: str) -> pd.DataFrame | None:
    """
    Load one RoadSens CSV and map to common schema.

    Parameters
    ----------
    csv_path     : path to the CSV
    default_label: if not None, all rows get this label (Normal Road files)
    session_prefix: prefix for session_id (e.g. 'rs_normal' or 'rs_anomaly')
    """
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except Exception as e:
        logger.warning(f"Could not read {csv_path.name}: {e}")
        return None

    cols = set(df.columns)

    # ── Accelerometer ──
    # RoadSens uses 'accelerometer_x/y/z' (confirmed from data inspection)
    if "accelerometer_x" in cols:
        acc_x = df["accelerometer_x"].astype(float)
        acc_y = df["accelerometer_y"].astype(float)
        acc_z = df["accelerometer_z"].astype(float) if "accelerometer_z" in cols else pd.Series(np.nan, index=df.index)
    else:
        # Fallback: heuristic column detection
        acc_cols = [c for c in df.columns if "acc" in c.lower() and not "uncalibrated" in c.lower()
                    and not "total" in c.lower() and not "gravity" in c.lower()]
        x_cols = [c for c in acc_cols if c.endswith("_x") or c.lower() == "accx"]
        y_cols = [c for c in acc_cols if c.endswith("_y") or c.lower() == "accy"]
        z_cols = [c for c in acc_cols if c.endswith("_z") or c.lower() == "accz"]
        if not x_cols or not y_cols:
            logger.debug(f"  Skipping {csv_path.name}: cannot find acc_x/y columns")
            return None
        acc_x = df[x_cols[0]].astype(float)
        acc_y = df[y_cols[0]].astype(float)
        acc_z = df[z_cols[0]].astype(float) if z_cols else pd.Series(np.nan, index=df.index)

    # ── Gyroscope ──
    if "gyroscope_x" in cols:
        gyro_x = df["gyroscope_x"].astype(float)
        gyro_y = df["gyroscope_y"].astype(float)
        gyro_z = df["gyroscope_z"].astype(float) if "gyroscope_z" in cols else pd.Series(np.nan, index=df.index)
    else:
        gyro_x = pd.Series(np.nan, index=df.index)
        gyro_y = pd.Series(np.nan, index=df.index)
        gyro_z = pd.Series(np.nan, index=df.index)

    # ── Timestamp ──
    if "seconds_elapsed" in cols:
        timestamp = pd.to_numeric(df["seconds_elapsed"], errors="coerce")
    else:
        timestamp = pd.Series(np.arange(len(df), dtype=float) * 0.01)

    # ── Label ──
    if default_label is not None:
        label = pd.Series([default_label] * len(df))
    elif "annotation_text" in cols:
        label = df["annotation_text"].apply(_map_annotation)
    else:
        label = pd.Series(["NORMAL"] * len(df))

    # ── Assemble output ──
    acc_mag = np.sqrt(acc_x**2 + acc_y.fillna(0)**2 + acc_z.fillna(0)**2)

    out = pd.DataFrame({
        "source":      "roadsens",
        "session_id":  f"{session_prefix}_{csv_path.stem}",
        "source_file": csv_path.name,
        "timestamp":   timestamp,
        "acc_x":       acc_x,
        "acc_y":       acc_y,
        "acc_z":       acc_z,
        "gyro_x":      gyro_x,
        "gyro_y":      gyro_y,
        "gyro_z":      gyro_z,
        "acc_mag":     acc_mag,
        "label":       label,
    })

    out.dropna(subset=["acc_x", "acc_y", "timestamp"], inplace=True)
    return out[SCHEMA_COLS]


# ── Main loader ────────────────────────────────────────────────────────────────
def load_roadsens() -> pd.DataFrame:
    """
    Extract RARs (if needed) and load all RoadSens CSVs.

    Returns
    -------
    pd.DataFrame with SCHEMA_COLS, labels in {NORMAL, POTHOLE, BUMP}
    """
    ok_normal, ok_anomaly = extract_all()
    frames = []

    # ── Normal road CSVs → all rows labeled NORMAL ──
    if ok_normal:
        normal_csvs = sorted(_NORMAL_EXTRACT.glob("**/*.csv"))
        logger.info(f"Normal Road: {len(normal_csvs)} CSV files found.")
        for p in normal_csvs:
            df = _load_csv(p, default_label="NORMAL", session_prefix="rs_normal")
            if df is not None and len(df) > 0:
                frames.append(df)
    else:
        logger.warning("Normal Road extraction failed or skipped.")

    # ── Road anomaly CSVs → annotation_text decides label per row ──
    if ok_anomaly:
        anomaly_csvs = sorted(_ANOMALY_EXTRACT.glob("**/*.csv"))
        logger.info(f"Road Anomalies: {len(anomaly_csvs)} CSV files found.")
        for p in anomaly_csvs:
            df = _load_csv(p, default_label=None, session_prefix="rs_anomaly")
            if df is not None and len(df) > 0:
                frames.append(df)
    else:
        logger.warning("Road Anomalies extraction failed or skipped.")

    if not frames:
        logger.warning("No RoadSens data loaded. Returning empty DataFrame.")
        return pd.DataFrame(columns=SCHEMA_COLS)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"RoadSens total: {len(combined):,} rows | "
        f"Labels: {combined['label'].value_counts().to_dict()}"
    )
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_roadsens()
    if len(df) > 0:
        out = DATA_PROCESSED / "roadsens_raw.parquet"
        df.to_parquet(out, index=False)
        print(f"\n✓ Saved {len(df):,} rows → {out}")
        print(df["label"].value_counts())
    else:
        print("No data loaded from RoadSens.")
