# PRANA Dataset Notes
Generated: 2026-10-27 06:40 UTC

## Summary

```
  source  n_rows  n_sessions                                                 label_distribution                                                                           missing_per_col constant_channels  duplicate_rows  ts_ordering_issues clipping_suspects  estimated_hz  tiny_sessions_lt50
   darus  211238          41 {"NORMAL": 181587, "CRASH": 14047, "BUMP": 11703, "POTHOLE": 3901} {"acc_x": 0, "acc_y": 0, "acc_z": 0, "gyro_x": 0, "gyro_y": 0, "gyro_z": 0, "acc_mag": 0}              none              11                   2              none        2000.0                   0
roadsens  616863         101              {"NORMAL": 332026, "BUMP": 171380, "POTHOLE": 113457} {"acc_x": 0, "acc_y": 0, "acc_z": 0, "gyro_x": 1, "gyro_y": 1, "gyro_z": 1, "acc_mag": 0}              none               0                   0              none         100.0                   0
  kaggle    8000          80                                    {"NORMAL": 7000, "CRASH": 1000} {"acc_x": 0, "acc_y": 0, "acc_z": 0, "gyro_x": 0, "gyro_y": 0, "gyro_z": 0, "acc_mag": 0}              none               0                   0              none           1.0                   0
```

## Source-Specific Notes

### DaRUS Motorcycle Simulation Dataset
- **Type**: Simulation (multi-body dynamics), NOT real crash recordings.
- **Label source**: `CrashLabelML` (0=NORMAL, 1=CRASH) for main files;
  scenario filename for control files.
- **Curbstone policy**: Mapped to BUMP (user decision, 12 Oct 2026).
- **Units**: Linear acceleration m/s², angular velocity rad/s.
- **Session grouping**: Each CSV file = one session.
- **Dev mode**: 20% random subsample used during development.

### RoadSens-4M
- **Type**: Real road-anomaly recordings (NOT crash data).
- **Usage**: NORMAL, POTHOLE, BUMP only — no CRASH labels assigned from peaks.
- **Gyroscope**: May be absent in some files — left as NaN, NOT zero-filled.
- **Licence**: Confirm Figshare attribution before publication.
- **RAR extracted via Python rarfile.**

### Kaggle Smartphone IMU Dataset
- **Type**: GENERATED/SIMULATED data (synthetic timestamps from 2026-01-01).
- **⚠ Limitation**: This is NOT verified real-world motorcycle crash data.
  Sensor is a smartphone, not a motorcycle-mounted IMU.
  Should NOT be described as real crash recordings.
- **Classes available**: CRASH and NORMAL only (no POTHOLE/BUMP).
- **Session grouping**: Artificial — every 100 rows = one session (80 groups).
- **Units**: Assumed m/s² (Z-axis ≈9.8 at rest confirms g-scale).

## Known Risks & Shortcut Concerns
- DaRUS is simulation data; generalisation to real motorcycle crashes is unverified.
- Source probe (probe.py) should be run to detect dataset-specific artefacts.
- If source probe achieves high accuracy, investigate responsible features.
- LOSO evaluation should be the primary measure of cross-source generalisation.

## Label Policy
| PRANA Label | DaRUS | RoadSens | Kaggle |
|---|---|---|---|
| NORMAL | CrashLabelML=0, Randomstate_* | Normal Road CSVs | Crash_Label=0 |
| POTHOLE | Pothole_5_Out.csv | Pothole-annotated CSVs | — |
| BUMP | Speedbump/Curbstone CSVs | Bump-annotated CSVs | — |
| CRASH | CrashLabelML=1, ISO*/Iso_* | — | Crash_Label=1 |