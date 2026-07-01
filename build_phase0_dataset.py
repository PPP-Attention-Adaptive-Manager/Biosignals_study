"""
build_phase0_dataset.py  (v2 — simplified)
=============================================
Builds the Phase 0 cache for the Cog Lab gate test.

This REPLACES the previous build_phase0_dataset.py entirely. That version
assumed compute_swell_counts() took a pandas DataFrame with a 'button'
column — that interface didn't match how extract_session() actually
works (raw numpy arrays + boolean masks). This version calls the now-
patched build_xy(), which does all windowing and alignment internally —
there is no separate windowing loop left to write here.

For each subject, saves:
  X_counts        — (n_windows, 16) SWELL-style counts (Version 1's input)
  X_input_biosig  — (n_windows, 3)  hr_mean, hrv_rmssd, eda_scl
                                     (the 3 EXTRA inputs that make Version 2)
  Y_remaining     — (n_windows, 10) the 10 targets BOTH versions try to predict
  starts          — window start timestamps, for sanity-checking alignment

V1 = X_counts                      -> Y_remaining
V2 = concat(X_counts, X_input_biosig) -> Y_remaining
Both predict the SAME targets, so accuracy is directly comparable —
that comparison is the actual gate test (next script, not yet written).
"""

from __future__ import annotations   # FIX — makes tuple[float,float] etc.
                                       # work on Python 3.8, not just 3.9+.
                                       # Must be the first real line in the file.

import os
import json
import numpy as np
import biosignals_targets as BT

COG_LAB_DIR = os.path.expanduser("~/biosignals_data/cog_lab")
OUT_DIR = os.path.expanduser("~/biosignals_data/proxy_cache_swellstyle")
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE = {"S2", "S17"}   # S2 = no HCI, S17 = byte-identical duplicate of S1

INPUT_TARGETS = ["hr_mean", "hrv_rmssd", "eda_scl"]
REMAINING_TARGETS = [
    "acc_movement", "acc_jerk",
    "eda_tonic_slope", "eda_phasic_count",
    "resp_bpm",
    "eeg_theta_alpha", "eeg_engagement", "eeg_alpha_asym",
    "fnirs_hbo_slope_L", "fnirs_hbo_slope_R",
]


def load_task_window(subject_dir: str) -> tuple[float, float]:
    """
    Reads PB_description.json, returns (task_start_ts, task_end_ts).

    CONFIRMED real structure (verified against S1):
        {
          "<lesson name>": {
            "Task": [start_ts, end_ts],
            "Eliminate": [...]
          }
        }
    The top-level key is the lesson name and may differ per subject
    (S1's is "ECG lesson") — grabbed generically below rather than
    hardcoded, since it's the only key at that level.

    KNOWN LIMITATION — not yet handled: "Eliminate" may list sub-intervals
    to exclude WITHIN the Task window (artifacts, interruptions). It was
    empty for S1. This function only crops to the outer Task start/end —
    it does NOT remove any Eliminate sub-intervals. If other subjects have
    a non-empty Eliminate list, windows overlapping those periods will
    still be included. Check before trusting results past S1:

        for sid in subjects:
            pb = json.load(open(f"cog_lab/{sid}/D3_{sid}_PB_description.json"))
            elim = next(iter(pb.values()))["Eliminate"]
            if elim: print(sid, elim)
    """
    pb_files = [f for f in os.listdir(subject_dir) if f.endswith("PB_description.json")]
    pb_path = os.path.join(subject_dir, pb_files[0])
    with open(pb_path) as f:
        pb = json.load(f)
    lesson_data = next(iter(pb.values()))   # top-level key name varies per subject
    t_start, t_end = lesson_data["Task"]
    return float(t_start), float(t_end)


def build_subject(subject_id: str):
    subject_dir = os.path.join(COG_LAB_DIR, subject_id)
    t_start, t_end = load_task_window(subject_dir)

    X, swell_counts, Y, starts, x_names, swell_names, y_names = BT.build_xy(
        subject_dir, subject_id, source="coglab",
        t_start=t_start, t_end=t_end,
    )

    if swell_counts.shape[0] == 0:
        raise ValueError(
            f"{subject_id}: 0 windows produced. This usually means the "
            f"Task timestamps ({t_start}, {t_end}) don't share the same "
            f"scale as mouse/keyboard 'time' columns — check whether one "
            f"is epoch time and the other is session-relative."
        )

    idx = {name: i for i, name in enumerate(y_names)}
    inputs    = Y[:, [idx[n] for n in INPUT_TARGETS]]       # hr_mean, hrv_rmssd, eda_scl
    remaining = Y[:, [idx[n] for n in REMAINING_TARGETS]]   # the 10 V2 prediction targets

    np.savez(
        os.path.join(OUT_DIR, f"{subject_id}.npz"),
        X_counts=swell_counts,
        X_input_biosig=inputs,
        Y_remaining=remaining,
        starts=starts,
        swell_names=np.array(swell_names),
        input_names=np.array(INPUT_TARGETS),
        remaining_names=np.array(REMAINING_TARGETS),
    )
    return swell_counts.shape[0]


if __name__ == "__main__":
    subjects = sorted(
        d for d in os.listdir(COG_LAB_DIR)
        if d.startswith("S") and d[1:].isdigit() and d not in EXCLUDE
    )
    print(f"Building Phase 0 dataset for {len(subjects)} subjects: {subjects}\n")
    for sid in subjects:
        try:
            n_windows = build_subject(sid)
            print(f"  {sid}: done — {n_windows} windows")
        except Exception as e:
            print(f"  {sid}: FAILED — {type(e).__name__}: {e}")

    print(f"\nSaved to: {OUT_DIR}")
    print("Next: sanity-check checklist below, then write the V1-vs-V2 gate script.")
