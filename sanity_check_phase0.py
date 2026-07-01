"""
sanity_check_phase0.py
========================
Run this BEFORE trusting the gate test results. Checks every open
question flagged during Phase 0 construction:

  1. Shape consistency — X_counts, X_input_biosig, Y_remaining all have
     the same row count per subject (they should, since they came from
     the same build_xy() call, but worth confirming directly).
  2. NaN columns — SnRightClicked/SnDoubleClicked/SnDragged should be
     100% NaN (never observed in Cog Lab's event vocabulary). If they're
     NOT all-NaN, something upstream changed unexpectedly.
  3. HR/RMSSD/SCL plausibility — quick range check. HR should sit
     roughly 40-180 bpm, RMSSD should be a small positive ms value,
     SCL should not contain extreme outliers.
  4. The 'Eliminate' field across all 16 subjects — was empty for S1,
     unknown for the other 15. If non-empty anywhere, those subjects
     may have artifact periods inside the Task window that aren't
     currently excluded.
  5. Mouse Down / Left Click timestamp overlap — checks whether a
     single physical click produces two log rows (which would inflate
     SnLeftClicked / ms_click_rate by 2x).
"""

import os
import json
import numpy as np
import pandas as pd

CACHE_DIR = os.path.expanduser("~/biosignals_data/proxy_cache_swellstyle")
COG_LAB_DIR = os.path.expanduser("~/biosignals_data/cog_lab")
EXCLUDE = {"S2", "S17"}

subjects = sorted(
    d for d in os.listdir(COG_LAB_DIR)
    if d.startswith("S") and d[1:].isdigit() and d not in EXCLUDE
)

print("=" * 70)
print("1. SHAPE CONSISTENCY")
print("=" * 70)
total_windows = 0
for sid in subjects:
    z = np.load(os.path.join(CACHE_DIR, f"{sid}.npz"), allow_pickle=True)
    n_x, n_b, n_y = z["X_counts"].shape[0], z["X_input_biosig"].shape[0], z["Y_remaining"].shape[0]
    ok = "OK" if n_x == n_b == n_y else "MISMATCH"
    print(f"  {sid}: X_counts={n_x} X_input_biosig={n_b} Y_remaining={n_y}  [{ok}]")
    total_windows += n_x
print(f"\n  Total windows across all subjects: {total_windows}")

print()
print("=" * 70)
print("2. NaN COLUMNS — SnRightClicked / SnDoubleClicked / SnDragged")
print("=" * 70)
z0 = np.load(os.path.join(CACHE_DIR, f"{subjects[0]}.npz"), allow_pickle=True)
swell_names = list(z0["swell_names"])
all_X = np.vstack([np.load(os.path.join(CACHE_DIR, f"{s}.npz"), allow_pickle=True)["X_counts"]
                    for s in subjects])
print(f"  {'column':20s} {'%NaN':>8s}  expected")
for i, name in enumerate(swell_names):
    pct_nan = 100 * np.isnan(all_X[:, i]).mean()
    expected = "100% NaN" if name in ("SnRightClicked", "SnDoubleClicked", "SnDragged") else "0% NaN"
    flag = "OK" if (pct_nan > 99 if "100%" in expected else pct_nan < 1) else "UNEXPECTED"
    print(f"  {name:20s} {pct_nan:7.1f}%  {expected:10s} [{flag}]")

print()
print("=" * 70)
print("3. HR / RMSSD / SCL PLAUSIBILITY")
print("=" * 70)
all_biosig = np.vstack([np.load(os.path.join(CACHE_DIR, f"{s}.npz"), allow_pickle=True)["X_input_biosig"]
                         for s in subjects])
input_names = list(z0["input_names"])
for i, name in enumerate(input_names):
    col = all_biosig[:, i]
    finite = col[np.isfinite(col)]
    pct_nan = 100 * (1 - len(finite) / len(col))
    if len(finite) > 0:
        print(f"  {name:12s} min={finite.min():8.2f}  mean={finite.mean():8.2f}  "
              f"max={finite.max():8.2f}  NaN={pct_nan:.1f}%")
    else:
        print(f"  {name:12s} ALL NaN — something is broken")

print("\n  Plausibility check:")
hr_col = all_biosig[:, input_names.index("hr_mean")]
hr_finite = hr_col[np.isfinite(hr_col)]
if len(hr_finite):
    out_of_range = ((hr_finite < 40) | (hr_finite > 180)).mean() * 100
    print(f"    hr_mean outside 40-180 bpm: {out_of_range:.1f}% of windows "
          f"({'OK' if out_of_range < 5 else 'CHECK THIS'})")

print()
print("=" * 70)
print("4. 'Eliminate' FIELD ACROSS ALL SUBJECTS")
print("=" * 70)
any_nonempty = False
for sid in subjects:
    subject_dir = os.path.join(COG_LAB_DIR, sid)
    pb_files = [f for f in os.listdir(subject_dir) if f.endswith("PB_description.json")]
    with open(os.path.join(subject_dir, pb_files[0])) as f:
        pb = json.load(f)
    lesson_data = next(iter(pb.values()))
    elim = lesson_data.get("Eliminate", [])
    if elim:
        print(f"  {sid}: Eliminate = {elim}  <-- NON-EMPTY, not currently excluded from windows")
        any_nonempty = True
if not any_nonempty:
    print("  All 16 subjects have an empty Eliminate list — nothing to fix here.")

print()
print("=" * 70)
print("5. Mouse Down / Left Click TIMESTAMP OVERLAP (double-count check)")
print("=" * 70)
any_overlap = False
for sid in subjects:
    mouse_path = os.path.join(COG_LAB_DIR, sid, "HCI", f"D3_{sid}_mouse.csv")
    df = pd.read_csv(mouse_path)
    clicks = df[df["type"].isin(["Mouse Down", "Left Click"])].sort_values("time")
    if len(clicks) < 2:
        continue
    # check for (Mouse Down, Left Click) pairs sharing the same or near-identical timestamp
    t = clicks["time"].to_numpy()
    types = clicks["type"].to_numpy()
    gaps = np.diff(t)
    same_ts_diff_type = np.sum((gaps < 0.01) & (types[:-1] != types[1:]))
    pct = 100 * same_ts_diff_type / max(len(clicks) - 1, 1)
    if pct > 1:
        print(f"  {sid}: {pct:.1f}% of click rows are <10ms apart with different types "
              f"-- LIKELY DOUBLE-LOGGED")
        any_overlap = True
if not any_overlap:
    print("  No subject shows >1% suspiciously-paired Mouse Down / Left Click rows.")
    print("  SnLeftClicked / ms_click_rate likely safe to use as-is.")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("Review any [UNEXPECTED] or [MISMATCH] flags above, and any non-empty")
print("Eliminate lists or double-count warnings, before trusting the gate")
print("test results in the next script.")
