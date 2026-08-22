"""
diagnose_clock_offset.py
===========================
Checks whether the remaining ~20-60s mouse_gap is a roughly CONSTANT
clock-origin offset (fixable with a single correction) or genuinely
variable per subject/window (a harder, real problem).

Reports the SIGNED offset (mouse_start - target_start), not just the
absolute gap -- a consistent sign and magnitude across subjects would
confirm a fixed clock-origin mismatch between proxy_cache_swellstyle's
`starts` array and the raw mouse CSV's own `time` column.

Run from: ~/biosignals_data/
"""
import os, sys
import numpy as np

BASE = os.path.expanduser("~/biosignals_data")
sys.path.insert(0, os.path.join(BASE, "aam_proxy", "encoders"))
CACHE_DIR = os.path.join(BASE, "data", "cache", "proxy_cache_swellstyle")
MOUSE_STATS_DIR = os.path.join(BASE, "aam_proxy", "encoders", "mouse_stats")
COG_LAB_DIR = os.path.join(BASE, "data", "cog_lab")
EXCLUDE = {"S2", "S17"}

all_dirs = sorted(d for d in os.listdir(COG_LAB_DIR)
                  if d.startswith("S") and d[1:].isdigit())
subjects = [d for d in all_dirs if d not in EXCLUDE]

print(f"{'subj':6s} {'median_offset_s':>16s} {'mean_offset_s':>14s} "
      f"{'std_offset_s':>13s} {'target_t0':>14s} {'mouse_t0':>14s}")
print("-" * 78)

offsets_all = []
for sid in subjects:
    npz_path = os.path.join(CACHE_DIR, f"{sid}.npz")
    starts_path = os.path.join(MOUSE_STATS_DIR, f"{sid}_mouse_starts.npy")
    if not (os.path.isfile(npz_path) and os.path.isfile(starts_path)):
        continue
    z = np.load(npz_path, allow_pickle=True)
    target_starts = z["starts"]
    mouse_starts = np.load(starts_path)

    offsets = []
    for t in target_starts:
        gaps = mouse_starts - t
        nearest = gaps[np.argmin(np.abs(gaps))]
        offsets.append(nearest)
    offsets = np.array(offsets)
    offsets_all.extend(offsets.tolist())

    print(f"  {sid:6s} {np.median(offsets):16.2f} {np.mean(offsets):14.2f} "
          f"{np.std(offsets):13.2f} {target_starts[0]:14.1f} {mouse_starts[0]:14.1f}")

offsets_all = np.array(offsets_all)
print(f"\nPopulation-wide median offset: {np.median(offsets_all):.2f}s")
print(f"Population-wide std of offset: {np.std(offsets_all):.2f}s")
print()
print("If median_offset is roughly the SAME sign/magnitude across most")
print("subjects and population std is small relative to that median --")
print("this is a fixable constant clock-origin correction.")
print("If it varies a lot subject-to-subject with no consistent sign,")
print("that points to a real per-subject synchronization issue, not a")
print("single global constant.")


print("\n" + "="*78)
print("FOLLOW-UP: is target_starts itself evenly spaced at 30s, or gappy?")
print("="*78)
print(f"{'subj':6s} {'n_targets':>10s} {'gap_median_s':>13s} {'gap_std_s':>10s} "
      f"{'gap_max_s':>10s} {'pct_gaps_gt_35s':>16s}")
print("-" * 68)

for sid in subjects:
    npz_path = os.path.join(CACHE_DIR, f"{sid}.npz")
    if not os.path.isfile(npz_path):
        continue
    z = np.load(npz_path, allow_pickle=True)
    target_starts = np.sort(z["starts"])
    gaps = np.diff(target_starts)
    pct_gt35 = 100 * (gaps > 35).mean()
    print(f"  {sid:6s} {len(target_starts):10d} {np.median(gaps):13.2f} "
          f"{np.std(gaps):10.2f} {np.max(gaps):10.2f} {pct_gt35:16.1f}%")

print("\nIf gap_median is ~30s with LOW std and few gaps>35s -- target")
print("windows are evenly spaced, and the offset variance seen above is")
print("a real mystery worth digging into further.")
print("If gaps are irregular / median far from 30s / many gaps>35s --")
print("that fully explains the offset variance as benign discretization,")
print("not a bug. Safe to treat 'nearest match' alignment as the best")
print("available, document the ~20s jitter as a known limitation, and")
print("proceed.")
