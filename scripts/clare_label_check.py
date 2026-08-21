"""
Checks two things directly from CLARE's own data, no internet needed:
1. Is complexity level ordering fixed (0->1->2->3 for everyone) or
   randomized? Fixed order would confound "level" with "time in session".
2. Do self-reported Labels actually increase with level number,
   per participant? If not, that's evidence of a real mislabel.
"""
import pandas as pd, os, glob

BASE = os.path.expanduser("~/biosignals_data/data/clare/doi-10.5683-sp3-h0aelt")

print("=== 1. SESSION ORDERING (from ECG timestamps if absolute clock exists) ===")
# CLARE timestamps are relative (start at 0 per file), so we can't recover
# absolute session order from timestamps alone. Check file modification
# times as a weak proxy, and check for any session/trial index elsewhere.
ecg_dir = os.path.join(BASE, "ECG")
pids = sorted(os.listdir(ecg_dir))[:3]
for pid in pids:
    files = sorted(glob.glob(os.path.join(ecg_dir, pid, "ecg_data_experiment_*.csv")))
    print(f"\nP{pid}:")
    for f in files:
        mtime = os.path.getmtime(f)
        print(f"  {os.path.basename(f)}  mtime={mtime}")

print("\n=== 2. LABEL vs LEVEL CONSISTENCY, per participant ===")
labels_dir = os.path.join(BASE, "Labels")
pids_all = sorted([f.replace('.csv','') for f in os.listdir(labels_dir)])

increasing, not_increasing = [], []
for pid in pids_all:
    path = os.path.join(labels_dir, f"{pid}.csv")
    df = pd.read_csv(path)
    means = df.mean()
    is_monotonic = all(means[f"level_{i}"] <= means[f"level_{i+1}"] + 0.5
                       for i in range(3))  # small tolerance
    print(f"  P{pid}: " + "  ".join(f"L{i}={means[f'level_{i}']:.2f}" for i in range(4))
          + f"  {'OK' if is_monotonic else 'NOT INCREASING'}")
    (increasing if is_monotonic else not_increasing).append(pid)

print(f"\nMonotonically increasing (or flat): {len(increasing)}/{len(pids_all)}")
print(f"NOT increasing: {len(not_increasing)}/{len(pids_all)}  -> {not_increasing}")
