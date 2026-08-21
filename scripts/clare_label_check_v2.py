import pandas as pd, os

BASE = os.path.expanduser("~/biosignals_data/data/clare/doi-10.5683-sp3-h0aelt")
labels_dir = os.path.join(BASE, "Labels")
pids_all = sorted([f.replace('.csv','') for f in os.listdir(labels_dir)])

increasing, not_increasing, skipped = [], [], []

for pid in pids_all:
    path = os.path.join(labels_dir, f"{pid}.csv")
    df = pd.read_csv(path)
    have = [c for c in ["level_0","level_1","level_2","level_3"] if c in df.columns]
    if len(have) < 4:
        print(f"  P{pid}: only {have} -- SKIPPED (incomplete)")
        skipped.append(pid)
        continue
    means = df[have].mean()
    is_monotonic = all(means[f"level_{i}"] <= means[f"level_{i+1}"] + 0.5
                       for i in range(3))
    print(f"  P{pid}: " + "  ".join(f"L{i}={means[f'level_{i}']:.2f}" for i in range(4))
          + f"  {'OK' if is_monotonic else 'NOT INCREASING'}")
    (increasing if is_monotonic else not_increasing).append(pid)

n_total = len(increasing) + len(not_increasing)
print(f"\n=== SUMMARY ===")
print(f"Skipped (incomplete labels): {len(skipped)} -> {skipped}")
print(f"Monotonically increasing: {len(increasing)}/{n_total} ({100*len(increasing)/n_total:.0f}%)")
print(f"NOT increasing:           {len(not_increasing)}/{n_total} ({100*len(not_increasing)/n_total:.0f}%)")
print(f"  -> {not_increasing}")
