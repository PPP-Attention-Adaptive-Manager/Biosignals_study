import pandas as pd
import os

BASE = os.path.expanduser("~/biosignals_data/data/clare/doi-10.5683-sp3-h0aelt")
PID = "1026"   # first participant with complete data across all 4 levels

print("=" * 70)
print("ECG")
print("=" * 70)
for f in ["ecg_data_baseline_0.csv", "ecg_data_experiment_0.csv"]:
    path = os.path.join(BASE, "ECG", PID, f)
    df = pd.read_csv(path, nrows=5)
    print(f"\n{f}")
    print(f"  columns: {list(df.columns)}")
    print(f"  shape (first 5 rows): {df.shape}")
    print(df.head(3).to_string())
    full = pd.read_csv(path)
    print(f"  full length: {len(full)} rows")
    if "timestamp" in [c.lower() for c in df.columns] or "time" in [c.lower() for c in df.columns]:
        tcol = [c for c in df.columns if "time" in c.lower()][0]
        dur = full[tcol].iloc[-1] - full[tcol].iloc[0]
        print(f"  duration: {dur:.1f} (units unclear, check README says seconds)")

print("\n" + "=" * 70)
print("EEG")
print("=" * 70)
for f in ["eeg_baseline_0.csv", "eeg_data_exp_0.csv"]:
    path = os.path.join(BASE, "EEG", PID, f)
    df = pd.read_csv(path, nrows=5)
    print(f"\n{f}")
    print(f"  columns: {list(df.columns)}")
    print(df.head(3).to_string())
    full = pd.read_csv(path)
    print(f"  full length: {len(full)} rows")

print("\n" + "=" * 70)
print("EDA")
print("=" * 70)
path = os.path.join(BASE, "EDA", PID, "eda_data_experiment_0.csv")
df = pd.read_csv(path, nrows=5)
print(f"  columns: {list(df.columns)}")
print(df.head(3).to_string())

print("\n" + "=" * 70)
print("LABELS")
print("=" * 70)
path = os.path.join(BASE, "Labels", f"{PID}.csv")
df = pd.read_csv(path)
print(f"  columns: {list(df.columns)}")
print(f"  shape: {df.shape}")
print(df.head(10).to_string())
