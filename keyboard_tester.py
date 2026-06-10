import os
import pandas as pd, numpy as np

DATASET_DIR = os.path.expanduser("~/biosignals_data/cog_lab")

def analyze_kb(path):
    df = pd.read_csv(path)
    t  = df["time"].to_numpy(float)
    kc = df["key_code"].to_numpy()
    # consecutive same-key runs
    runs, i = [], 0
    while i < len(kc):
        j = i
        while j + 1 < len(kc) and kc[j + 1] == kc[i]:
            j += 1
        runs.append((i, j)); i = j + 1
    rl = np.array([e - s + 1 for s, e in runs])
    first, sub = [], []
    for s, e in runs:
        if e - s + 1 >= 4:
            d = np.diff(t[s:e + 1]); first.append(d[0]); sub += d[1:].tolist()
    return {
        "types": df["type"].unique().tolist(),
        "rows": len(df),
        "singleton_%": round(100 * (rl == 1).mean(), 1),
        "first_gap_ms": round(np.median(first) * 1000, 1) if first else None,
        "repeat_gap_ms": round(np.median(sub) * 1000, 1) if sub else None,
    }

for sid in sorted(os.listdir(DATASET_DIR), key=lambda s: int(s[1:]) if s[1:].isdigit() else 999):
    p = os.path.join(DATASET_DIR, sid, "HCI", f"D3_{sid}_keyboard.csv")
    if not os.path.exists(p):
        continue
    r = analyze_kb(p)
    print(f"{sid:4s} | {r['types']} | rows={r['rows']:5d} | "
          f"singletons={r['singleton_%']:5.1f}% | "
          f"first={r['first_gap_ms']} ms | repeat={r['repeat_gap_ms']} ms")