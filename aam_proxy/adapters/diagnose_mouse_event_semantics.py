"""
diagnose_mouse_event_semantics.py
====================================
Two real questions the batch run surfaced, neither safe to guess at:

1. Are Left Click/Right Click REDUNDANT with Mouse Down/Up (a
   synthesized "completed click" event on top of the raw press/release
   pair), or genuinely distinct information? Checked by looking at
   whether each Click event's timestamp falls within an existing
   Down->Up window for the same button.

2. Is the wild speed variance (S1 mean=6.5k px/s vs S5 mean=63.8k,
   p95=485k) driven by inconsistent SAMPLING RATE across subjects
   (small/uneven dt between Mouse Move samples), which would make raw
   consecutive-sample speed non-comparable across subjects and need a
   different computation (e.g. minimum dt floor, or resampling onto a
   fixed time grid) rather than the naive dx/dt used so far.

Run from: ~/biosignals_data/
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.expanduser("~/biosignals_data")
COG_LAB_DIR = os.path.join(BASE, "data", "cog_lab")


def check_click_redundancy(sid):
    path = os.path.join(COG_LAB_DIR, sid, "HCI", f"D3_{sid}_mouse.csv")
    df = pd.read_csv(path).sort_values("time").reset_index(drop=True)

    downs = df[df["type"] == "Mouse Down"][["time", "button"]].to_numpy()
    ups   = df[df["type"] == "Mouse Up"][["time", "button"]].to_numpy()
    left_clicks = df[df["type"] == "Left Click"]["time"].to_numpy()

    if len(left_clicks) == 0 or len(downs) == 0:
        return None

    # for each Left Click, is there a Mouse Down within +/- 500ms?
    matched = 0
    for ct in left_clicks:
        if np.any(np.abs(downs[:, 0] - ct) < 0.5):
            matched += 1

    return {
        "n_left_click": len(left_clicks),
        "n_mouse_down": len(downs),
        "n_mouse_up": len(ups),
        "pct_left_click_matched_to_down": 100 * matched / len(left_clicks),
    }


def check_sampling_rate(sid):
    path = os.path.join(COG_LAB_DIR, sid, "HCI", f"D3_{sid}_mouse.csv")
    df = pd.read_csv(path)
    moves = df[df["type"] == "Mouse Move"].sort_values("time")
    ts = moves["time"].to_numpy(float)
    if len(ts) < 10:
        return None
    dt = np.diff(ts) * 1000   # ms
    dt = dt[dt > 0]
    return {
        "n_moves": len(ts),
        "dt_p10_ms": float(np.percentile(dt, 10)),
        "dt_median_ms": float(np.median(dt)),
        "dt_p90_ms": float(np.percentile(dt, 90)),
        "pct_dt_under_5ms": float((dt < 5).mean() * 100),
        "implied_max_rate_hz": float(1000 / np.percentile(dt, 10)),
    }


SUBJECTS = ["S1","S3","S4","S5","S6","S7","S8","S9","S10","S11",
           "S12","S13","S14","S15","S16","S18"]

print("=" * 90)
print("PART 1 — Left Click redundancy with Mouse Down")
print("=" * 90)
print(f"{'subj':6s} {'n_LClick':>9s} {'n_Down':>8s} {'n_Up':>6s} "
      f"{'%_matched_to_Down':>18s}")
print("-" * 55)
for sid in SUBJECTS:
    r = check_click_redundancy(sid)
    if r:
        print(f"  {sid:6s} {r['n_left_click']:9d} {r['n_mouse_down']:8d} "
              f"{r['n_mouse_up']:6d} {r['pct_left_click_matched_to_down']:17.1f}%")

print("\n" + "=" * 90)
print("PART 2 — Mouse Move sampling rate consistency across subjects")
print("=" * 90)
print(f"{'subj':6s} {'n_moves':>8s} {'dt_p10_ms':>10s} {'dt_median':>10s} "
      f"{'dt_p90':>8s} {'%dt<5ms':>9s} {'implied_max_hz':>15s}")
print("-" * 78)
for sid in SUBJECTS:
    r = check_sampling_rate(sid)
    if r:
        print(f"  {sid:6s} {r['n_moves']:8d} {r['dt_p10_ms']:10.2f} "
              f"{r['dt_median_ms']:10.2f} {r['dt_p90_ms']:8.2f} "
              f"{r['pct_dt_under_5ms']:8.1f}% {r['implied_max_rate_hz']:15.1f}")

print("""
INTERPRETATION:
If implied_max_rate_hz varies wildly across subjects (some ~60Hz browser-
throttled, others much higher/unthrottled), that directly explains the
speed inconsistency -- raw dx/dt is NOT comparable across subjects with
different effective sampling rates. Fix would be resampling onto a fixed
time grid (e.g. 60Hz) before computing dx/dy/speed, not just a tighter
dt floor.
""")
