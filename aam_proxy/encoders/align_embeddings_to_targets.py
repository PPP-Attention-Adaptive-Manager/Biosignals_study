"""
align_embeddings_to_targets.py
=================================
Aligns three different windowing schemes onto ONE common time grid
before any head-to-head test can run correctly:

  keyboard embeddings : keystroke-count windows (20 strokes/window),
                         IRREGULAR time duration -- varies with typing
                         speed, no fixed spacing
  mouse stats         : 30s time windows, 15s stride
  acc_jerk/eeg_engagement targets : 30s time windows, from
                         proxy_cache_swellstyle's own `starts` array

Naively zipping these by array index would silently pair a keyboard
embedding from one moment with a target from a different moment --
exactly the kind of quiet misalignment this project has spent all
session catching in other forms.

METHOD
--------
For each TARGET window (the anchor -- these define ground truth):
  mouse:    nearest-start match against the mouse stats windows
            (should be near-exact, both use ~30s windows)
  keyboard: MEAN-POOL every keyboard embedding whose window CENTER
            falls inside [target_start, target_start+30s) -- since
            multiple keystroke-count windows can fall inside one 30s
            span, or (during a pause) NONE might.

Reports match quality explicitly -- time gap for mouse matches, count
of pooled keyboard windows (including the 0-match case) -- rather than
silently assuming alignment worked.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, sys, glob
import numpy as np

BASE = os.path.expanduser("~/biosignals_data")
sys.path.insert(0, os.path.join(BASE, "aam_proxy", "adapters"))
from coglab_keyboard_adapter import adapt_coglab_keyboard

CACHE_DIR = os.path.join(BASE, "data", "cache", "proxy_cache_swellstyle")
KB_EMB_DIR = os.path.join(BASE, "aam_proxy", "encoders", "keyboard_embeddings")
MOUSE_STATS_DIR = os.path.join(BASE, "aam_proxy", "encoders", "mouse_stats")
COG_LAB_DIR = os.path.join(BASE, "data", "cog_lab")

EXCLUDE = {"S2", "S17"}
TARGET_WINDOW_S = 30.0

KB_WINDOW_SIZE = 20
KB_STRIDE = 10


def get_keyboard_window_times(sid):
    """Recompute the TIME span (start, end, center) for each keyboard
    embedding window -- needed for alignment since the embeddings
    themselves don't carry timestamps, only the adapter's raw events do."""
    kb_path = os.path.join(COG_LAB_DIR, sid, "HCI", f"D3_{sid}_keyboard.csv")
    events = adapt_coglab_keyboard(kb_path)
    # events have no absolute timestamp field (adapter strips it, keeping
    # only code/hold/ikl) -- recompute from the raw file directly instead
    import pandas as pd
    df = pd.read_csv(kb_path)
    df = df[df["type"] == "Keyboard Keydown"].sort_values("time").reset_index(drop=True)
    ts = df["time"].to_numpy(float)

    centers = []
    i = 0
    while i + KB_WINDOW_SIZE <= len(ts):
        w_ts = ts[i:i + KB_WINDOW_SIZE]
        centers.append((w_ts[0] + w_ts[-1]) / 2)
        i += KB_STRIDE
    return np.array(centers)


def align_subject(sid):
    npz_path = os.path.join(CACHE_DIR, f"{sid}.npz")
    kb_emb_path = os.path.join(KB_EMB_DIR, f"{sid}_keyboard_emb.npy")
    mouse_path = os.path.join(MOUSE_STATS_DIR, f"{sid}_mouse_stats.npy")

    if not all(os.path.isfile(p) for p in [npz_path, kb_emb_path, mouse_path]):
        return None

    z = np.load(npz_path, allow_pickle=True)
    target_starts = z["starts"]                       # absolute unix timestamps
    remaining_names = list(z["remaining_names"])
    acc_jerk_idx = remaining_names.index("acc_jerk")
    eeg_eng_idx = remaining_names.index("eeg_engagement")
    Y = z["Y_remaining"]

    kb_emb = np.load(kb_emb_path)              # (n_kb_windows, 64)
    kb_centers = get_keyboard_window_times(sid) # absolute unix timestamps
    mouse_stats = np.load(mouse_path)          # (n_mouse_windows, 22)

    # FIXED: load the REAL start time saved per surviving window, rather
    # than reconstructing it from a dense-index assumption that breaks
    # the moment any window gets dropped by extract_window_stats()'s own
    # MIN_EVENTS_FOR_WINDOW filter. This was the actual bug: mouse_gap_p90
    # ranged from 0.73s (few dropped windows) to 591.98s (many dropped
    # windows, index drift compounding) -- confirming exactly this
    # mechanism, not a marginal alignment offset.
    mouse_starts_path = os.path.join(MOUSE_STATS_DIR, f"{sid}_mouse_starts.npy")
    if not os.path.isfile(mouse_starts_path):
        raise FileNotFoundError(
            f"{mouse_starts_path} not found -- re-run run_mouse_stats.py "
            f"with the fix that saves real per-window start times before "
            f"using this alignment script.")
    mouse_starts = np.load(mouse_starts_path)

    rows = []
    kb_pool_counts = []
    mouse_gaps = []
    for i, t_start in enumerate(target_starts):
        t_end = t_start + TARGET_WINDOW_S

        # keyboard: mean-pool windows whose center falls inside [t_start, t_end)
        in_range = (kb_centers >= t_start) & (kb_centers < t_end)
        kb_pool_counts.append(int(in_range.sum()))
        if in_range.sum() == 0:
            kb_feat = np.full(64, np.nan)
        else:
            kb_feat = kb_emb[in_range].mean(axis=0)

        # mouse: nearest-start match
        gaps = np.abs(mouse_starts - t_start)
        nearest = np.argmin(gaps)
        mouse_gaps.append(float(gaps[nearest]))
        mouse_feat = mouse_stats[nearest] if gaps[nearest] < TARGET_WINDOW_S else np.full(22, np.nan)

        rows.append({
            "participant": sid, "target_idx": i,
            "kb_feat": kb_feat, "mouse_feat": mouse_feat,
            "acc_jerk": Y[i, acc_jerk_idx],
            "eeg_engagement": Y[i, eeg_eng_idx],
        })

    return rows, np.array(kb_pool_counts), np.array(mouse_gaps)


def main():
    print("=" * 78)
    print("ALIGNMENT DIAGNOSTIC — before trusting any head-to-head numbers")
    print("=" * 78)

    all_dirs = sorted(d for d in os.listdir(COG_LAB_DIR)
                      if d.startswith("S") and d[1:].isdigit())
    subjects = [d for d in all_dirs if d not in EXCLUDE]

    all_rows = []
    print(f"\n{'subj':6s} {'n_targets':>10s} {'kb_0match_pct':>14s} "
          f"{'kb_mean_pool':>13s} {'mouse_gap_p90_s':>16s}")
    print("-" * 65)

    for sid in subjects:
        result = align_subject(sid)
        if result is None:
            print(f"  {sid:6s}  missing one of npz/kb_emb/mouse_stats")
            continue
        rows, kb_counts, mouse_gaps = result
        all_rows.extend(rows)
        pct_zero = 100 * (kb_counts == 0).mean()
        print(f"  {sid:6s} {len(rows):10d} {pct_zero:13.1f}% "
              f"{kb_counts[kb_counts>0].mean() if (kb_counts>0).any() else 0:13.2f} "
              f"{np.percentile(mouse_gaps, 90):16.2f}")

    print(f"\nTotal aligned rows: {len(all_rows)}")
    print("\nINTERPRETATION:")
    print("  kb_0match_pct: fraction of target windows with NO keyboard")
    print("  activity at all in that 30s span -- expected to be nonzero")
    print("  (people pause typing), NOT expected to be near 100%.")
    print("  mouse_gap_p90_s: 90th percentile of |nearest mouse window")
    print("  start - target start|, in seconds. Should be small (<15s,")
    print("  half the mouse stride) if both windowing schemes are truly")
    print("  aligned to the same session clock.")

    out_path = os.path.join(BASE, "aam_proxy", "encoders", "aligned_rows.npy")
    np.save(out_path, np.array(all_rows, dtype=object), allow_pickle=True)
    print(f"\nSaved {len(all_rows)} aligned rows to {out_path}")


if __name__ == "__main__":
    main()
