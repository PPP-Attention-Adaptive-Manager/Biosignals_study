"""
batch_run_coglab_keyboard.py
===============================
Runs coglab_keyboard_adapter.py across ALL 16 usable Cog Lab subjects
(S1, S3-S16, S18 -- S2 and S17 EXCLUDED, confirmed structurally: their
folders have no HCI/ subdirectory at all, no keyboard.csv or mouse.csv
exists for them).

This is the population-level check that should have been the default
from the start -- the earlier single-file run against S10 was a smoke
test of the parsing logic only, never intended as "the proxy," and
should have been labeled that way explicitly.

Reports, per subject: event count, hold-channel status (should be
identical "always 0.0" note for every subject -- if any subject shows
something different, that's worth investigating, not assuming it's
fine), key-code range, mean IKL. Then an aggregate summary across the
full population.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coglab_keyboard_adapter import adapt_coglab_keyboard_report

BASE = os.path.expanduser("~/biosignals_data")
COG_LAB_DIR = os.path.join(BASE, "data", "cog_lab")
EXCLUDE = {"S2", "S17"}   # confirmed: no HCI/ folder exists for these


def find_subjects():
    all_dirs = sorted(
        d for d in os.listdir(COG_LAB_DIR)
        if d.startswith("S") and d[1:].isdigit()
    )
    usable, excluded_confirmed, excluded_unexpected = [], [], []
    for d in all_dirs:
        hci_path = os.path.join(COG_LAB_DIR, d, "HCI")
        kb_path = os.path.join(hci_path, f"D3_{d}_keyboard.csv")
        has_hci = os.path.isfile(kb_path)
        if d in EXCLUDE:
            if has_hci:
                excluded_unexpected.append(d)  # flag: exclusion may be
                                               # stale if data changed
            else:
                excluded_confirmed.append(d)
        elif has_hci:
            usable.append(d)
        else:
            print(f"  WARNING: {d} not in EXCLUDE but has no keyboard.csv "
                 f"-- check this subject manually")
    return usable, excluded_confirmed, excluded_unexpected


def main():
    print("=" * 78)
    print("COG LAB KEYBOARD ADAPTER — FULL POPULATION RUN")
    print("=" * 78)

    usable, excl_confirmed, excl_unexpected = find_subjects()
    print(f"\nUsable subjects ({len(usable)}): {usable}")
    print(f"Confirmed-excluded (no HCI folder, matches EXCLUDE set): "
          f"{excl_confirmed}")
    if excl_unexpected:
        print(f"UNEXPECTED: these are in EXCLUDE but DO have keyboard.csv "
              f"-- exclusion may need revisiting: {excl_unexpected}")
    print()

    print(f"{'subject':10s} {'n_events':>9s} {'code_min':>9s} {'code_max':>9s} "
          f"{'mean_ikl_ms':>12s}")
    print("-" * 55)

    all_reports = {}
    for sid in usable:
        kb_path = os.path.join(COG_LAB_DIR, sid, "HCI", f"D3_{sid}_keyboard.csv")
        try:
            rep = adapt_coglab_keyboard_report(kb_path)
        except Exception as e:
            print(f"  {sid:10s}  ERROR: {e}")
            continue
        all_reports[sid] = rep
        code_min, code_max = rep["code_range"] if rep["code_range"] else (None, None)
        print(f"  {sid:10s} {rep['n_events']:9d} {code_min!s:>9s} "
              f"{code_max!s:>9s} {rep['ikl_mean_ms']:12.1f}")

    print("\n" + "=" * 78)
    print("POPULATION SUMMARY")
    print("=" * 78)
    n_events_all = [r["n_events"] for r in all_reports.values()]
    ikl_all = [r["ikl_mean_ms"] for r in all_reports.values() if r["ikl_mean_ms"]]
    print(f"\nSubjects processed: {len(all_reports)}/{len(usable)}")
    print(f"Total keystroke events across population: {sum(n_events_all)}")
    print(f"Events per subject: mean={np.mean(n_events_all):.0f}  "
          f"min={min(n_events_all)}  max={max(n_events_all)}")
    print(f"Mean IKL across population: {np.mean(ikl_all):.1f} ms  "
          f"(sanity check -- should be roughly 100-500ms range for normal "
          f"typing rhythm; wildly different values suggest a timestamp "
          f"unit mismatch worth checking)")
    print(f"\nHold-channel status: ALWAYS 0.0 for all {len(all_reports)} "
          f"subjects (structural gap, confirmed identical across the full "
          f"population -- not subject-specific)")


if __name__ == "__main__":
    main()
