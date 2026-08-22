"""
batch_run_coglab_mouse.py
============================
Runs coglab_mouse_adapter.py across all 15 confirmed-clean subjects
(S1, S3-S16, S18 -- S2 excluded: no HCI folder; S17 excluded: HCI
folder confirmed byte-identical duplicate of S1's, verified via diff).

Reports event-type breakdown per subject FIRST -- if any subject shows
an unmapped type the others don't, that's worth a look before trusting
the population-level numbers.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coglab_mouse_adapter import adapt_coglab_mouse_report

BASE = os.path.expanduser("~/biosignals_data")
COG_LAB_DIR = os.path.join(BASE, "data", "cog_lab")

# S17 excluded here for a CONFIRMED reason (byte-identical duplicate of
# S1, verified via diff earlier), not the same uncertain reasoning that
# was corrected for the keyboard batch run.
EXCLUDE = {"S2", "S17"}


def main():
    print("=" * 78)
    print("COG LAB MOUSE ADAPTER — FULL POPULATION RUN")
    print("=" * 78)

    all_dirs = sorted(d for d in os.listdir(COG_LAB_DIR)
                      if d.startswith("S") and d[1:].isdigit())
    subjects = [d for d in all_dirs if d not in EXCLUDE]
    print(f"\nSubjects ({len(subjects)}): {subjects}")
    print(f"Excluded: {sorted(EXCLUDE)} (S2: no HCI folder; "
          f"S17: confirmed duplicate of S1, verified via diff)\n")

    all_unmapped = set()
    print(f"{'subject':10s} {'n_events':>9s} {'speed_mean':>11s} "
          f"{'speed_p95':>10s}  unmapped types")
    print("-" * 78)

    for sid in subjects:
        path = os.path.join(COG_LAB_DIR, sid, "HCI", f"D3_{sid}_mouse.csv")
        if not os.path.isfile(path):
            print(f"  {sid:10s}  no mouse.csv found")
            continue
        rep = adapt_coglab_mouse_report(path)
        all_unmapped.update(rep["unmapped_types"])
        sm = rep["speed_mean_px_s"] or 0
        sp = rep["speed_p95_px_s"] or 0
        flag = f"  {rep['unmapped_types']}" if rep["unmapped_types"] else ""
        print(f"  {sid:10s} {rep['n_events']:9d} {sm:11.1f} {sp:10.1f}{flag}")

    print("\n" + "=" * 78)
    if all_unmapped:
        print(f"UNMAPPED TYPES SEEN ACROSS POPULATION: {all_unmapped}")
        print("Add these to MAPPED_TYPES in coglab_mouse_adapter.py before")
        print("trusting the event_type field downstream.")
    else:
        print("No unmapped event types across the full population --")
        print("MAPPED_TYPES covers everything actually present.")


if __name__ == "__main__":
    main()
