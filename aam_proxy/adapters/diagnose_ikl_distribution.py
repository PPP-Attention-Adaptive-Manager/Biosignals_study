"""
diagnose_ikl_distribution.py
===============================
Distinguishes two explanations for the population mean IKL (1011ms,
well above the naive 100-500ms expectation for continuous typing):

  A. Burst+pause structure (EXPECTED for a reading/learning task):
     most inter-keydown gaps are short (real typing rhythm), a small
     tail of much longer gaps (reading/thinking between bursts) drags
     the MEAN up without meaning the underlying rhythm signal is broken.
     Signature: median/p25 look normal (100-400ms), mean >> median.

  B. Systematically elevated (would suggest our diff-based computation
     doesn't match Record Tool's actual interval_ms convention):
     Signature: median ITSELF is elevated, not just the mean -- the
     whole distribution is shifted, not just a long tail.

Run from: ~/biosignals_data/
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coglab_keyboard_adapter import ikl_distribution

BASE = os.path.expanduser("~/biosignals_data")
COG_LAB_DIR = os.path.join(BASE, "data", "cog_lab")
SUBJECTS = ["S1","S3","S4","S5","S6","S7","S8","S9","S10","S11",
           "S12","S13","S14","S15","S16","S17","S18"]  # S17 included
                                                        # this time --
                                                        # correction

print(f"{'subj':6s} {'n':>6s} {'p10':>7s} {'p25':>7s} {'median':>7s} "
      f"{'p75':>7s} {'p90':>7s} {'mean':>8s}  {'%<500ms':>8s} {'%>5s':>6s}")
print("-" * 82)

for sid in SUBJECTS:
    path = os.path.join(COG_LAB_DIR, sid, "HCI", f"D3_{sid}_keyboard.csv")
    if not os.path.isfile(path):
        print(f"  {sid:6s}  no keyboard.csv found")
        continue
    d = ikl_distribution(path)
    if not d:
        continue
    print(f"  {sid:6s} {d['n']:6d} {d['p10']:7.0f} {d['p25']:7.0f} "
          f"{d['median']:7.0f} {d['p75']:7.0f} {d['p90']:7.0f} "
          f"{d['mean']:8.0f}  {100*d['pct_under_500ms']:7.1f}% "
          f"{100*d['pct_over_5000ms']:5.1f}%")

print()
print("If median stays in the 100-400ms range while mean is much higher")
print("(explanation A) -- the encoder's own log1p+clip(0,20000) already")
print("handles this correctly, no adapter fix needed, just proceed.")
print()
print("If median itself is also elevated (explanation B) -- the diff-")
print("based IKL computation likely needs a session/idle-reset rule")
print("before it matches what the encoder was actually trained on.")
