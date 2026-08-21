"""
diagnose_cca_leakage.py
==========================
Checks whether train_biosignal_proxy.py's CCA cross-validation
(ShuffleSplit(n_splits=50, test_size=0.2), NOT grouped by participant)
leaks participant identity across train/test the same way CLARE's own
10-fold vs LOSO gap (85.58% -> 72.70%) demonstrated.

agg has 75 rows: 25 participants x 3 conditions (N/I/T), one row per
participant-condition mean. Random row-level ShuffleSplit can put 1-2
of a participant's 3 rows in TRAIN and the 3rd in TEST -- true
participant-level leakage, since per-participant z-scoring removes raw
baseline differences but not necessarily within-person patterns shared
across their own condition rows.

Compares:
  A. ShuffleSplit(50, test_size=0.2)     -- exactly what's in the script
  B. GroupShuffleSplit(50, test_size=0.2) -- same design, but respects
                                             participant boundaries
  C. True LOSO (25 folds, one participant's all 3 rows held out together)

If A inflates relative to B/C, the CCA result needs the same
"NOT reproducible as stated" treatment as HR_rising/RMSSD_rising.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, glob, warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, LeaveOneGroupOut

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
_matches = glob.glob(os.path.join(BASE, "data", "swell_kw", "**",
                                   "Behavioral-features - per minute.xlsx"),
                     recursive=True)
if not _matches:
    raise FileNotFoundError("SWELL-KW file not found")
SWELL_FILE = _matches[0]
print(f"Using: {SWELL_FILE}\n")

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]
PHYSIO_COLS_ACTIVE = ["HR", "RMSSD"]


def build_agg():
    df = pd.read_excel(SWELL_FILE)
    df = df[df["Condition"].isin(["N", "I", "T"])].copy()
    for c in ["HR", "RMSSD", "SCL"]:
        df[c] = df[c].replace(999, np.nan)

    agg = df.groupby(["PP","Condition"])[HCI_COLS + PHYSIO_COLS_ACTIVE].mean().reset_index()
    for col in HCI_COLS + PHYSIO_COLS_ACTIVE:
        agg[col+"_z"] = agg.groupby("PP")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))
    clean = agg[["PP"] + [c+"_z" for c in HCI_COLS] + [c+"_z" for c in PHYSIO_COLS_ACTIVE]].dropna()
    print(f"agg table after dropna: {len(clean)} rows, "
          f"{clean.PP.nunique()} participants "
          f"(expect <=75 rows if some PP-condition means were dropped)\n")
    return clean


def eval_scheme(Xz, Ycca, groups, splitter, label, grouped):
    cv_rs = []
    if grouped:
        split_iter = splitter.split(Xz, groups=groups)
    else:
        split_iter = splitter.split(Xz)
    n_folds_used = 0
    n_folds_leaked = 0
    for tr, te in split_iter:
        c = CCA(n_components=1, max_iter=2000).fit(Xz[tr], Ycca[tr])
        Xt, Yt = c.transform(Xz[te], Ycca[te])
        if np.std(Xt) > 1e-9 and np.std(Yt) > 1e-9:
            cv_rs.append(np.corrcoef(Xt[:,0], Yt[:,0])[0,1])
            n_folds_used += 1
        # check for leakage: does any participant appear in BOTH tr and te?
        if len(set(groups[tr]) & set(groups[te])) > 0:
            n_folds_leaked += 1

    mean_r = float(np.mean(cv_rs)) if cv_rs else np.nan
    std_r  = float(np.std(cv_rs))  if cv_rs else np.nan
    print(f"{label}")
    print(f"  CV r: {mean_r:.3f} +/- {std_r:.3f}  ({n_folds_used} folds)")
    print(f"  Folds with a participant appearing in BOTH train and test: "
          f"{n_folds_leaked}/{n_folds_used}")
    print()
    return mean_r, n_folds_leaked


def main():
    print("=" * 78)
    print("CCA LEAKAGE DIAGNOSTIC")
    print("Historical CV r cited this entire session: 0.581, 0.549")
    print("=" * 78)
    print()

    clean = build_agg()
    Xz = clean[[c+"_z" for c in HCI_COLS]].to_numpy(float)
    Ycca = clean[[c+"_z" for c in PHYSIO_COLS_ACTIVE]].to_numpy(float)
    groups = clean["PP"].to_numpy()

    print("=== A: ShuffleSplit(50, test_size=0.2) -- EXACTLY what the script runs ===")
    r_a, leak_a = eval_scheme(Xz, Ycca, groups,
                              ShuffleSplit(n_splits=50, test_size=0.2, random_state=42),
                              "ShuffleSplit (NOT grouped)", grouped=False)

    print("=== B: GroupShuffleSplit(50, test_size=0.2) -- same design, respects participants ===")
    r_b, leak_b = eval_scheme(Xz, Ycca, groups,
                              GroupShuffleSplit(n_splits=50, test_size=0.2, random_state=42),
                              "GroupShuffleSplit", grouped=True)

    print("=== C: True LOSO (LeaveOneGroupOut, one participant's rows held out together) ===")
    r_c, leak_c = eval_scheme(Xz, Ycca, groups, LeaveOneGroupOut(),
                              "LeaveOneGroupOut", grouped=True)

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"\n  A (current script, NOT grouped):  CV r = {r_a:.3f}  "
          f"(leaked folds: {leak_a})")
    print(f"  B (GroupShuffleSplit):             CV r = {r_b:.3f}  "
          f"(leaked folds: {leak_b})")
    print(f"  C (true LOSO):                      CV r = {r_c:.3f}  "
          f"(leaked folds: {leak_c})")

    if leak_a > 0 and abs(r_a - r_b) > 0.05:
        print(f"""
  [LEAKAGE CONFIRMED] {leak_a} of the current script's folds have the
  same participant in both train and test. Properly grouped validation
  gives a meaningfully different number ({r_b:.3f} vs {r_a:.3f}).
  The 0.581/0.549 figures cited throughout this project need the same
  correction already applied to HR_rising/RMSSD_rising/RMSSD_magnitude.
""")
    elif leak_a > 0 and abs(r_a - r_b) <= 0.05:
        print(f"""
  [LEAKAGE PRESENT BUT SMALL IMPACT] {leak_a} folds do leak a participant
  across train/test, but the grouped alternative ({r_b:.3f}) is close to
  the ungrouped figure ({r_a:.3f}). Per-participant z-scoring appears to
  have removed most of the exploitable leakage. Still recommend switching
  to GroupShuffleSplit/LOSO going forward for correctness, but the
  headline number may not need retraction.
""")
    else:
        print("\n  No leakage detected in the current scheme's folds.")


if __name__ == "__main__":
    main()
