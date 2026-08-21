"""
diagnose_rmssd_magnitude.py
==============================
Resolves the RMSSD_magnitude discrepancy: 84.7% (cited historically as
"the strongest single result in the whole project") vs 69.9% (fresh
retrain of train_biosignal_proxy.py after the path-fix patch).

THE LEADING HYPOTHESIS
-------------------------
train_biosignal_proxy.py's train_direction_models() validates
RMSSD_magnitude with:

    gkf = GroupKFold(n_splits=min(10, len(np.unique(groups))))

On 25 SWELL-KW participants, min(10, 25) = 10 -- meaning each fold
holds out ~2-3 participants AT ONCE, not one participant at a time.
This is NOT the same validation scheme as true LOSO (n_splits=25,
exactly one participant held out per fold), even though both get
loosely described as "LOSO-style" in this project's own comments.

If the historical 84.7% figure was computed with TRUE per-participant
LOSO (as most other direction/magnitude numbers in this project were),
and the current script uses GroupKFold(10) instead, that is a genuine
methodological difference -- not a bug exactly, but two different
validation schemes being compared as if they were the same number.

GroupKFold with few folds on a moderate number of groups is also more
sensitive to which specific participants land in which fold: with only
10 folds, some folds hold out multiple participants whose combined
class balance (after the tertile split) could be unusually skewed,
which can pull average accuracy down for reasons unrelated to real
signal strength.

WHAT THIS SCRIPT DOES
------------------------
Reproduces the EXACT SAME feature/target construction as
train_biosignal_proxy.py's RMSSD_magnitude block, then evaluates it
THREE ways on the identical data:

  A. GroupKFold(n_splits=10)     -- what the current script actually runs
  B. True per-participant LOSO   -- n_splits = n_participants (25)
  C. Per-fold class balance check -- are any GroupKFold(10) folds
                                     degenerate (near-single-class)?

If (B) lands near 84.7% and (A) lands near 69.9%, the discrepancy is
explained: two different validation schemes, not an unstable target or
a data bug. If (B) does NOT reproduce 84.7% either, the historical
number needs a different explanation (possibly computed on an earlier
data version, or a different feature set) and should not be cited
without further digging.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, glob, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")

_matches = glob.glob(os.path.join(BASE, "data", "swell_kw", "**",
                                   "Behavioral-features - per minute.xlsx"),
                     recursive=True)
if not _matches:
    raise FileNotFoundError(
        f"Could not find SWELL-KW file under {os.path.join(BASE,'data','swell_kw')}")
SWELL_FILE = _matches[0]
print(f"Using SWELL-KW file: {SWELL_FILE}\n")

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]


def build_rmssd_magnitude_data(exact_original_filter=False):
    """
    exact_original_filter=True reproduces train_biosignal_proxy.py's
    ACTUAL row selection: dropna on HR_rising, RMSSD_rising, AND
    SCL_rising SIMULTANEOUSLY (all three physio deltas must be non-null
    for a row to survive) -- not just RMSSD_delta alone. This is a real
    construction difference from the first version of this diagnostic
    (which used the less restrictive RMSSD_delta-only filter, 1062 rows/
    23 participants) and needs to be ruled out before concluding
    anything about whether 69.9%/84.7% are reproducible.
    """
    df = pd.read_excel(SWELL_FILE)
    df = df[df["Condition"].isin(["N", "I", "T"])].copy()
    for c in ["HR", "RMSSD", "SCL"]:
        df[c] = df[c].replace(999, np.nan)

    df_s = df.sort_values(["PP", "Condition"]).copy()
    for target in ["HR", "RMSSD", "SCL"]:
        df_s[f"{target}_delta"]  = df_s.groupby(["PP", "Condition"])[target].diff()
        df_s[f"{target}_rising"] = (df_s[f"{target}_delta"] > 0).astype(float)
    for col in HCI_COLS:
        df_s[col + "_delta"] = df_s.groupby(["PP", "Condition"])[col].diff()

    if exact_original_filter:
        need_cols = ["HR_rising", "RMSSD_rising", "SCL_rising"]
        df_clean = df_s.dropna(subset=need_cols)
        print("Using EXACT original filter: dropna on HR_rising + "
              "RMSSD_rising + SCL_rising simultaneously")
    else:
        df_clean = df_s.dropna(subset=["RMSSD_delta"])
        print("Using RMSSD_delta-only filter (broader, from the first "
              "version of this diagnostic)")

    HCI_DELTA = [c + "_delta" for c in HCI_COLS]
    X = np.nan_to_num(df_clean[HCI_DELTA].to_numpy(float))
    train_mu = X.mean(0); train_sd = X.std(0) + 1e-9
    Xz = (X - train_mu) / train_sd

    rmssd_delta = df_clean["RMSSD_delta"].to_numpy(float)
    thresholds = np.nanpercentile(rmssd_delta, [33.3, 66.7])
    y_mag = np.digitize(rmssd_delta, thresholds)
    groups = df_clean["PP"].to_numpy()

    print(f"Rows after dropna: {len(df_clean)}")
    print(f"Participants: {len(np.unique(groups))}")
    print(f"Tertile thresholds on RMSSD_delta: {thresholds}")
    print(f"Class distribution (0=fall,1=flat,2=rise): "
          f"{np.bincount(y_mag)}  "
          f"({100*np.bincount(y_mag)/len(y_mag)}%)\n")

    return Xz, y_mag, groups


def eval_groupkfold(Xz, y_mag, groups, n_splits, label):
    print(f"=== {label} (n_splits={n_splits}) ===")
    gkf = GroupKFold(n_splits=n_splits)
    accs, fold_details = [], []
    for i, (tr, te) in enumerate(gkf.split(Xz, y_mag, groups)):
        m = RandomForestClassifier(200, min_samples_leaf=5,
                                   class_weight="balanced",
                                   random_state=0, n_jobs=-1)
        m.fit(Xz[tr], y_mag[tr])
        acc = accuracy_score(y_mag[te], m.predict(Xz[te]))
        accs.append(acc)

        te_participants = sorted(set(groups[te]))
        te_class_counts = np.bincount(y_mag[te], minlength=3)
        te_class_pct = te_class_counts / te_class_counts.sum()
        degenerate = te_class_pct.max() > 0.80
        fold_details.append((i, te_participants, acc, te_class_pct, degenerate))

    print(f"  Mean accuracy: {np.mean(accs):.3f}  (folds: {len(accs)})")
    print(f"  Per-fold: {[f'{a:.3f}' for a in accs]}")
    n_degenerate = sum(1 for f in fold_details if f[4])
    if n_degenerate:
        print(f"  WARNING: {n_degenerate}/{len(fold_details)} folds have a "
              f">80% single-class test set (degenerate, unreliable accuracy)")
        for i, parts, acc, pct, deg in fold_details:
            if deg:
                print(f"    Fold {i}: participants={parts}  "
                      f"class%={pct.round(2)}  acc={acc:.3f}")
    print()
    return np.mean(accs), accs


def main():
    print("=" * 78)
    print("RMSSD_MAGNITUDE DISCREPANCY DIAGNOSTIC")
    print("Historical: 84.7%   Fresh retrain: 69.9%")
    print("=" * 78)
    print()

    print("### PASS 1: broader RMSSD_delta-only filter (first diagnostic run) ###\n")
    Xz, y_mag, groups = build_rmssd_magnitude_data(exact_original_filter=False)
    n_participants = len(np.unique(groups))
    acc_gkf10, _ = eval_groupkfold(Xz, y_mag, groups, min(10, n_participants),
                                    "A: GroupKFold(10)")
    acc_loso, _  = eval_groupkfold(Xz, y_mag, groups, n_participants,
                                    "B: True per-participant LOSO")

    print("### PASS 2: EXACT original filter (HR_rising+RMSSD_rising+SCL_rising) ###\n")
    Xz2, y_mag2, groups2 = build_rmssd_magnitude_data(exact_original_filter=True)
    n_participants2 = len(np.unique(groups2))
    acc_gkf10_exact, _ = eval_groupkfold(Xz2, y_mag2, groups2,
                                         min(10, n_participants2),
                                         "C: GroupKFold(10), EXACT original filter")
    acc_loso_exact, _  = eval_groupkfold(Xz2, y_mag2, groups2, n_participants2,
                                         "D: True LOSO, EXACT original filter")

    print(f"\nPASS 2 summary: GroupKFold(10)={acc_gkf10_exact:.3f}  "
          f"LOSO={acc_loso_exact:.3f}  (n_rows={len(y_mag2)}, "
          f"n_participants={n_participants2})\n")

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"\n  GroupKFold(10) accuracy: {acc_gkf10:.3f}  ({100*acc_gkf10:.1f}%)")
    print(f"  True LOSO accuracy:      {acc_loso:.3f}  ({100*acc_loso:.1f}%)")
    print(f"\n  Historical figure: 84.7%    Fresh retrain figure: 69.9%")

    dist_to_hist_gkf  = abs(100*acc_gkf10 - 84.7)
    dist_to_hist_loso = abs(100*acc_loso  - 84.7)
    dist_to_fresh_gkf  = abs(100*acc_gkf10 - 69.9)
    dist_to_fresh_loso = abs(100*acc_loso  - 69.9)

    print(f"\n  |GroupKFold(10) - 69.9|  = {dist_to_fresh_gkf:.1f}")
    print(f"  |True LOSO - 84.7|       = {dist_to_hist_loso:.1f}")

    if dist_to_fresh_gkf < 3 and dist_to_hist_loso < 5:
        print("""
  [EXPLAINED] The discrepancy is a validation-scheme mismatch, not an
  unstable target or a data bug. GroupKFold(10) reproduces the 69.9%
  fresh-retrain figure; true per-participant LOSO reproduces something
  close to the historical 84.7%. The two numbers were never measuring
  the same thing, despite both being informally called "LOSO-style" in
  this project's comments.

  FIX: change train_biosignal_proxy.py's RMSSD_magnitude evaluation
  from GroupKFold(min(10, n_participants)) to true per-participant
  LOSO (n_splits = n_participants), matching every other direction/
  magnitude accuracy reported in this project. Re-run and re-save
  metadata.json with the corrected, apples-to-apples number.
""")
    elif dist_to_hist_loso < 5:
        print("""
  [PARTIALLY EXPLAINED] True LOSO reproduces something close to 84.7%,
  confirming that's the right validation scheme -- but GroupKFold(10)
  doesn't cleanly reproduce 69.9% either. Check the per-fold degenerate
  warnings above for whether a specific unlucky fold assignment (not
  just the coarser fold count) is dragging the GroupKFold number down.
""")
    else:
        print("""
  [NOT EXPLAINED BY VALIDATION SCHEME ALONE] Neither GroupKFold(10) nor
  true LOSO cleanly reproduces 84.7% on this data. The historical figure
  may have been computed on a different data snapshot, a different
  feature set, or a different random seed's fold assignment. Do not cite
  84.7% as reproducible until its exact source script/run is located.
  The number to trust going forward is whichever validation scheme is
  used consistently -- true LOSO is recommended since it matches every
  other accuracy figure in this project.
""")


if __name__ == "__main__":
    main()
