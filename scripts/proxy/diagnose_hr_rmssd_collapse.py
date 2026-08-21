"""
diagnose_hr_rmssd_collapse.py
================================
Tests directly whether HR_rising/RMSSD_rising's historical 79.9%/78.1%
were inflated by the SAME NaN-handling bug that inflated
RMSSD_magnitude's 84.7%/69.9%, just producing a different symptom:
instead of dumping missing deltas into one bin (digitize's behavior),
(NaN > 0).astype(float) silently made missing deltas "falling" (0.0)
rather than excluding them.

If HR/RMSSD have substantial real missingness (confirmed: only 1062/2688
rows survive a correct per-target filter, vs SCL's 2082/2688 -- HR/RMSSD
are the SPARSE ones, not SCL), then the pre-fix "79.9%/78.1%" numbers
were trained and tested with many "falling" labels that were actually
missing-data artifacts, not genuine physiological decreases. If
missingness itself correlates with HCI behavior (plausible: sensor
adjustment, movement, specific task phases), a classifier could exploit
that correlation and look like it's predicting HR/RMSSD direction when
it's actually detecting "is this a missing-sensor-reading row."

THREE COMPARISONS
--------------------
A. OLD buggy construction: NaN delta -> False -> 0.0 ("falling"),
   ALL 2688 rows kept, true per-participant LOSO (25 folds).
   This reproduces exactly what the ORIGINAL, pre-fix pipeline did.

B. NEW correct construction: NaN delta -> NaN, dropped via dropna,
   only genuine deltas kept (1062 rows, 23 participants), true LOSO.
   This is what the fixed train_biosignal_proxy.py now runs.

C. Missingness-as-target sanity check: can HCI features predict WHICH
   rows have a missing HR/RMSSD delta at all? If yes, that's direct
   proof missingness correlates with behavior -- the exact mechanism
   that would let (A) exploit it.

If (A) reproduces ~79.9%/78.1% and (C) shows HCI can predict
missingness above chance, the historical numbers are confirmed
artifacts of the same bug class as RMSSD_magnitude, just less visibly.
If (A) does NOT reproduce the historical figures either, something else
entirely explains them and this needs a different investigation.

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
    raise FileNotFoundError("SWELL-KW file not found")
SWELL_FILE = _matches[0]
print(f"Using: {SWELL_FILE}\n")

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]


def load_base():
    df = pd.read_excel(SWELL_FILE)
    df = df[df["Condition"].isin(["N", "I", "T"])].copy()
    for c in ["HR", "RMSSD", "SCL"]:
        df[c] = df[c].replace(999, np.nan)
    df_s = df.sort_values(["PP", "Condition"]).copy()
    df_s["HR_delta"]    = df_s.groupby(["PP","Condition"])["HR"].diff()
    df_s["RMSSD_delta"] = df_s.groupby(["PP","Condition"])["RMSSD"].diff()
    for col in HCI_COLS:
        df_s[col+"_delta"] = df_s.groupby(["PP","Condition"])[col].diff()
    return df_s


def loso_acc(Xz, y, groups):
    n_p = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_p)
    accs = []
    for tr, te in gkf.split(Xz, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(200, min_samples_leaf=5,
                                   class_weight="balanced",
                                   random_state=0, n_jobs=-1)
        m.fit(Xz[tr], y[tr])
        accs.append(accuracy_score(y[te], m.predict(Xz[te])))
    return float(np.mean(accs)) if accs else np.nan, len(accs)


def zscore(X):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    return (X - mu) / sd


HCI_DELTA = [c+"_delta" for c in HCI_COLS]


def comparison_a_old_buggy(df_s, target):
    """Reproduces the ORIGINAL pre-fix behavior: NaN delta -> False -> 0.0,
    every row kept (2688 total)."""
    print(f"=== A: OLD buggy construction ({target}) — all rows kept ===")
    delta = df_s[f"{target}_delta"].to_numpy(float)
    y = (delta > 0).astype(int)   # exact original bug: NaN>0 -> False -> 0
    X = np.nan_to_num(df_s[HCI_DELTA].to_numpy(float))
    Xz = zscore(X)
    groups = df_s["PP"].to_numpy()
    acc, nf = loso_acc(Xz, y, groups)
    n_fake_falling = int(((delta_isna := df_s[f"{target}_delta"].isna()) & (y==0)).sum())
    print(f"  Rows: {len(df_s)}  Participants: {len(np.unique(groups))}")
    print(f"  'Falling' labels that are actually MISSING data: {n_fake_falling}/{(y==0).sum()} "
          f"of all class-0 rows")
    print(f"  LOSO accuracy: {acc:.3f}  ({nf} folds)\n")
    return acc


def comparison_b_new_correct(df_s, target):
    print(f"=== B: NEW correct construction ({target}) — only genuine deltas ===")
    df_clean = df_s.dropna(subset=[f"{target}_delta"])
    y = (df_clean[f"{target}_delta"].to_numpy(float) > 0).astype(int)
    X = np.nan_to_num(df_clean[HCI_DELTA].to_numpy(float))
    Xz = zscore(X)
    groups = df_clean["PP"].to_numpy()
    acc, nf = loso_acc(Xz, y, groups)
    print(f"  Rows: {len(df_clean)}  Participants: {len(np.unique(groups))}")
    print(f"  LOSO accuracy: {acc:.3f}  ({nf} folds)\n")
    return acc


def comparison_c_missingness_predictable(df_s, target):
    """Can HCI features predict WHICH rows have a missing delta?
    If yes -> missingness correlates with behavior -> mechanism for
    comparison A's exploit is confirmed live.

    BUG FIXED: is_missing must be a numpy array before positional
    fold-index lookups (y[tr]/y[te] from GroupKFold). Left as a pandas
    Series, it kept df_s's original (sorted-but-not-reset) row-label
    index, so y[tr] with GroupKFold's positional integers triggered
    LABEL-based lookup and crashed with a KeyError the moment a fold's
    positions didn't match real index labels.
    """
    print(f"=== C: can HCI predict MISSINGNESS itself ({target})? ===")
    is_missing = df_s[f"{target}_delta"].isna().to_numpy().astype(int)
    X = np.nan_to_num(df_s[HCI_DELTA].to_numpy(float))
    Xz = zscore(X)
    groups = df_s["PP"].to_numpy()
    acc, nf = loso_acc(Xz, is_missing, groups)
    chance = max(is_missing.mean(), 1-is_missing.mean())
    print(f"  Missing rate: {is_missing.mean():.3f}  chance={chance:.3f}")
    print(f"  LOSO accuracy predicting MISSINGNESS from HCI: {acc:.3f}  "
          f"(over chance: {acc-chance:+.3f})\n")
    return acc, chance


def main():
    print("=" * 78)
    print("HR/RMSSD COLLAPSE DIAGNOSTIC")
    print("Historical: HR=79.9%  RMSSD=78.1%   Fresh (fixed): HR=58.6%  RMSSD=46.3%")
    print("=" * 78)
    print()

    df_s = load_base()

    results = {}
    for target, hist in [("HR", 0.799), ("RMSSD", 0.781)]:
        print(f"\n{'#'*78}\nTARGET: {target}  (historical figure: {hist:.3f})\n{'#'*78}\n")
        acc_a = comparison_a_old_buggy(df_s, target)
        acc_b = comparison_b_new_correct(df_s, target)
        acc_c, chance_c = comparison_c_missingness_predictable(df_s, target)
        results[target] = {"old_buggy": acc_a, "new_correct": acc_b,
                           "missingness_pred": acc_c, "missingness_chance": chance_c,
                           "historical": hist}

        print(f"  SUMMARY for {target}:")
        print(f"    Historical figure:        {hist:.3f}")
        print(f"    A (old buggy, reproduce): {acc_a:.3f}  "
              f"(diff from historical: {acc_a-hist:+.3f})")
        print(f"    B (new correct):          {acc_b:.3f}")
        print(f"    C (missingness predictable from HCI): {acc_c:.3f} "
              f"vs chance {chance_c:.3f}  (over: {acc_c-chance_c:+.3f})")

        if abs(acc_a - hist) < 0.03:
            print(f"    -> [CONFIRMED] Old buggy construction reproduces the "
                  f"historical {hist:.1%} almost exactly.")
            if acc_c - chance_c > 0.03:
                print(f"    -> [MECHANISM CONFIRMED] HCI predicts missingness "
                      f"above chance -- the historical number was very likely")
                print(f"       exploiting 'is this sensor reading missing' as a "
                      f"proxy signal, not genuine {target} direction.")
        else:
            print(f"    -> Old buggy construction does NOT cleanly reproduce "
                  f"{hist:.1%} either -- historical figure's source still unclear.")

    print("\n" + "=" * 78)
    print("OVERALL VERDICT")
    print("=" * 78)
    print("""
If A reproduced the historical numbers and C shows missingness is
predictable from HCI: the 79.9%/78.1% figures were never real
directional signal. Both HR_rising and RMSSD_rising need the same
flagged/excluded treatment already applied to RMSSD_magnitude and SCL.
That would leave the CCA result (0.549-0.581, condition-level means,
NOT delta-based, structurally unaffected by this bug) as the only
surviving Tier 1 component.

If A did NOT reproduce the historical numbers: something else explains
them (different data snapshot, different feature set) and that source
needs to be located before citing 79.9%/78.1% again.
""")


if __name__ == "__main__":
    main()
