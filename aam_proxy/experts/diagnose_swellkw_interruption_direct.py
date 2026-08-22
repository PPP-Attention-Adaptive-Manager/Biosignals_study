"""
diagnose_swellkw_interruption_direct.py
==========================================
The one untested, legitimate path in the interruption-detection question:
can HCI behavior directly predict SWELL-KW's Condition label (N/I/T),
with NO biosignal intermediary anywhere in the pipeline? This is
structurally different from every SWELL-KW result invalidated earlier
today (HR_rising, RMSSD_rising, CCA) -- those all tried to predict
CARDIAC/SKIN signals from HCI. This predicts the CONDITION LABEL itself,
which is what the router actually needs (interruption_events > 0 or not).

TWO FRAMINGS
--------------
A. 3-class: N vs I vs T (matches the historical 42.6% number's likely
   framing -- never re-validated with true LOSO + empirical chance)
B. Binary: I (interruption present) vs {N,T} combined -- directly
   matches what session_router.py's interruption_events>0 check needs

BOTH use:
  - per-minute HCI counts (raw, not delta -- absolute activity level
    during a condition block is the natural signal here, unlike the
    direction/magnitude framing used for continuous physio targets)
  - true per-participant LeaveOneGroupOut (no ShuffleSplit/GroupKFold(n)
    substitution -- the exact leak that inflated the CCA/direction
    results earlier)
  - empirical chance from actual per-fold class distribution
  - permutation control

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, glob, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
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


def zscore_within(X, groups):
    Xz = np.zeros_like(X, dtype=float)
    for u in np.unique(groups):
        m = groups == u
        Xz[m] = (X[m]-np.nanmean(X[m],0))/(np.nanstd(X[m],0)+1e-9)
    return Xz


def true_loso(X, y, groups, n_classes, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    logo = LeaveOneGroupOut()
    accs, chances = [], []
    for tr, te in logo.split(X, y, groups):
        ok_tr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ok_te = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if ok_tr.sum() < 15 or ok_te.sum() < 3: continue
        ytr, yte = y[tr][ok_tr].astype(int), y[te][ok_te].astype(int)
        if len(np.unique(ytr)) < 2: continue
        if shuffle: ytr = rng.permutation(ytr)
        m = RandomForestClassifier(200, min_samples_leaf=5,
                                   class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
        m.fit(X[tr][ok_tr], ytr)
        pred = m.predict(X[te][ok_te])
        accs.append(accuracy_score(yte, pred))
        counts = np.bincount(yte, minlength=n_classes)
        chances.append(counts.max()/counts.sum())
    return (float(np.mean(accs)), float(np.mean(chances)), len(accs)) if accs else (np.nan,np.nan,0)


def main():
    print("="*78)
    print("SWELL-KW: HCI -> Condition, DIRECT (no biosignal intermediary)")
    print("Historical reference: 42.6% (3-class), never re-validated")
    print("="*78)

    df = pd.read_excel(SWELL_FILE)
    df = df[df["Condition"].isin(["N","I","T"])].copy()
    print(f"\n{len(df)} rows, {df.PP.nunique()} participants\n")

    X = np.nan_to_num(df[HCI_COLS].to_numpy(float))
    groups = df["PP"].to_numpy()
    Xz = zscore_within(X, groups)

    cond_map = {"N":0, "I":1, "T":2}
    y3 = df["Condition"].map(cond_map).to_numpy()

    print("=== A: 3-class (N vs I vs T) ===")
    acc, chance, nf = true_loso(Xz, y3, groups, n_classes=3)
    perm, _, _ = true_loso(Xz, y3, groups, n_classes=3, shuffle=True)
    over = acc - chance
    perm_over = perm - chance
    print(f"  chance={chance:.3f}  acc={acc:.3f}  over={over:+.3f}  "
          f"perm_acc={perm:.3f}  perm_over={perm_over:+.3f}  ({nf} folds)")
    real_vs_perm = over - perm_over
    verdict_a = "REAL" if over>0.03 and real_vs_perm>0.03 else "NOT REAL -- matches permutation"
    print(f"  real_vs_perm={real_vs_perm:+.3f}  -> {verdict_a}")

    print("\n=== B: binary (Interruption vs {N,T}) — matches router's actual need ===")
    y_bin = (df["Condition"] == "I").astype(int).to_numpy()
    acc_b, chance_b, nf_b = true_loso(Xz, y_bin, groups, n_classes=2)
    perm_b, _, _ = true_loso(Xz, y_bin, groups, n_classes=2, shuffle=True)
    over_b = acc_b - chance_b
    perm_over_b = perm_b - chance_b
    print(f"  chance={chance_b:.3f}  acc={acc_b:.3f}  over={over_b:+.3f}  "
          f"perm_acc={perm_b:.3f}  perm_over={perm_over_b:+.3f}  ({nf_b} folds)")
    real_vs_perm_b = over_b - perm_over_b
    verdict_b = "REAL" if over_b>0.03 and real_vs_perm_b>0.03 else "NOT REAL -- matches permutation"
    print(f"  real_vs_perm={real_vs_perm_b:+.3f}  -> {verdict_b}")

    print("\n" + "="*78)
    print("VERDICT")
    print("="*78)
    print(f"\n  3-class (N/I/T):        {verdict_a}")
    print(f"  Binary (I vs not-I):    {verdict_b}")
    if verdict_a == "REAL" or verdict_b == "REAL":
        print("\n  At least one framing survives true LOSO + permutation.")
        print("  This is a genuine, defensible interruption-branch signal --")
        print("  safe to wire into session_router.py as a real expert.")
    else:
        print("\n  Neither framing survives. HCI does not directly predict")
        print("  interruption condition either, even without a biosignal")
        print("  intermediary. The interruption branch stays")
        print("  no_expert_available -- confirmed, not just assumed.")


if __name__ == "__main__":
    main()
