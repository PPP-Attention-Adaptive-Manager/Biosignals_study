"""
sense42_engagement_gap_test.py
=================================
Tests whether Pass B (physio+HCI+task) genuinely beats Pass A (physio
alone) at predicting frontal_theta/frontal_alpha, using a PAIRED
per-fold comparison rather than a difference of two independently-
estimated aggregate R2 numbers.

WHY THIS IS NECESSARY
------------------------
sense42_engagement_test.py's "gap(B-A)" compares two R2 values, each
already averaged across LOSO folds -- but the DIFFERENCE between two
noisy estimates has its own sampling variance, never checked. For
frontal_alpha specifically: A=-0.068 (below zero, not individually
real) and B=+0.028 (below the 0.03 threshold, also not individually
real) -- yet gap=+0.096 was reported as "adds real value." Neither
endpoint was validated on its own; the gap between two shaky numbers
isn't automatically meaningful.

METHOD
--------
For each LOSO fold (same participant held out), fit BOTH Pass A and
Pass B models on the SAME training data, predict on the SAME held-out
participant, and record the R2 DIFFERENCE for that fold. Then test
whether this per-fold difference is CONSISTENTLY positive (paired
t-test + Wilcoxon across folds) -- the correct way to ask "does adding
these features help," using the natural pairing (same held-out
person, same train/test split) instead of independently-estimated
aggregates.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
from scipy.stats import ttest_rel, wilcoxon

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
EV_CSV = os.path.join(BASE, "outputs", "sense42_v2_events.csv")

HCI_COLS = ["SnKeyStrokes","SnChars","SnSpecialKeys","SnDirectionKeys",
            "SnErrorKeys","SnSpaces","CharactersRatio","ErrorKeyRatio",
            "SnLeftClicked","SnMouseDistance","SnMouseAct"]
PHYSIO_PRED = ["engagement_index", "hr_mean", "hrv_rmssd", "resp_bpm"]
APPS = ["mail", "notes", "file_mgr", "browser", "trash"]
TARGETS = ["frontal_theta", "frontal_alpha", "theta_alpha_ratio"]


def zscore_within(df, cols, group="participant"):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = df.groupby(group)[c].transform(
                lambda x: (x-x.mean())/(x.std()+1e-9) if x.std()>1e-9 else 0.0)
    return out


def paired_fold_comparison(dfz, target, pred_a, pred_b, groups):
    Xa = np.nan_to_num(dfz[pred_a].to_numpy(float))
    Xb = np.nan_to_num(dfz[pred_b].to_numpy(float))
    y  = dfz[target].to_numpy(float)

    logo = LeaveOneGroupOut()
    diffs, r2a_list, r2b_list = [], [], []

    for tr, te in logo.split(Xa, y, groups):
        ok_tr = np.isfinite(Xb[tr]).all(1) & np.isfinite(y[tr])
        ok_te = np.isfinite(Xb[te]).all(1) & np.isfinite(y[te])
        if ok_tr.sum() < 15 or ok_te.sum() < 3:
            continue
        ytr, yte = y[tr][ok_tr], y[te][ok_te]

        ma = RandomForestRegressor(200, min_samples_leaf=5, random_state=0, n_jobs=-1)
        ma.fit(Xa[tr][ok_tr], ytr)
        r2a = r2_score(yte, ma.predict(Xa[te][ok_te]))

        mb = RandomForestRegressor(200, min_samples_leaf=5, random_state=0, n_jobs=-1)
        mb.fit(Xb[tr][ok_tr], ytr)
        r2b = r2_score(yte, mb.predict(Xb[te][ok_te]))

        diffs.append(r2b - r2a)
        r2a_list.append(r2a); r2b_list.append(r2b)

    return np.array(diffs), np.array(r2a_list), np.array(r2b_list)


def main():
    print("="*78)
    print("SENSE-42: paired per-fold test of whether B genuinely beats A")
    print("="*78)

    df = pd.read_csv(EV_CSV)
    for c in ["frontal_theta", "frontal_alpha"]:
        v = df[c].to_numpy(float)
        df[c] = np.where(v > 0, np.log10(v), np.nan)

    need = PHYSIO_PRED + TARGETS
    df = df.dropna(subset=[c for c in need if c in df.columns])
    hci_avail = [c for c in HCI_COLS if c in df.columns]
    for a in APPS:
        df[f"app_{a}"] = (df["app"] == a).astype(float) if "app" in df.columns else 0.0
    app_cols = [f"app_{a}" for a in APPS]

    print(f"Loaded {len(df)} events, {df.participant.nunique()} participants\n")

    groups = df["participant"].to_numpy()
    pred_a = PHYSIO_PRED
    pred_b = PHYSIO_PRED + hci_avail + app_cols
    dfz = zscore_within(df, pred_a + pred_b + TARGETS)

    print(f"{'target':18s} {'mean diff':>10s} {'paired t':>9s} {'p(t)':>8s} "
          f"{'p(Wilcox)':>10s} {'n folds':>8s}  verdict")
    print("-"*84)

    for tgt in TARGETS:
        diffs, r2a, r2b = paired_fold_comparison(dfz, tgt, pred_a, pred_b, groups)
        if len(diffs) < 8:
            print(f"  {tgt:16s}  too few folds ({len(diffs)})")
            continue
        t_stat, p_t = ttest_rel(r2b, r2a)
        try:
            w_stat, p_w = wilcoxon(r2b, r2a)
        except ValueError:
            p_w = np.nan

        n_improved = int((diffs > 0).sum())
        verdict = ("REAL improvement" if p_t < 0.05 and diffs.mean() > 0
                  else "not significant -- gap not distinguishable from fold noise")
        print(f"  {tgt:16s} {diffs.mean():+10.3f} {t_stat:9.2f} {p_t:8.4f} "
              f"{p_w:10.4f} {len(diffs):8d}  {verdict}")
        print(f"    ({n_improved}/{len(diffs)} folds improved with B; "
              f"mean R2: A={r2a.mean():+.3f} B={r2b.mean():+.3f})")


if __name__ == "__main__":
    main()
