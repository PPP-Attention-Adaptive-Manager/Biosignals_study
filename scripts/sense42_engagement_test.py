"""
sense42_engagement_test.py
=============================
Same question as stew_engagement_test.py / clare_engagement_test.py,
on SENSE-42, using the already-cached sense42_v2_events.csv (no
re-extraction needed -- frontal_theta, frontal_alpha, theta_alpha_ratio,
engagement_index, hr_mean, hrv_rmssd, resp_bpm, full HCI columns, and
app/task labels are all already there).

TWO PASSES
------------
A. engagement_index + hr_mean + hrv_rmssd + resp_bpm
   -> predict frontal_theta, frontal_alpha, theta_alpha_ratio
   Same physio-only design as the CLARE test (ECG+EDA there,
   ECG+resp here -- SENSE-42 has respiration, CLARE doesn't).

B. Same predictors + full HCI feature set + task identity (app one-hot)
   -> same targets
   Tests whether adding behavioral + task context improves EEG
   reconstruction beyond physiology alone -- the "all that + HCI and
   especially the task" request.

CAVEAT CARRIED FROM STEW
----------------------------
engagement_index = beta/(alpha+theta). Frontal alpha and frontal theta
are literally the denominator of the predictor being used to predict
them, so part of any positive result here could be mathematical
entanglement rather than purely physiological. theta_alpha_ratio uses
the identical two components combined differently -- if it does NOT
show the same inflated relationship, that's evidence against pure
circularity (as it was in the STEW result). Watch this the same way
for every target below.

Same discipline as every check this session: true per-participant
LOSO, permutation control on every target.

Run from: ~/biosignals_data/
Output:   outputs/sense42_engagement_results.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
EV_CSV = os.path.join(BASE, "outputs", "sense42_v2_events.csv")
OUT_JSON = os.path.join(BASE, "outputs", "sense42_engagement_results.json")

HCI_COLS = ["SnKeyStrokes","SnChars","SnSpecialKeys","SnDirectionKeys",
            "SnErrorKeys","SnSpaces","CharactersRatio","ErrorKeyRatio",
            "SnLeftClicked","SnMouseDistance","SnMouseAct"]

PHYSIO_PRED = ["engagement_index", "hr_mean", "hrv_rmssd", "resp_bpm"]
TARGETS = ["frontal_theta", "frontal_alpha", "theta_alpha_ratio"]
APPS = ["mail", "notes", "file_mgr", "browser", "trash"]

MIN_FOLDS = 8


def zscore_within(df, cols, group="participant"):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = df.groupby(group)[c].transform(
                lambda x: (x-x.mean())/(x.std()+1e-9) if x.std()>1e-9 else 0.0)
    return out


def loso_r2(X, y, groups, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    logo = LeaveOneGroupOut()
    scores = []
    for tr, te in logo.split(X, y, groups):
        ok_tr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ok_te = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if ok_tr.sum() < 15 or ok_te.sum() < 3:
            continue
        ytr = y[tr][ok_tr]
        if shuffle:
            ytr = rng.permutation(ytr)
        m = RandomForestRegressor(200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
        m.fit(X[tr][ok_tr], ytr)
        scores.append(r2_score(y[te][ok_te], m.predict(X[te][ok_te])))
    return float(np.mean(scores)) if scores else np.nan, len(scores)


def run_pass(df, pred_cols, label, groups):
    avail = [c for c in pred_cols if c in df.columns]
    dfz = zscore_within(df, avail + TARGETS)
    X = np.nan_to_num(dfz[avail].to_numpy(float))

    print(f"\n--- {label} ---")
    print(f"Predictors ({len(avail)}): {avail}")
    print(f"{'target':22s} {'R2 real':>9s} {'R2 perm':>9s} {'over':>8s}")
    print("-"*54)

    res = {}
    for tgt in TARGETS:
        y = dfz[tgt].to_numpy(float)
        r2r, nf = loso_r2(X, y, groups)
        r2p, _  = loso_r2(X, y, groups, shuffle=True)
        if nf < MIN_FOLDS:
            print(f"  {tgt:20s}  insufficient folds ({nf})")
            continue
        flag = "  *** REAL" if r2r > 0.03 and r2r - r2p > 0.03 else ""
        print(f"  {tgt:20s} {r2r:9.3f} {r2p:9.3f} {r2r-r2p:+8.3f}{flag}")
        res[tgt] = {"r2_real": r2r, "r2_perm": r2p, "n_folds": nf}
    return res


def main():
    print("="*78)
    print("SENSE-42: does engagement_index (+physio, +HCI/task) predict EEG?")
    print("="*78)

    if not os.path.isfile(EV_CSV):
        print(f"Missing {EV_CSV}"); return

    df = pd.read_csv(EV_CSV)

    # BUG FIX: frontal_theta/frontal_alpha in sense42_v2_events.csv are
    # RAW, un-logged band power (~1e-13 to 1e-5 V^2/Hz -- confirmed via
    # the trait-level correlation diagnostic earlier this session). Per-
    # participant z-scoring on a variable spanning 8 orders of magnitude,
    # dominated by rare extreme-power windows, produces a near-constant
    # z-scored target for most rows -- which is exactly what produced
    # the degenerate "0.000 real, 0.000 perm" result in the first run of
    # this script. theta_alpha_ratio was unaffected (it's a dimensionless
    # ratio, already scale-normalized) and gave a clean, trustworthy
    # result (0.240, over+0.277) precisely because it never hit this bug.
    # Same fix already applied in sense42_task_identity_eeg.py and
    # sense42_trait_level_correlation.py -- just missing here.
    for _c in ["frontal_theta", "frontal_alpha"]:
        if _c in df.columns:
            _v = df[_c].to_numpy(float)
            df[_c] = np.where(_v > 0, np.log10(_v), np.nan)
    print("frontal_theta/frontal_alpha log10-transformed before use "
          "(fixes the degenerate-zscore bug from the previous run)")

    need = PHYSIO_PRED + TARGETS
    df = df.dropna(subset=[c for c in need if c in df.columns])
    print(f"Loaded {len(df)} events, {df.participant.nunique()} participants "
          f"(after dropping rows missing any predictor/target)\n")

    hci_avail = [c for c in HCI_COLS if c in df.columns]
    for a in APPS:
        df[f"app_{a}"] = (df["app"] == a).astype(float) if "app" in df.columns else 0.0
    app_cols = [f"app_{a}" for a in APPS]

    groups = df["participant"].to_numpy()

    results = {}
    results["A_physio_only"] = run_pass(
        df, PHYSIO_PRED, "PASS A: engagement + HR + RMSSD + resp_bpm", groups)
    results["B_physio_plus_hci_task"] = run_pass(
        df, PHYSIO_PRED + hci_avail + app_cols,
        "PASS B: A + full HCI feature set + task identity (app one-hot)", groups)

    print("\n" + "="*78)
    print("A vs B comparison — does HCI+task add anything beyond physio alone?")
    print("="*78)
    for tgt in TARGETS:
        a = results["A_physio_only"].get(tgt, {}).get("r2_real")
        b = results["B_physio_plus_hci_task"].get(tgt, {}).get("r2_real")
        if a is not None and b is not None:
            print(f"  {tgt:22s} A={a:+.3f}  B={b:+.3f}  "
                  f"gap(B-A)={b-a:+.3f}  "
                  f"{'HCI/task adds real value' if b-a>0.03 else 'no meaningful addition'}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
