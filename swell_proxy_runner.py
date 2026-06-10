"""
swell_proxy_runner.py
======================
SWELL-KW biosignal proxy gate:
  HCI minute features (mouse + keyboard + app-switching) → HR / RMSSD / SCL

Same LOSO harness logic as run_baseline.py, adapted for SWELL-KW structure:
  - Source: Behavioral-features - per minute.xlsx  (all signals merged, 1 row/minute)
  - 25 participants, 3 work conditions each (~45-60 min/condition)
  - 999 = NaN marker in physiology columns
  - Relax condition excluded (no knowledge-work HCI, resting physiology)

Models run per target:
  V0  mean predictor (R²=0 floor)
  V1  RidgeCV        (linear gate)
  V3  Random Forest  (nonlinear gate)
  CON condition-only ridge (confound check: is the model just learning "condition X = higher HR"?)

Outputs per target:
  V0_R2, V1_R2, V3_R2, CON_R2, verdict

Run:
    pip install scikit-learn openpyxl --quiet
    python swell_proxy_runner.py

Edit SWELL_DIR at the top if your path differs.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------------ config
SWELL_DIR   = os.path.expanduser("~/biosignals_data/swell-kw")
DATA_FILE   = os.path.join(SWELL_DIR, "Behavioral-features - per minute.xlsx")
NAN_MARKER  = 999
EXCLUDE_COND = {"R"}          # relax / baseline — no knowledge-work HCI
R2_GOOD      = 0.2

# ------------------------------------------------------------------ feature sets
HCI_COLS = [
    # mouse
    "SnMouseAct", "SnLeftClicked", "SnRightClicked", "SnDoubleClicked",
    "SnWheel", "SnDragged", "SnMouseDistance",
    # keyboard
    "SnKeyStrokes", "SnChars", "SnSpecialKeys", "SnDirectionKeys",
    "SnErrorKeys", "SnShortcutKeys", "SnSpaces",
    # app-switching  ← the channel Cog Lab had ZERO of
    "SnAppChange", "SnTabfocusChange",
    # derived
    "CharactersRatio", "ErrorKeyRatio",
]
TARGET_COLS  = ["HR", "RMSSD", "SCL"]
COND_MAP     = {"N": 0, "I": 1, "T": 2}    # neutral / interruptions / time-pressure


# ------------------------------------------------------------------ loading
def load_swell() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)

    # 999 → NaN in physiology
    for col in TARGET_COLS:
        if col in df.columns:
            df[col] = df[col].replace(NAN_MARKER, np.nan)

    # drop relax condition
    df = df[~df["Condition"].isin(EXCLUDE_COND)].copy()

    # encode condition as integer (for confound test)
    df["_cond_int"] = df["Condition"].map(COND_MAP).fillna(-1).astype(float)

    return df


def build_arrays(df: pd.DataFrame, pp: str):
    """Return (X_hci, X_cond, Y) for one participant, per-user z-scored."""
    sub = df[df["PP"] == pp].copy()

    X = sub[HCI_COLS].to_numpy(dtype=float)
    C = sub[["_cond_int"]].to_numpy(dtype=float)   # 1-feature condition baseline
    Y = sub[TARGET_COLS].to_numpy(dtype=float)

    # per-user z-score X (not C — it's already 0/1/2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X,  axis=0, keepdims=True) + 1e-9
    X = (X - mu) / sd

    # per-user z-score Y per target
    for j in range(Y.shape[1]):
        col = Y[:, j]
        fin = np.isfinite(col)
        if fin.sum() > 1:
            m, s = col[fin].mean(), col[fin].std() + 1e-9
            Y[:, j] = (col - m) / s

    return X, C, Y


# ------------------------------------------------------------------ LOSO eval
def loso(data: dict, subs: list[str]):
    """data[pp] = (X_hci, X_cond, Y).  Returns dicts of R² arrays per variant."""
    F, T = len(subs), len(TARGET_COLS)
    r2 = {k: np.full((F, T), np.nan) for k in ("v0", "v1", "v3", "con")}

    for fi, held in enumerate(subs):
        Xtr = np.vstack([data[s][0] for s in subs if s != held])
        Ctr = np.vstack([data[s][1] for s in subs if s != held])
        Ytr = np.vstack([data[s][2] for s in subs if s != held])
        Xte, Cte, Yte = data[held]

        xtr_ok = np.isfinite(Xtr).all(axis=1)
        xte_ok = np.isfinite(Xte).all(axis=1)

        for j in range(T):
            tr = xtr_ok & np.isfinite(Ytr[:, j])
            te = xte_ok & np.isfinite(Yte[:, j])
            if tr.sum() < 30 or te.sum() < 5:
                continue
            yt  = Yte[te, j]
            sst = ((yt - yt.mean()) ** 2).sum() + 1e-12

            def r2_score(model, Xfit, Xpred):
                model.fit(Xfit[tr], Ytr[tr, j])
                p = model.predict(Xpred[te])
                return 1 - ((yt - p) ** 2).sum() / sst

            # V0 mean
            r2["v0"][fi, j] = 1 - ((yt - Ytr[tr, j].mean()) ** 2).sum() / sst

            # V1 ridge
            r2["v1"][fi, j] = r2_score(
                RidgeCV(alphas=[0.1, 1, 10, 100]), Xtr, Xte)

            # V3 RF
            r2["v3"][fi, j] = r2_score(
                RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                      max_depth=10, n_jobs=-1, random_state=0),
                Xtr, Xte)

            # CON condition-only ridge (confound check)
            ctr_ok = np.isfinite(Ctr[:, 0])
            cte_ok = np.isfinite(Cte[:, 0])
            tr_c   = tr & ctr_ok
            te_c   = te & cte_ok
            if tr_c.sum() >= 10 and te_c.sum() >= 3:
                yt_c   = Yte[te_c, j]
                sst_c  = ((yt_c - yt_c.mean()) ** 2).sum() + 1e-12
                m = RidgeCV(alphas=[0.1, 1, 10]).fit(Ctr[tr_c], Ytr[tr_c, j])
                p = m.predict(Cte[te_c])
                r2["con"][fi, j] = 1 - ((yt_c - p) ** 2).sum() / sst_c

    return r2


# ------------------------------------------------------------------ verdict
def verdict(v1, v3, con):
    best = max(v1, v3)
    if best < 0.1:
        return "flat (no signal)"
    if con > 0.7 * best:
        return "CONDITION CONFOUND"    # model just learned condition identity
    if v3 > v1 + 0.1:
        return "nonlinear signal"
    return "weak linear signal" if best < R2_GOOD else "recoverable signal ✓"


# ------------------------------------------------------------------ main
def main():
    print(f"Loading {DATA_FILE} …")
    df = load_swell()

    subs = sorted(df["PP"].unique().tolist())
    print(f"Participants: {len(subs)}  |  Rows (work conditions only): {len(df)}")
    print(f"Conditions kept: {sorted(df['Condition'].unique().tolist())}\n")

    data = {}
    for pp in subs:
        X, C, Y = build_arrays(df, pp)
        n_rows = X.shape[0]
        dead   = [TARGET_COLS[j] for j in range(Y.shape[1])
                  if np.all(~np.isfinite(Y[:, j]))]
        flag   = f"  <-- all-NaN targets: {dead}" if dead else ""
        print(f"  {pp:5s}  rows={n_rows:4d}{flag}")
        data[pp] = (X, C, Y)
    print()

    r2 = loso(data, subs)

    # report
    print(f"{'target':8s} {'V0_R2':>7s} {'V1_R2':>7s} {'V3_R2':>7s} "
          f"{'CON_R2':>7s} {'folds':>6s}  verdict")
    print("-" * 70)
    for j, name in enumerate(TARGET_COLS):
        v0  = np.nanmean(r2["v0"][:, j])
        v1  = np.nanmean(r2["v1"][:, j])
        v3  = np.nanmean(r2["v3"][:, j])
        con = np.nanmean(r2["con"][:, j])
        folds = int(np.isfinite(r2["v1"][:, j]).sum())
        print(f"{name:8s} {v0:7.3f} {v1:7.3f} {v3:7.3f} {con:7.3f} "
              f"{folds:6d}  {verdict(v1, v3, con)}")
    print("-" * 70)

    good = int((np.nanmean(r2["v3"], axis=0) > R2_GOOD).sum())
    print(f"\nV3 RF: {good}/{len(TARGET_COLS)} targets with mean LOSO R² > {R2_GOOD}")
    print()
    print("Reading it:")
    print("  CONDITION CONFOUND → model learned condition identity, not within-condition HCI→physio")
    print("  recoverable signal → real HCI–physiology link, proceed to proxy training")
    print("  flat               → no signal on this dataset either")
    print()
    print("Key difference from Cog Lab: SWELL-KW has 3 deliberately varied stress conditions")
    print("and includes app-switching (SnAppChange / SnTabfocusChange) which Cog Lab had zero of.")
    print("A CONDITION CONFOUND result is still informative — it means HCI distinguishes")
    print("conditions (stress levels), which validates the AAM premise even if within-condition")
    print("minute-level prediction is hard.")


if __name__ == "__main__":
    main()