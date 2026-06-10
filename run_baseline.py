"""
biosignal_proxy/run_baseline.py
================================
V0 (mean predictor) + V1 (RidgeCV) baseline gate for the HCI->biosignal proxy.

Pipeline:
  per subject -> build_xy (hci_features + biosignal_targets, cropped to PB 'Task' window)
              -> per-user z-score X and Y (each subject by its OWN stats; leak-free)
  LOSO        -> for each held-out subject, per-target fit on the other 15, score on the held-out
  report      -> per-target R2 (mean +/- std over folds), MAE, valid folds, and the
                 headline "N/12 targets with mean LOSO R2 > 0.2".

This is the GATE: if V1 (and later V3) can't beat ~0 on any target, HCI does not
predict biosignals on this data — which is itself the answer to the circular-
validation question. If several targets clear R2>0.2, proceed to V2-V7.

Run:
    pip install scikit-learn --quiet     # if needed
    python run_baseline.py

Files hci_features.py and biosignal_targets.py must sit in the same folder.
Per-subject (X,Y) are cached to proxy_cache/ so reruns are instant.
"""

from __future__ import annotations
import os, json, glob
import numpy as np
from sklearn.linear_model import RidgeCV

import hci_features as H
import biosignals_targets as B

# --------------------------------------------------------------------------- config
DATASET_DIR = os.path.expanduser("~/biosignals_data/cog_lab")
EXCLUDE     = {"S2", "S17"}          # S2: no HCI folder.  S17: byte-identical duplicate of S1.
WINDOW_S, STRIDE_S = 30.0, 15.0
CACHE_DIR   = "proxy_cache"
R2_GOOD     = 0.2                    # "recoverable" threshold for the headline count

# ----------------------------------------------------------------------- data loading
def pb_task_window(subject_root: str, sid: str):
    """Read the PB_description 'Task' [start, end] — the canonical crop window."""
    cands = glob.glob(os.path.join(subject_root, f"D3_{sid}_PB_description.json"))
    if not cands:
        return None, None
    with open(cands[0]) as f:
        d = json.load(f)
    first = next(iter(d.values()))           # top key varies ("ECG lesson", ...)
    task = first.get("Task")
    if task and len(task) >= 2:
        return float(task[0]), float(task[1])
    return None, None

def list_subjects():
    subs = [d for d in os.listdir(DATASET_DIR)
            if d.startswith("S") and d[1:].isdigit() and d not in EXCLUDE
            and os.path.isdir(os.path.join(DATASET_DIR, d, "HCI"))]
    return sorted(subs, key=lambda s: int(s[1:]))

def load_subject(sid: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = os.path.join(CACHE_DIR, f"{sid}.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        return z["X"], z["Y"]
    root = os.path.join(DATASET_DIR, sid)
    ts, te = pb_task_window(root, sid)
    X, Y, starts, _, _ = B.build_xy(root, sid, source="coglab",
                                    t_start=ts, t_end=te,
                                    window_s=WINDOW_S, stride_s=STRIDE_S)
    np.savez(cpath, X=X, Y=Y)
    return X, Y

# ------------------------------------------------------------------- normalization
def zscore_self(M: np.ndarray) -> np.ndarray:
    """Per-subject z-score, NaN-safe (NaNs stay NaN). Each subject uses only its own stats."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(M, axis=0, keepdims=True)
        sd = np.nanstd(M, axis=0, keepdims=True) + 1e-9
    return (M - mu) / sd

# --------------------------------------------------------------------------- LOSO eval
def loso_eval(data: dict, subs: list[str], n_targets: int):
    """data[sid] = (Xz, Yz). Returns r2[fold, target], mae[fold, target] for V0 and V1."""
    F = len(subs)
    r2_v0 = np.full((F, n_targets), np.nan)
    r2_v1 = np.full((F, n_targets), np.nan)
    mae_v1 = np.full((F, n_targets), np.nan)

    for fi, held in enumerate(subs):
        Xtr = np.vstack([data[s][0] for s in subs if s != held])
        Ytr = np.vstack([data[s][1] for s in subs if s != held])
        Xte, Yte = data[held]

        x_tr_ok = np.isfinite(Xtr).all(axis=1)
        x_te_ok = np.isfinite(Xte).all(axis=1)

        for j in range(n_targets):
            tr = x_tr_ok & np.isfinite(Ytr[:, j])
            te = x_te_ok & np.isfinite(Yte[:, j])
            if tr.sum() < 30 or te.sum() < 5:
                continue
            yt = Yte[te, j]
            ss_tot = ((yt - yt.mean()) ** 2).sum() + 1e-12

            # V0 — predict training mean
            v0_pred = Ytr[tr, j].mean()
            r2_v0[fi, j] = 1 - ((yt - v0_pred) ** 2).sum() / ss_tot

            # V1 — RidgeCV (internal CV on the training fold only -> no leak)
            model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            model.fit(Xtr[tr], Ytr[tr, j])
            pred = model.predict(Xte[te])
            r2_v1[fi, j]  = 1 - ((yt - pred) ** 2).sum() / ss_tot
            mae_v1[fi, j] = np.abs(yt - pred).mean()

    return r2_v0, r2_v1, mae_v1

# --------------------------------------------------------------------------- main
def main():
    subs = list_subjects()
    print(f"subjects ({len(subs)}): {subs}\n")

    data = {}
    for s in subs:
        X, Y = load_subject(s)
        dead = [B.TARGET_NAMES[j] for j in range(Y.shape[1]) if np.all(~np.isfinite(Y[:, j]))]
        flag = f"  <-- all-NaN targets: {dead}" if dead else ""
        print(f"  {s:4s}  windows={X.shape[0]:4d}{flag}")
        data[s] = (H.per_user_zscore(X), zscore_self(Y))
    print()

    yn = B.TARGET_NAMES
    r2_v0, r2_v1, mae_v1 = loso_eval(data, subs, len(yn))

    print(f"{'target':20s} {'V0_R2':>7s} {'V1_R2':>7s} {'V1_std':>7s} {'V1_MAE':>7s} {'folds':>6s}")
    print("-" * 60)
    for j, name in enumerate(yn):
        c = r2_v1[:, j]; valid = np.isfinite(c)
        print(f"{name:20s} "
              f"{np.nanmean(r2_v0[:, j]):7.3f} "
              f"{np.nanmean(c):7.3f} "
              f"{np.nanstd(c):7.3f} "
              f"{np.nanmean(mae_v1[:, j]):7.3f} "
              f"{valid.sum():6d}")
    good = int((np.nanmean(r2_v1, axis=0) > R2_GOOD).sum())
    print("-" * 60)
    print(f"\nV1 ridge: {good}/{len(yn)} targets with mean LOSO R2 > {R2_GOOD}")
    print("(V0_R2 should sit near 0 — it's the floor. Targets where V1_R2 >> 0 are the recoverable ones.)")

if __name__ == "__main__":
    main()