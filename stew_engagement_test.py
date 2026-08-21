"""
stew_engagement_test.py
==========================
Tests whether eeg_engagement (beta/(alpha+theta)) -- STEW's strongest
paired-contrast marker (rest vs multitask, d~0.4-0.7 across earlier
runs) -- can PREDICT the rest of a person's EEG picture from itself
alone: frontal_theta, frontal_alpha, theta_alpha_ratio.

WHY THIS IS A NEW, DIFFERENT QUESTION
------------------------------------------
Every prior STEW test asked "does engagement differ between rest and
multitask" (a paired contrast, real and confirmed). This asks something
structurally different: "if you only had engagement_index, could you
reconstruct the rest of the EEG state" -- i.e. is it a good one-number
SUMMARY of broader cortical activity, window to window, within the
SAME person. This is a genuinely different, harder claim, and has never
been tested.

METHOD
--------
Uses the SAME validated STEW pipeline as stew_analysis.py (dual-band
filtering, IAF correction, log-relative power, 5s/50%-overlap windows).
Reuses per-window features already computed there rather than
re-extracting.

Per-participant z-scoring, then LOSO (across the 45 rated participants)
predicting each OTHER EEG feature FROM engagement_index alone, plus a
permutation control on every target -- same discipline as every check
run today.

Run from: wherever stew_analysis.py and the STEW Dataset folder live
Output:   stew_engagement_results.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# Hardcoded directly -- confirmed via:
#   find ~/biosignals_data -iname "ratings.txt"
#   -> /home/hefouzinho/biosignals_data/data/stew/ratings.txt
# No more --data argument passing; that path kept silently failing to
# reach the actual file on disk across three copy attempts.
DATA_DIR = os.path.expanduser("~/biosignals_data/data/stew/STEW Dataset")
OUT_JSON = os.path.expanduser("~/biosignals_data/outputs/stew_engagement_results.json")

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"'{DATA_DIR}' not found. If STEW moved, edit DATA_DIR directly "
        f"at the top of this file -- do not rely on --data, it is not "
        f"parsed by this script.")
if not os.path.isfile(os.path.join(DATA_DIR, "ratings.txt")):
    raise FileNotFoundError(
        f"DATA_DIR exists but ratings.txt is missing inside it: {DATA_DIR}")

CHANNELS = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
FS = 128
EPOCH_S = 5
OVERLAP = 0.5
FRONTAL = ["AF3","F3","F4","AF4"]

THETA = (4, 8); ALPHA_FIXED = (8, 13); BETA = (13, 30)


def load_ratings():
    path = os.path.join(DATA_DIR, "ratings.txt")
    ratings = {}
    with open(path) as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) == 3:
                ratings[int(parts[0])] = True
    return ratings


def load_eeg(subject, condition):
    path = os.path.join(DATA_DIR, f"sub{subject:02d}_{condition}.txt")
    if not os.path.isfile(path):
        return None
    data = pd.read_csv(path, sep=r"\s+", header=None).to_numpy(float)
    return data - data.mean(axis=0)


def bandpass(data, lo, hi, fs=FS, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, data, axis=0)


def find_iaf(data, fs=FS):
    occ_idx = [CHANNELS.index(c) for c in ["O1","O2"] if c in CHANNELS]
    nper = min(data.shape[0], int(fs*2))
    f, psd = welch(data, fs=fs, nperseg=nper, axis=0)
    psd_occ = psd[:, occ_idx].mean(axis=1)
    m = (f>=7)&(f<=13)
    if m.sum()==0: return 10.0
    return float(f[m][np.argmax(psd_occ[m])])


def epoch_features(data, iaf, fs=FS):
    f_idx = [CHANNELS.index(c) for c in FRONTAL if c in CHANNELS]
    win = int(EPOCH_S*fs); step = int(win*(1-OVERLAP))
    rows, pos = [], 0
    while pos+win <= data.shape[0]:
        seg = data[pos:pos+win]
        nper = min(seg.shape[0], int(fs*2))
        f, psd = welch(seg, fs=fs, nperseg=nper, axis=0)
        theta = psd[(f>=THETA[0])&(f<THETA[1])].mean(axis=0)
        alpha = psd[(f>=max(1,iaf-2))&(f<iaf+2)].mean(axis=0)
        beta  = psd[(f>=BETA[0])&(f<BETA[1])].mean(axis=0)
        ft, fa, fb = theta[f_idx].mean(), alpha[f_idx].mean(), beta[f_idx].mean()
        def slog(x): return float(np.log10(x)) if x>0 else np.nan
        rows.append({
            "log_frontal_theta": slog(ft),
            "log_frontal_alpha": slog(fa),
            "theta_alpha_ratio": float(ft/(fa+1e-15)),
            "engagement_index":  float(fb/(fa+ft+1e-15)),
        })
        pos += step
    return pd.DataFrame(rows)


TARGETS = ["log_frontal_theta", "log_frontal_alpha", "theta_alpha_ratio"]


def zscore_within(df, cols, group="subject"):
    out = df.copy()
    for c in cols:
        out[c] = df.groupby(group)[c].transform(
            lambda x: (x-x.mean())/(x.std()+1e-9) if x.std()>1e-9 else 0.0)
    return out


def loso_r2(X, y, groups, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    logo = LeaveOneGroupOut()
    scores = []
    for tr, te in logo.split(X, y, groups):
        ok_tr = np.isfinite(X[tr,0]) & np.isfinite(y[tr])
        ok_te = np.isfinite(X[te,0]) & np.isfinite(y[te])
        if ok_tr.sum() < 20 or ok_te.sum() < 3: continue
        ytr = y[tr][ok_tr]
        if shuffle: ytr = rng.permutation(ytr)
        m = RandomForestRegressor(200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
        m.fit(X[tr][ok_tr], ytr)
        pred = m.predict(X[te][ok_te])
        scores.append(r2_score(y[te][ok_te], pred))
    return float(np.mean(scores)) if scores else np.nan, len(scores)


def main():
    print("="*78); print("STEW: does eeg_engagement predict the rest of the EEG picture?")
    print("="*78)

    ratings = load_ratings()
    all_rows = []
    for sid in sorted(ratings.keys()):
        for cond in ["lo","hi"]:
            data = load_eeg(sid, cond)
            if data is None: continue
            data_f = bandpass(data, 1, 40)
            iaf = find_iaf(data_f)
            feats = epoch_features(data_f, iaf)
            feats["subject"] = sid
            feats["condition"] = cond
            all_rows.append(feats)
    df = pd.concat(all_rows, ignore_index=True)
    print(f"Loaded {len(df)} windows, {df.subject.nunique()} subjects\n")

    dfz = zscore_within(df, ["engagement_index"] + TARGETS)
    X = np.nan_to_num(dfz[["engagement_index"]].to_numpy(float))
    groups = df["subject"].to_numpy()

    print(f"{'target':22s} {'R2 (real)':>10s} {'R2 (perm)':>10s} {'over':>8s}")
    print("-"*54)
    results = {}
    for tgt in TARGETS:
        y = dfz[tgt].to_numpy(float)
        r2_real, nf = loso_r2(X, y, groups)
        r2_perm, _  = loso_r2(X, y, groups, shuffle=True)
        flag = "  *** REAL" if r2_real > 0.03 and r2_real - r2_perm > 0.03 else ""
        print(f"  {tgt:20s} {r2_real:10.3f} {r2_perm:10.3f} {r2_real-r2_perm:+8.3f}{flag}")
        results[tgt] = {"r2_real": r2_real, "r2_perm": r2_perm, "n_folds": nf}

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
