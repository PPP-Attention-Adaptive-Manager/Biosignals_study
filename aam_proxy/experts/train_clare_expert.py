"""
train_clare_expert.py
========================
Builds the FROZEN, deployable CLARE expert -- structurally different
from Cog Lab: this one requires a REAL EEG calibration reading as
input (engagement_index computed from actual EEG), then extends it
into frontal_alpha and theta_alpha_ratio using cardiac/EDA signal on
top. It never predicts EEG from HCI alone -- that's the whole point of
the calibration-anchored design.

TARGETS -- frontal_theta DELIBERATELY EXCLUDED
----------------------------------------------------
clare_engagement_test.py found log_frontal_theta at R2=0.887-0.890 --
almost certainly mostly mathematical entanglement, since
engagement_index = beta/(alpha+theta) has theta directly in its own
denominator. Including it here would ship a circular "prediction" that
mostly un-computes its own input. Only the two LESS entangled, still
genuinely real targets are trained:
    frontal_alpha:         R2 0.057 (A) -> 0.129 (B), REAL, improved by
                            adding cardiac/EDA
    theta_alpha_ratio:      R2 0.309 (A) -> 0.349 (B), REAL
Both confirmed clean against the temporal-site (TP9/TP10) control in
the original test -- no target there crossed the significance bar.

PASS USED: B (engagement + HR + RMSSD + EDA tonic mean/slope) -- this
beat engagement-alone for both retained targets, so it's the version
worth deploying, not the simpler Pass A.

Extraction pipeline (clean_stream, ecg_window, eda_window, eeg_window,
process_participant) copied VERBATIM from clare_engagement_test.py --
same paper-validated filters (5-15Hz ECG, 8-12Hz alpha, 60Hz notch)
already confirmed against the CLARE paper's own numbers earlier in
this project.

Run from: ~/biosignals_data/
Output:   aam_proxy/experts/clare_expert_artifacts/
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch, iirnotch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
import joblib

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
CLARE_ROOT = os.path.join(BASE, "data", "clare", "doi-10.5683-sp3-h0aelt")
ECG_DIR = os.path.join(CLARE_ROOT, "ECG")
EDA_DIR = os.path.join(CLARE_ROOT, "EDA")
EEG_DIR = os.path.join(CLARE_ROOT, "EEG")
OUT_DIR = os.path.join(BASE, "aam_proxy", "experts", "clare_expert_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

ECG_LEADS = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]
PRIMARY_LEAD = "ECG LL-RA CAL"
EDA_COL = "GSR Conductance CAL"
ECG_SF, EDA_SF, EEG_SF = 512.0, 128.0, 256.0
FRONTAL  = ["AF7", "AF8"]
TEMPORAL = ["TP9", "TP10"]
WINDOW_S = 30.0; OVERLAP = 0.5
MIN_HR, MAX_HR = 40, 140
MIN_FOLDS = 8

PREDICTORS = ["frontal_engagement_index", "hr_mean", "hrv_rmssd",
             "eda_tonic_mean", "eda_tonic_slope"]
TARGETS = ["log_frontal_alpha", "frontal_theta_alpha_ratio"]   # theta excluded, see docstring


# ── verbatim extraction pipeline from clare_engagement_test.py ─────────

def clean_stream(path, value_cols, primary_col):
    raw = pd.read_csv(path)
    df = raw.dropna(subset=value_cols, how="all").copy()
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp", keep="first")
    df = df.dropna(subset=[primary_col]).reset_index(drop=True)
    return df


def ecg_window(ts, sig, w0, w1, sf=ECG_SF):
    mask = (ts>=w0)&(ts<w1)
    if mask.sum() < sf*5: return np.nan, np.nan
    seg = sig[mask]
    b,a = butter(3,[5/(sf/2),15/(sf/2)],btype="band")
    z = filtfilt(b,a,seg); z=(z-z.mean())/(z.std()+1e-9)
    peaks = np.array([])
    for h in [2.5,2.0,1.5,1.0]:
        peaks,_ = find_peaks(z, distance=int(0.35*sf), height=h)
        if len(peaks)>8: break
    if len(peaks)<6: return np.nan, np.nan
    seg_ts = ts[mask]; rr = np.diff(seg_ts[peaks])
    rr = rr[(rr>0.33)&(rr<1.5)]
    if len(rr)<4: return np.nan, np.nan
    hr = float(60.0/np.median(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2))*1000) if len(rr)>=8 else np.nan
    return hr, rmssd


def eda_window(ts, sig, w0, w1, sf=EDA_SF):
    mask = (ts>=w0)&(ts<w1)
    if mask.sum() < sf*5: return np.nan, np.nan
    seg = sig[mask]
    if not np.all(np.isfinite(seg)) or len(seg)<10: return np.nan, np.nan
    return float(np.mean(seg)), float(np.polyfit(np.arange(len(seg)), seg, 1)[0])


def eeg_window(ts, data, chans, w0, w1, sf=EEG_SF):
    mask = (ts>=w0)&(ts<w1)
    if mask.sum() < sf*5: return None
    seg = data[:, mask]
    b,a = butter(3,[1/(sf/2), min(40,sf/2-1)/(sf/2)], btype="band")
    seg_f = filtfilt(b,a,seg,axis=1)
    if sf/2>60:
        bn,an = iirnotch(60.0,30.0,sf); seg_f = filtfilt(bn,an,seg_f,axis=1)
    nper = min(seg_f.shape[1], int(sf*4))
    f, psd = welch(seg_f, fs=sf, nperseg=nper, axis=1)
    total = psd[:,(f>=4)&(f<40)].mean(axis=1)+1e-15
    theta = psd[:,(f>=4)&(f<8)].mean(axis=1)/total
    alpha = psd[:,(f>=8)&(f<12)].mean(axis=1)/total
    beta  = psd[:,(f>=12)&(f<31)].mean(axis=1)/total
    def slog(x): return float(np.log10(x)) if x>0 else np.nan
    out = {}
    for site, names in [("frontal",FRONTAL), ("temporal",TEMPORAL)]:
        idx = [chans.index(c) for c in names if c in chans]
        if not idx: continue
        t_,a_,b_ = theta[idx].mean(), alpha[idx].mean(), beta[idx].mean()
        out[f"log_{site}_theta"] = slog(t_)
        out[f"log_{site}_alpha"] = slog(a_)
        out[f"{site}_theta_alpha_ratio"] = float(t_/(a_+1e-15))
        out[f"{site}_engagement_index"]  = float(b_/(a_+t_+1e-15))
    return out


def process_participant(pid):
    rows = []
    for level in range(4):
        ecg_path = os.path.join(ECG_DIR, pid, f"ecg_data_experiment_{level}.csv")
        eda_path = os.path.join(EDA_DIR, pid, f"eda_data_experiment_{level}.csv")
        eeg_path = os.path.join(EEG_DIR, pid, f"eeg_data_exp_{level}.csv")
        if not all(os.path.isfile(p) for p in (ecg_path,eda_path,eeg_path)): continue
        try:
            ecg_df = clean_stream(ecg_path, ECG_LEADS, PRIMARY_LEAD)
            ecg_ts = ecg_df["Timestamp"].to_numpy(); ecg_sig = ecg_df[PRIMARY_LEAD].to_numpy(float)
            eda_df = clean_stream(eda_path, [EDA_COL], EDA_COL)
            eda_ts = eda_df["Timestamp"].to_numpy(); eda_sig = eda_df[EDA_COL].to_numpy(float)
            eeg_df = pd.read_csv(eeg_path).dropna()
            chans = [c for c in FRONTAL+TEMPORAL if c in eeg_df.columns]
            if not all(c in chans for c in FRONTAL): continue
            eeg_ts = eeg_df["Timestamp"].to_numpy(); eeg_data = eeg_df[chans].to_numpy(float).T
        except Exception:
            continue
        t_end = min(ecg_ts[-1], eda_ts[-1], eeg_ts[-1])
        step = WINDOW_S*(1-OVERLAP); w0 = 0.0
        while w0+WINDOW_S <= t_end:
            w1 = w0+WINDOW_S
            hr, rmssd = ecg_window(ecg_ts, ecg_sig, w0, w1)
            tonic_mean, tonic_slope = eda_window(eda_ts, eda_sig, w0, w1)
            eeg_f = eeg_window(eeg_ts, eeg_data, chans, w0, w1)
            if eeg_f is not None and hr is not None and MIN_HR<=hr<=MAX_HR:
                row = {"participant":pid,"level":level,"window_start":w0,
                      "hr_mean":hr,"hrv_rmssd":rmssd,
                      "eda_tonic_mean":tonic_mean,"eda_tonic_slope":tonic_slope}
                row.update(eeg_f); rows.append(row)
            w0 += step
    return rows

# ── end verbatim block ──────────────────────────────────────────────────


def zscore_within(df, cols, group="participant"):
    out = df.copy()
    for c in cols:
        out[c] = df.groupby(group)[c].transform(
            lambda x: (x-x.mean())/(x.std()+1e-9) if x.std()>1e-9 else 0.0)
    return out


def loso_r2(X, y, groups, seed=0):
    logo = LeaveOneGroupOut()
    scores = []
    for tr, te in logo.split(X, y, groups):
        ok_tr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ok_te = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if ok_tr.sum()<15 or ok_te.sum()<3: continue
        m = RandomForestRegressor(200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
        m.fit(X[tr][ok_tr], y[tr][ok_tr])
        scores.append(r2_score(y[te][ok_te], m.predict(X[te][ok_te])))
    return float(np.mean(scores)) if scores else np.nan, len(scores)


def main():
    print("=" * 78)
    print("TRAINING FROZEN CLARE EXPERT")
    print("Requires REAL EEG calibration input (engagement_index)")
    print(f"Targets: {TARGETS}  (frontal_theta excluded -- circular, see docstring)")
    print("=" * 78)

    if not os.path.isdir(ECG_DIR):
        print(f"\nCLARE data not found at {ECG_DIR}")
        return

    pids = sorted(os.listdir(ECG_DIR))
    all_rows = []
    for pid in pids:
        rows = process_participant(pid)
        print(f"  P{pid}: {len(rows)} windows")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} windows, {df.participant.nunique()} participants\n")

    dfz = zscore_within(df, PREDICTORS + TARGETS)
    X = np.nan_to_num(dfz[PREDICTORS].to_numpy(float))
    groups = df["participant"].to_numpy()

    train_mu = df[PREDICTORS].mean().to_numpy()
    train_sd = df[PREDICTORS].std().to_numpy() + 1e-9

    metadata = {"predictors": PREDICTORS, "targets": {},
               "train_mu": train_mu.tolist(), "train_sd": train_sd.tolist(),
               "requires_real_eeg_calibration": True,
               "excluded_target": "log_frontal_theta -- R2=0.887-0.890 in "
                                  "testing, mostly circular (engagement_index "
                                  "shares theta in its own denominator)"}

    for tgt in TARGETS:
        y = dfz[tgt].to_numpy(float)
        r2, nf = loso_r2(X, y, groups)
        print(f"  {tgt:28s} true-LOSO R2 = {r2:.3f}  ({nf} folds)")
        metadata["targets"][tgt] = {"loso_r2": r2, "n_folds": nf}

        m_final = RandomForestRegressor(200, min_samples_leaf=5,
                                        random_state=0, n_jobs=-1)
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        m_final.fit(X[ok], y[ok])
        joblib.dump(m_final, os.path.join(OUT_DIR, f"{tgt}_model.pkl"))

    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/")
    print("This expert only fires when session_router detects a real EEG")
    print("calibration reading -- never from HCI alone.")


if __name__ == "__main__":
    main()
