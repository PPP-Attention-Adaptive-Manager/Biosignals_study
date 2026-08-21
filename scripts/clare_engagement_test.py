"""
clare_engagement_test.py
===========================
Same question as stew_engagement_test.py, on CLARE: does eeg_engagement
predict the rest of a person's EEG picture (frontal_theta, frontal_alpha,
theta_alpha_ratio)?

TWO PASSES
------------
A. engagement_index ALONE as predictor
B. engagement_index + ECG (hr_mean, rmssd) + EDA (tonic_mean, tonic_slope)
   as predictors -- does adding real cardiac/electrodermal signal on top
   of one real EEG measurement help reconstruct the rest of the EEG
   picture better than engagement alone?

Reuses the paper-validated CLARE pipeline (5-15Hz ECG bandpass, 8-12Hz
alpha, 60Hz notch, 30s/50%-overlap windows) already built and confirmed
against the CLARE paper's own numbers earlier in this project.

Same discipline as every check today: true per-participant LOSO,
permutation control on every target, temporal-site (TP9/TP10) control
reported alongside frontal results so any "signal" can be checked
against a site that shouldn't show a cognitive effect.

Run from: ~/biosignals_data/
Output:   outputs/clare_engagement_results.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch, iirnotch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
CLARE_ROOT = os.path.join(BASE, "data", "clare", "doi-10.5683-sp3-h0aelt")
ECG_DIR = os.path.join(CLARE_ROOT, "ECG")
EDA_DIR = os.path.join(CLARE_ROOT, "EDA")
EEG_DIR = os.path.join(CLARE_ROOT, "EEG")
OUT_JSON = os.path.join(BASE, "outputs", "clare_engagement_results.json")

ECG_LEADS = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]
PRIMARY_LEAD = "ECG LL-RA CAL"
EDA_COL = "GSR Conductance CAL"
ECG_SF, EDA_SF, EEG_SF = 512.0, 128.0, 256.0
FRONTAL  = ["AF7", "AF8"]
TEMPORAL = ["TP9", "TP10"]
WINDOW_S = 30.0; OVERLAP = 0.5
MIN_HR, MAX_HR = 40, 140
MIN_FOLDS = 8


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


TARGETS = ["log_frontal_theta", "log_frontal_alpha", "frontal_theta_alpha_ratio"]
TEMPORAL_TARGETS = ["log_temporal_theta", "log_temporal_alpha", "temporal_theta_alpha_ratio"]


def zscore_within(df, cols, group="participant"):
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
        ok_tr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ok_te = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if ok_tr.sum()<15 or ok_te.sum()<3: continue
        ytr = y[tr][ok_tr]
        if shuffle: ytr = rng.permutation(ytr)
        m = RandomForestRegressor(200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
        m.fit(X[tr][ok_tr], ytr)
        scores.append(r2_score(y[te][ok_te], m.predict(X[te][ok_te])))
    return float(np.mean(scores)) if scores else np.nan, len(scores)


def run_pass(df, pred_cols, label, all_targets):
    groups = df["participant"].to_numpy()
    dfz = zscore_within(df, pred_cols + all_targets)
    X = np.nan_to_num(dfz[pred_cols].to_numpy(float))
    print(f"\n--- {label} (predictors: {pred_cols}) ---")
    print(f"{'target':30s} {'R2 real':>9s} {'R2 perm':>9s} {'over':>8s}")
    print("-"*60)
    res = {}
    for tgt in all_targets:
        if tgt not in df.columns: continue
        y = dfz[tgt].to_numpy(float)
        r2r, nf = loso_r2(X, y, groups)
        r2p, _  = loso_r2(X, y, groups, shuffle=True)
        tag = "  <- TEMPORAL CONTROL" if "temporal" in tgt else ""
        flag = " ***" if r2r>0.03 and r2r-r2p>0.03 else ""
        print(f"  {tgt:28s} {r2r:9.3f} {r2p:9.3f} {r2r-r2p:+8.3f}{flag}{tag}")
        res[tgt] = {"r2_real": r2r, "r2_perm": r2p, "n_folds": nf}
    return res


def main():
    print("="*78); print("CLARE: does eeg_engagement predict the rest of the EEG picture?")
    print("="*78)

    pids = sorted(os.listdir(ECG_DIR))
    all_rows = []
    for pid in pids:
        rows = process_participant(pid)
        print(f"  P{pid}: {len(rows)} windows")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} windows, {df.participant.nunique()} participants")

    all_targets = TARGETS + TEMPORAL_TARGETS
    results = {}
    results["A_engagement_alone"] = run_pass(
        df, ["frontal_engagement_index"], "PASS A: engagement alone", all_targets)
    results["B_engagement_plus_cardiac_eda"] = run_pass(
        df, ["frontal_engagement_index","hr_mean","hrv_rmssd",
            "eda_tonic_mean","eda_tonic_slope"],
        "PASS B: engagement + ECG + EDA", all_targets)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
