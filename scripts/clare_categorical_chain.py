"""
clare_categorical_chain.py
=============================
NEW test: predict EEG direction/magnitude from ECG+EDA direction/
magnitude -- categorical-to-categorical, not raw-value-to-categorical.

WHY THIS IS A GENUINELY DIFFERENT HYPOTHESIS
------------------------------------------------
Stage 2b (clare_replicate_and_chain.py) used raw z-scored ECG/EDA
VALUES as continuous predictors of EEG direction. That was null.

This script converts BOTH sides to the same categorical language first:
    ecg_hr_mean        -> ecg_hr_mean_dir (rising/falling)
                        -> ecg_hr_mean_mag (falling/flat/rising, tertile)
    eda_tonic_slope     -> eda_tonic_slope_dir / _mag
    ... (same for every ECG/EDA feature)
predicting:
    log_frontal_theta   -> log_frontal_theta_dir / _mag
    ... (same for every EEG feature)

The hypothesis: raw physiological VALUES are noisy and individually
variable, but the PATTERN of change (is HR currently rising, by how
much) might carry cleaner information about arousal transitions that
maps onto EEG state changes -- a genuinely different question from
"do raw values correlate," closer in spirit to what the SWELL-KW proxy
does (direction/magnitude framing beat raw regression everywhere in
this project).

WINDOWING
-----------
Reverts to 30s sliding windows (50% overlap), matching
clare_pooled_test.py / clare_expanded_test.py -- NOT the paper's 10s
segments used in Stage 1, since this test has no dependency on the
paper's label timing. 30s gives more stable HRV estimates than 10s.

EDA processed for the first time at 30s resolution here (was only done
at 10s in clare_replicate_and_chain.py).

Run from: ~/biosignals_data/
Output:   outputs/clare_categorical_chain_results.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch, iirnotch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
CLARE_ROOT = os.path.join(BASE, "data", "clare", "doi-10.5683-sp3-h0aelt")
ECG_DIR = os.path.join(CLARE_ROOT, "ECG")
EDA_DIR = os.path.join(CLARE_ROOT, "EDA")
EEG_DIR = os.path.join(CLARE_ROOT, "EEG")
OUT_JSON = os.path.join(BASE, "outputs", "clare_categorical_chain_results.json")

ECG_LEADS = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]
PRIMARY_LEAD = "ECG LL-RA CAL"
EDA_COL = "GSR Conductance CAL"
ECG_SF, EDA_SF, EEG_SF = 512.0, 128.0, 256.0

FRONTAL  = ["AF7", "AF8"]
TEMPORAL = ["TP9", "TP10"]

WINDOW_S = 30.0
OVERLAP  = 0.5
MIN_HR, MAX_HR = 40, 140
MIN_FOLDS = 8
STAR_BAR = 0.03


def clean_stream(path, value_cols, primary_col):
    raw = pd.read_csv(path)
    df = raw.dropna(subset=value_cols, how="all").copy()
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp", keep="first")
    df = df.dropna(subset=[primary_col]).reset_index(drop=True)
    return df


def ecg_window(ts, sig, w0, w1, sf=ECG_SF):
    mask = (ts >= w0) & (ts < w1)
    if mask.sum() < sf * 5:
        return np.nan, np.nan
    seg = sig[mask]
    b, a = butter(3, [5/(sf/2), 15/(sf/2)], btype="band")
    z = filtfilt(b, a, seg)
    z = (z - z.mean()) / (z.std() + 1e-9)
    peaks = np.array([])
    for h in [2.5, 2.0, 1.5, 1.0]:
        peaks, _ = find_peaks(z, distance=int(0.35*sf), height=h)
        if len(peaks) > 8:
            break
    if len(peaks) < 6:
        return np.nan, np.nan
    seg_ts = ts[mask]
    rr = np.diff(seg_ts[peaks])
    rr = rr[(rr > 0.33) & (rr < 1.5)]
    if len(rr) < 4:
        return np.nan, np.nan
    hr = float(60.0/np.median(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2))*1000) if len(rr)>=8 else np.nan
    return hr, rmssd


def eda_window(ts, sig, w0, w1, sf=EDA_SF):
    mask = (ts >= w0) & (ts < w1)
    if mask.sum() < sf * 5:
        return np.nan, np.nan
    seg = sig[mask]
    if not np.all(np.isfinite(seg)) or len(seg) < 10:
        return np.nan, np.nan
    tonic_mean = float(np.mean(seg))
    tonic_slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])
    return tonic_mean, tonic_slope


def eeg_window(ts, data, chans, w0, w1, sf=EEG_SF):
    mask = (ts >= w0) & (ts < w1)
    if mask.sum() < sf * 5:
        return None
    seg = data[:, mask]
    b, a = butter(3, [1/(sf/2), min(40,sf/2-1)/(sf/2)], btype="band")
    seg_f = filtfilt(b, a, seg, axis=1)
    if sf/2 > 60:
        b_n, a_n = iirnotch(60.0, 30.0, sf)
        seg_f = filtfilt(b_n, a_n, seg_f, axis=1)
    nper = min(seg_f.shape[1], int(sf * 4))
    f, psd = welch(seg_f, fs=sf, nperseg=nper, axis=1)
    total = psd[:, (f>=4)&(f<40)].mean(axis=1) + 1e-15
    theta = psd[:, (f>=4)&(f<8)].mean(axis=1)  / total
    alpha = psd[:, (f>=8)&(f<12)].mean(axis=1) / total
    beta  = psd[:, (f>=12)&(f<31)].mean(axis=1)/ total

    def safelog(x):
        return float(np.log10(x)) if x > 0 else np.nan

    f_idx = [chans.index(c) for c in FRONTAL if c in chans]
    t_, a_, b_ = theta[f_idx].mean(), alpha[f_idx].mean(), beta[f_idx].mean()
    return {
        "log_frontal_theta": safelog(t_),
        "log_frontal_alpha": safelog(a_),
        "log_frontal_beta":  safelog(b_),
        "frontal_theta_alpha_ratio": float(t_/(a_+1e-15)),
        "frontal_engagement_index":  float(b_/(a_+t_+1e-15)),
    }


def process_participant(pid):
    rows = []
    for level in range(4):
        ecg_path = os.path.join(ECG_DIR, pid, f"ecg_data_experiment_{level}.csv")
        eda_path = os.path.join(EDA_DIR, pid, f"eda_data_experiment_{level}.csv")
        eeg_path = os.path.join(EEG_DIR, pid, f"eeg_data_exp_{level}.csv")
        if not all(os.path.isfile(p) for p in (ecg_path, eda_path, eeg_path)):
            continue
        try:
            ecg_df = clean_stream(ecg_path, ECG_LEADS, PRIMARY_LEAD)
            ecg_ts = ecg_df["Timestamp"].to_numpy()
            ecg_sig = ecg_df[PRIMARY_LEAD].to_numpy(float)

            eda_df = clean_stream(eda_path, [EDA_COL], EDA_COL)
            eda_ts = eda_df["Timestamp"].to_numpy()
            eda_sig = eda_df[EDA_COL].to_numpy(float)

            eeg_df = pd.read_csv(eeg_path).dropna()
            chans = [c for c in FRONTAL + TEMPORAL if c in eeg_df.columns]
            if not all(c in chans for c in FRONTAL):
                continue
            eeg_ts = eeg_df["Timestamp"].to_numpy()
            eeg_data = eeg_df[chans].to_numpy(float).T
        except Exception:
            continue

        t_end = min(ecg_ts[-1], eda_ts[-1], eeg_ts[-1])
        step = WINDOW_S * (1 - OVERLAP)
        w0 = 0.0
        while w0 + WINDOW_S <= t_end:
            w1 = w0 + WINDOW_S
            hr, rmssd = ecg_window(ecg_ts, ecg_sig, w0, w1)
            tonic_mean, tonic_slope = eda_window(eda_ts, eda_sig, w0, w1)
            eeg_f = eeg_window(eeg_ts, eeg_data, chans, w0, w1)
            if eeg_f is not None and hr is not None and MIN_HR <= hr <= MAX_HR:
                row = {"participant": pid, "level": level, "window_start": w0,
                      "ecg_hr_mean": hr, "ecg_rmssd": rmssd,
                      "eda_tonic_mean": tonic_mean, "eda_tonic_slope": tonic_slope}
                row.update(eeg_f)
                rows.append(row)
            w0 += step
    return rows


def build_models():
    m = {}
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = lambda seed: CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=seed,
            verbose=0, allow_writing_files=False)
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = lambda seed: XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0)
    except ImportError:
        pass
    m["RF"] = lambda seed: RandomForestClassifier(
        150, min_samples_leaf=3, class_weight="balanced",
        random_state=seed, n_jobs=-1)
    return m


def direction_labels(series, groups):
    out = np.full(len(series), np.nan)
    s, g = series.to_numpy(float), groups
    for u in np.unique(g):
        idx = np.where(g==u)[0]
        for k in range(1, len(idx)):
            i, j = idx[k-1], idx[k]
            if np.isfinite(s[i]) and np.isfinite(s[j]):
                out[j] = float(s[j] > s[i])
    return out


def magnitude_labels(series, groups):
    """3-class: falling/flat/rising, tertile-split on the per-participant
    window-to-window delta. Same design as clare_expanded_test.py."""
    out = np.full(len(series), np.nan)
    s, g = series.to_numpy(float), groups
    for u in np.unique(g):
        idx = np.where(g==u)[0]
        if len(idx) < 4: continue
        deltas = np.diff(s[idx])
        valid = np.isfinite(deltas)
        if valid.sum() < 4: continue
        lo, hi = np.nanpercentile(deltas[valid], [33.3, 66.7])
        if hi <= lo: continue
        for k in range(1, len(idx)):
            d = s[idx[k]] - s[idx[k-1]]
            if not np.isfinite(d): continue
            cls = 0 if d <= lo else (2 if d >= hi else 1)
            out[idx[k]] = float(cls)
    return out


ECG_EDA_FEATS = ["ecg_hr_mean", "ecg_rmssd", "eda_tonic_mean", "eda_tonic_slope"]
EEG_FEATS = ["log_frontal_theta", "log_frontal_alpha", "log_frontal_beta",
            "frontal_theta_alpha_ratio", "frontal_engagement_index"]


def build_categorical_predictors(df, groups):
    """For each ECG/EDA feature, compute its OWN direction (2-class) and
    magnitude (3-class) labels -- these become the categorical predictor
    columns, not the raw z-scored values."""
    cols = {}
    for feat in ECG_EDA_FEATS:
        if feat not in df.columns: continue
        cols[f"{feat}_dir"] = direction_labels(df[feat], groups)
        cols[f"{feat}_mag"] = magnitude_labels(df[feat], groups)
    return pd.DataFrame(cols)


def loso_predict(X, y, groups, model_fn, n_classes, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    accs, bases = [], []
    for held in np.unique(groups):
        tr, te = groups!=held, groups==held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum()<15 or ote.sum()<3: continue
        ytr, yte = y[tr][otr].astype(int), y[te][ote].astype(int)
        if len(np.unique(ytr))<2: continue
        if shuffle: ytr = rng.permutation(ytr)
        try:
            m = model_fn(seed)
            m.fit(X[tr][otr], ytr)
            accs.append(accuracy_score(yte, m.predict(X[te][ote])))
            counts = np.bincount(yte, minlength=n_classes)
            bases.append(counts.max()/counts.sum())
        except Exception:
            continue
    if len(accs) < MIN_FOLDS:
        return np.nan, np.nan, len(accs)
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


def main():
    print("=" * 88)
    print("CLARE CATEGORICAL CHAIN — (ECG+EDA direction/magnitude) -> (EEG direction/magnitude)")
    print("=" * 88)
    print("""
Predictors are CATEGORICAL (direction: rising/falling, magnitude:
falling/flat/rising of ECG/EDA features), not raw z-scored values --
tests whether the PATTERN of physiological change carries information
raw-value regression missed.
""")

    if not os.path.isdir(ECG_DIR):
        print(f"ECG dir not found: {ECG_DIR}"); return

    pids = sorted(os.listdir(ECG_DIR))
    all_rows = []
    for pid in pids:
        rows = process_participant(pid)
        print(f"  P{pid}: {len(rows)} windows")
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo usable windows."); return
    df = pd.DataFrame(all_rows)
    groups = df["participant"].to_numpy()
    print(f"\nTotal: {len(df)} windows, {df.participant.nunique()} participants")

    pred_df = build_categorical_predictors(df, groups)
    pred_cols = list(pred_df.columns)
    print(f"\nCategorical predictors ({len(pred_cols)}): {pred_cols}")
    X = pred_df.to_numpy(float)

    models = build_models()
    print(f"Models: {list(models.keys())}")

    results = {}
    test_counter = {"total": 0, "starred": 0}

    for target_kind, label_fn, n_classes in [("direction", direction_labels, 2),
                                             ("magnitude", magnitude_labels, 3)]:
        print(f"\n{'='*88}")
        print(f"TARGET TYPE: EEG {target_kind}")
        print(f"{'='*88}")
        print(f"{'EEG feature':30s} {'model':10s} {'chance':>8s} {'acc':>8s} "
              f"{'perm':>8s} {'over':>8s}")
        print("-" * 78)

        for feat in EEG_FEATS:
            if feat not in df.columns: continue
            y = label_fn(df[feat], groups)

            best = None
            for mname, mfn in models.items():
                acc, base, nf = loso_predict(X, y, groups, mfn, n_classes)
                if not np.isfinite(acc) or nf < MIN_FOLDS: continue
                over = acc - base
                test_counter["total"] += 1
                if over > STAR_BAR: test_counter["starred"] += 1
                if best is None or over > best[3]:
                    best = (mname, acc, base, over, nf)

            if best is None:
                print(f"  {feat:28s}  insufficient folds"); continue
            mname, acc, base, over, nf = best
            perm, _, _ = loso_predict(X, y, groups, models[mname], n_classes, shuffle=True)
            perm_over = perm - base if np.isfinite(perm) else np.nan
            real_vs_perm = over - perm_over if np.isfinite(perm_over) else np.nan
            flag = ""
            if over > STAR_BAR and np.isfinite(real_vs_perm) and real_vs_perm > STAR_BAR:
                flag = "  *** REAL"
            print(f"  {feat:28s} {mname:10s} {base:8.3f} {acc:8.3f} "
                  f"{perm:8.3f} {over:+8.3f}{flag}")

            results[f"{target_kind}__{feat}"] = {
                "model": mname, "acc": acc, "chance": base,
                "perm": float(perm) if np.isfinite(perm) else None,
                "over": over,
                "real_vs_perm": float(real_vs_perm) if np.isfinite(real_vs_perm) else None,
                "n_folds": nf}

    print(f"\n{'='*88}")
    print("SUMMARY")
    print(f"{'='*88}")
    print(f"Tests run: {test_counter['total']}   "
          f"above +{STAR_BAR:.2f}: {test_counter['starred']}")
    real_hits = {k for k,v in results.items()
                if v.get("real_vs_perm") and v["over"]>STAR_BAR and v["real_vs_perm"]>STAR_BAR}
    print(f"Flagged REAL: {len(real_hits)}  -> {sorted(real_hits)}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
