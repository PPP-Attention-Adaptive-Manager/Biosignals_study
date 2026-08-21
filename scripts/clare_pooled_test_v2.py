"""
clare_pooled_test_v2.py
=========================
Re-run of clare_pooled_test.py with the CLARE paper's own validated
preprocessing, adopted as a pre-emptive rigor check rather than an
expectation of a different answer.

ORIGINAL RESULT (clare_pooled_results.json, generic filters):
    cca_frontal:  CV r = -0.087 +/- 0.130
    cca_temporal: CV r = +0.078 +/- 0.186   (control beats frontal)
    All 6 model/target combos: real accuracy ~matches permuted accuracy
    (real_vs_perm in [-0.01, +0.006])
This is a clean null -- small stable SD, control beating signal, real
indistinguishable from permuted across three model families. Not
expected to change qualitatively here. This rerun exists so that if
asked "did you use this dataset's own validated preprocessing," the
answer is yes, closing that objection pre-emptively.

THREE CHANGES, PAPER-SOURCED (Bhatti et al., CLARE, Section IV-A/B):
  1. ECG bandpass 5-15 Hz (was 0.5-40 Hz) -- paper's exact filter for
     R-peak detection on this Shimmer device.
  2. EEG alpha band 8-12 Hz (was 8-13 Hz) -- paper's exact band edge.
  3. 60 Hz notch filter, Q=30, added before EEG band-power extraction
     -- paper specifies this for Muse's 256 Hz signal; previously
     unfiltered.

Everything else identical to the original: level-agnostic pooling
across all 4 experiment-phase files per participant, 30s/50%-overlap
sliding windows, CCA + RF/XGBoost/CatBoost direction classification,
permutation control, TP9/TP10 temporal-site control.

Run from: ~/biosignals_data/
Output:   outputs/clare_pooled_results_v2.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
CLARE_ROOT = os.path.join(BASE, "data", "clare", "doi-10.5683-sp3-h0aelt")
ECG_DIR = os.path.join(CLARE_ROOT, "ECG")
EEG_DIR = os.path.join(CLARE_ROOT, "EEG")
OUT_JSON = os.path.join(BASE, "outputs", "clare_pooled_results_v2.json")

ECG_LEADS = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]
PRIMARY_LEAD = "ECG LL-RA CAL"
ECG_SF = 512.0
EEG_SF = 256.0

FRONTAL  = ["AF7", "AF8"]
TEMPORAL = ["TP9", "TP10"]

WINDOW_S = 30.0
OVERLAP  = 0.5
MIN_HR, MAX_HR = 40, 140
MIN_FOLDS = 8


def clean_ecg(path):
    raw = pd.read_csv(path)
    df = raw.dropna(subset=ECG_LEADS, how="all").copy()
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp", keep="first")
    df = df.dropna(subset=[PRIMARY_LEAD]).reset_index(drop=True)
    return df


def ecg_window_hr_rmssd(ts, sig, w0, w1, sf=ECG_SF):
    """
    Bandpass 5-15 Hz, matching the CLARE paper's own preprocessing
    (Section IV-A: "Butterworth bandpass filter with a passband
    frequency of 5-15 Hz" for R-peak detection on this exact Shimmer
    device). Previously used a generic 0.5-40 Hz band -- this narrower
    filter is specifically tuned to suppress EMG, powerline noise, and
    baseline wander for this hardware.
    """
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
    hr = float(60.0 / np.median(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)) * 1000) if len(rr) >= 8 else np.nan
    return hr, rmssd


def eeg_window_features(ts, data, chans, w0, w1, sf=EEG_SF):
    """
    Alpha band 8-12 Hz (paper: "Alpha (8-12) Hz"), not the 8-13 Hz used
    elsewhere in this project for other datasets -- CLARE's own paper
    specifies this exact boundary for their Muse-derived features.
    Adds a 60 Hz notch filter (paper: "notch filter at 60 Hz with a
    quality factor of 30") before band-power extraction, since 60 Hz
    powerline noise sits well within Muse's 256 Hz Nyquist range and
    could otherwise contaminate adjacent bins.
    """
    mask = (ts >= w0) & (ts < w1)
    if mask.sum() < sf * 5:
        return None
    seg = data[:, mask]
    b, a = butter(3, [1/(sf/2), min(40,sf/2-1)/(sf/2)], btype="band")
    seg_f = filtfilt(b, a, seg, axis=1)

    # 60 Hz notch, Q=30 (matches paper spec exactly)
    from scipy.signal import iirnotch
    if sf/2 > 60:
        b_notch, a_notch = iirnotch(60.0, 30.0, sf)
        seg_f = filtfilt(b_notch, a_notch, seg_f, axis=1)

    nper = min(seg_f.shape[1], int(sf * 4))
    f, psd = welch(seg_f, fs=sf, nperseg=nper, axis=1)
    total = psd[:, (f>=4)&(f<40)].mean(axis=1) + 1e-15
    theta = psd[:, (f>=4)&(f<8)].mean(axis=1) / total
    alpha = psd[:, (f>=8)&(f<12)].mean(axis=1) / total   # 8-12Hz, paper spec

    f_idx = [chans.index(c) for c in FRONTAL if c in chans]
    t_idx = [chans.index(c) for c in TEMPORAL if c in chans]

    def safelog(x):
        return float(np.log10(x)) if x > 0 else np.nan

    out = {}
    if f_idx:
        out["log_frontal_theta"] = safelog(theta[f_idx].mean())
        out["log_frontal_alpha"] = safelog(alpha[f_idx].mean())
    if t_idx:
        out["log_temporal_theta"] = safelog(theta[t_idx].mean())
        out["log_temporal_alpha"] = safelog(alpha[t_idx].mean())
    return out


def process_participant_pooled(pid):
    rows = []
    for level in range(4):
        ecg_path = os.path.join(ECG_DIR, pid, f"ecg_data_experiment_{level}.csv")
        eeg_path = os.path.join(EEG_DIR, pid, f"eeg_data_exp_{level}.csv")
        if not (os.path.isfile(ecg_path) and os.path.isfile(eeg_path)):
            continue
        try:
            ecg_df = clean_ecg(ecg_path)
            ecg_ts = ecg_df["Timestamp"].to_numpy()
            ecg_sig = ecg_df[PRIMARY_LEAD].to_numpy(float)

            eeg_df = pd.read_csv(eeg_path).dropna()
            chans = [c for c in FRONTAL + TEMPORAL if c in eeg_df.columns]
            if not all(c in chans for c in FRONTAL):
                continue
            eeg_ts = eeg_df["Timestamp"].to_numpy()
            eeg_data = eeg_df[chans].to_numpy(float).T
        except Exception:
            continue

        t_end = min(ecg_ts[-1], eeg_ts[-1])
        step = WINDOW_S * (1 - OVERLAP)
        w0 = 0.0
        while w0 + WINDOW_S <= t_end:
            w1 = w0 + WINDOW_S
            hr, rmssd = ecg_window_hr_rmssd(ecg_ts, ecg_sig, w0, w1)
            eeg_f = eeg_window_features(eeg_ts, eeg_data, chans, w0, w1)
            if eeg_f is not None and hr is not None and MIN_HR <= hr <= MAX_HR:
                row = {"participant": pid, "level": level,
                      "window_start": w0, "hr_mean": hr, "hrv_rmssd": rmssd}
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


def zscore_within(X, g):
    Xz = np.zeros_like(X, dtype=float)
    for u in np.unique(g):
        m = g == u
        Xz[m] = (X[m]-X[m].mean(0))/(X[m].std(0)+1e-9)
    return Xz


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


def loso_cca(X, y, groups):
    y2 = y.reshape(-1,1)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y2, groups = X[ok], y2[ok], groups[ok]
    if len(X) < 30:
        return np.nan, np.nan, 0
    rs = []
    for held in np.unique(groups):
        tr, te = groups!=held, groups==held
        if tr.sum()<20 or te.sum()<3: continue
        try:
            c = CCA(n_components=1, max_iter=2000).fit(X[tr], y2[tr])
            Xt, Yt = c.transform(X[te], y2[te])
            if np.std(Xt[:,0])>1e-9 and np.std(Yt[:,0])>1e-9:
                rs.append(np.corrcoef(Xt[:,0], Yt[:,0])[0,1])
        except Exception:
            pass
    return (float(np.nanmean(rs)), float(np.nanstd(rs)), len(rs)) if rs else (np.nan,np.nan,0)


def loso_direction(X, y, groups, model_fn, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    accs, bases = [], []
    for held in np.unique(groups):
        tr, te = groups!=held, groups==held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum()<20 or ote.sum()<4: continue
        ytr, yte = y[tr][otr].astype(int), y[te][ote].astype(int)
        if len(np.unique(ytr))<2: continue
        if shuffle:
            ytr = rng.permutation(ytr)
        try:
            m = model_fn(seed)
            m.fit(X[tr][otr], ytr)
            accs.append(accuracy_score(yte, m.predict(X[te][ote])))
            bases.append(max(yte.mean(), 1-yte.mean()))
        except Exception:
            continue
    if len(accs) < MIN_FOLDS:
        return np.nan, np.nan, len(accs)
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


def main():
    print("=" * 78)
    print("CLARE POOLED TEST — level-agnostic, multi-model, permutation-checked")
    print("=" * 78)
    print("""
Level-based contrasts set aside: only 10/19 participants (53%) show
self-reported load monotonically increasing across level_0->level_3.
This pools all experiment-phase windows regardless of level number.
""")

    if not os.path.isdir(ECG_DIR):
        print(f"ECG dir not found: {ECG_DIR}"); return

    pids = sorted(os.listdir(ECG_DIR))
    all_rows = []
    for pid in pids:
        rows = process_participant_pooled(pid)
        print(f"  P{pid}: {len(rows)} windows")
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo usable windows."); return
    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} windows, {df.participant.nunique()} participants")

    models = build_models()
    print(f"Models: {list(models.keys())}\n")

    groups = df["participant"].to_numpy()
    X = zscore_within(df[["hr_mean","hrv_rmssd"]].to_numpy(float), groups)

    results = {}

    print("=" * 78)
    print("CCA — HR+RMSSD -> EEG (continuous, z-scored per participant)")
    print("=" * 78)
    for name, cols in [("frontal", ["log_frontal_theta","log_frontal_alpha"]),
                       ("temporal_control", ["log_temporal_theta","log_temporal_alpha"])]:
        avail = [c for c in cols if c in df.columns]
        if len(avail) < 2: continue
        Y = zscore_within(df[avail].to_numpy(float), groups)
        m, s, nf = loso_cca(X, Y[:,0], groups)
        print(f"  {name:20s} CV r = {m:.3f} +/- {s:.3f}  ({nf} folds)")
        results[f"cca_{name}"] = {"cv_r": m, "cv_sd": s, "n_folds": nf}

    fr = results.get("cca_frontal",{}).get("cv_r") or 0
    tc = results.get("cca_temporal_control",{}).get("cv_r") or 0
    print(f"  gap (frontal-temporal) = {fr-tc:+.3f}")

    print("\n" + "=" * 78)
    print("DIRECTION CLASSIFICATION — all models, with permutation control")
    print("=" * 78)

    for target_name, col in [("frontal_theta","log_frontal_theta"),
                             ("temporal_theta_CONTROL","log_temporal_theta")]:
        if col not in df.columns:
            continue
        y = direction_labels(df[col], groups)
        print(f"\n{target_name}:")
        print(f"  {'model':10s} {'chance':>8s} {'acc':>8s} {'perm':>8s} {'over':>8s}  verdict")
        print("  " + "-"*66)
        results.setdefault("direction", {})[target_name] = {}
        for mname, mfn in models.items():
            acc, base, nf = loso_direction(X, y, groups, mfn)
            perm, _, _ = loso_direction(X, y, groups, mfn, shuffle=True)
            if nf < MIN_FOLDS:
                print(f"  {mname:10s}  too few folds ({nf})")
                continue
            over = (acc or 0) - (base or 0)
            perm_over = (perm or 0) - (base or 0)
            real_vs_perm = over - perm_over
            verdict = ("REAL" if over > 0.03 and real_vs_perm > 0.03 else
                      "matches permutation" if abs(real_vs_perm) < 0.02 else
                      "weak/inconclusive")
            print(f"  {mname:10s} {base:8.3f} {acc:8.3f} {perm:8.3f} {over:+8.3f}  {verdict}")
            results["direction"][target_name][mname] = {
                "acc": acc, "chance": base, "perm": perm, "over": over,
                "real_vs_perm": real_vs_perm, "n_folds": nf}

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print("""
Real signal:      real 'over' clearly exceeds permuted 'over', frontal
                   exceeds temporal control by a real margin.
Heterogeneity:     large SD in CCA but direction models (which don't
                   depend on linear structure) also show a consistent
                   real-vs-perm gap -- would suggest per-participant
                   variability in a REAL effect rather than pure noise.
Noise:             real accuracy ~matches permuted accuracy. SD>mean in
                   CCA was noise, not signal, regardless of model choice.
""")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
