"""
clare_expanded_test.py
=========================
Addresses three real gaps in the previous CLARE tests:

  1. SUBGROUP RESTRICTION: runs the full battery on all 19 participants
     AND separately on just the 10 where self-reported load actually
     increased monotonically level_0->level_3 (from clare_label_check_v2.py:
     1026,1194,1337,1419,1624,1629,1688,1717,1818,1981). The full-sample
     result pools people whose complexity manipulation may not have
     landed as intended; the subgroup is a cleaner, if smaller, test.

  2. EXPANDED EEG FEATURES: previous tests used only theta and alpha.
     Added: beta (12-31Hz, paper's own band definition), theta/alpha
     ratio, engagement index (beta/(alpha+theta) -- one of STEW's
     strongest markers, never computed for CLARE), and frontal
     asymmetry (log(AF8_alpha)-log(AF7_alpha), a distinct literature
     marker from raw band power). Gamma (31-128Hz per the paper)
     deliberately excluded -- dry-electrode Muse gamma is highly
     EMG-contamination-prone and the paper's own bandpass for our
     total-power normalizer only extends to 40Hz.

  3. SEPARATE PREDICTORS: previous CCA combined HR+RMSSD into one
     linear combination before correlating with a combined EEG score --
     if theta and alpha move in OPPOSITE directions under load (theta
     up, alpha down, both physiologically expected), a single combined
     CCA component can partially cancel that structure out. This script
     tests HR-only, RMSSD-only, and HR+RMSSD-combined as three separate
     predictor sets, against each EEG feature INDIVIDUALLY, not
     pre-combined.

MULTIPLE COMPARISONS
-----------------------
3 predictor sets x ~12 EEG features x 2 samples (full + subgroup) is a
large number of tests. Total count and threshold-crossing count are
reported explicitly, same convention used throughout this project, so
an isolated positive can be weighed against how many tests were run
rather than treated as a discovery on its own. A hit is only treated as
worth trusting if it REPLICATES between the full sample and the
subgroup, not just appearing in one.

Reuses the paper-validated preprocessing from clare_pooled_test_v2.py
(ECG 5-15Hz bandpass, EEG 8-12Hz alpha, 60Hz notch, Q=30).

Run from: ~/biosignals_data/
Output:   outputs/clare_expanded_results_v2.json
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
EEG_DIR = os.path.join(CLARE_ROOT, "EEG")
OUT_JSON = os.path.join(BASE, "outputs", "clare_expanded_results_v2.json")

ECG_LEADS = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]
PRIMARY_LEAD = "ECG LL-RA CAL"
ECG_SF = 512.0
EEG_SF = 256.0

FRONTAL  = ["AF7", "AF8"]
TEMPORAL = ["TP9", "TP10"]

WINDOW_S = 30.0
OVERLAP  = 0.5
MIN_HR, MAX_HR = 40, 140
MIN_FOLDS_FULL = 8
MIN_FOLDS_SUB  = 6

STAR_BAR = 0.03

MONOTONIC_PIDS = ["1026","1194","1337","1419","1624",
                  "1629","1688","1717","1818","1981"]


def clean_ecg(path):
    raw = pd.read_csv(path)
    df = raw.dropna(subset=ECG_LEADS, how="all").copy()
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp", keep="first")
    df = df.dropna(subset=[PRIMARY_LEAD]).reset_index(drop=True)
    return df


def ecg_window_hr_rmssd(ts, sig, w0, w1, sf=ECG_SF):
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

    out = {}
    for site, names in [("frontal", FRONTAL), ("temporal", TEMPORAL)]:
        idx = [chans.index(c) for c in names if c in chans]
        if not idx:
            continue
        t_, a_, b_ = theta[idx].mean(), alpha[idx].mean(), beta[idx].mean()
        out[f"log_{site}_theta"] = safelog(t_)
        out[f"log_{site}_alpha"] = safelog(a_)
        out[f"log_{site}_beta"]  = safelog(b_)
        out[f"{site}_theta_alpha_ratio"] = float(t_/(a_+1e-15))
        out[f"{site}_engagement_index"]  = float(b_/(a_+t_+1e-15))

    if "AF7" in chans and "AF8" in chans:
        a7 = alpha[chans.index("AF7")]
        a8 = alpha[chans.index("AF8")]
        out["frontal_asymmetry"] = float(np.log10(a8+1e-15) - np.log10(a7+1e-15))
    if "TP9" in chans and "TP10" in chans:
        t9  = alpha[chans.index("TP9")]
        t10 = alpha[chans.index("TP10")]
        out["temporal_asymmetry"] = float(np.log10(t10+1e-15) - np.log10(t9+1e-15))

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
    """2-class: is this window's value higher than the PREVIOUS window
    (within participant)? Same framing used for HR_rising/RMSSD_rising
    throughout this project."""
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
    """
    3-class: falling / flat / rising, based on the WINDOW-TO-WINDOW
    DELTA, tertile-split per participant. Mirrors RMSSD_magnitude in
    train_biosignal_proxy.py -- the strongest single result in the whole
    project (84.7% on SWELL-KW), so this is the natural next thing to
    test here.

    Tertile split (33rd/67th percentile of the delta), not a median
    split -- a median split on a value with many repeated/near-zero
    deltas can produce severe class imbalance (this broke the
    questionnaire RF result earlier in the project: a 0.794 accuracy
    that was actually BELOW the 0.798 majority-class baseline). Tertile
    splitting on continuous EEG deltas is less prone to that specific
    failure, but the majority-baseline check in loso_predict() still
    catches it if it happens anyway.
    """
    out = np.full(len(series), np.nan)
    s, g = series.to_numpy(float), groups
    for u in np.unique(g):
        idx = np.where(g==u)[0]
        if len(idx) < 4:
            continue
        deltas = np.diff(s[idx])
        valid = np.isfinite(deltas)
        if valid.sum() < 4:
            continue
        lo, hi = np.nanpercentile(deltas[valid], [33.3, 66.7])
        for k in range(1, len(idx)):
            d = s[idx[k]] - s[idx[k-1]]
            if not np.isfinite(d):
                continue
            if hi <= lo:          # degenerate case: no spread
                continue
            cls = 0 if d <= lo else (2 if d >= hi else 1)
            out[idx[k]] = float(cls)
    return out


def loso_predict(X, y, groups, model_fn, min_folds, n_classes=2, seed=0, shuffle=False):
    """
    Generalized to n_classes: majority baseline computed as the largest
    class proportion in the test fold, not the binary-only
    max(mean, 1-mean). This matters for magnitude (3-class: fall/flat/
    rise) -- using the binary formula there would silently under-report
    the true chance level.
    """
    rng = np.random.default_rng(seed)
    accs, bases = [], []
    for held in np.unique(groups):
        tr, te = groups!=held, groups==held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum()<15 or ote.sum()<3: continue
        ytr, yte = y[tr][otr].astype(int), y[te][ote].astype(int)
        if len(np.unique(ytr)) < 2: continue
        if shuffle:
            ytr = rng.permutation(ytr)
        try:
            m = model_fn(seed)
            m.fit(X[tr][otr], ytr)
            accs.append(accuracy_score(yte, m.predict(X[te][ote])))
            # majority-class baseline, generalized to n classes
            counts = np.bincount(yte, minlength=n_classes)
            bases.append(counts.max() / counts.sum())
        except Exception:
            continue
    if len(accs) < min_folds:
        return np.nan, np.nan, len(accs)
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


# kept as an alias so nothing upstream breaks
def loso_direction(X, y, groups, model_fn, min_folds, seed=0, shuffle=False):
    return loso_predict(X, y, groups, model_fn, min_folds, n_classes=2,
                        seed=seed, shuffle=shuffle)


EEG_FEATS = [
    "log_frontal_theta", "log_frontal_alpha", "log_frontal_beta",
    "frontal_theta_alpha_ratio", "frontal_engagement_index", "frontal_asymmetry",
    "log_temporal_theta", "log_temporal_alpha", "log_temporal_beta",
    "temporal_theta_alpha_ratio", "temporal_engagement_index", "temporal_asymmetry",
]

PREDICTOR_SETS = {
    "HR-only":    ["hr_mean"],
    "RMSSD-only": ["hrv_rmssd"],
    "HR+RMSSD":   ["hr_mean", "hrv_rmssd"],
}


def run_battery(df, label, min_folds):
    print("\n" + "=" * 88)
    print(f"BATTERY -- {label}  (n={df.participant.nunique()} participants, "
          f"{len(df)} windows)")
    print("=" * 88)

    groups = df["participant"].to_numpy()
    models = build_models()
    test_counter = {"total": 0, "starred": 0}
    results = {}

    TARGET_TYPES = {
        "direction": (direction_labels, 2),
        "magnitude": (magnitude_labels, 3),
    }

    for target_type, (label_fn, n_classes) in TARGET_TYPES.items():
        for pred_name, pred_cols in PREDICTOR_SETS.items():
            avail_p = [c for c in pred_cols if c in df.columns]
            X = zscore_within(df[avail_p].to_numpy(float), groups)

            print(f"\n--- {target_type} | predictor: {pred_name} "
                  f"(chance = majority class, {n_classes}-class) ---")
            print(f"{'EEG feature':30s} {'best model':10s} {'chance':>8s} "
                  f"{'acc':>8s} {'perm':>8s} {'over':>8s}")
            print("-" * 84)

            for feat in EEG_FEATS:
                if feat not in df.columns:
                    continue
                y = label_fn(df[feat], groups)

                best = None
                for mname, mfn in models.items():
                    acc, base, nf = loso_predict(X, y, groups, mfn, min_folds,
                                                 n_classes=n_classes)
                    if nf < min_folds or not np.isfinite(acc):
                        continue
                    over = acc - base
                    test_counter["total"] += 1
                    if over > STAR_BAR:
                        test_counter["starred"] += 1
                    if best is None or over > best[3]:
                        best = (mname, acc, base, over, nf)

                if best is None:
                    print(f"  {feat:28s}  insufficient folds")
                    continue
                mname, acc, base, over, nf = best
                perm, _, _ = loso_predict(X, y, groups, models[mname], min_folds,
                                          n_classes=n_classes, shuffle=True)
                perm_over = perm - base if np.isfinite(perm) else np.nan
                real_vs_perm = over - perm_over if np.isfinite(perm_over) else np.nan
                flag = ""
                if over > STAR_BAR and np.isfinite(real_vs_perm) and real_vs_perm > STAR_BAR:
                    flag = "  *** REAL"
                print(f"  {feat:28s} {mname:10s} {base:8.3f} {acc:8.3f} "
                      f"{perm:8.3f} {over:+8.3f}{flag}")

                key = f"{target_type}__{pred_name}__{feat}"
                results[key] = {
                    "target_type": target_type, "predictor": pred_name,
                    "feature": feat, "model": mname, "n_classes": n_classes,
                    "acc": acc, "chance": base,
                    "perm": float(perm) if np.isfinite(perm) else None,
                    "over": over,
                    "real_vs_perm": float(real_vs_perm) if np.isfinite(real_vs_perm) else None,
                    "n_folds": nf}

    print(f"\nTests run: {test_counter['total']}   "
          f"above +{STAR_BAR:.2f}: {test_counter['starred']} "
          f"({100*test_counter['starred']/max(test_counter['total'],1):.0f}%)")

    return results, test_counter


def main():
    print("=" * 88)
    print("CLARE EXPANDED TEST -- full feature set, separate predictors, subgroup")
    print("=" * 88)

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
    df_full = pd.DataFrame(all_rows)
    print(f"\nFull sample: {len(df_full)} windows, "
          f"{df_full.participant.nunique()} participants")

    df_sub = df_full[df_full.participant.isin(MONOTONIC_PIDS)].reset_index(drop=True)
    print(f"Monotonic subgroup: {len(df_sub)} windows, "
          f"{df_sub.participant.nunique()} participants "
          f"(expected 10: {MONOTONIC_PIDS})")

    all_results = {}
    all_results["full_sample"], ctr_full = run_battery(
        df_full, "FULL SAMPLE (19 participants)", MIN_FOLDS_FULL)
    all_results["monotonic_subgroup"], ctr_sub = run_battery(
        df_sub, "MONOTONIC-LOAD SUBGROUP (10 participants)", MIN_FOLDS_SUB)

    print("\n" + "=" * 88)
    print("OVERALL SUMMARY")
    print("=" * 88)
    total_tests = ctr_full["total"] + ctr_sub["total"]
    total_star  = ctr_full["starred"] + ctr_sub["starred"]
    print(f"\nTotal tests across both samples: {total_tests}")
    print(f"Above +{STAR_BAR:.2f} threshold: {total_star} "
          f"({100*total_star/max(total_tests,1):.1f}%)")
    print("At this many tests, some threshold crossings are expected from")
    print("noise alone. A crossing only means something if real_vs_perm is")
    print("also clearly positive (printed inline above) AND it replicates")
    print("between the full sample and the subgroup, not just one or the other.")

    full_real = {k for k,v in all_results["full_sample"].items()
                if v.get("real_vs_perm") and v["over"]>STAR_BAR and v["real_vs_perm"]>STAR_BAR}
    sub_real  = {k for k,v in all_results["monotonic_subgroup"].items()
                if v.get("real_vs_perm") and v["over"]>STAR_BAR and v["real_vs_perm"]>STAR_BAR}
    replicated = full_real & sub_real
    print(f"\nFlagged as real in full sample: {len(full_real)}  -> {sorted(full_real)}")
    print(f"Flagged as real in subgroup:     {len(sub_real)}  -> {sorted(sub_real)}")
    print(f"Replicated in BOTH:              {len(replicated)}  -> {sorted(replicated)}")
    if replicated:
        print("\n*** These replicated hits are the only results worth trusting. ***")
    else:
        print("\nNo hit replicated across both samples -- treat any single-sample")
        print("flag as a multiple-comparisons false positive, not a finding.")

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
