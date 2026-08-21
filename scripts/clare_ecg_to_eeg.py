"""
clare_ecg_to_eeg.py
=====================
The experiment this whole CLARE detour exists for: does cardiac state
predict EEG under GENUINE induced difficulty contrast, with real,
separately-recorded ECG and EEG hardware (Shimmer ECG + Muse EEG --
NOT a shared amplifier, so cardiac field artifact is not a structural
risk here the way it was in SENSE-42).

WHY THIS TEST FINALLY ANSWERS THE OPEN QUESTION
----------------------------------------------------
Every prior attempt at cardiac/EEG coupling had a confound baked in:

  SENSE-42:  ECG + EEG on the SAME BioSemi amplifier -> CFA contamination
             confirmed directly (control features lit up as much or more
             than cognitive ones in every cardiac->EEG test run).
  STEW:      EEG only, no ECG at all -- couldn't test this link.

CLARE has separately-recorded ECG (Shimmer, validated clean at 512Hz,
99% of files give plausible HR 44.7-107.0 bpm) and EEG (Muse, 256Hz,
TP9/AF7/AF8/TP10), on 20 participants, across 4 GENUINELY induced
complexity levels (MATB-II task), each with its own baseline. This is
the first time cardiac->EEG can be tested without the shared-amplifier
confound AND with real difficulty contrast simultaneously.

DESIGN
--------
Two complementary tests, same structure that worked on STEW:

  TEST A -- paired contrast (STEW-style)
    baseline_N vs experiment_N, and level_0 vs level_3 (min vs max
    complexity), for HR/RMSSD (predictor) and frontal EEG theta/alpha/
    ratio (target). Paired across participants, Cohen's d + t-test.

  TEST B -- LOSO regression (does HR/RMSSD level PREDICT EEG level)
    CCA: HR+RMSSD -> frontal EEG features, LOSO across 20 participants.
    This is the actual proxy-relevant question: can cardiac state alone
    predict cortical state under real induced contrast.

CHANNEL ADAPTATION FOR MUSE (4 channels, no midline, no occipital)
-----------------------------------------------------------------------
STEW/SENSE-42 used AF3+F3+F4+AF4 (frontal) and O1+O2 (occipital control).
Muse only has TP9, AF7, AF8, TP10 -- no Fz, no true occipital site.
    FRONTAL   = AF7, AF8   (closest available to STEW's frontal cluster)
    TEMPORAL  = TP9, TP10  (used as the CONTROL here instead of
                           occipital -- temporal sites are further from
                           expected cognitive-load frontal theta/alpha
                           effects, so if TEMPORAL shows the same pattern
                           as FRONTAL, that's the artifact-warning signal,
                           same logic as occipital_delta elsewhere in
                           this project, just a different site because
                           Muse's montage doesn't include one)

FEATURE RECIPE
----------------
Same STEW-validated pipeline: log-relative band power (theta 4-8Hz,
alpha 8-13Hz normalized by 4-40Hz total), no IAF correction attempted
here (fixed bands, to keep this directly comparable to STEW numbers).

Run from: ~/biosignals_data/
Output:   outputs/clare_ecg_eeg_results.json
"""
from __future__ import annotations
import os, glob, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.stats import ttest_rel
from sklearn.cross_decomposition import CCA

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
CLARE_ROOT = os.path.join(BASE, "data", "clare", "doi-10.5683-sp3-h0aelt")
ECG_DIR = os.path.join(CLARE_ROOT, "ECG")
EEG_DIR = os.path.join(CLARE_ROOT, "EEG")
OUT_JSON = os.path.join(BASE, "outputs", "clare_ecg_eeg_results.json")

ECG_LEADS = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]
PRIMARY_LEAD = "ECG LL-RA CAL"
ECG_SF = 512.0   # confirmed via cleaning report: std=0.0 across all files
EEG_SF = 256.0

FRONTAL  = ["AF7", "AF8"]
TEMPORAL = ["TP9", "TP10"]   # control site, see docstring

MIN_HR, MAX_HR = 40, 140
MIN_FOLDS = 8


def clean_ecg(path):
    raw = pd.read_csv(path)
    df = raw.dropna(subset=ECG_LEADS, how="all").copy()
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp", keep="first")
    df = df.dropna(subset=[PRIMARY_LEAD]).reset_index(drop=True)
    return df


def ecg_hr_rmssd(df, lead=PRIMARY_LEAD, sf=ECG_SF):
    sig = df[lead].to_numpy(float)
    if len(sig) < sf * 5:
        return np.nan, np.nan
    b, a = butter(3, [0.5/(sf/2), 40/(sf/2)], btype="band")
    z = filtfilt(b, a, sig)
    z = (z - z.mean()) / (z.std() + 1e-9)
    peaks = np.array([])
    for h in [2.5, 2.0, 1.5, 1.0]:
        peaks, _ = find_peaks(z, distance=int(0.35*sf), height=h)
        if len(peaks) > 20:
            break
    if len(peaks) < 10:
        return np.nan, np.nan
    ts = df["Timestamp"].to_numpy()
    rr = np.diff(ts[peaks])
    rr = rr[(rr > 0.33) & (rr < 1.5)]
    if len(rr) < 5:
        return np.nan, np.nan
    hr = float(60.0 / np.median(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)) * 1000) if len(rr) >= 10 else np.nan
    return hr, rmssd


def eeg_features(path, sf=EEG_SF):
    df = pd.read_csv(path).dropna()
    if len(df) < sf * 5:
        return None
    chans = [c for c in FRONTAL + TEMPORAL if c in df.columns]
    if not all(c in df.columns for c in FRONTAL):
        return None

    data = df[chans].to_numpy(float).T
    b, a = butter(3, [1/(sf/2), min(40,sf/2-1)/(sf/2)], btype="band")
    data_f = filtfilt(b, a, data, axis=1)

    nper = min(data_f.shape[1], int(sf * 4))
    f, psd = welch(data_f, fs=sf, nperseg=nper, axis=1)
    total = psd[:, (f>=4)&(f<40)].mean(axis=1) + 1e-15
    theta = psd[:, (f>=4)&(f<8)].mean(axis=1) / total
    alpha = psd[:, (f>=8)&(f<13)].mean(axis=1) / total

    f_idx = [chans.index(c) for c in FRONTAL if c in chans]
    t_idx = [chans.index(c) for c in TEMPORAL if c in chans]

    def safelog(x):
        return float(np.log10(x)) if x > 0 else np.nan

    ft, fa = theta[f_idx].mean(), alpha[f_idx].mean()
    tt, ta = (theta[t_idx].mean(), alpha[t_idx].mean()) if t_idx else (np.nan, np.nan)

    return {
        "log_frontal_theta": safelog(ft),
        "log_frontal_alpha": safelog(fa),
        "frontal_theta_alpha_ratio": float(ft/(fa+1e-15)),
        "log_temporal_theta": safelog(tt) if np.isfinite(tt) else np.nan,
        "log_temporal_alpha": safelog(ta) if np.isfinite(ta) else np.nan,
        "temporal_theta_alpha_ratio": float(tt/(ta+1e-15)) if np.isfinite(tt) else np.nan,
    }


def process_participant(pid):
    rows = []
    for level in range(4):
        for phase, ecg_tag, eeg_tag in [
            ("baseline", f"ecg_data_baseline_{level}.csv", f"eeg_baseline_{level}.csv"),
            ("experiment", f"ecg_data_experiment_{level}.csv", f"eeg_data_exp_{level}.csv"),
        ]:
            ecg_path = os.path.join(ECG_DIR, pid, ecg_tag)
            eeg_path = os.path.join(EEG_DIR, pid, eeg_tag)
            if not (os.path.isfile(ecg_path) and os.path.isfile(eeg_path)):
                continue
            try:
                ecg_df = clean_ecg(ecg_path)
                hr, rmssd = ecg_hr_rmssd(ecg_df)
                eeg_f = eeg_features(eeg_path)
            except Exception:
                continue
            if eeg_f is None or hr is None or not (MIN_HR <= hr <= MAX_HR):
                continue
            row = {"participant": pid, "level": level, "phase": phase,
                  "hr_mean": hr, "hrv_rmssd": rmssd}
            row.update(eeg_f)
            rows.append(row)
    return rows


def test_a_contrasts(df):
    print("\n" + "=" * 78)
    print("TEST A -- PAIRED CONTRASTS (STEW-style)")
    print("Reference, STEW rest vs multitask: log_frontal_theta d=+0.71***")
    print("=" * 78)

    contrasts = []
    for level in range(4):
        b = df[(df.phase=="baseline") & (df.level==level)].set_index("participant")
        e = df[(df.phase=="experiment") & (df.level==level)].set_index("participant")
        contrasts.append((f"level{level}_baseline_vs_exp", b, e,
                          f"rest vs task, complexity level {level}"))
    l0 = df[(df.phase=="experiment") & (df.level==0)].set_index("participant")
    l3 = df[(df.phase=="experiment") & (df.level==3)].set_index("participant")
    contrasts.append(("exp_level0_vs_level3", l0, l3, "lowest vs highest complexity"))

    feats = ["hr_mean", "hrv_rmssd", "log_frontal_theta", "log_frontal_alpha",
             "frontal_theta_alpha_ratio", "log_temporal_theta", "log_temporal_alpha"]

    results = {}
    for name, a, b, label in contrasts:
        common = a.index.intersection(b.index)
        if len(common) < MIN_FOLDS:
            print(f"\n{name} ({label}): too few paired ({len(common)})")
            continue
        print(f"\n{name} -- {label}  (n={len(common)})")
        print(f"  {'feature':28s} {'a_mean':>9s} {'b_mean':>9s} "
              f"{'diff':>8s} {'t':>7s} {'p':>8s} {'d':>7s}")
        res = {}
        for feat in feats:
            va = a.loc[common, feat].to_numpy(float)
            vb = b.loc[common, feat].to_numpy(float)
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() < MIN_FOLDS:
                continue
            va, vb = va[ok], vb[ok]
            diff = vb - va
            t, p = ttest_rel(vb, va)
            d = float(diff.mean() / (diff.std(ddof=1)+1e-9))
            star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
            tag = "  <- TEMPORAL CONTROL" if "temporal" in feat else ""
            print(f"  {feat:28s} {va.mean():9.3f} {vb.mean():9.3f} "
                  f"{diff.mean():+8.3f} {t:7.2f} {p:8.4f}{star:<4s}{d:+7.2f}{tag}")
            res[feat] = {"a": float(va.mean()), "b": float(vb.mean()),
                        "diff": float(diff.mean()), "p": float(p), "d": d}
        results[name] = res
    return results


def test_b_cca(df):
    print("\n" + "=" * 78)
    print("TEST B -- LOSO CCA: HR + RMSSD -> frontal EEG")
    print("Does cardiac state predict cortical state under real induced contrast?")
    print("=" * 78)

    d = df.dropna(subset=["hr_mean","hrv_rmssd","log_frontal_theta",
                          "log_frontal_alpha","log_temporal_theta",
                          "log_temporal_alpha"])
    groups = d["participant"].to_numpy()

    def zscore(X, g):
        Xz = np.zeros_like(X, dtype=float)
        for u in np.unique(g):
            m = g == u
            Xz[m] = (X[m]-X[m].mean(0))/(X[m].std(0)+1e-9)
        return Xz

    X = zscore(d[["hr_mean","hrv_rmssd"]].to_numpy(float), groups)

    results = {}
    for name, cols in [("frontal (cognitive)", ["log_frontal_theta","log_frontal_alpha"]),
                       ("temporal (control)", ["log_temporal_theta","log_temporal_alpha"])]:
        Y = zscore(d[cols].to_numpy(float), groups)
        cv_rs = []
        for held in np.unique(groups):
            tr, te = groups!=held, groups==held
            if tr.sum()<20 or te.sum()<3: continue
            try:
                c = CCA(n_components=1, max_iter=2000).fit(X[tr], Y[tr])
                Xt, Yt = c.transform(X[te], Y[te])
                if np.std(Xt[:,0])>1e-9 and np.std(Yt[:,0])>1e-9:
                    cv_rs.append(np.corrcoef(Xt[:,0], Yt[:,0])[0,1])
            except Exception:
                pass
        cv_m = float(np.nanmean(cv_rs)) if cv_rs else np.nan
        cv_s = float(np.nanstd(cv_rs)) if cv_rs else np.nan
        print(f"  {name:22s} LOSO CV r = {cv_m:.3f} +/- {cv_s:.3f}  ({len(cv_rs)} folds)")
        results[name] = {"cv_r": cv_m, "cv_sd": cv_s, "n_folds": len(cv_rs)}

    fr = results.get("frontal (cognitive)",{}).get("cv_r") or 0
    tc = results.get("temporal (control)",{}).get("cv_r") or 0
    print(f"\n  gap (frontal - temporal) = {fr-tc:+.3f}")
    if fr > 0.25 and fr > tc + 0.05:
        print("  -> REAL: frontal exceeds control, above 0.25 bar")
    elif tc >= fr - 0.05:
        print("  -> ARTIFACT WARNING: control matches frontal")
    else:
        print("  -> NULL")
    return results


def main():
    print("=" * 78)
    print("CLARE: ECG -> EEG under genuine induced-difficulty contrast")
    print("=" * 78)
    print("""
Separately-recorded hardware (Shimmer ECG, Muse EEG) -- no shared-
amplifier CFA risk, unlike SENSE-42. Real 4-level complexity contrast --
unlike SENSE-42's naturalistic design.
""")

    if not os.path.isdir(ECG_DIR) or not os.path.isdir(EEG_DIR):
        print("ECG or EEG directory not found -- check CLARE_ROOT path.")
        return

    pids = sorted(os.listdir(ECG_DIR))
    print(f"Participants: {len(pids)}\n")

    all_rows = []
    for pid in pids:
        rows = process_participant(pid)
        print(f"  P{pid}: {len(rows)} usable phase-level rows")
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo usable rows -- check file matching between ECG/EEG folders.")
        return

    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} rows, {df.participant.nunique()} participants")

    results = {}
    results["test_a"] = test_a_contrasts(df)
    results["test_b"] = test_b_cca(df)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
