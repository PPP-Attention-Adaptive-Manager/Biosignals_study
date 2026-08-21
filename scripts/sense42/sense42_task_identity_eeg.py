"""
sense42_task_identity_eeg.py
==============================
Tests whether task IDENTITY produces EEG differences in SENSE-42 across
ALL pairwise combinations of task categories using itertools.combinations.
"""
from __future__ import annotations
import os, sys, glob, gc, json, warnings
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
from scipy.stats import ttest_rel
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

BASE     = os.path.expanduser("~/biosignals_data")
SENSE    = os.path.join(BASE, "data", "sense_42")
EEG_DIR  = os.path.join(SENSE, "EEG_cleaned")
EEG_ZIP  = os.path.join(SENSE, "EEG_cleaned.zip")
BDF_DIR  = os.path.join(SENSE, "EEG_raw")
BDF_ZIP  = os.path.join(SENSE, "EEG_raw.zip")
CSV_DIR  = os.path.join(SENSE, "Behavioural", "CSV")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FEAT = os.path.join(OUT_DIR, "sense42_task_eeg_features.csv")
OUT_JSON = os.path.join(OUT_DIR, "sense42_task_eeg_results.json")

KEEP_FILES = False
EPOCH_S    = 5.0
OVERLAP    = 0.5
MIN_FOLDS  = 8

# STEW channel roles, mapped onto SENSE-42's 32-channel BioSemi montage
FRONTAL   = ['AF3', 'F3', 'Fz', 'F4', 'AF4']
POSTERIOR = ['P3', 'Pz', 'P4', 'O1', 'O2']
OCCIPITAL = ['O1', 'Oz', 'O2']

THETA = (4, 8); ALPHA_FIXED = (8, 13); BETA = (13, 30); DELTA = (1, 4)

TASK_CODES = {
     7: "mail_homescreen",    9: "mail_notification",   11: "mail_content",
    13: "file_manager_homescreen", 15: "file_manager_dragging",
    17: "file_manager_opening", 19: "trash_bin_homescreen",
    21: "trash_bin_select",  23: "trash_bin_confirm",
    25: "notes_homescreen",  27: "notes_repeat",
    29: "browser_homescreen",31: "browser_navigation", 33: "browser_content",
}
TASK_APP = {
    "mail_homescreen":"mail","mail_notification":"mail","mail_content":"mail",
    "file_manager_homescreen":"file_mgr","file_manager_dragging":"file_mgr",
    "file_manager_opening":"file_mgr",
    "trash_bin_homescreen":"trash","trash_bin_select":"trash","trash_bin_confirm":"trash",
    "notes_homescreen":"notes","notes_repeat":"notes",
    "browser_homescreen":"browser","browser_navigation":"browser","browser_content":"browser",
}

FEAT_COLS = [
    "log_frontal_theta", "log_posterior_alpha_fixed", "log_posterior_alpha_iaf",
    "log_frontal_alpha_fixed", "log_frontal_alpha_iaf",
    "theta_alpha_ratio_fixed", "theta_alpha_ratio_iaf",
    "log_occipital_delta", "fp_ratio",
]


# ══════════════════════════════════════════════════════════════════════
# Triggers + calibration window from raw BDF
# ══════════════════════════════════════════════════════════════════════

def get_bdf_triggers(pid):
    bdf = os.path.join(BDF_DIR, f"P{pid}.bdf")
    tmp = False
    if not os.path.isfile(bdf):
        rc = os.system(f'unzip -q -o "{BDF_ZIP}" "EEG_raw/P{pid}.bdf" '
                       f'-d "{SENSE}" 2>/dev/null')
        tmp = rc == 0 and os.path.isfile(bdf)
    if not os.path.isfile(bdf):
        return None, None, None
    try:
        raw = mne.io.read_raw_bdf(bdf, preload=False, verbose=False)
        sf  = raw.info["sfreq"]
        ev  = mne.find_events(raw, stim_channel="Status",
                              min_duration=1/sf, verbose=False)
        ev[:, 2] = ev[:, 2] & 0xFF

        task_events = [(int(s), int(c), float(s/sf))
                       for s, _, c in ev if c in TASK_CODES]

        calib_start = next((int(s) for s, _, c in ev if c == 1), None)
        exp_begin   = next((int(s) for s, _, c in ev if c == 3), None)
        calib_bounds = (calib_start, exp_begin) if (calib_start and exp_begin) else None

        return task_events, calib_bounds, sf
    except Exception as e:
        print(f"    trigger error: {e}")
        return None, None, None
    finally:
        if tmp and not KEEP_FILES and os.path.isfile(bdf):
            os.remove(bdf)


# ══════════════════════════════════════════════════════════════════════
# Cleaned EEG loading
# ══════════════════════════════════════════════════════════════════════

def load_cleaned_eeg(pid):
    path = os.path.join(EEG_DIR, f"P{pid}_100Hz_downsampled.set")
    tmp = False
    if not os.path.isfile(path):
        rc = os.system(f'unzip -q -o "{EEG_ZIP}" '
                       f'"EEG_cleaned/P{pid}_100Hz_downsampled.set" '
                       f'-d "{SENSE}" 2>/dev/null')
        tmp = rc == 0 and os.path.isfile(path)
    if not os.path.isfile(path):
        return None, None, False
    try:
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
        return raw, raw.info["sfreq"], tmp
    except Exception as e:
        print(f"    .set load error: {e}")
        return None, None, tmp


# ══════════════════════════════════════════════════════════════════════
# STEW-recipe feature extraction (dual-band, IAF-corrected, log-relative)
# ══════════════════════════════════════════════════════════════════════

def bandpass(data, lo, hi, fs, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, data, axis=1)


def compute_psd(seg, fs):
    nper = min(seg.shape[1], int(fs * 2))
    f, psd = welch(seg, fs=fs, nperseg=nper, axis=1)
    return f, psd


def band_power(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return psd[:, m].mean(axis=1) if m.sum() else np.zeros(psd.shape[0])


def find_iaf(calib_seg_cog, fs, occ_idx):
    if calib_seg_cog.shape[1] < int(fs * 4):
        return 10.0
    f, psd = compute_psd(calib_seg_cog, fs)
    psd_occ = psd[occ_idx].mean(axis=0)
    m = (f >= 7) & (f <= 13)
    if m.sum() == 0 or psd_occ[m].sum() == 0:
        return 10.0
    return float(f[m][np.argmax(psd_occ[m])])


def window_features(seg_cog, seg_wide, fs, iaf, f_idx, p_idx, occ_idx):
    f, psd = compute_psd(seg_cog, fs)
    total_p = band_power(psd, f, 4, 40) + 1e-15

    theta = band_power(psd, f, *THETA) / total_p
    beta  = band_power(psd, f, *BETA)  / total_p
    alpha_fixed = band_power(psd, f, *ALPHA_FIXED) / total_p
    alpha_iaf   = band_power(psd, f, max(1.0, iaf-2), iaf+2) / total_p

    f_w, psd_w = compute_psd(seg_wide, fs)
    total_p_w  = band_power(psd_w, f_w, 1, 45) + 1e-15
    delta_rel  = band_power(psd_w, f_w, *DELTA) / total_p_w

    def safelog(x):
        return float(np.log10(x)) if x > 0 else np.nan

    ft   = theta[f_idx].mean()
    fa_f = alpha_fixed[f_idx].mean()
    fa_i = alpha_iaf[f_idx].mean()
    pa_f = alpha_fixed[p_idx].mean()
    pa_i = alpha_iaf[p_idx].mean()
    od   = delta_rel[occ_idx].mean()

    return {
        "log_frontal_theta":         safelog(ft),
        "log_frontal_alpha_fixed":   safelog(fa_f),
        "log_frontal_alpha_iaf":     safelog(fa_i),
        "log_posterior_alpha_fixed": safelog(pa_f),
        "log_posterior_alpha_iaf":   safelog(pa_i),
        "theta_alpha_ratio_fixed":   float(ft / (fa_f + 1e-15)),
        "theta_alpha_ratio_iaf":     float(ft / (fa_i + 1e-15)),
        "log_occipital_delta":       safelog(od),
        "fp_ratio":                  float(ft / (pa_f + 1e-15)),
    }


# ══════════════════════════════════════════════════════════════════════
# Per-participant processing
# ══════════════════════════════════════════════════════════════════════

def process_participant(pid):
    print(f"  P{pid}", end="", flush=True)

    task_events, calib_bounds, bdf_sf = get_bdf_triggers(pid)
    if not task_events:
        print(" no triggers"); return None
    print(f" {len(task_events)} triggers", end="")

    eeg_raw, eeg_sf, set_tmp = load_cleaned_eeg(pid)
    if eeg_raw is None:
        print(" no cleaned EEG"); return None
    eeg_data = eeg_raw.get_data()

    def ch_idx(names):
        return [eeg_raw.ch_names.index(c) for c in names if c in eeg_raw.ch_names]
    f_idx, p_idx, occ_idx = ch_idx(FRONTAL), ch_idx(POSTERIOR), ch_idx(OCCIPITAL)
    if not f_idx or not p_idx:
        print(" missing channels")
        if set_tmp: os.remove(os.path.join(EEG_DIR, f"P{pid}_100Hz_downsampled.set"))
        return None

    n_samp = eeg_data.shape[1]

    def bdf_to_cleaned(bdf_sample):
        return int(bdf_sample / 1024 * eeg_sf)

    eeg_cog  = bandpass(eeg_data, 4, 40, eeg_sf)
    eeg_wide = bandpass(eeg_data, 1, 45, eeg_sf)

    iaf = 10.0
    if calib_bounds:
        c0 = max(0, bdf_to_cleaned(calib_bounds[0]))
        c1 = min(n_samp, bdf_to_cleaned(calib_bounds[1]))
        if c1 > c0:
            iaf = find_iaf(eeg_cog[:, c0:c1], eeg_sf, occ_idx)
    print(f" IAF={iaf:.1f}Hz", end="")

    task_events.sort(key=lambda x: x[0])
    starts = [s for s, _, _ in task_events]
    stretches = {}
    for i, (s, code, _) in enumerate(task_events):
        nxt = starts[i+1] if i+1 < len(task_events) else int(1024 * (n_samp/eeg_sf))
        s0, s1 = bdf_to_cleaned(s), bdf_to_cleaned(nxt)
        s0, s1 = max(0, s0), min(n_samp, s1)
        if s1 - s0 < int(eeg_sf * 2):
            continue
        app = TASK_APP.get(TASK_CODES[code], "unknown")
        stretches.setdefault(app, []).append((s0, s1))

    win = int(EPOCH_S * eeg_sf)
    step = int(win * (1 - OVERLAP))
    rows = []
    for app, segs in stretches.items():
        n_win = 0
        for s0, s1 in segs:
            pos = s0
            while pos + win <= s1:
                fe = window_features(
                    eeg_cog[:, pos:pos+win], eeg_wide[:, pos:pos+win],
                    eeg_sf, iaf, f_idx, p_idx, occ_idx)
                fe.update({"participant": f"P{pid}", "app": app,
                          "onset_s": float(pos / eeg_sf)})
                rows.append(fe)
                n_win += 1
                pos += step
        if n_win:
            print(f" {app}:{n_win}", end="")
    print()

    del eeg_data, eeg_cog, eeg_wide, eeg_raw
    gc.collect()
    if set_tmp and not KEEP_FILES:
        sp = os.path.join(EEG_DIR, f"P{pid}_100Hz_downsampled.set")
        if os.path.isfile(sp): os.remove(sp)

    return pd.DataFrame(rows) if rows else None


# ══════════════════════════════════════════════════════════════════════
# Dynamic All-Pairs Task Contrast Analysis
# ══════════════════════════════════════════════════════════════════════

def run_contrasts(df):
    print("\n" + "=" * 82)
    print("ALL PAIRWISE TASK-IDENTITY CONTRASTS (paired across participants)")
    print("=" * 82)

    unique_apps = sorted([a for a in df["app"].unique() if a != "unknown"])
    all_pairs = list(combinations(unique_apps, 2))
    print(f"Testing {len(all_pairs)} total pairs across categories: {unique_apps}")

    results = {}
    for app_a, app_b in all_pairs:
        sub_a = df[df.app == app_a].groupby("participant")[FEAT_COLS].mean()
        sub_b = df[df.app == app_b].groupby("participant")[FEAT_COLS].mean()
        common = sub_a.index.intersection(sub_b.index)
        
        if len(common) < MIN_FOLDS:
            print(f"\n{app_a} vs {app_b}: too few paired participants ({len(common)} < {MIN_FOLDS})")
            continue
            
        a = sub_a.loc[common]; b = sub_b.loc[common]

        print(f"\n{app_a} vs {app_b}")
        print(f"  paired participants: {len(common)}")
        print(f"  {'Feature':28s} {app_a[:6]:>9s} {app_b[:6]:>9s} "
              f"{'diff':>8s} {'t':>7s} {'p':>8s} {'d':>7s}")
        print("  " + "-" * 76)

        contrast_res = {}
        for feat in FEAT_COLS:
            va, vb = a[feat].to_numpy(), b[feat].to_numpy()
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() < MIN_FOLDS:
                continue
            va, vb = va[ok], vb[ok]
            diff = va - vb
            t, p = ttest_rel(va, vb)
            d = float(diff.mean() / (diff.std(ddof=1) + 1e-9))
            star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
            tag = "  <- CONTROL" if feat == "log_occipital_delta" else ""
            print(f"  {feat:28s} {va.mean():9.3f} {vb.mean():9.3f} "
                  f"{diff.mean():+8.3f} {t:7.2f} {p:8.4f} {d:+7.2f}{star}{tag}")
            contrast_res[feat] = {"a": float(va.mean()), "b": float(vb.mean()),
                                  "diff": float(diff.mean()), "t": float(t),
                                  "p": float(p), "cohens_d": d}

        results[f"{app_a}_vs_{app_b}"] = contrast_res

        ctrl = contrast_res.get("log_occipital_delta", {})
        cog_ds = [v["cohens_d"] for k, v in contrast_res.items()
                  if k != "log_occipital_delta"]
        if ctrl and cog_ds:
            mean_cog_d = np.mean(np.abs(cog_ds))
            ctrl_d = abs(ctrl["cohens_d"])
            print(f"\n  Control check: mean |d| cognitive={mean_cog_d:.3f}  "
                  f"control |d|={ctrl_d:.3f}")
            if ctrl["p"] < 0.05 and ctrl_d > mean_cog_d * 0.5:
                print("  WARNING: control moving nearly as much as cognitive features")
            else:
                print("  Control flat relative to cognitive effects — trustworthy")

    return results


def main():
    print("=" * 82)
    print("SENSE-42 ALL-PAIRS TASK-IDENTITY EEG CONTRAST")
    print("=" * 82)

    if os.path.isfile(OUT_FEAT):
        df = pd.read_csv(OUT_FEAT)
        done = set(df.participant.unique())
        print(f"Cached: {len(df)} rows, {len(done)} participants\n")
    else:
        df, done = pd.DataFrame(), set()

    pids = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(os.path.join(CSV_DIR, "*.csv"))})
    pids = [p for p in pids if f"P{p}" not in done and p != "005"]

    if pids:
        print(f"To process: {len(pids)} participants\n")
        for i, pid in enumerate(pids, 1):
            print(f"[{i:2d}/{len(pids)}]", end="")
            try:
                r = process_participant(pid)
            except Exception as e:
                print(f"    ERROR: {e}"); continue
            if r is not None and len(r):
                df = pd.concat([df, r], ignore_index=True) if len(df) else r
                df.to_csv(OUT_FEAT, index=False)

    if df.empty:
        print("\nNo data collected."); return

    print(f"\nTotal: {len(df)} windows, {df.participant.nunique()} participants")
    print("\nWindows per task type:")
    print(df.app.value_counts().to_string())

    results = run_contrasts(df)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved all pair results to: {OUT_JSON}")


if __name__ == "__main__":
    main()