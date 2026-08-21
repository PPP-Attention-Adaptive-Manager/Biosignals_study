"""
sense42_cleaned_eeg_experiments.py
====================================
Re-runs EXP2, EXP3, EXP4 using ICA-cleaned EEG instead of raw BDF.

THE PROBLEM DIAGNOSED
----------------------
32 channels → 28 ICA components in the cleaned .set files.
Four components were removed by the dataset authors before publication.
These likely include cardiac field artifact (CFA) components — heartbeats
leave a broadband trace on scalp EEG when ECG and EEG share one amplifier.

Our previous EXP2/3/4/5 results used EEG from the RAW BDF:
    sense42_trigger_extract_v2.py -> raw = mne.io.read_raw_bdf(bdf)

So every EEG feature in sense42_v2_events.csv is contaminated.
The CFA warning (controls CV r=0.307 ≈ cognitive CV r=0.342) was valid.

THE FIX
--------
Use EEG from the cleaned .set files (ICA-corrected, 100 Hz).
Use trigger TIMESTAMPS from the raw BDF (same recording, correct clock).
The two share the same time axis (both start at BioSemi recording start).

This script:
  1. Loads trigger events from the raw BDF (Status channel, same as before)
  2. Loads ICA-cleaned EEG from the .set at 100 Hz
  3. Extracts band powers from the CLEANED signal per task epoch
  4. Joins to HCI (from sense42_v2_events.csv) and cardiac (from .fif/.wav)
  5. Re-runs EXP2 (HCI→EEG), EXP3 (cardiac→EEG), EXP4 (HCI+cardiac→EEG)
  6. Reports results BEFORE and AFTER ICA cleaning side by side

THE DEFINITIVE TEST
--------------------
If cleaned EXP2 CV r stays near 0.342:
    -> Genuine HCI-cortical coupling. CFA was not the explanation.
    -> EXP2 result stands.

If cleaned EXP2 CV r drops toward zero:
    -> CFA was inflating the result. Raw BDF EEG was contaminated.
    -> Only the cleaned results are trustworthy.

The comparison between raw and cleaned is the paper's methodological
contribution: showing ICA artifact removal changes/doesn't change the
HCI-EEG coupling estimate.

Run from: ~/biosignals_data/
Input:
    data/sense_42/EEG_cleaned/P00N_100Hz_downsampled.set  (cleaned EEG)
    data/sense_42/EEG_raw/P00N.bdf OR EEG_raw.zip         (triggers only)
    outputs/sense42_v2_events.csv                           (HCI + old EEG)
    data/sense_42/ECG/P00N.fif                             (cardiac)
    data/sense_42/Respiration/P00N.wav                     (respiration)
Output:
    outputs/sense42_cleaned_features.csv
    outputs/sense42_cleaned_results.json
"""
from __future__ import annotations
import os, sys, json, glob, gc, warnings
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, butter, filtfilt, decimate
from scipy.io import wavfile
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

BASE    = os.path.expanduser("~/biosignals_data")
SENSE   = os.path.join(BASE, "data", "sense_42")
EEG_DIR = os.path.join(SENSE, "EEG_cleaned")
EEG_ZIP = os.path.join(SENSE, "EEG_cleaned.zip")
BDF_DIR = os.path.join(SENSE, "EEG_raw")
BDF_ZIP = os.path.join(SENSE, "EEG_raw.zip")
ECG_DIR = os.path.join(SENSE, "ECG")
RSP_DIR = os.path.join(SENSE, "Respiration")
EV_CSV  = os.path.join(BASE, "outputs", "sense42_v2_events.csv")
OUT_CSV = os.path.join(BASE, "outputs", "sense42_cleaned_features.csv")
OUT_J   = os.path.join(BASE, "outputs", "sense42_cleaned_results.json")

KEEP_FILES = False
RESP_FS    = 32.0
MIN_EPOCH  = 2.0
MAX_EPOCH  = 120.0
MIN_FOLDS  = 8

FRONTAL   = ['AF3','F3','Fz','F4','AF4']
POSTERIOR = ['P3','Pz','P4']
OCCIPITAL = ['O1','Oz','O2']
BANDS     = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30)}

TASK_CODES = {
    7:"mail_homescreen",9:"mail_notification",11:"mail_content",
    13:"file_manager_homescreen",15:"file_manager_dragging",
    17:"file_manager_opening",19:"trash_bin_homescreen",
    21:"trash_bin_select",23:"trash_bin_confirm",
    25:"notes_homescreen",27:"notes_repeat",29:"browser_homescreen",
    31:"browser_navigation",33:"browser_content",
}
TASK_APP = {
    "mail_homescreen":"mail","mail_notification":"mail","mail_content":"mail",
    "file_manager_homescreen":"file_mgr","file_manager_dragging":"file_mgr",
    "file_manager_opening":"file_mgr",
    "trash_bin_homescreen":"trash","trash_bin_select":"trash","trash_bin_confirm":"trash",
    "notes_homescreen":"notes","notes_repeat":"notes",
    "browser_homescreen":"browser","browser_navigation":"browser","browser_content":"browser",
}

HCI_COLS  = ["SnKeyStrokes","SnChars","SnSpecialKeys","SnDirectionKeys",
             "SnErrorKeys","SnSpaces","CharactersRatio","ErrorKeyRatio",
             "SnLeftClicked","SnMouseDistance","SnMouseAct"]
CARDIAC   = ["hr_mean","hrv_rmssd"]
RESP_COLS = ["resp_bpm","resp_amp"]
EEG_COG   = ["frontal_theta","frontal_alpha","theta_alpha_ratio",
             "engagement_index","posterior_alpha"]
EEG_CTRL  = ["occipital_delta","broadband_amplitude"]
EEG_ALL   = EEG_COG + EEG_CTRL


# ══════════════════════════════════════════════════════════════════════
# EEG band powers (log10, on ICA-cleaned signal)
# ══════════════════════════════════════════════════════════════════════

def band_powers(seg, sfreq, idx):
    """seg: (n_channels, n_samples). Returns dict of log10 band powers."""
    nper = min(seg.shape[1], int(sfreq * 2))
    if nper < int(sfreq * 0.5):
        return None
    f, psd = welch(seg, fs=sfreq, nperseg=nper, axis=1)
    bp = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp[name] = psd[:, m].mean(axis=1) if m.sum() else np.zeros(seg.shape[0])
    m_bb = (f >= 1) & (f < 40)
    bp["broadband"] = psd[:, m_bb].mean(axis=1)

    def safe_log(x):
        return float(np.log10(x)) if x > 0 else np.nan

    ft  = bp["theta"][idx["f"]].mean()
    fa  = bp["alpha"][idx["f"]].mean()
    fb  = bp["beta"][idx["f"]].mean()
    pa  = bp["alpha"][idx["p"]].mean() if idx["p"] else fa
    od  = bp["delta"][idx["o"]].mean() if idx["o"] else bp["delta"].mean()
    bb  = bp["broadband"].mean()
    return {
        "frontal_theta":       safe_log(ft),
        "frontal_alpha":       safe_log(fa),
        "theta_alpha_ratio":   float(ft / (fa + 1e-15)),   # ratio, not log
        "engagement_index":    float(fb / (fa + ft + 1e-15)),
        "posterior_alpha":     safe_log(pa),
        "occipital_delta":     safe_log(od),
        "broadband_amplitude": safe_log(bb),
    }


# ══════════════════════════════════════════════════════════════════════
# Cardiac helpers
# ══════════════════════════════════════════════════════════════════════

def ecg_features(seg, sf):
    if seg is None or len(seg) < int(sf * 5):
        return np.nan, np.nan, 0
    b, a = butter(3, [0.5/(sf/2), 40/(sf/2)], btype="band")
    z = filtfilt(b, a, seg); z = (z - z.mean()) / (z.std() + 1e-9)
    pk, _ = find_peaks(z, distance=int(0.35*sf), height=2.0)
    if len(pk) < 4:
        return np.nan, np.nan, len(pk)
    rr = np.diff(pk)/sf; rr = rr[(rr > 0.33) & (rr < 1.5)]
    if len(rr) < 3:
        return np.nan, np.nan, len(pk)
    return (float(60/np.median(rr)),
            float(np.sqrt(np.mean(np.diff(rr)**2))*1000) if len(rr)>=10 else np.nan,
            int(len(rr)))


def resp_features(seg32):
    if seg32 is None or len(seg32) < int(RESP_FS * 8):
        return np.nan, np.nan
    b, a = butter(3, [0.1/(RESP_FS/2), 0.5/(RESP_FS/2)], btype="band")
    x = filtfilt(b, a, seg32.astype(float))
    if not np.all(np.isfinite(x)) or x.std() < 1e-12:
        return np.nan, np.nan
    z = (x - x.mean()) / (x.std() + 1e-9)
    pk, _ = find_peaks(z, distance=int(1.5*RESP_FS), prominence=0.3)
    if len(pk) < 3:
        return np.nan, np.nan
    bpm = 60 / np.mean(np.diff(pk)/RESP_FS)
    return (float(bpm) if 5 < bpm < 60 else np.nan), float(x.std())


# ══════════════════════════════════════════════════════════════════════
# Per-participant extraction
# ══════════════════════════════════════════════════════════════════════

def extract_triggers_from_bdf(pid):
    """Get task-event triggers from raw BDF. Returns list of (sample, code, time)."""
    bdf = os.path.join(BDF_DIR, f"P{pid}.bdf")
    tmp = False
    if not os.path.isfile(bdf):
        rc = os.system(f'unzip -q -o "{BDF_ZIP}" "EEG_raw/P{pid}.bdf" '
                       f'-d "{SENSE}" 2>/dev/null')
        tmp = rc == 0 and os.path.isfile(bdf)
    if not os.path.isfile(bdf):
        return [], None, tmp
    try:
        raw = mne.io.read_raw_bdf(bdf, preload=False, verbose=False)
        ev  = mne.find_events(raw, stim_channel="Status", verbose=False)
        ev[:, 2] = ev[:, 2] & 0xFF
        sf  = raw.info["sfreq"]
        # only task triggers
        task_ev = [(int(s), int(c), float(s/sf))
                   for s, _, c in ev if c in TASK_CODES]
        return task_ev, sf, tmp
    except Exception as e:
        print(f"    BDF trigger error: {e}")
        return [], None, tmp
    finally:
        if tmp and not KEEP_FILES and os.path.isfile(bdf):
            os.remove(bdf)


def load_cleaned_set(pid):
    """Load ICA-cleaned EEG .set. Returns (raw_mne, sfreq) or (None, None)."""
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


def process_participant(pid, ev_df):
    print(f"  P{pid}", end="", flush=True)

    # ── Triggers from BDF ─────────────────────────────────────────────
    task_ev, bdf_sf, bdf_tmp = extract_triggers_from_bdf(pid)
    if not task_ev:
        print(" no triggers"); return []

    # ── Cleaned EEG ───────────────────────────────────────────────────
    eeg_raw, eeg_sf, set_tmp = load_cleaned_set(pid)
    if eeg_raw is None:
        print(" no cleaned EEG"); return []
    eeg_data = eeg_raw.get_data()   # (32_or_less, n_samples) — ICA cleaned

    def ch_idx(names):
        return [eeg_raw.ch_names.index(c) for c in names if c in eeg_raw.ch_names]
    idx = {"f": ch_idx(FRONTAL), "p": ch_idx(POSTERIOR), "o": ch_idx(OCCIPITAL)}
    if not idx["f"]:
        print(" no frontal channels")
        if set_tmp: os.remove(os.path.join(EEG_DIR, f"P{pid}_100Hz_downsampled.set"))
        return []
    print(f" {eeg_data.shape[1]/eeg_sf:.0f}s cleaned EEG "
          f"({eeg_raw.info['nchan']} ch after ICA)", end="")

    # ── ECG from .fif ────────────────────────────────────────────────
    fif = os.path.join(ECG_DIR, f"P{pid}.fif")
    ecg_sig, ecg_sf_fif = None, None
    if os.path.isfile(fif):
        try:
            r = mne.io.read_raw_fif(fif, preload=True, verbose=False)
            ecg_chs = [c for c in r.ch_names if "ECG" in c.upper()]
            if ecg_chs:
                ecg_sig = r.get_data(picks=[ecg_chs[0]])[0]
                ecg_sf_fif = r.info["sfreq"]
        except Exception:
            pass

    # ── Respiration from .wav ────────────────────────────────────────
    wav = os.path.join(RSP_DIR, f"P{pid}.wav")
    resp_sig = None
    if os.path.isfile(wav):
        try:
            _, s = wavfile.read(wav)
            resp_sig = s[:, 0].astype(float) if s.ndim > 1 else s.astype(float)
        except Exception:
            pass

    print(f" | ECG {'y' if ecg_sig is not None else 'n'}"
          f" resp {'y' if resp_sig is not None else 'n'}")

    # ── HCI from v2 events ───────────────────────────────────────────
    p_hci = ev_df[ev_df.participant == f"P{pid}"].copy()
    hci_avail = [c for c in HCI_COLS if c in p_hci.columns]

    # ── Epoch extraction ─────────────────────────────────────────────
    n_samp_eeg = eeg_data.shape[1]
    rows = []
    trig_starts = [t for t, c, _ in task_ev]

    for i, (samp, code, onset) in enumerate(task_ev):
        task = TASK_CODES[code]
        # epoch: trigger onset to next trigger
        nxt = trig_starts[i+1] if i+1 < len(task_ev) else n_samp_eeg * (1024/eeg_sf)
        dur = (nxt - samp) / 1024.0  # BDF sample rate
        if dur < MIN_EPOCH:
            continue

        # map BDF sample to cleaned EEG sample
        # BDF: 1024 Hz, cleaned: 100 Hz, same recording start
        eeg_s0 = int(samp / 1024 * eeg_sf)
        eeg_s1 = min(int((samp/1024 + min(dur, MAX_EPOCH)) * eeg_sf), n_samp_eeg)
        if eeg_s1 <= eeg_s0:
            continue

        bp = band_powers(eeg_data[:, eeg_s0:eeg_s1], eeg_sf, idx)
        if bp is None:
            continue

        rec = {"participant": f"P{pid}", "onset_s": onset,
               "task": task, "app": TASK_APP.get(task, "unknown"),
               "duration_s": float(min(dur, MAX_EPOCH))}
        rec.update(bp)

        # ECG: from .fif (t=0 ≈ BDF t=0 since same BioSemi recording)
        if ecg_sig is not None:
            e0 = int(onset * ecg_sf_fif)
            e1 = min(int((onset + min(dur, MAX_EPOCH)) * ecg_sf_fif), len(ecg_sig))
            hr, rmssd, nb = ecg_features(ecg_sig[e0:e1], ecg_sf_fif)
            rec["hr_mean"] = hr; rec["hrv_rmssd"] = rmssd; rec["n_beats"] = nb
        else:
            rec["hr_mean"] = rec["hrv_rmssd"] = np.nan; rec["n_beats"] = 0

        # Respiration: from .wav (32 Hz, same recording)
        if resp_sig is not None:
            r0 = int(onset * RESP_FS)
            r1 = min(int((onset + min(dur, MAX_EPOCH)) * RESP_FS), len(resp_sig))
            rb, ra = resp_features(resp_sig[r0:r1])
            rec["resp_bpm"] = rb; rec["resp_amp"] = ra
        else:
            rec["resp_bpm"] = rec["resp_amp"] = np.nan

        # HCI: from v2 events (event-index matched, correct)
        # match by onset_s within ±2s
        hci_match = p_hci[(p_hci.onset_s >= onset - 2) & (p_hci.onset_s <= onset + 2)]
        if len(hci_match):
            for c in hci_avail:
                rec[c] = float(hci_match.iloc[0][c])
        rows.append(rec)

    del eeg_data, eeg_raw
    gc.collect()
    if set_tmp:
        sp = os.path.join(EEG_DIR, f"P{pid}_100Hz_downsampled.set")
        if not KEEP_FILES and os.path.isfile(sp):
            os.remove(sp)

    return rows


# ══════════════════════════════════════════════════════════════════════
# Analysis (CCA + direction RF, LOSO)
# ══════════════════════════════════════════════════════════════════════

def zscore_within(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = df.groupby("participant")[c].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return out


def loso_cca(X, Y, groups, n_comp=2):
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y, groups = X[ok], Y[ok], groups[ok]
    if len(X) < 200:
        return [np.nan]*n_comp, [np.nan]*n_comp, [np.nan]*n_comp
    try:
        cca = CCA(n_components=n_comp, max_iter=3000)
        Xs, Ys = cca.fit_transform(X, Y)
        tr = [float(np.corrcoef(Xs[:,i], Ys[:,i])[0,1]) for i in range(n_comp)]
    except Exception:
        return [np.nan]*n_comp, [np.nan]*n_comp, [np.nan]*n_comp
    cv = [[] for _ in range(n_comp)]
    for held in np.unique(groups):
        a, b = groups != held, groups == held
        if a.sum() < 100 or b.sum() < 5: continue
        try:
            c = CCA(n_components=n_comp, max_iter=3000).fit(X[a], Y[a])
            Xt, Yt = c.transform(X[b], Y[b])
            for i in range(n_comp):
                if np.std(Xt[:,i]) > 1e-9 and np.std(Yt[:,i]) > 1e-9:
                    cv[i].append(np.corrcoef(Xt[:,i], Yt[:,i])[0,1])
        except Exception:
            pass
    cvm = [float(np.nanmean(c)) if c else np.nan for c in cv]
    cvs = [float(np.nanstd(c))  if c else np.nan for c in cv]
    return tr, cvm, cvs


def run_exp(name, df, x_cols, y_cols):
    avail_x = [c for c in x_cols if c in df.columns]
    avail_y = [c for c in y_cols if c in df.columns]
    if not avail_x or not avail_y:
        print(f"  {name}: missing columns"); return {}

    dz = zscore_within(df, avail_x + avail_y)
    X  = np.nan_to_num(dz[avail_x].to_numpy(float))
    Y  = np.nan_to_num(dz[avail_y].to_numpy(float))
    g  = df.participant.to_numpy()

    n_comp = min(2, len(avail_x), len(avail_y))
    tr, cvm, cvs = loso_cca(X, Y, g, n_comp)

    # CFA check
    ctrl_avail = [c for c in EEG_CTRL if c in df.columns]
    ctrl_cca = [np.nan]
    if ctrl_avail and any(c in avail_y for c in EEG_COG):
        dz2 = zscore_within(df, avail_x + ctrl_avail)
        Yc  = np.nan_to_num(dz2[ctrl_avail].to_numpy(float))
        _, ctrl_cca, _ = loso_cca(X, Yc, g, min(1, len(avail_x), len(ctrl_avail)))

    print(f"\n  {name}")
    print(f"    CCA Component 1: train r={tr[0]:.3f}  CV r={cvm[0]:.3f}±{cvs[0]:.3f}")
    if ctrl_avail and any(c in avail_y for c in EEG_COG):
        gap = (cvm[0] or 0) - (ctrl_cca[0] or 0)
        flag = ("  -> REAL (cognitive > control)" if gap > 0.05 else
                "  -> ARTIFACT (control ≥ cognitive)" if gap < 0 else
                "  -> AMBIGUOUS")
        print(f"    Control CCA CV r={ctrl_cca[0]:.3f}  gap={gap:+.3f}{flag}")

    verdict = ("*** REAL" if (cvm[0] or 0) > 0.30 and
                             (ctrl_cca[0] or 0) < (cvm[0] or 0) - 0.05
               else "~ marginal" if (cvm[0] or 0) > 0.15
               else "null")
    print(f"    Overall: {verdict}")
    return {"cv_r": cvm[0], "cv_sd": cvs[0], "train_r": tr[0],
            "ctrl_cv_r": ctrl_cca[0], "verdict": verdict}


# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 74)
    print("SENSE-42 CLEANED EEG EXPERIMENTS")
    print("ICA-cleaned .set files (32→28 channels, 4 components removed)")
    print("=" * 74)

    # Load existing v2 events for HCI and old (raw) EEG
    ev_df = pd.read_csv(EV_CSV) if os.path.isfile(EV_CSV) else pd.DataFrame()
    hci_present = [c for c in HCI_COLS if c in ev_df.columns]
    print(f"v2 events: {len(ev_df)} rows, {len(hci_present)} HCI cols")

    # Load cached if available
    if os.path.isfile(OUT_CSV):
        print(f"Loading cached cleaned features: {OUT_CSV}")
        df = pd.read_csv(OUT_CSV)
        done = set(df.participant.unique())
    else:
        df, done = pd.DataFrame(), set()

    # Find participants to process
    pids = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(os.path.join(
                       os.path.dirname(EV_CSV), "../data/sense_42/Behavioural/CSV/*.csv"))})
    # Fallback: from v2 events
    if not pids and len(ev_df):
        pids = sorted({p.replace("P","") for p in ev_df.participant.unique()})
    pids = [p for p in pids if f"P{p}" not in done and p != "005"]

    print(f"To process: {len(pids)} participants\n")

    for pid in pids:
        try:
            rows = process_participant(pid, ev_df)
        except Exception as e:
            print(f"\n  P{pid} ERROR: {e}"); continue
        if rows:
            new = pd.DataFrame(rows)
            df = pd.concat([df, new], ignore_index=True) if len(df) else new
            df.to_csv(OUT_CSV, index=False)

    if df.empty:
        print("No data collected."); return

    print(f"\n{'='*74}")
    print(f"DATASET: {len(df)} event rows, {df.participant.nunique()} participants")
    print("Coverage on CLEANED EEG:")
    for c in EEG_COG + EEG_CTRL + ["hr_mean","hrv_rmssd","resp_bpm"]:
        if c in df.columns:
            n = df[c].notna().sum()
            print(f"  {c:24s} {n:6d}/{len(df)} ({100*n/len(df):.1f}%)")

    hci_avail   = [c for c in HCI_COLS   if c in df.columns]
    cardiac     = [c for c in ["hr_mean","hrv_rmssd"] if c in df.columns]
    resp        = [c for c in ["resp_bpm","resp_amp"] if c in df.columns]

    print(f"\n{'='*74}")
    print("EXPERIMENTS ON ICA-CLEANED EEG")
    print("Comparing against raw BDF results:")
    print("  EXP2 (HCI→EEG raw):           CV r = 0.342  ctrl = 0.307")
    print("  EXP3 (cardiac→EEG raw):        CV r = 0.172  ctrl = 0.216")
    print("  EXP4 (HCI+cardiac→EEG raw):    CV r = 0.203  ctrl = 0.168")
    print(f"{'='*74}")

    results = {}
    results["exp2_cleaned"] = run_exp(
        "EXP2 cleaned: HCI → EEG", df, hci_avail, EEG_COG)

    results["exp3_cleaned"] = run_exp(
        "EXP3 cleaned: cardiac → EEG", df, cardiac + resp, EEG_COG)

    results["exp4_cleaned"] = run_exp(
        "EXP4 cleaned: HCI + cardiac → EEG", df, hci_avail + cardiac, EEG_COG)

    # Per-task for EXP2 (the most important one)
    if "app" in df.columns:
        print("\n  EXP2 per task type (cleaned):")
        task_res = {}
        for task in df.app.value_counts().index:
            sub = df[df.app == task].reset_index(drop=True)
            if len(sub) < 200 or sub.participant.nunique() < MIN_FOLDS:
                continue
            dz = zscore_within(sub, hci_avail + EEG_COG)
            X  = np.nan_to_num(dz[hci_avail].to_numpy(float))
            Y  = np.nan_to_num(dz[EEG_COG].to_numpy(float))
            g  = sub.participant.to_numpy()
            _, cvm, cvs = loso_cca(X, Y, g, 1)
            task_res[task] = {"cv_r": cvm[0], "n": int(len(sub))}
            flag = " ***" if (cvm[0] or 0) > 0.30 else \
                   " *"   if (cvm[0] or 0) > 0.15 else ""
            print(f"    {task:12s} n={len(sub):5d}  CV r={cvm[0]:.3f}±{cvs[0]:.3f}{flag}")
        results["exp2_per_task"] = task_res

    print(f"\n{'='*74}")
    print("INTERPRETATION")
    print(f"{'='*74}")
    r2c = results.get("exp2_cleaned", {}).get("cv_r", np.nan)
    r2ctrl = results.get("exp2_cleaned", {}).get("ctrl_cv_r", np.nan)
    print(f"""
  EXP2 cleaned: CV r = {r2c:.3f}  ctrl = {r2ctrl:.3f}
  EXP2 raw:     CV r = 0.342   ctrl = 0.307
  Change:        {(r2c or 0) - 0.342:+.3f} on cognitive  {(r2ctrl or 0) - 0.307:+.3f} on control

  If cognitive drops more than control after ICA cleaning:
    -> CFA was specifically inflating cognitive targets (unlikely)
  If both drop similarly:
    -> CFA was broadband and ICA removed it from both equally
  If cognitive stays near 0.342 and control drops:
    -> The HCI-EEG coupling is REAL — ICA removed CFA but not the signal
  If both stay similar to raw:
    -> ICA didn't remove cardiac components (check which 4 were removed)
""")

    results["n_rows"] = int(len(df))
    results["n_participants"] = int(df.participant.nunique())
    with open(OUT_J, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {OUT_J}")


if __name__ == "__main__":
    main()
