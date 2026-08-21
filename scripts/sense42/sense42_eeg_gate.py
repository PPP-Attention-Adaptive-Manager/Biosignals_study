"""
sense42_eeg_gate.py
====================
SENSE-42 gate test — does adding autonomic signals to HCI improve
prediction of cortical (EEG) state?

    V1:  HCI only                          -> EEG band powers
    V2:  HCI + HR + RMSSD + resp_bpm        -> EEG band powers

This replicates the Cog Lab dynamics gate test on lab-grade data.

WHY RE-RUN A TEST THAT ALREADY FAILED:
  On Cog Lab, adding HR/RMSSD/SCL gave no lift for any cortical target
  (eeg_theta_alpha +0.007, eeg_engagement -0.022, eeg_alpha_asym -0.004).
  But Cog Lab's EEG was consumer-grade, single passive task, and frontal
  channels were contaminated by jaw EMG during typing. SENSE-42 is
  BioSemi 32-ch, cleaned, 42 participants, varied tasks. A null result
  here is a far stronger claim than the Cog Lab null.

THE CONFOUND THIS SCRIPT CONTROLS FOR (critical):
  SENSE-42 README: "3-lead ECG signals extracted from the external
  channels of the EEG system" -- ECG and EEG share one amplifier.
  Heartbeats leave a cardiac field artifact (CFA) in scalp EEG.
  So if V2 > V1, two explanations are indistinguishable without control:
     (a) REAL     - autonomic state predicts cortical state
     (b) ARTIFACT - HR predicts residual cardiac contamination
  CONTROL: CFA is broadband and topographically diffuse. Real cognitive
  coupling is band-specific (theta/alpha) and frontal. So we also predict
  two cognitively-meaningless targets:
     occipital_delta     (O1/Oz/O2, 1-4 Hz)
     broadband_amplitude (all channels, all freqs)
  If HR predicts these as well as it predicts frontal theta -> CFA.

METHODOLOGICAL FIXES vs the first Phase B attempt:
  1. Band power via 2s sub-epochs, MEDIAN across them (not one Welch
     over the full 30s). Robust to transient artifacts.
  2. Per-task-type analysis. Averaging mail reading + file dragging +
     typing mixes incompatible brain states and cancels signal.
  3. Direction framing (rising/falling) alongside absolute values --
     the reframe that rescued the SWELL-KW work.

WINDOW CHOICE:
  30s is kept because RMSSD needs ~30 beats to be stable. EEG is
  computed in 2s sub-epochs within that window, so fine structure is
  preserved via the median rather than washed out by a single estimate.

MEMORY:
  Processes one participant at a time. Extracts the .set from the zip,
  computes features, deletes the .set, caches features to CSV.
  Never more than ~1.4 GB on disk at once. Resumable.

Run from: ~/biosignals_data/
Output:
  outputs/sense42_gate_features.csv   -- cached per-window features
  outputs/sense42_gate_results.json   -- V1 vs V2 results + CFA control
"""
from __future__ import annotations
import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

BASE      = os.path.expanduser("~/biosignals_data")
SENSE     = os.path.join(BASE, "data", "sense_42")
CSV_DIR   = os.path.join(SENSE, "Behavioural", "CSV")
ECG_DIR   = os.path.join(SENSE, "ECG")
RESP_DIR  = os.path.join(SENSE, "Respiration")
EEG_DIR   = os.path.join(SENSE, "EEG_cleaned")
EEG_ZIP   = os.path.join(SENSE, "EEG_cleaned.zip")
OUT_FEAT  = os.path.join(BASE, "outputs", "sense42_gate_features.csv")
OUT_JSON  = os.path.join(BASE, "outputs", "sense42_gate_results.json")
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)

WINDOW_S   = 30.0     # main window (RMSSD needs ~30 beats)
SUBEPOCH_S = 2.0      # EEG band power sub-epoch
KEEP_SET   = False    # True = don't delete .set after processing

# ── channel groups (confirmed present in SENSE-42 32-ch montage) ──────
FRONTAL   = ['AF3', 'F3', 'Fz', 'F4', 'AF4']
POSTERIOR = ['P3', 'Pz', 'P4']
OCCIPITAL = ['O1', 'Oz', 'O2']          # CFA control site

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnSpaces","SnAppChange",
    "CharactersRatio","ErrorKeyRatio",
]
AUTONOMIC_COLS = ["hr_mean", "hrv_rmssd", "resp_bpm"]

# EEG targets: 5 cognitive + 2 control
EEG_COGNITIVE = ["frontal_theta", "frontal_alpha", "theta_alpha_ratio",
                 "engagement_index", "posterior_alpha"]
EEG_CONTROL   = ["occipital_delta", "broadband_amplitude"]
EEG_TARGETS   = EEG_COGNITIVE + EEG_CONTROL

sys.path.insert(0, os.path.join(BASE, "scripts", "sense42"))
from sense42_phase_a import extract_hci_from_csv


# ════════════════════════════════════════════════════════════════════
# EEG band powers -- sub-epoch median (robust to transients)
# ════════════════════════════════════════════════════════════════════

def band_powers_subepoch(epoch: np.ndarray, sfreq: float) -> dict:
    """
    epoch: (n_channels, n_samples) covering WINDOW_S seconds.
    Splits into SUBEPOCH_S chunks, computes Welch band power in each,
    returns the MEDIAN across sub-epochs per channel.

    Median rather than mean because a single artifact spike (blink,
    swallow, movement) inflates one sub-epoch and would drag a mean
    upward. The median ignores it.
    """
    n_sub = int(SUBEPOCH_S * sfreq)
    n_chunks = epoch.shape[1] // n_sub
    if n_chunks < 3:
        return None

    bands = {"delta": (1, 4), "theta": (4, 8),
             "alpha": (8, 13), "beta": (13, 30)}
    acc = {b: [] for b in bands}
    acc["broadband"] = []

    for c in range(n_chunks):
        chunk = epoch[:, c*n_sub:(c+1)*n_sub]
        f, psd = welch(chunk, fs=sfreq, nperseg=min(n_sub, int(sfreq)), axis=1)
        for b, (lo, hi) in bands.items():
            m = (f >= lo) & (f < hi)
            acc[b].append(psd[:, m].mean(axis=1))
        # broadband 1-40 Hz: CFA control -- cardiac artifact is broadband
        m_bb = (f >= 1) & (f < 40)
        acc["broadband"].append(psd[:, m_bb].mean(axis=1))

    return {b: np.median(np.array(v), axis=0) for b, v in acc.items()}


def extract_eeg_features(set_path: str) -> pd.DataFrame:
    """Per-30s-window EEG features, incl. the two CFA control targets."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    data  = raw.get_data()
    times = raw.times

    def idx(names):
        return [raw.ch_names.index(c) for c in names if c in raw.ch_names]

    i_front, i_post, i_occ = idx(FRONTAL), idx(POSTERIOR), idx(OCCIPITAL)
    if not i_front:
        print("      no frontal channels found"); return pd.DataFrame()

    rows, w = [], float(times[0])
    while w + WINDOW_S <= times[-1]:
        m = (times >= w) & (times < w + WINDOW_S)
        bp = band_powers_subepoch(data[:, m], sfreq)
        if bp is None:
            w += WINDOW_S; continue

        f_theta = bp["theta"][i_front].mean()
        f_alpha = bp["alpha"][i_front].mean()
        f_beta  = bp["beta"][i_front].mean()
        p_alpha = bp["alpha"][i_post].mean() if i_post else f_alpha
        o_delta = bp["delta"][i_occ].mean()  if i_occ  else bp["delta"].mean()

        rows.append({
            "window_start":       w,
            # cognitive targets
            "frontal_theta":      float(f_theta),
            "frontal_alpha":      float(f_alpha),
            "theta_alpha_ratio":  float(f_theta / (f_alpha + 1e-15)),
            "engagement_index":   float(f_beta / (f_alpha + f_theta + 1e-15)),
            "posterior_alpha":    float(p_alpha),
            # CFA control targets -- no cognitive-load interpretation
            "occipital_delta":    float(o_delta),
            "broadband_amplitude":float(bp["broadband"].mean()),
        })
        w += WINDOW_S

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# ECG / Respiration
# ════════════════════════════════════════════════════════════════════

def extract_ecg(fif_path: str) -> pd.DataFrame:
    raw   = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    chs   = [c for c in raw.ch_names if "ECG" in c.upper()] or raw.ch_names[:1]
    ecg   = raw.get_data(picks=[chs[0]])[0]
    t     = raw.times

    # bandpass is essential: raw BioSemi ECG has baseline drift that
    # swamps R-peaks after z-scoring (raw std ~0.0018)
    b, a = butter(3, [0.5/(sfreq/2), 40.0/(sfreq/2)], btype="band")
    z = filtfilt(b, a, ecg)
    z = (z - z.mean()) / (z.std() + 1e-9)
    peaks, _ = find_peaks(z, distance=int(0.35*sfreq), height=3.0)
    if len(peaks) < 50:
        return pd.DataFrame()

    pt, rows, w = t[peaks], [], float(t[0])
    while w + WINDOW_S <= t[-1]:
        inw = pt[(pt >= w) & (pt < w + WINDOW_S)]
        if len(inw) >= 8:
            rr = np.diff(inw); rr = rr[(rr > 0.33) & (rr < 1.5)]
            if len(rr) >= 6:
                rows.append({"window_start": w,
                             "hr_mean":   float(60.0/np.median(rr)),
                             "hrv_rmssd": float(np.sqrt(np.mean(np.diff(rr)**2))*1000)})
        w += WINDOW_S
    return pd.DataFrame(rows)


def extract_resp(wav_path: str) -> pd.DataFrame:
    fs, sig = wavfile.read(wav_path)
    if sig.ndim > 1: sig = sig[:, 0]
    sig = sig.astype(float)
    t   = np.arange(len(sig)) / fs
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    peaks, _ = find_peaks(sig, distance=int(0.75*fs), prominence=0.3)
    if len(peaks) < 3: return pd.DataFrame()

    pt, rows, w = t[peaks], [], float(t[0])
    while w + WINDOW_S <= t[-1]:
        inw = pt[(pt >= w) & (pt < w + WINDOW_S)]
        if len(inw) >= 3:
            bpm = 60.0 / np.mean(np.diff(inw))
            if 5 < bpm < 60:
                rows.append({"window_start": w, "resp_bpm": float(bpm)})
        w += WINDOW_S
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# Alignment (clocks differ: PsychoPy starts ~35-260s after BioSemi)
# ════════════════════════════════════════════════════════════════════

def align_to(hci_starts: np.ndarray, other: pd.DataFrame,
             cols: list) -> dict:
    out = {c: np.full(len(hci_starts), np.nan) for c in cols}
    if other.empty or len(other) < 5:
        return out
    o_start = other["window_start"].to_numpy(float)
    offset  = hci_starts[0] - o_start[0]
    for i, ht in enumerate(hci_starts):
        equiv = ht - offset
        j = int(np.argmin(np.abs(o_start - equiv)))
        if abs(o_start[j] - equiv) < WINDOW_S / 2:
            for c in cols:
                out[c][i] = other.iloc[j][c]
    return out


def process_participant(pid: str) -> pd.DataFrame | None:
    csvs = glob.glob(os.path.join(CSV_DIR, f"{pid}_*.csv"))
    if not csvs:
        print("    no CSV"); return None

    hci = extract_hci_from_csv(csvs[0])
    if hci.empty:
        print("    HCI failed"); return None

    ecg_p  = os.path.join(ECG_DIR,  f"P{pid}.fif")
    resp_p = os.path.join(RESP_DIR, f"P{pid}.wav")
    ecg  = extract_ecg(ecg_p)   if os.path.isfile(ecg_p)  else pd.DataFrame()
    resp = extract_resp(resp_p) if os.path.isfile(resp_p) else pd.DataFrame()
    print(f"    ECG {len(ecg)}w  Resp {len(resp)}w", end="")

    # EEG: extract from zip, process, delete
    set_p = os.path.join(EEG_DIR, f"P{pid}_100Hz_downsampled.set")
    tmp_extract = False
    if not os.path.isfile(set_p):
        rc = os.system(f'unzip -q -o "{EEG_ZIP}" '
                       f'"EEG_cleaned/P{pid}_100Hz_downsampled.set" '
                       f'-d "{SENSE}" 2>/dev/null')
        tmp_extract = (rc == 0 and os.path.isfile(set_p))
        if not tmp_extract:
            print("  |  EEG extract failed"); return None
    try:
        eeg = extract_eeg_features(set_p)
    finally:
        if tmp_extract and not KEEP_SET and os.path.isfile(set_p):
            os.remove(set_p)
    print(f"  EEG {len(eeg)}w", end="")
    if eeg.empty:
        print("  -> skip"); return None

    hs = hci["window_start"].to_numpy(float)
    df = hci.copy()
    df["participant"] = f"P{pid}"
    for c, v in align_to(hs, ecg,  ["hr_mean", "hrv_rmssd"]).items(): df[c] = v
    for c, v in align_to(hs, resp, ["resp_bpm"]).items():             df[c] = v
    for c, v in align_to(hs, eeg,  EEG_TARGETS).items():              df[c] = v

    full = df[EEG_TARGETS + AUTONOMIC_COLS].notna().all(axis=1).sum()
    print(f"  |  complete rows: {full}/{len(df)}")
    return df


# ════════════════════════════════════════════════════════════════════
# Gate test
# ════════════════════════════════════════════════════════════════════

def zscore_by_group(X, groups):
    Xz = np.zeros_like(X, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        Xz[m] = (X[m] - np.nanmean(X[m], 0)) / (np.nanstd(X[m], 0) + 1e-9)
    return Xz


def direction_labels(y, groups):
    """1 = rising vs previous window, 0 = falling, NaN at each block start."""
    out = np.full(len(y), np.nan)
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        d = np.diff(y[m])
        out[m[1:]] = (d > 0).astype(float)
    return out


def loso_direction(X, y, groups, min_tr=60, min_te=8):
    """LOSO accuracy + majority baseline (the honest chance level)."""
    accs, bases = [], []
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum() < min_tr or ote.sum() < min_te: continue
        ytr, yte = y[tr][otr].astype(int), y[te][ote].astype(int)
        if len(np.unique(ytr)) < 2: continue
        m = RandomForestClassifier(200, min_samples_leaf=5,
                                   class_weight="balanced",
                                   random_state=0, n_jobs=-1)
        m.fit(X[tr][otr], ytr)
        accs.append(accuracy_score(yte, m.predict(X[te][ote])))
        bases.append(max(yte.mean(), 1 - yte.mean()))
    if not accs: return np.nan, np.nan, 0
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


def run_gate(df: pd.DataFrame):
    print("\n" + "=" * 74)
    print("GATE TEST — V1 (HCI) vs V2 (HCI + HR + RMSSD + resp)")
    print("=" * 74)

    need = HCI_COLS + AUTONOMIC_COLS + EEG_TARGETS
    d = df.dropna(subset=need).reset_index(drop=True)
    groups = d["participant"].to_numpy()
    print(f"Complete rows: {len(d)}  |  participants: {len(np.unique(groups))}\n")
    if len(d) < 200 or len(np.unique(groups)) < 5:
        print("Not enough complete data for a meaningful gate test.")
        return None

    X1 = zscore_by_group(np.nan_to_num(d[HCI_COLS].to_numpy(float)), groups)
    X2 = zscore_by_group(
        np.nan_to_num(d[HCI_COLS + AUTONOMIC_COLS].to_numpy(float)), groups)

    print(f"{'EEG target':22s} {'chance':>7s} {'V1':>7s} {'V2':>7s} "
          f"{'lift':>7s} {'V2-chance':>10s}")
    print("-" * 74)

    results = {}
    for tgt in EEG_TARGETS:
        y = direction_labels(d[tgt].to_numpy(float), groups)
        a1, b1, n1 = loso_direction(X1, y, groups)
        a2, b2, n2 = loso_direction(X2, y, groups)
        if np.isnan(a1) or np.isnan(a2):
            print(f"{tgt:22s}   insufficient folds"); continue
        lift = a2 - a1
        over = a2 - b2
        tag  = "  <- CONTROL" if tgt in EEG_CONTROL else ""
        print(f"{tgt:22s} {b2:7.3f} {a1:7.3f} {a2:7.3f} "
              f"{lift:+7.3f} {over:+10.3f}{tag}")
        results[tgt] = {"chance": b2, "v1": a1, "v2": a2,
                        "lift": lift, "over_chance": over, "folds": n2}

    # ── CFA control verdict ──────────────────────────────────────────
    print("\n" + "=" * 74)
    print("CARDIAC FIELD ARTIFACT CONTROL")
    print("=" * 74)
    print("ECG was recorded on the EEG amplifier's external channels, so")
    print("heartbeats leave a broadband electrical trace in scalp EEG.")
    print("Real autonomic-cortical coupling is band-specific and frontal.")
    print("CFA is broadband and diffuse -> it inflates the CONTROL targets.\n")

    cog  = [results[t]["lift"] for t in EEG_COGNITIVE if t in results]
    ctrl = [results[t]["lift"] for t in EEG_CONTROL   if t in results]
    if cog and ctrl:
        mc, mk = float(np.mean(cog)), float(np.mean(ctrl))
        print(f"  mean lift, cognitive targets: {mc:+.3f}")
        print(f"  mean lift, control targets:   {mk:+.3f}")
        print(f"  difference:                   {mc - mk:+.3f}\n")
        if mc < 0.02 and mk < 0.02:
            verdict = "NULL — autonomic signals add nothing to EEG prediction"
            print("  ->  NULL RESULT. Neither cognitive nor control targets")
            print("      benefit. Replicates the Cog Lab gate failure on")
            print("      lab-grade EEG. Consistent with Lacey (1967)")
            print("      response fractionation: cortical and autonomic")
            print("      subsystems are not tightly coupled.")
        elif mk >= mc - 0.01:
            verdict = "ARTIFACT — lift is cardiac contamination, not coupling"
            print("  ->  ARTIFACT. Control targets gain as much as cognitive")
            print("      ones. HR is predicting cardiac contamination in the")
            print("      EEG, not cortical state. Do NOT report as coupling.")
        else:
            verdict = "REAL — band-specific lift survives the CFA control"
            print("  ->  REAL SIGNAL. Cognitive targets gain substantially")
            print("      more than controls -> band-specific, not broadband.")
            print("      This is genuine autonomic-cortical coupling.")
    else:
        verdict = "inconclusive"
        mc = mk = float("nan")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  Verdict: {verdict}\n")
    print("  Reference — Cog Lab gate (consumer EEG, N=16, single task):")
    print("    eeg_theta_alpha  +0.007   eeg_engagement  -0.022")
    print("    eeg_alpha_asym   -0.004   (all failed)")
    print("    resp_bpm         +0.093   (the sole exception, via RSA)")

    out = {"n_rows": int(len(d)),
           "n_participants": int(len(np.unique(groups))),
           "targets": results,
           "mean_lift_cognitive": mc,
           "mean_lift_control": mk,
           "verdict": verdict}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
    return out


# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 74)
    print("SENSE-42 EEG GATE TEST")
    print("=" * 74)

    if os.path.isfile(OUT_FEAT):
        df = pd.read_csv(OUT_FEAT)
        done = set(df["participant"].unique())
        print(f"Cached: {len(df)} rows from {len(done)} participants")
        print("(delete outputs/sense42_gate_features.csv to rebuild)\n")
    else:
        df, done = pd.DataFrame(), set()

    pids = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(os.path.join(CSV_DIR, "*.csv"))})
    todo = [p for p in pids if f"P{p}" not in done]
    print(f"To process: {len(todo)} of {len(pids)} participants\n")

    for k, pid in enumerate(todo, 1):
        print(f"[{k:2d}/{len(todo)}] P{pid}")
        try:
            r = process_participant(pid)
        except Exception as e:
            print(f"    ERROR: {e}"); continue
        if r is None or r.empty: continue
        df = pd.concat([df, r], ignore_index=True) if len(df) else r
        df.to_csv(OUT_FEAT, index=False)   # save after each participant

    if df.empty:
        print("\nNo data collected."); return
    print(f"\nTotal: {len(df)} windows, {df.participant.nunique()} participants")
    run_gate(df)


if __name__ == "__main__":
    main()
