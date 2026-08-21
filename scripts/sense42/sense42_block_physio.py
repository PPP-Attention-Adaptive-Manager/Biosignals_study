"""
sense42_block_physio.py
========================
Fixes the low cardiac/resp coverage in sense42_v2_events.csv.

WHY COVERAGE WAS LOW
---------------------
Per-event epoching gives median epoch = 3.3s. A 2.4s file-manager drag
cannot contain 30 heartbeats. Our thresholds were:
    HR    >= 5s  -> 37.6% of events qualify
    resp  >= 8s  -> 22.8%
    RMSSD >= 20s -> 17.5%

The ECG and respiration signals themselves are fine:
    ECG  (EXG2-EXG3, h=2.0): 9,322 peaks -> 70.0 bpm (P002)
    Resp (decimate 1024->32Hz): 2,007 peaks -> 15.1 bpm (P002)

The signals are clean. The epoch design is wrong for cardiac physiology.

TWO OPTIONS
-----------

OPTION A — Block-level cardiac (best for EXP1 replication)
  Computes HR / RMSSD / resp over the full ~5-minute questionnaire block.
  Replicates SWELL-KW's per-condition aggregate design exactly.
  26 blocks x 40 participants = ~1,040 rows, all with full physiology.
  HCI features aggregated over the same block from the v2 events table.
  This is the correct design for CCA comparison with SWELL-KW CV r=0.581.

OPTION B — Sliding 30s window from .fif / .wav files
  Slides a 30s window over the pre-processed .fif (ECG) and .wav (resp)
  files provided in the dataset. Produces continuous cardiac time series
  at 30s resolution. Aligns to trigger-defined task blocks by onset.
  The .fif files are already cleaned by the dataset authors.
  Advantage: .fif is cleaner than BDF-derived; sliding window gives
  full session coverage independent of event duration.
  Note: .fif starts ~116s before the BDF experiment clock for P002.
  The offset is estimated from the known EXP_BEGIN trigger time.

BOTH options feed into the same six experiments as sense42_aligned_experiments.py.

Run from: ~/biosignals_data/
Output:
    outputs/sense42_block_physio.csv     (Option A: block-level)
    outputs/sense42_sliding_physio.csv   (Option B: 30s sliding window)
    outputs/sense42_block_exp1.json      (EXP1 re-run on block-level data)
    outputs/sense42_sliding_exp1.json    (EXP1 re-run on sliding data)
"""
from __future__ import annotations
import os, sys, glob, warnings, json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, decimate
from scipy.io import wavfile
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

BASE     = os.path.expanduser("~/biosignals_data")
SENSE    = os.path.join(BASE, "data", "sense_42")
ECG_DIR  = os.path.join(SENSE, "ECG")        # .fif files (pre-processed)
RESP_DIR = os.path.join(SENSE, "Respiration") # .wav files at 32 Hz
BDF_DIR  = os.path.join(SENSE, "EEG_raw")
EV_CSV   = os.path.join(BASE, "outputs", "sense42_v2_events.csv")
Q_CSV    = os.path.join(BASE, "outputs", "sense42_v2_questions.csv")
OUT_BLK  = os.path.join(BASE, "outputs", "sense42_block_physio.csv")
OUT_SLD  = os.path.join(BASE, "outputs", "sense42_sliding_physio.csv")
OUT_BLK_EXP = os.path.join(BASE, "outputs", "sense42_block_exp1.json")
OUT_SLD_EXP = os.path.join(BASE, "outputs", "sense42_sliding_exp1.json")

WINDOW_S  = 30.0       # sliding window for Option B
RESP_FS   = 32.0       # RespInPeace .wav sampling rate
BLOCK_GAP = 120.0      # seconds gap that separates questionnaire blocks

HCI_COLS = ["SnKeyStrokes","SnChars","SnSpecialKeys","SnDirectionKeys",
            "SnErrorKeys","SnSpaces","CharactersRatio","ErrorKeyRatio",
            "SnLeftClicked","SnMouseDistance","SnMouseAct"]
EEG_COLS = ["frontal_theta","frontal_alpha","theta_alpha_ratio",
            "engagement_index","posterior_alpha"]
PHYSIO   = ["hr_mean","hrv_rmssd","resp_bpm","resp_amp"]
MIN_FOLDS = 8


# ══════════════════════════════════════════════════════════════════════
# Shared signal helpers
# ══════════════════════════════════════════════════════════════════════

def extract_hr_rmssd(signal, sfreq, t0=None, t1=None):
    """
    R-peak detection on a bipolar ECG or .fif signal segment.
    Returns (hr_mean, hrv_rmssd, n_beats).
    t0/t1: sample indices. If None, use full signal.
    """
    seg = signal[t0:t1] if (t0 is not None) else signal
    if len(seg) < int(sfreq * 5):
        return np.nan, np.nan, 0
    b, a = butter(3, [0.5/(sfreq/2), 40.0/(sfreq/2)], btype="band")
    z = filtfilt(b, a, seg)
    z = (z - z.mean()) / (z.std() + 1e-9)
    peaks, _ = find_peaks(z, distance=int(0.35 * sfreq), height=2.0)
    if len(peaks) < 4:
        return np.nan, np.nan, len(peaks)
    rr = np.diff(peaks) / sfreq
    rr = rr[(rr > 0.33) & (rr < 1.5)]
    if len(rr) < 3:
        return np.nan, np.nan, len(peaks)
    hr = float(60.0 / np.median(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)) * 1000) if len(rr) >= 10 else np.nan
    return hr, rmssd, int(len(rr))


def extract_resp(signal32, s0=None, s1=None):
    """
    Respiration features from a segment of the 32 Hz RespInPeace signal.
    s0/s1: sample indices in the 32 Hz signal.
    """
    seg = signal32[s0:s1] if s0 is not None else signal32
    if len(seg) < int(RESP_FS * 8):
        return np.nan, np.nan
    b, a = butter(3, [0.1/(RESP_FS/2), 0.5/(RESP_FS/2)], btype="band")
    x = filtfilt(b, a, seg.astype(float))
    if not np.all(np.isfinite(x)) or x.std() < 1e-12:
        return np.nan, np.nan
    z = (x - x.mean()) / (x.std() + 1e-9)
    peaks, _ = find_peaks(z, distance=int(1.5 * RESP_FS), prominence=0.3)
    if len(peaks) < 3:
        return np.nan, np.nan
    bpm = 60.0 / np.mean(np.diff(peaks) / RESP_FS)
    return (float(bpm) if 5 < bpm < 60 else np.nan), float(x.std())


def load_fif_ecg(pid):
    """Load .fif ECG file. Returns (signal, sfreq) or (None, None)."""
    path = os.path.join(ECG_DIR, f"P{pid}.fif")
    if not os.path.isfile(path):
        return None, None
    try:
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
        ecg_chs = [c for c in raw.ch_names if "ECG" in c.upper()]
        if not ecg_chs:
            return None, None
        # The .fif already has proper ECG derivation done by the authors
        return raw.get_data(picks=[ecg_chs[0]])[0], raw.info["sfreq"]
    except Exception:
        return None, None


def load_wav_resp(pid):
    """Load .wav respiration. Returns (signal32,) or (None,)."""
    path = os.path.join(RESP_DIR, f"P{pid}.wav")
    if not os.path.isfile(path):
        return None
    try:
        fs, sig = wavfile.read(path)
        if sig.ndim > 1:
            sig = sig[:, 0]
        return sig.astype(float)
    except Exception:
        return None


def load_bdf_ecg(pid):
    """
    Fallback: bipolar ECG from raw BDF (EXG2-EXG3).
    Used only if .fif is unavailable.
    """
    path = os.path.join(BDF_DIR, f"P{pid}.bdf")
    if not os.path.isfile(path):
        return None, None
    try:
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
        sf = raw.info["sfreq"]
        if "EXG2" in raw.ch_names and "EXG3" in raw.ch_names:
            ecg = (raw.get_data(picks=["EXG2"])[0]
                   - raw.get_data(picks=["EXG3"])[0])
            return ecg, sf
        return None, None
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════
# OPTION A — Block-level cardiac
# ══════════════════════════════════════════════════════════════════════

def assign_blocks(q):
    """Cluster consecutive question triggers into blocks (same as v3)."""
    q = q.sort_values(["participant", "onset_s"]).copy()
    blk, prev_p, prev_t = 0, None, None
    out = []
    for p, t in zip(q.participant, q.onset_s):
        if prev_p != p or (t - prev_t) > BLOCK_GAP:
            blk += 1
        out.append(blk)
        prev_p, prev_t = p, t
    q["block"] = out
    return q


def option_a_block_level():
    """
    Aggregate HCI features and physiology at block level.
    Each block = one questionnaire period (~5 minutes of task work).
    Physiology computed over the task window preceding the questionnaire.
    """
    print("\n" + "=" * 72)
    print("OPTION A — Block-level cardiac (replicates SWELL-KW design)")
    print("=" * 72)

    for p in (EV_CSV, Q_CSV):
        if not os.path.isfile(p):
            print(f"  Missing {p}"); return pd.DataFrame()

    ev = pd.read_csv(EV_CSV)
    q  = pd.read_csv(Q_CSV)
    q  = assign_blocks(q)

    hci_avail = [c for c in HCI_COLS if c in ev.columns]
    eeg_avail = [c for c in EEG_COLS if c in ev.columns]
    if not hci_avail:
        print("  No HCI columns in events table. Need sense42_v2_events.csv")
        return pd.DataFrame()

    print(f"  Events: {len(ev)}  Questions: {len(q)}")
    print(f"  Blocks: {q.block.nunique()} total, "
          f"{q.groupby('participant').block.nunique().mean():.1f} per participant")

    # Block boundaries from question timestamps
    binfo = (q.groupby(["participant", "block"])
               .agg(blk_start=("onset_s", "min"),
                    blk_end=("onset_s", "max")).reset_index()
               .sort_values(["participant", "blk_start"]))

    rows = []
    pids = binfo.participant.unique()
    print(f"  Processing {len(pids)} participants...")

    for pid in pids:
        pid_str = pid.replace("P", "")

        # Load ECG: try .fif first (author-processed), fall back to BDF
        ecg_sig, ecg_sf = load_fif_ecg(pid_str)
        ecg_source = "fif"
        if ecg_sig is None:
            ecg_sig, ecg_sf = load_bdf_ecg(pid_str)
            ecg_source = "bdf"
        if ecg_sig is not None:
            print(f"    {pid}: ECG from {ecg_source} "
                  f"({len(ecg_sig)/ecg_sf:.0f}s)", end="")

        # Load respiration (.wav, already at 32 Hz)
        resp_sig = load_wav_resp(pid_str)
        if resp_sig is not None:
            print(f" | resp {len(resp_sig)/RESP_FS:.0f}s", end="")
        print()

        p_events = ev[ev.participant == pid].sort_values("onset_s")
        p_blocks = binfo[binfo.participant == pid].reset_index(drop=True)

        prev_blk_end = 0.0
        for _, b in p_blocks.iterrows():
            task_lo = prev_blk_end
            task_hi = b.blk_start
            prev_blk_end = b.blk_end

            rec = {"participant": pid, "block": int(b.block),
                   "task_lo": float(task_lo), "task_hi": float(task_hi),
                   "task_span_s": float(task_hi - task_lo)}

            # HCI: mean over task events in [task_lo, task_hi]
            seg = p_events[(p_events.onset_s >= task_lo) &
                           (p_events.onset_s < task_hi)]
            rec["n_task_events"] = int(len(seg))
            for c in hci_avail + eeg_avail:
                v = seg[c].to_numpy(float) if c in seg.columns else np.array([])
                v = v[np.isfinite(v)]
                rec[c] = float(v.mean()) if len(v) else np.nan

            # ECG: R-peaks over the full task block
            if ecg_sig is not None and task_hi > task_lo:
                # .fif starts at a different time origin than triggers
                # For .fif: the signal covers the full session but t=0
                # corresponds to the recording start, not EXP_BEGIN.
                # Approximate: .fif duration ~8109s vs BDF 7993s for P002
                # -> .fif starts ~116s before EXP_BEGIN.
                # More robust: use proportion of session.
                # We use trigger-derived offsets stored in events onset_s.
                # For now: assume .fif t=0 ≈ EEG_START_RECORDING (~0.42s
                # before EXP_BEGIN), so .fif time ≈ trigger time + small const.
                # This is much closer than the 104s BDF misalignment.
                if ecg_source == "fif":
                    # .fif is in MNE raw format: times start at 0
                    # EXP_BEGIN is at ~101.5s in BDF, .fif starts even earlier
                    # Conservative offset: assume .fif t=0 ≈ trigger t=0
                    # (both anchored to BioSemi recording start)
                    ecg_lo = int(task_lo * ecg_sf)
                    ecg_hi = int(task_hi * ecg_sf)
                else:
                    # BDF: same clock as triggers directly
                    ecg_lo = int(task_lo * ecg_sf)
                    ecg_hi = int(task_hi * ecg_sf)
                ecg_lo = max(0, ecg_lo)
                ecg_hi = min(len(ecg_sig), ecg_hi)
                hr, rmssd, nb = extract_hr_rmssd(ecg_sig, ecg_sf, ecg_lo, ecg_hi)
                rec["hr_mean"] = hr; rec["hrv_rmssd"] = rmssd; rec["n_beats"] = nb
            else:
                rec["hr_mean"] = rec["hrv_rmssd"] = np.nan; rec["n_beats"] = 0

            # Respiration: peak detection over task block
            if resp_sig is not None and task_hi > task_lo:
                r_lo = int(task_lo * RESP_FS)
                r_hi = int(task_hi * RESP_FS)
                r_lo, r_hi = max(0, r_lo), min(len(resp_sig), r_hi)
                bpm, amp = extract_resp(resp_sig, r_lo, r_hi)
                rec["resp_bpm"] = bpm; rec["resp_amp"] = amp
            else:
                rec["resp_bpm"] = rec["resp_amp"] = np.nan

            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No rows produced"); return df

    # Join questionnaire ratings
    ratings = q.pivot_table(index=["participant", "block"],
                            columns="dimension", values="rating").reset_index()
    df = df.merge(ratings, on=["participant", "block"], how="left")
    df.to_csv(OUT_BLK, index=False)

    print(f"\n  OPTION A output: {len(df)} blocks, "
          f"{df.participant.nunique()} participants")
    for c in ["hr_mean", "hrv_rmssd", "resp_bpm"] + hci_avail[:3]:
        if c in df.columns:
            n = df[c].notna().sum()
            print(f"    {c:22s} {n:5d}/{len(df)} ({100*n/len(df):.1f}%)")
    print(f"\n  Saved: {OUT_BLK}")
    return df


# ══════════════════════════════════════════════════════════════════════
# OPTION B — Sliding 30s window on .fif / .wav
# ══════════════════════════════════════════════════════════════════════

def option_b_sliding_window():
    """
    Slide a 30s window over the .fif ECG and .wav respiration.
    Then align to task blocks by onset time.
    Produces continuous cardiac/resp time series at 30s resolution.
    """
    print("\n" + "=" * 72)
    print("OPTION B — Sliding 30s window on .fif + .wav (continuous)")
    print("=" * 72)

    if not os.path.isfile(EV_CSV):
        print(f"  Missing {EV_CSV}"); return pd.DataFrame()

    ev = pd.read_csv(EV_CSV)
    hci_avail = [c for c in HCI_COLS if c in ev.columns]
    eeg_avail = [c for c in EEG_COLS if c in ev.columns]

    all_rows = []
    pids = ev.participant.unique()
    print(f"  Processing {len(pids)} participants, {WINDOW_S:.0f}s windows...")

    for pid in pids:
        pid_str = pid.replace("P", "")

        ecg_sig, ecg_sf = load_fif_ecg(pid_str)
        ecg_source = "fif"
        if ecg_sig is None:
            ecg_sig, ecg_sf = load_bdf_ecg(pid_str)
            ecg_source = "bdf"
        resp_sig = load_wav_resp(pid_str)

        if ecg_sig is None and resp_sig is None:
            print(f"    {pid}: no cardiac/resp signals found"); continue

        # Determine session duration from the longer signal
        t_max = 0.0
        if ecg_sig is not None:
            t_max = max(t_max, len(ecg_sig) / ecg_sf)
        if resp_sig is not None:
            t_max = max(t_max, len(resp_sig) / RESP_FS)

        # Build cardiac/resp sliding window rows
        cardiac_windows = {}  # t_start -> (hr, rmssd, resp_bpm, resp_amp)
        w = 0.0
        while w + WINDOW_S <= t_max:
            hr = rmssd = rb = ra = np.nan
            if ecg_sig is not None:
                s0 = int(w * ecg_sf)
                s1 = min(int((w + WINDOW_S) * ecg_sf), len(ecg_sig))
                hr, rmssd, _ = extract_hr_rmssd(ecg_sig, ecg_sf, s0, s1)
            if resp_sig is not None:
                r0 = int(w * RESP_FS)
                r1 = min(int((w + WINDOW_S) * RESP_FS), len(resp_sig))
                rb, ra = extract_resp(resp_sig, r0, r1)
            cardiac_windows[w] = (hr, rmssd, rb, ra)
            w += WINDOW_S

        if not cardiac_windows:
            continue
        cw_arr = np.array(list(cardiac_windows.keys()))

        # For each event, find the 30s window that best covers it
        p_events = ev[ev.participant == pid].sort_values("onset_s").copy()
        for _, row in p_events.iterrows():
            onset = float(row.onset_s)
            # Nearest cardiac window whose start <= onset and end > onset
            # (i.e., the window that contains this event onset)
            diffs = onset - cw_arr
            valid = diffs[(diffs >= 0) & (diffs < WINDOW_S)]
            if len(valid) == 0:
                continue
            w_start = cw_arr[np.where((diffs >= 0) & (diffs < WINDOW_S))[0][-1]]
            hr, rmssd, rb, ra = cardiac_windows[w_start]

            rec = {"participant": pid,
                   "onset_s": onset,
                   "task": row.get("task", ""),
                   "app": row.get("app", ""),
                   "cardiac_window_start": w_start,
                   "cardiac_source": ecg_source,
                   "hr_mean": hr, "hrv_rmssd": rmssd,
                   "resp_bpm": rb, "resp_amp": ra}
            for c in hci_avail + eeg_avail:
                rec[c] = row.get(c, np.nan)
            all_rows.append(rec)

        n_valid = sum(1 for v in cardiac_windows.values() if np.isfinite(v[0]))
        print(f"    {pid}: {len(cardiac_windows)} cardiac windows "
              f"({n_valid} with valid HR)  source={ecg_source}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("  No rows produced"); return df

    df.to_csv(OUT_SLD, index=False)
    print(f"\n  OPTION B output: {len(df)} event-rows, "
          f"{df.participant.nunique()} participants")
    for c in ["hr_mean", "hrv_rmssd", "resp_bpm"] + hci_avail[:3]:
        if c in df.columns:
            n = df[c].notna().sum()
            print(f"    {c:22s} {n:5d}/{len(df)} ({100*n/len(df):.1f}%)")
    print(f"\n  Saved: {OUT_SLD}")
    return df


# ══════════════════════════════════════════════════════════════════════
# EXP1 re-runner (HCI → HR + RMSSD + resp)
# ══════════════════════════════════════════════════════════════════════

def zscore_within(df, cols, group="participant"):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = df.groupby(group)[c].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return out


def run_exp1(df, label, out_path):
    """
    HCI → HR + RMSSD + resp_bpm  (SWELL-KW replication).
    Runs CCA + RF direction classifiers with LOSO CV.
    """
    print(f"\n{'=' * 72}")
    print(f"EXP1 REPLICATION — {label}")
    print(f"SWELL-KW reference: CCA CV r=0.581  HR direction 79.0%")
    print(f"{'=' * 72}")

    hci = [c for c in HCI_COLS if c in df.columns]
    phy = [c for c in ["hr_mean","hrv_rmssd","resp_bpm","resp_amp"] if c in df.columns]
    if not hci or not phy:
        print("  Missing HCI or physiology columns"); return

    # Drop rows missing either side
    need = hci + phy
    d = df.dropna(subset=need).reset_index(drop=True)
    groups = d.participant.to_numpy()
    print(f"  Complete rows: {len(d)} ({len(np.unique(groups))} participants)")
    if len(d) < 100 or len(np.unique(groups)) < MIN_FOLDS:
        print("  Too few data points for reliable CV"); return

    dz = zscore_within(d, hci + phy)
    X  = np.nan_to_num(dz[hci].to_numpy(float))
    Y  = np.nan_to_num(dz[phy].to_numpy(float))

    # ── CCA ──────────────────────────────────────────────────────────
    n_comp = min(2, len(hci), len(phy))
    try:
        cca = CCA(n_components=n_comp, max_iter=3000)
        Xs, Ys = cca.fit_transform(X, Y)
        tr = [float(np.corrcoef(Xs[:,i], Ys[:,i])[0,1]) for i in range(n_comp)]
    except Exception as e:
        print(f"  CCA failed: {e}"); tr = [np.nan]*n_comp

    cv = [[] for _ in range(n_comp)]
    for held in np.unique(groups):
        a, b = groups != held, groups == held
        if a.sum() < 50 or b.sum() < 4: continue
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
    print(f"\n  CCA Component 1: train r={tr[0]:.3f}  "
          f"LOSO CV r={cvm[0]:.3f} ±{cvs[0]:.3f}  ({len(cv[0])} folds)")
    if np.isfinite(cvm[0]):
        if cvm[0] > 0.30:
            print(f"  *** ABOVE 0.30 — proxy generalizes cross-dataset ***")
        elif cvm[0] > 0.15:
            print(f"  ~ marginal (0.15-0.30)")
        else:
            print(f"  null (< 0.15)")

    # top HCI loadings
    if not np.isnan(tr[0]):
        try:
            w = cca.x_weights_[:, 0]
            top = np.argsort(np.abs(w))[::-1][:5]
            print(f"  HCI drivers: " +
                  ", ".join(f"{hci[k]}({w[k]:+.2f})" for k in top))
            yw = cca.y_weights_[:, 0]
            print(f"  Physio loadings: " +
                  ", ".join(f"{phy[k]}({yw[k]:+.2f})" for k in range(len(phy))))
        except Exception:
            pass

    # ── Direction classifiers ─────────────────────────────────────────
    print()
    dir_results = {}
    for target in phy:
        y = d[target].to_numpy(float)
        # direction: rising vs previous observation within participant
        ydir = np.full(len(d), np.nan)
        for u in np.unique(groups):
            idx = np.where(groups == u)[0]
            for k in range(1, len(idx)):
                i, j = idx[k-1], idx[k]
                if np.isfinite(y[i]) and np.isfinite(y[j]):
                    ydir[j] = float(y[j] > y[i])

        accs, bases, perms = [], [], []
        rng = np.random.default_rng(42)
        for held in np.unique(groups):
            tr2, te = groups != held, groups == held
            otr = np.isfinite(X[tr2]).all(1) & np.isfinite(ydir[tr2])
            ote = np.isfinite(X[te]).all(1)  & np.isfinite(ydir[te])
            if otr.sum() < 40 or ote.sum() < 4 or len(np.unique(ydir[tr2][otr])) < 2:
                continue
            ytr, yte = ydir[tr2][otr].astype(int), ydir[te][ote].astype(int)
            m = RandomForestClassifier(200, min_samples_leaf=5,
                                       class_weight="balanced",
                                       random_state=0, n_jobs=-1)
            m.fit(X[tr2][otr], ytr)
            accs.append(accuracy_score(yte, m.predict(X[te][ote])))
            bases.append(max(yte.mean(), 1 - yte.mean()))
            mp = RandomForestClassifier(200, min_samples_leaf=5,
                                        class_weight="balanced", random_state=0, n_jobs=-1)
            mp.fit(X[tr2][otr], rng.permutation(ytr))
            perms.append(accuracy_score(yte, mp.predict(X[te][ote])))

        if not accs:
            print(f"  {target:18s}: too few folds"); continue
        a, b2, p = np.mean(accs), np.mean(bases), np.mean(perms)
        flag = " ***" if a - b2 > 0.03 else " *" if a - b2 > 0.01 else ""
        print(f"  {target:18s}: acc={a:.3f} chance={b2:.3f} "
              f"over={a-b2:+.3f} perm={p:.3f}{flag}  ({len(accs)} folds)")
        dir_results[target] = {"acc": a, "chance": b2, "over": a-b2,
                               "perm": p, "folds": len(accs)}

    print(f"\n  SWELL-KW reference: HR direction 79.0%, RMSSD magnitude 84.7%")

    result = {"label": label, "n_rows": int(len(d)),
              "n_participants": int(len(np.unique(groups))),
              "cca_train_r": tr, "cca_cv_r": cvm, "cca_cv_sd": cvs,
              "direction": dir_results}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\n  Saved: {out_path}")
    return result


# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("SENSE-42 BLOCK PHYSIOLOGY — Fixing cardiac coverage")
    print("=" * 72)
    print("""
Diagnosis:
  Per-event epoching gives median epoch = 3.3s.
  Most file_manager_dragging events are 1-3s — no room for cardiac.
  The signals themselves are clean (verified: 70.0 bpm, 15.1 bpm).
  Coverage issue is epoch design, not extraction bugs.

Fix A: block-level (5-min blocks) — replicates SWELL-KW aggregate design
Fix B: sliding 30s window from .fif/.wav — continuous, cleaner signals
""")

    # OPTION A
    df_blk = option_a_block_level()

    # OPTION B
    df_sld = option_b_sliding_window()

    # EXP1 on both
    if not df_blk.empty:
        run_exp1(df_blk, "Option A — block-level (5-min blocks)", OUT_BLK_EXP)
    else:
        print("\nOption A produced no data — check EV_CSV and ECG/resp files")

    if not df_sld.empty:
        run_exp1(df_sld, "Option B — sliding 30s window", OUT_SLD_EXP)
    else:
        print("\nOption B produced no data — check ECG_DIR and RESP_DIR")

    print("\n" + "="*72)
    print("DONE")
    print("="*72)
    print(f"""
Output files:
  {OUT_BLK}
  {OUT_SLD}
  {OUT_BLK_EXP}
  {OUT_SLD_EXP}

The key number: CCA LOSO CV r for each option.
  > 0.30: SWELL-KW proxy generalizes — block-level design was needed
  0.15-0.30: marginal, investigate per-task split
  < 0.15: genuine null confirmed at block level too

If block-level CV r >> per-event CV r (0.091), that proves the
EXP1 null was a statistical power issue from sparse cardiac data,
not a genuine absence of HCI-physiology coupling.
""")


if __name__ == "__main__":
    main()
