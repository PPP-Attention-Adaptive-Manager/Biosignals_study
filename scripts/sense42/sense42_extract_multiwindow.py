"""
sense42_extract_multiwindow.py
===============================
Re-extracts ALL SENSE-42 features at 1s, 2s and 4s windows.

WHY THIS EXISTS:
  Every previous SENSE-42 result used 30-second windows. That length was
  chosen because RMSSD needs ~30 heartbeats to stabilise -- a CARDIAC
  constraint. But it was then applied to EEG as well, where frontal theta
  bursts during working memory last 1-2 seconds. Averaging band power
  over 30s flattens exactly the dynamics that carry cognitive signal.
  One modality's requirement was allowed to dictate the window for all
  modalities, and that was a design error.

  An earlier attempt computed 2s sub-epochs and took the MEDIAN across
  them. That is a robustness fix against artifact spikes. It is NOT a
  resolution fix -- it still collapses to one scalar per 30s.

WHAT CHANGES AT SHORT WINDOWS:
  Some features simply cannot be computed and are dropped rather than
  faked:

    hrv_rmssd   needs ~30 beats  -> IMPOSSIBLE below ~25s. DROPPED.
    resp_bpm    needs >=3 breaths -> at 4s you get ~0.25 breaths. DROPPED.

  They are replaced by instantaneous equivalents that are valid at any
  resolution:

    hr_inst     RR series interpolated to a continuous curve, sampled at
                the window centre. Standard practice in HRV work.
    rr_local    local RR interval at window centre (inverse of hr_inst,
                kept separately because trees split differently on it)
    hr_slope    change in instantaneous HR across the window
    resp_amp    respiratory amplitude at window centre (continuous 32 Hz)
    resp_phase_sin / resp_phase_cos
                Hilbert phase of the respiratory cycle, sin/cos encoded
                so the model sees it as circular rather than a jump from
                2*pi back to 0. Inhalation vs exhalation phase is exactly
                the variable RSA predicts should matter.
    resp_slope  respiratory amplitude change across the window

FREQUENCY RESOLUTION CAVEAT (important, do not ignore):
  Welch frequency resolution = 1 / window_length.
      4s -> 0.25 Hz : theta (4-8 Hz) spans 16 bins.  Reliable.
      2s -> 0.50 Hz : theta spans 8 bins.            Reliable.
      1s -> 1.00 Hz : theta spans 4 bins.            Marginal.
                      delta (1-4 Hz) spans 3 bins and needs 2-3 cycles
                      of a 1 Hz oscillation, which does not fit in 1s.
                      occipital_delta at 1s is UNRELIABLE and is flagged
                      as such in the output.
  This is a hard physical limit, not an implementation choice. Report 1s
  delta results with that caveat or exclude them.

METHOD:
  EEG band powers come from a single scipy spectrogram call per window
  size per participant (non-overlapping segments), rather than looping
  Welch thousands of times. Same result, far faster.

  Band power is NOT linear under aggregation, so 2s is computed directly
  from the signal rather than by averaging two 1s estimates.

MEMORY:
  One participant at a time. EEG .set extracted from the zip, all three
  window sizes computed in one pass, .set deleted. Never more than
  ~1.4 GB on disk. Resumable -- rows already present in the output CSVs
  are skipped on restart.

Run from: ~/biosignals_data/
Output:
  outputs/sense42_feat_1s.csv
  outputs/sense42_feat_2s.csv
  outputs/sense42_feat_4s.csv
"""
from __future__ import annotations
import os, sys, glob, warnings
import numpy as np
import pandas as pd
from scipy.signal import (spectrogram, find_peaks, butter, filtfilt, hilbert)
from scipy.io import wavfile
from scipy.interpolate import interp1d
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

BASE     = os.path.expanduser("~/biosignals_data")
SENSE    = os.path.join(BASE, "data", "sense_42")
CSV_DIR  = os.path.join(SENSE, "Behavioural", "CSV")
ECG_DIR  = os.path.join(SENSE, "ECG")
RESP_DIR = os.path.join(SENSE, "Respiration")
EEG_DIR  = os.path.join(SENSE, "EEG_cleaned")
EEG_ZIP  = os.path.join(SENSE, "EEG_cleaned.zip")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOWS  = [1.0, 2.0, 4.0]
KEEP_SET = False

FRONTAL   = ['AF3', 'F3', 'Fz', 'F4', 'AF4']
POSTERIOR = ['P3', 'Pz', 'P4']
OCCIPITAL = ['O1', 'Oz', 'O2']

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

sys.path.insert(0, os.path.join(BASE, "scripts", "sense42"))
from sense42_phase_a import extract_hci_from_csv   # reused for event parsing


# ══════════════════════════════════════════════════════════════════════
# HCI -- reuse the list-based event parser, then bin at arbitrary width
# ══════════════════════════════════════════════════════════════════════

def hci_events_from_csv(csv_path: str):
    """
    Re-derive the raw event lists that extract_hci_from_csv() builds
    internally, so they can be binned at any window width.
    Returns (keystroke_events, mouse_events, task_timeline).
    """
    import ast as _ast
    df = pd.read_csv(csv_path, low_memory=False)
    cols = df.columns.tolist()
    FRAME_RATE = 60.0

    KB_MAP = {
        'calibration_typing_key.keys':                     'calibration_typing.started',
        'mail.mail_content_user_key_release.keys':         'mail_content.started',
        'notes.notes_repeat_keyboard.keys':                'notes_repeat.started',
        'browser.browser_navigation_user_key_release.keys':'browser_navigation.started',
    }

    keystrokes = []
    for kb_col, started_col in KB_MAP.items():
        if kb_col not in cols or started_col not in cols:
            continue
        rt_col = kb_col.replace('.keys', '.rt')
        if rt_col not in cols:
            continue
        for _, row in df.iterrows():
            kv = str(row.get(kb_col, '')); rv = str(row.get(rt_col, ''))
            st = pd.to_numeric(row.get(started_col), errors='coerce')
            if kv in ('', 'nan', 'None', '[]') or not np.isfinite(st):
                continue
            try:
                ks, rts = _ast.literal_eval(kv), _ast.literal_eval(rv)
                if isinstance(ks, list) and isinstance(rts, list):
                    for k, rt in zip(ks, rts):
                        keystrokes.append((float(st) + float(rt), str(k)))
            except Exception:
                pass

    mouse = []
    for xc in [c for c in cols if c.endswith('.x') and 'mouse' in c.lower()]:
        prefix, yc, lbc = xc[:-2], xc[:-2] + '.y', xc[:-2] + '.leftButton'
        sc = next((s for s in [prefix + '.started',
                               '.'.join(prefix.split('.')[1:]) + '.started']
                   if s in cols), None)
        if sc is None:
            continue
        for _, row in df.iterrows():
            st = pd.to_numeric(row.get(sc), errors='coerce')
            xv = str(row.get(xc, ''))
            if xv in ('', 'nan', 'None', '[]') or not np.isfinite(st):
                continue
            try:
                xs = _ast.literal_eval(xv)
                ys = _ast.literal_eval(str(row.get(yc, '[]'))) if yc in cols else [0.] * len(xs)
                lb = _ast.literal_eval(str(row.get(lbc, '[]'))) if lbc in cols else [0] * len(xs)
                n = min(len(xs), len(ys), len(lb))
                for i in range(n):
                    mouse.append((float(st) + i / FRAME_RATE,
                                  float(xs[i]), float(ys[i]), int(lb[i])))
            except Exception:
                pass

    TASK_MAP = {
        'mail_homescreen': 'mail', 'mail_notification': 'mail', 'mail_content': 'mail',
        'file_manager_homescreen': 'file_mgr', 'file_manager_dragging': 'file_mgr',
        'file_manager_opening': 'file_mgr',
        'trash_bin_homescreen': 'trash', 'trash_bin_select': 'trash',
        'trash_bin_confirm': 'trash',
        'notes_homescreen': 'notes', 'notes_repeat': 'notes',
        'browser_homescreen': 'browser', 'browser_navigation': 'browser',
        'browser_content': 'browser',
    }
    timeline = []
    for task, app in TASK_MAP.items():
        sc = task + '.started'
        if sc in cols:
            for v in pd.to_numeric(df[sc], errors='coerce').dropna():
                timeline.append((float(v), app))
    timeline.sort()

    keystrokes.sort(); mouse.sort()
    return keystrokes, mouse, timeline


def bin_hci(keystrokes, mouse, timeline, win: float,
            t0: float, t1: float) -> pd.DataFrame:
    """Bin pre-parsed HCI events into windows of width `win`."""
    SPECIAL = {'backspace','delete','lshift','rshift','lctrl','rctrl','lalt',
               'ralt','tab','escape','return','enter','caps_lock','lsuper',
               'comma','period','semicolon','slash','backslash','quote'}
    DIRECT  = {'left','right','up','down','pageup','pagedown','home','end'}

    kt = np.array([t for t, _ in keystrokes]) if keystrokes else np.empty(0)
    kk = [k for _, k in keystrokes]
    mt = np.array([t for t, _, _, _ in mouse]) if mouse else np.empty(0)
    mx = np.array([x for _, x, _, _ in mouse]) if mouse else np.empty(0)
    my = np.array([y for _, _, y, _ in mouse]) if mouse else np.empty(0)
    ml = np.array([l for _, _, _, l in mouse]) if mouse else np.empty(0)
    tl_t = np.array([t for t, _ in timeline]) if timeline else np.empty(0)

    def app_at(t):
        if len(tl_t) == 0:
            return 'unknown'
        i = np.searchsorted(tl_t, t, side='right') - 1
        return timeline[i][1] if i >= 0 else 'unknown'

    rows, w = [], t0
    while w + win <= t1:
        we = w + win
        ki = np.where((kt >= w) & (kt < we))[0]
        keys = [kk[i] for i in ki]
        total = len(keys)
        back  = sum(1 for k in keys if k in ('backspace', 'delete'))
        space = sum(1 for k in keys if k == 'space')
        dire  = sum(1 for k in keys if k in DIRECT)
        spec  = sum(1 for k in keys if k in SPECIAL)
        prnt  = sum(1 for k in keys if k not in SPECIAL and k not in DIRECT and len(k) == 1)

        mi = np.where((mt >= w) & (mt < we))[0]
        dist = clicks = act = 0.0
        if len(mi) > 1:
            dx, dy = np.diff(mx[mi]), np.diff(my[mi])
            lb = ml[mi]
            dist   = float(np.sqrt(dx**2 + dy**2).sum())
            clicks = int(((lb[1:] == 1) & (lb[:-1] == 0)).sum())
            act    = float(((np.abs(dx) > 0.001) | (np.abs(dy) > 0.001)).mean())

        rows.append({
            "window_start": w,
            "task_type": app_at(w),
            "SnKeyStrokes": total, "SnChars": prnt, "SnSpecialKeys": spec,
            "SnDirectionKeys": dire, "SnErrorKeys": back, "SnSpaces": space,
            "CharactersRatio": prnt / max(total, 1),
            "ErrorKeyRatio":   back / max(total, 1),
            "SnLeftClicked": clicks, "SnMouseDistance": dist, "SnMouseAct": act,
            "SnAppChange": int(((tl_t >= w) & (tl_t < we)).sum()),
        })
        w += win
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Cardiac -- instantaneous, valid at any window width
# ══════════════════════════════════════════════════════════════════════

def cardiac_series(fif_path: str):
    """
    Returns (t_grid, hr_inst) -- instantaneous heart rate on a 10 Hz grid.

    RMSSD is deliberately NOT computed. It requires ~30 successive beats
    for a stable estimate; at 1-4s windows there are 1-7 beats. Reporting
    RMSSD at those widths would be a fabricated number.
    """
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sf  = raw.info["sfreq"]
    chs = [c for c in raw.ch_names if "ECG" in c.upper()] or raw.ch_names[:1]
    sig = raw.get_data(picks=[chs[0]])[0]
    t   = raw.times

    # bandpass is essential: raw BioSemi ECG baseline drift swamps R-peaks
    b, a = butter(3, [0.5 / (sf / 2), 40.0 / (sf / 2)], btype="band")
    z = filtfilt(b, a, sig)
    z = (z - z.mean()) / (z.std() + 1e-9)
    peaks, _ = find_peaks(z, distance=int(0.35 * sf), height=3.0)
    if len(peaks) < 50:
        return None, None

    pt = t[peaks]
    rr = np.diff(pt)
    ok = (rr > 0.33) & (rr < 1.5)
    if ok.sum() < 30:
        return None, None
    rr_t, rr_v = pt[1:][ok], rr[ok]

    grid = np.arange(t[0], t[-1], 0.1)          # 10 Hz
    f = interp1d(rr_t, 60.0 / rr_v, kind="linear",
                 bounds_error=False, fill_value=np.nan)
    return grid, f(grid)


def respiration_series(wav_path: str):
    """
    Returns (t, amplitude, phase_sin, phase_cos).

    resp_bpm is deliberately NOT computed -- at 4s you get ~0.25 breaths.
    Instead the continuous amplitude and the Hilbert phase are used.
    Phase is sin/cos encoded so the model sees it as circular; a raw
    phase would jump from 2*pi to 0 mid-cycle and trees would split on
    that discontinuity.
    """
    fs, sig = wavfile.read(wav_path)
    if sig.ndim > 1:
        sig = sig[:, 0]
    sig = sig.astype(float)
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    t = np.arange(len(sig)) / fs

    b, a = butter(3, [0.05 / (fs / 2), 1.0 / (fs / 2)], btype="band")
    filt = filtfilt(b, a, sig)
    an = hilbert(filt)
    return t, np.abs(an), np.sin(np.angle(an)), np.cos(np.angle(an))


# ══════════════════════════════════════════════════════════════════════
# EEG -- spectrogram once per window size
# ══════════════════════════════════════════════════════════════════════

def eeg_bandpowers(set_path: str, win: float) -> pd.DataFrame:
    """
    One spectrogram call, non-overlapping segments of width `win`.
    Band power is not linear under aggregation, so each window size is
    computed directly from the signal rather than by averaging finer bins.
    """
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    sf   = raw.info["sfreq"]
    data = raw.get_data()

    def idx(names):
        return [raw.ch_names.index(c) for c in names if c in raw.ch_names]
    i_f, i_p, i_o = idx(FRONTAL), idx(POSTERIOR), idx(OCCIPITAL)
    if not i_f:
        return pd.DataFrame()

    nper = int(win * sf)
    if nper < 8:
        return pd.DataFrame()

    f, tt, Sxx = spectrogram(data, fs=sf, nperseg=nper, noverlap=0, axis=-1)
    # Sxx: (n_channels, n_freqs, n_times)

    bp = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp[name] = Sxx[:, m, :].mean(axis=1) if m.sum() else np.zeros(Sxx.shape[::2])
    m_bb = (f >= 1) & (f < 40)
    bp["broadband"] = Sxx[:, m_bb, :].mean(axis=1)

    ft = bp["theta"][i_f].mean(0)
    fa = bp["alpha"][i_f].mean(0)
    fb = bp["beta"][i_f].mean(0)
    pa = bp["alpha"][i_p].mean(0) if i_p else fa
    od = bp["delta"][i_o].mean(0) if i_o else bp["delta"].mean(0)

    return pd.DataFrame({
        "window_start":        tt - win / 2.0,   # spectrogram t = segment centre
        "frontal_theta":       ft,
        "frontal_alpha":       fa,
        "theta_alpha_ratio":   ft / (fa + 1e-15),
        "engagement_index":    fb / (fa + ft + 1e-15),
        "posterior_alpha":     pa,
        "occipital_delta":     od,
        "broadband_amplitude": bp["broadband"].mean(0),
    })


# ══════════════════════════════════════════════════════════════════════
# Alignment
# ══════════════════════════════════════════════════════════════════════

def align_nearest(hci_starts, other: pd.DataFrame, cols, win):
    """
    Clock offset correction. PsychoPy starts 35-260s after BioSemi, so
    the offset is estimated from the first window of each stream and the
    nearest match within half a window is taken.
    """
    out = {c: np.full(len(hci_starts), np.nan) for c in cols}
    if other is None or other.empty or len(other) < 5:
        return out
    ot = other["window_start"].to_numpy(float)
    off = hci_starts[0] - ot[0]
    vals = {c: other[c].to_numpy(float) for c in cols}
    for i, ht in enumerate(hci_starts):
        j = int(np.argmin(np.abs(ot - (ht - off))))
        if abs(ot[j] - (ht - off)) < win / 2:
            for c in cols:
                out[c][i] = vals[c][j]
    return out


def sample_series(t_grid, values, centres, off):
    """Sample a continuous series at window centres (offset-corrected)."""
    out = np.full(len(centres), np.nan)
    if t_grid is None or values is None:
        return out
    for i, c in enumerate(centres):
        j = int(np.argmin(np.abs(t_grid - (c - off))))
        if abs(t_grid[j] - (c - off)) < 1.0:
            out[i] = values[j]
    return out


# ══════════════════════════════════════════════════════════════════════

EEG_COLS = ["frontal_theta", "frontal_alpha", "theta_alpha_ratio",
            "engagement_index", "posterior_alpha",
            "occipital_delta", "broadband_amplitude"]


def process_participant(pid: str, done: dict):
    """Extract all three window sizes for one participant in a single pass."""
    todo = [w for w in WINDOWS if f"P{pid}" not in done.get(w, set())]
    if not todo:
        print("    already done"); return {}

    csvs = glob.glob(os.path.join(CSV_DIR, f"{pid}_*.csv"))
    if not csvs:
        print("    no CSV"); return {}

    print("    parsing HCI events...", end="", flush=True)
    keystrokes, mouse, timeline = hci_events_from_csv(csvs[0])
    if not keystrokes and not mouse:
        print(" none"); return {}
    all_t = ([t for t, _ in keystrokes] + [t for t, _, _, _ in mouse])
    t0, t1 = min(all_t), max(all_t)
    print(f" {len(keystrokes)} keys, {len(mouse)} mouse frames")

    ecg_p  = os.path.join(ECG_DIR,  f"P{pid}.fif")
    resp_p = os.path.join(RESP_DIR, f"P{pid}.wav")
    hr_t, hr_v = cardiac_series(ecg_p) if os.path.isfile(ecg_p) else (None, None)
    if os.path.isfile(resp_p):
        rs_t, rs_a, rs_s, rs_c = respiration_series(resp_p)
    else:
        rs_t = rs_a = rs_s = rs_c = None
    print(f"    cardiac {'ok' if hr_t is not None else 'MISSING'} | "
          f"resp {'ok' if rs_t is not None else 'MISSING'}")

    set_p = os.path.join(EEG_DIR, f"P{pid}_100Hz_downsampled.set")
    tmp = False
    if not os.path.isfile(set_p):
        rc = os.system(f'unzip -q -o "{EEG_ZIP}" '
                       f'"EEG_cleaned/P{pid}_100Hz_downsampled.set" '
                       f'-d "{SENSE}" 2>/dev/null')
        tmp = (rc == 0 and os.path.isfile(set_p))
        if not tmp:
            print("    EEG extract failed"); return {}

    out = {}
    try:
        for win in todo:
            hci = bin_hci(keystrokes, mouse, timeline, win, t0, t1)
            if hci.empty:
                continue
            eeg = eeg_bandpowers(set_p, win)
            if eeg.empty:
                continue

            hs = hci["window_start"].to_numpy(float)
            centres = hs + win / 2.0
            df = hci.copy()
            df["participant"] = f"P{pid}"

            for c, v in align_nearest(hs, eeg, EEG_COLS, win).items():
                df[c] = v

            off_c = centres[0] - hr_t[0] if hr_t is not None else 0.0
            hr = sample_series(hr_t, hr_v, centres, off_c)
            df["hr_inst"]  = hr
            df["rr_local"] = 60.0 / np.where(np.isfinite(hr) & (hr > 0), hr, np.nan)
            df["hr_slope"] = np.concatenate([[np.nan], np.diff(hr)])

            off_r = centres[0] - rs_t[0] if rs_t is not None else 0.0
            amp = sample_series(rs_t, rs_a, centres, off_r)
            df["resp_amp"]       = amp
            df["resp_phase_sin"] = sample_series(rs_t, rs_s, centres, off_r)
            df["resp_phase_cos"] = sample_series(rs_t, rs_c, centres, off_r)
            df["resp_slope"]     = np.concatenate([[np.nan], np.diff(amp)])

            full = df[EEG_COLS + ["hr_inst", "resp_amp"]].notna().all(axis=1).sum()
            print(f"    win={win}s -> {len(df):6d} windows | complete {full:6d}")
            out[win] = df
    finally:
        if tmp and not KEEP_SET and os.path.isfile(set_p):
            os.remove(set_p)
    return out


def main():
    print("=" * 78)
    print("SENSE-42 MULTI-WINDOW EXTRACTION  (1s / 2s / 4s)")
    print("=" * 78)
    print("Replaces the 30s pipeline. RMSSD and resp_bpm are dropped -- they")
    print("cannot be computed at these widths -- and replaced by instantaneous")
    print("HR, HR slope, respiratory amplitude/phase/slope.\n")
    print("Frequency-resolution caveat: at 1s, Welch resolution is 1 Hz, so")
    print("occipital_delta (1-4 Hz) is UNRELIABLE. Theta at 1s spans only 4")
    print("bins. Treat 1s spectral results with caution; 2s and 4s are sound.\n")

    paths = {w: os.path.join(OUT_DIR, f"sense42_feat_{int(w)}s.csv") for w in WINDOWS}
    done, frames = {}, {}
    for w, p in paths.items():
        if os.path.isfile(p):
            d = pd.read_csv(p)
            frames[w] = d
            done[w] = set(d["participant"].unique())
            print(f"  {int(w)}s cache: {len(d)} rows, {len(done[w])} participants")
        else:
            frames[w], done[w] = pd.DataFrame(), set()

    pids = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(os.path.join(CSV_DIR, "*.csv"))})
    print(f"\nParticipants: {len(pids)}\n")

    for i, pid in enumerate(pids, 1):
        print(f"[{i:2d}/{len(pids)}] P{pid}")
        try:
            res = process_participant(pid, done)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        for w, df in res.items():
            frames[w] = pd.concat([frames[w], df], ignore_index=True) \
                        if len(frames[w]) else df
            frames[w].to_csv(paths[w], index=False)   # save after each

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    for w in WINDOWS:
        d = frames[w]
        if len(d):
            print(f"  {int(w)}s: {len(d):7d} rows, "
                  f"{d.participant.nunique()} participants  -> {paths[w]}")
    print("\nNext: python scripts/sense42/sense42_analyse_multiwindow.py")


if __name__ == "__main__":
    main()
