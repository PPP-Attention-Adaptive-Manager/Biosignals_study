"""
biosignal_proxy/biosignal_targets.py
=====================================
Target-side feature extractor for the biosignal proxy. Companion to
hci_features.py — it windows on the SAME grid (pass the `starts` array that
hci_features.extract_session returned) so X[i] and Y[i] describe the same 30s.

Per window it produces a ~12-dim biosignal feature vector:
  acc_movement, acc_jerk,
  hr_mean, hrv_rmssd,
  eda_tonic_slope, eda_phasic_count,
  resp_bpm,
  eeg_theta_alpha, eeg_engagement, eeg_alpha_asym,
  fnirs_hbo_slope_L, fnirs_hbo_slope_R

Honest limitations baked in as comments:
  * RMSSD at 100 Hz is coarse (10 ms RR resolution) — parabolic peak refine helps a little.
  * resp_bpm over 30 s sees only ~6-10 breaths -> low resolution.
  * EEG frontal channels are blink/EMG contaminated; extreme windows -> NaN.
  * fNIRS HbO is an APPROXIMATE log-ratio proxy, not full Beer-Lambert.

Windows that can't be computed (too few samples, artifact) return np.nan for that
signal's features; handle (drop / impute) when assembling the dataset.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, butter, filtfilt, detrend

TARGET_NAMES = [
    "acc_movement", "acc_jerk",
    "hr_mean", "hrv_rmssd",
    "eda_tonic_slope", "eda_phasic_count",
    "resp_bpm",
    "eeg_theta_alpha", "eeg_engagement", "eeg_alpha_asym",
    "fnirs_hbo_slope_L", "fnirs_hbo_slope_R",
]

def load_signal(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+")          # whitespace-delimited (the EDA fix)

def _fs(t: np.ndarray) -> float:
    d = np.diff(t)
    return 1.0 / np.median(d) if len(d) else np.nan

def _slice(t, *arrs, ws, we):
    m = (t >= ws) & (t < we)
    return (t[m],) + tuple(a[m] for a in arrs)

# ------------------------------------------------------------------ per-signal features
def f_acc(t, x, y, z):
    if len(t) < 5: return [np.nan, np.nan]
    mag = np.sqrt(x**2 + y**2 + z**2)
    return [float(np.std(mag)), float(np.mean(np.abs(np.diff(mag))))]

def _parabolic_refine(v, peaks, t, dt):
    out = []
    for p in peaks:
        if 0 < p < len(v) - 1:
            y0, y1, y2 = v[p-1], v[p], v[p+1]
            denom = (y0 - 2*y1 + y2)
            off = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
            out.append(t[p] + off * dt)
        else:
            out.append(t[p])
    return np.array(out)

def f_ecg(t, ecg):
    if len(t) < 50: return [np.nan, np.nan]
    fs = _fs(t); dt = 1.0 / fs
    v = (ecg - ecg.mean()) / (ecg.std() + 1e-9)
    peaks, _ = find_peaks(v, distance=int(0.4 * fs), height=2.0)   # <150 bpm, >2 SD
    if len(peaks) < 15:                                            # reject sparse windows
        return [np.nan, np.nan]
    rt = _parabolic_refine(v, peaks, t, dt)
    rr = np.diff(rt)
    rr = rr[(rr > 0.33) & (rr < 1.5)]                             # 40-180 bpm plausibility
    if len(rr) < 10: return [np.nan, np.nan]
    hr   = 60.0 / np.median(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2)) * 1000.0
    return [float(hr), float(rmssd)]

def f_eda(t, eda):
    if len(t) < 50: return [np.nan, np.nan]
    fs = _fs(t)
    b, a = butter(2, 0.05 / (fs / 2), btype="low")
    tonic = filtfilt(b, a, eda)
    slope = np.polyfit(t - t[0], tonic, 1)[0]
    phasic = eda - tonic
    pz = (phasic - phasic.mean()) / (phasic.std() + 1e-9)
    pk, _ = find_peaks(pz, prominence=0.5, distance=int(1.0 * fs))   # SCRs
    return [float(slope), float(len(pk))]

def f_rip(t, rip):
    if len(t) < 50: return [np.nan]
    fs = _fs(t)
    v = detrend(rip)
    f, P = welch(v, fs=fs, nperseg=min(len(v), 1024))
    band = (f >= 0.1) & (f <= 0.6)
    if not band.any(): return [np.nan]
    return [float(60.0 * f[band][np.argmax(P[band])])]

def _bandpowers(v, fs):
    f, P = welch(v, fs=fs, nperseg=min(len(v), 512))
    bp = lambda lo, hi: float(P[(f >= lo) & (f < hi)].sum())   # df cancels in ratios
    return bp(4, 8), bp(8, 13), bp(13, 30)        # theta, alpha, beta

def f_eeg(t, af7, af8):
    if len(t) < 50: return [np.nan, np.nan, np.nan]
    fs = _fs(t)
    # artifact reject: drop window if either channel has an extreme excursion
    for ch in (af7, af8):
        if np.abs(ch - ch.mean()).max() > 8 * (ch.std() + 1e-9):
            return [np.nan, np.nan, np.nan]
    th7, al7, be7 = _bandpowers(af7, fs)
    th8, al8, be8 = _bandpowers(af8, fs)
    theta, alpha, beta = (th7+th8)/2, (al7+al8)/2, (be7+be8)/2
    theta_alpha = theta / (alpha + 1e-12)
    engagement  = beta / (alpha + theta + 1e-12)
    alpha_asym  = np.log(al8 + 1e-12) - np.log(al7 + 1e-12)        # AF8 - AF7
    return [float(theta_alpha), float(engagement), float(alpha_asym)]

def _hbo_slope(t, i_red, i_ir):
    # APPROXIMATE HbO proxy: optical-density contrast between wavelengths, detrended slope.
    od_red = -np.log(i_red / (i_red.mean() + 1e-12))
    od_ir  = -np.log(i_ir  / (i_ir.mean()  + 1e-12))
    hbo    = od_ir - od_red                                        # IR tracks HbO; contrast vs red
    return float(np.polyfit(t - t[0], hbo, 1)[0])

def f_fnirs(t, r7, ir7, r8, ir8):
    if len(t) < 20: return [np.nan, np.nan]
    return [_hbo_slope(t, r7, ir7), _hbo_slope(t, r8, ir8)]

# ----------------------------------------------------------------- session-level driver
def extract_targets_session(biosig_dir: str, sid: str, starts: np.ndarray,
                            window_s: float = 30.0) -> np.ndarray:
    """
    biosig_dir : path to a subject's Biosignals/ folder
    sid        : e.g. 'S1'
    starts     : the window-start array returned by hci_features.extract_session
    Returns Y of shape (len(starts), len(TARGET_NAMES)), rows aligned to `starts`.
    """
    p = lambda name: f"{biosig_dir}/D3_{sid}_{name}.txt"
    acc = load_signal(p("ACC"));  ecg = load_signal(p("ECG"))
    eda = load_signal(p("EDA"));  rip = load_signal(p("RIP"))
    eeg = load_signal(p("EEG"));  fni = load_signal(p("fNIRS"))

    A = (acc["Timestamp"].to_numpy(float), acc["ACC_x"].to_numpy(float),
         acc["ACC_y"].to_numpy(float), acc["ACC_z"].to_numpy(float))
    E = (ecg["Timestamp"].to_numpy(float), ecg["ECG"].to_numpy(float))
    D = (eda["Timestamp"].to_numpy(float), eda["EDA"].to_numpy(float))
    R = (rip["Timestamp"].to_numpy(float), rip["RIP"].to_numpy(float))
    G = (eeg["Timestamp"].to_numpy(float), eeg["EEG_AF7"].to_numpy(float),
         eeg["EEG_AF8"].to_numpy(float))
    N = (fni["Timestamp"].to_numpy(float),
         fni["fNIRS_red_AF7"].to_numpy(float), fni["fNIRS_infrared_AF7"].to_numpy(float),
         fni["fNIRS_red_AF8"].to_numpy(float), fni["fNIRS_infrared_AF8"].to_numpy(float))

    rows = []
    for ws in starts:
        we = ws + window_s
        ta, ax, ay, az = _slice(*A, ws=ws, we=we)
        te, ev         = _slice(*E, ws=ws, we=we)
        td, dv         = _slice(*D, ws=ws, we=we)
        tr, rv         = _slice(*R, ws=ws, we=we)
        tg, g7, g8     = _slice(*G, ws=ws, we=we)
        tn, n1, n2, n3, n4 = _slice(*N, ws=ws, we=we)
        row = (f_acc(ta, ax, ay, az) + f_ecg(te, ev) + f_eda(td, dv)
               + f_rip(tr, rv) + f_eeg(tg, g7, g8) + f_fnirs(tn, n1, n2, n3, n4))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


# ------------------------------------------------------------------ (X, Y) pair builder
def build_xy(subject_root: str, sid: str, source: str = "coglab",
             t_start=None, t_end=None, window_s=30.0, stride_s=15.0):
    """
    subject_root : path to one subject folder containing HCI/ and Biosignals/
    Returns (X, Y, starts, x_names, y_names) with X and Y row-aligned.
    Pass t_start/t_end = the PB 'Task' window so behavior and physiology share a crop.
    """
    import hci_features as H
    X, starts = H.extract_session(
        f"{subject_root}/HCI/D3_{sid}_mouse.csv",
        f"{subject_root}/HCI/D3_{sid}_keyboard.csv",
        source=source, t_start=t_start, t_end=t_end,
        window_s=window_s, stride_s=stride_s,
    )
    Y = extract_targets_session(f"{subject_root}/Biosignals", sid, starts, window_s)
    return X, Y, starts, H.FEATURE_NAMES, TARGET_NAMES