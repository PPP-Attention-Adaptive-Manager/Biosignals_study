"""
sense42_phase_a.py  (v2 — fixed for actual CSV structure)
============================================================
SENSE-42 Behavioural CSV is a PsychoPy TRIAL-LEVEL log,
NOT a 144Hz frame log. Each row = one trial or routine iteration.
Columns are sparse and component-specific.

Key corrections from v1:
  1. Time: use task `.started` timestamps, not `thisN`
  2. HCI: extract from sparse per-component columns, not unified mouse.x
  3. Resp model shape: 16 features (13 + HR + RMSSD + SCL), not 18
  4. ECG: handle missing P001 gracefully, try alternate naming
"""
from __future__ import annotations
import os, sys, glob, warnings
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score
import joblib
import mne

warnings.filterwarnings("ignore")

BASE      = os.path.expanduser("~/biosignals_data")
SENSE_DIR = os.path.join(BASE, "data", "sense_42")
SWELL_FILE= os.path.join(BASE, "data", "swell_kw",
                          "Behavioral-features - per minute.xlsx")
MODEL_DIR = os.path.join(BASE, "models")
OUT_CSV   = os.path.join(BASE, "outputs", "sense42_phase_a_results.csv")

CSV_DIR   = os.path.join(SENSE_DIR, "Behavioural", "CSV")
ECG_DIR   = os.path.join(SENSE_DIR, "ECG")
RESP_DIR  = os.path.join(SENSE_DIR, "Respiration")

WINDOW_S  = 30.0

HCI_COLS_SWELL = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]

# These 3 are structurally absent (always NaN) — remove for resp model input
ALWAYS_NAN = {"SnRightClicked", "SnDoubleClicked", "SnDragged"}
# Also remove SnAppChange + SnTabfocusChange since resp model was trained on
# Cog Lab which had no app switching → 13 cols total matching training schema
RESP_MODEL_EXCLUDE = ALWAYS_NAN | {"SnAppChange", "SnTabfocusChange"}
RESP_COLS_13 = [c for c in HCI_COLS_SWELL if c not in RESP_MODEL_EXCLUDE]
# = 13 cols + HR + RMSSD + SCL = 16 features → matches resp_mu shape (16,)

TASK_TYPES = [
    "browser_content", "mail_content", "mail_notification",
    "file_manager_dragging", "file_manager_opening",
    "notes_repeat", "trash_bin_select",
    "browser_homescreen", "mail_homescreen",
    "file_manager_homescreen", "notes_homescreen",
]


# ── Step 1: train SWELL-KW proxy (unchanged) ─────────────────────────

def train_swell_models():
    print("Training proxy models on SWELL-KW...")
    df = pd.read_excel(SWELL_FILE)
    df = df[df["Condition"].isin(["N","I","T"])].copy()
    for c in ["HR","RMSSD","SCL"]:
        df[c] = df[c].replace(999, np.nan)

    agg = df.groupby(["PP","Condition"])[HCI_COLS_SWELL+["HR","RMSSD","SCL"]].mean().reset_index()
    for col in HCI_COLS_SWELL + ["HR","RMSSD","SCL"]:
        agg[col+"_z"] = agg.groupby("PP")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))
    clean = agg[[c+"_z" for c in HCI_COLS_SWELL] +
                 [c+"_z" for c in ["HR","RMSSD","SCL"]]].dropna()
    Xcca = clean[[c+"_z" for c in HCI_COLS_SWELL]].to_numpy(float)
    Ycca = clean[[c+"_z" for c in ["HR","RMSSD","SCL"]]].to_numpy(float)
    cca_mu = Xcca.mean(0); cca_sd = Xcca.std(0) + 1e-9
    cca = CCA(n_components=1, max_iter=2000)
    cca.fit((Xcca - cca_mu) / cca_sd, Ycca)
    print(f"  CCA: {len(Xcca)} condition-level rows")

    df_s = df.sort_values(["PP","Condition"]).copy()
    for col in ["HR","RMSSD"]:
        df_s[col+"_delta"]  = df_s.groupby(["PP","Condition"])[col].diff()
        df_s[col+"_rising"] = (df_s[col+"_delta"] > 0).astype(float)
    HCI_DELTA = [c+"_delta" for c in HCI_COLS_SWELL]
    for col in HCI_COLS_SWELL:
        df_s[col+"_delta"] = df_s.groupby(["PP","Condition"])[col].diff()
    df_asp = df_s.dropna(subset=["HR_rising","RMSSD_rising"])
    X = np.nan_to_num(df_asp[HCI_DELTA].to_numpy(float))
    train_mu = X.mean(0); train_sd = X.std(0) + 1e-9
    Xz = (X - train_mu) / train_sd
    rf_hr = RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced",
                                    random_state=0, n_jobs=-1)
    rf_hr.fit(Xz, df_asp["HR_rising"].to_numpy(int))
    rf_rmssd = RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced",
                                       random_state=0, n_jobs=-1)
    rf_rmssd.fit(Xz, df_asp["RMSSD_rising"].to_numpy(int))
    print(f"  RF: {len(Xz)} windows")
    return cca, cca_mu, cca_sd, rf_hr, rf_rmssd, train_mu, train_sd


# ── Step 2: extract HCI from SENSE-42 trial-level CSV ────────────────

def inspect_csv(csv_path: str):
    """Print column structure to help understand the CSV format."""
    df = pd.read_csv(csv_path, nrows=5, low_memory=False)
    print(f"\n  CSV columns ({len(df.columns)} total):")
    # Group by prefix
    from collections import defaultdict
    groups = defaultdict(list)
    for c in df.columns:
        prefix = c.split('.')[0].split('_')[0]
        groups[prefix].append(c)
    for prefix, cols in sorted(groups.items()):
        if len(cols) <= 3:
            print(f"    {prefix}: {cols}")
        else:
            print(f"    {prefix}: {cols[:3]} ... ({len(cols)} total)")
    return df.columns.tolist()


def extract_hci_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Correct list-based extraction from SENSE-42 PsychoPy CSV.

    The CSV stores behavioral data as Python list strings per trial row:
      - keyboard: .keys col = "['a','backspace','return',...]"
                  .rt  col  = "[1.23, 1.45, ...]"  (relative to task.started)
      - mouse:    .x col   = "[-0.99, -0.98, ...]"  (per-frame at 60Hz)
                  .leftButton col = "[0, 0, 1, 1, 0, ...]" (per-frame)
    Absolute timestamp = task_component.started + rt[i] (keyboard)
                       = task_component.started + frame_i/60 (mouse)
    """
    import ast as _ast
    df = pd.read_csv(csv_path, low_memory=False)
    cols = df.columns.tolist()

    FRAME_RATE    = 60.0
    SPECIAL_KEYS  = {'backspace','delete','lshift','rshift','lctrl','rctrl','lalt',
                     'ralt','tab','escape','return','enter','caps_lock','lsuper',
                     'f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12',
                     'comma','period','semicolon','slash','backslash','quote'}
    DIRECTION_KEYS = {'left','right','up','down','pageup','pagedown','home','end'}

    # keyboard column → corresponding task.started column
    KB_MAP = {
        'calibration_typing_key.keys':                     'calibration_typing.started',
        'mail.mail_content_user_key_release.keys':         'mail_content.started',
        'notes.notes_repeat_keyboard.keys':                'notes_repeat.started',
        'browser.browser_navigation_user_key_release.keys':'browser_navigation.started',
    }

    # ── 1. Build keystroke event list ────────────────────────────────
    keystroke_events = []   # (abs_time, key_name)
    for kb_col, started_col in KB_MAP.items():
        if kb_col not in cols or started_col not in cols:
            continue
        rt_col = kb_col.replace('.keys', '.rt')
        if rt_col not in cols:
            continue
        for _, row in df.iterrows():
            keys_val = str(row.get(kb_col, ''))
            rt_val   = str(row.get(rt_col, ''))
            started  = pd.to_numeric(row.get(started_col), errors='coerce')
            if keys_val in ('','nan','None','[]') or not np.isfinite(started):
                continue
            try:
                keys = _ast.literal_eval(keys_val)
                rts  = _ast.literal_eval(rt_val)
                if isinstance(keys, list) and isinstance(rts, list):
                    for k, rt in zip(keys, rts):
                        keystroke_events.append((float(started) + float(rt), str(k)))
            except Exception:
                pass

    # ── 2. Build mouse event list (per-frame at 60Hz) ────────────────
    mouse_events = []   # (abs_time, x, y, leftButton)
    mouse_x_cols = [c for c in cols if c.endswith('.x') and 'mouse' in c.lower()]
    for xc in mouse_x_cols:
        prefix = xc[:-2]
        yc  = prefix + '.y'
        lbc = prefix + '.leftButton'
        # find started column: try exact prefix, then drop nested scope prefix
        sc = None
        for sc_try in [prefix + '.started',
                       '.'.join(prefix.split('.')[1:]) + '.started']:
            if sc_try in cols:
                sc = sc_try; break
        if sc is None:
            continue
        for _, row in df.iterrows():
            started = pd.to_numeric(row.get(sc), errors='coerce')
            x_val   = str(row.get(xc, ''))
            if x_val in ('','nan','None','[]') or not np.isfinite(started):
                continue
            try:
                xs  = _ast.literal_eval(x_val)
                ys  = _ast.literal_eval(str(row.get(yc,'[]'))) if yc in cols else [0.]*len(xs)
                lbs = _ast.literal_eval(str(row.get(lbc,'[]'))) if lbc in cols else [0]*len(xs)
                n   = min(len(xs), len(ys), len(lbs))
                for i in range(n):
                    t = float(started) + i / FRAME_RATE
                    mouse_events.append((t, float(xs[i]),
                                         float(ys[i]) if i<len(ys) else 0.,
                                         int(lbs[i])  if i<len(lbs) else 0))
            except Exception:
                pass

    if not keystroke_events and not mouse_events:
        print("    No events extracted — check column names")
        return pd.DataFrame()

    # ── 3. App switching timeline ─────────────────────────────────────
    TASK_MAP = {
        'mail_homescreen':'mail',   'mail_notification':'mail',
        'mail_content':'mail',
        'file_manager_homescreen':'file_mgr', 'file_manager_dragging':'file_mgr',
        'file_manager_opening':'file_mgr',
        'trash_bin_homescreen':'trash', 'trash_bin_select':'trash',
        'trash_bin_confirm':'trash',
        'notes_homescreen':'notes', 'notes_repeat':'notes',
        'browser_homescreen':'browser', 'browser_navigation':'browser',
        'browser_content':'browser',
    }
    task_timeline = []   # (abs_time, app)
    for task, app in TASK_MAP.items():
        sc = task + '.started'
        if sc not in cols: continue
        for v in pd.to_numeric(df[sc], errors='coerce').dropna():
            task_timeline.append((float(v), app))
    task_timeline.sort()

    def get_app(t):
        for ts, app in reversed(task_timeline):
            if ts <= t: return app
        return 'unknown'

    # ── 4. Window aggregation ─────────────────────────────────────────
    keystroke_events.sort()
    mouse_events.sort()
    all_t = [t for t,_ in keystroke_events] + [t for t,_,_,_ in mouse_events]
    if not all_t:
        return pd.DataFrame()
    t_start = min(all_t); t_end = max(all_t)

    rows_out = []
    w = t_start
    while w + WINDOW_S <= t_end:
        w_end = w + WINDOW_S

        # keyboard
        k_win      = [(t,k) for t,k in keystroke_events if w <= t < w_end]
        total_keys = len(k_win)
        backspace  = sum(1 for _,k in k_win if k in ('backspace','delete'))
        spaces     = sum(1 for _,k in k_win if k == 'space')
        direction  = sum(1 for _,k in k_win if k in DIRECTION_KEYS)
        special    = sum(1 for _,k in k_win if k in SPECIAL_KEYS)
        printable  = sum(1 for _,k in k_win
                          if k not in SPECIAL_KEYS and k not in DIRECTION_KEYS and len(k)==1)
        chars_ratio = float(printable / max(total_keys, 1))
        error_ratio = float(backspace  / max(total_keys, 1))

        # mouse
        m_win = [(t,x,y,lb) for t,x,y,lb in mouse_events if w <= t < w_end]
        mouse_dist = left_clicks = mouse_act = 0.
        if len(m_win) > 1:
            xs  = np.array([x  for _,x,_,_ in m_win])
            ys  = np.array([y  for _,_,y,_ in m_win])
            lbs = np.array([lb for _,_,_,lb in m_win])
            dx  = np.diff(xs); dy = np.diff(ys)
            mouse_dist  = float(np.sqrt(dx**2 + dy**2).sum())
            left_clicks = int(((lbs[1:]==1)&(lbs[:-1]==0)).sum())
            mouse_act   = float(((np.abs(dx)>0.001)|(np.abs(dy)>0.001)).mean())

        app_change = sum(1 for ts,_ in task_timeline if w <= ts < w_end)

        rows_out.append({
            'window_start':    w,
            'window_end':      w_end,
            'task_type':       get_app(w),
            'SnKeyStrokes':    total_keys,
            'SnChars':         printable,
            'SnSpecialKeys':   special,
            'SnDirectionKeys': direction,
            'SnErrorKeys':     backspace,
            'SnShortcutKeys':  0,
            'SnSpaces':        spaces,
            'CharactersRatio': chars_ratio,
            'ErrorKeyRatio':   error_ratio,
            'SnLeftClicked':   left_clicks,
            'SnRightClicked':  0,
            'SnDoubleClicked': 0,
            'SnWheel':         0,
            'SnDragged':       0,
            'SnMouseDistance': mouse_dist,
            'SnMouseAct':      mouse_act,
            'SnAppChange':     app_change,
            'SnTabfocusChange':0,
        })
        w += WINDOW_S

    result = pd.DataFrame(rows_out)
    if not result.empty:
        print(f"    HCI: {len(result)} windows | "
              f"keystrokes/win avg: {result.SnKeyStrokes.mean():.1f} | "
              f"mouse_dist avg: {result.SnMouseDistance.mean():.2f} | "
              f"clicks avg: {result.SnLeftClicked.mean():.2f}")
    return result


# ── Step 3: ECG and Respiration (unchanged logic) ─────────────────────

def extract_ecg_features(fif_path: str) -> pd.DataFrame:
    from scipy.signal import butter, filtfilt
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    ecg_chs = [ch for ch in raw.ch_names
                if any(k in ch.upper() for k in ['ECG','EXG','BIO'])]
    if not ecg_chs: ecg_chs = raw.ch_names[:1]
    ecg = raw.get_data(picks=[ecg_chs[0]])[0]
    t   = raw.times

    # CRITICAL: bandpass filter before normalization
    # Raw BioSemi ECG has baseline drift — without this, R-peaks don't
    # stand out after z-scoring (confirmed: raw std=0.00184, post-filter
    # R-peaks reach 60σ with clear detection at height=3.0)
    b, a = butter(3, [0.5/(sfreq/2), 40.0/(sfreq/2)], btype='band')
    ecg_filt = filtfilt(b, a, ecg)
    ecg_z = (ecg_filt - ecg_filt.mean()) / (ecg_filt.std() + 1e-9)

    # height=3.0 after bandpass gives ~69 bpm (physiologically clean)
    peaks, _ = find_peaks(ecg_z, distance=int(0.35*sfreq), height=3.0)
    if len(peaks) < 10: return pd.DataFrame()
    peak_t = t[peaks]
    rows = []
    w = t[0]
    while w + WINDOW_S <= t[-1]:
        in_w = peak_t[(peak_t >= w) & (peak_t < w+WINDOW_S)]
        if len(in_w) >= 8:
            rr = np.diff(in_w); rr = rr[(rr>0.33)&(rr<1.5)]
            if len(rr) >= 6:
                rows.append({"window_start": w,
                              "hr_mean":    float(60./np.median(rr)),
                              "hrv_rmssd":  float(np.sqrt(np.mean(np.diff(rr)**2))*1000.)})
        w += WINDOW_S
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def extract_resp_features(wav_path: str) -> pd.DataFrame:
    fs, signal = wavfile.read(wav_path)
    if signal.ndim > 1: signal = signal[:,0]
    signal = signal.astype(float)
    t = np.arange(len(signal)) / fs
    signal = (signal - signal.mean()) / (signal.std() + 1e-9)
    peaks, _ = find_peaks(signal, distance=int(0.75*fs), prominence=0.3)
    if len(peaks) < 3: return pd.DataFrame()
    peak_t = t[peaks]
    rows = []
    w = t[0]
    while w + WINDOW_S <= t[-1]:
        in_w = peak_t[(peak_t >= w) & (peak_t < w+WINDOW_S)]
        if len(in_w) >= 3:
            ipi = np.diff(in_w)
            rbpm = float(60./np.mean(ipi))
            if 5 < rbpm < 60:
                rows.append({"window_start": w, "resp_bpm": rbpm})
        w += WINDOW_S
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Step 4: find ECG file with flexible naming ────────────────────────

def find_ecg_file(pid_str: str) -> str | None:
    """Try multiple possible ECG filenames for a participant."""
    candidates = [
        os.path.join(ECG_DIR, f"P{pid_str}.fif"),          # P001.fif
        os.path.join(ECG_DIR, f"P{int(pid_str):03d}.fif"),  # P001.fif
        os.path.join(ECG_DIR, f"{pid_str}.fif"),             # 001.fif
        os.path.join(ECG_DIR, f"p{pid_str}.fif"),            # p001.fif
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Try glob fallback
    matches = glob.glob(os.path.join(ECG_DIR, f"*{int(pid_str):03d}*.fif"))
    return matches[0] if matches else None


def find_resp_file(pid_str: str) -> str | None:
    candidates = [
        os.path.join(RESP_DIR, f"P{pid_str}.wav"),
        os.path.join(RESP_DIR, f"P{int(pid_str):03d}.wav"),
        os.path.join(RESP_DIR, f"{pid_str}.wav"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    matches = glob.glob(os.path.join(RESP_DIR, f"*{int(pid_str):03d}*.wav"))
    return matches[0] if matches else None


# ── Step 5: run everything ────────────────────────────────────────────

def direction_label(series):
    return (series.diff() > 0).astype(float).where(series.diff().notna())


def run_phase_a():
    print("\n" + "="*70)
    print("SENSE-42 PHASE A — Proxy Validation  (v2)")
    print("="*70)
    for d, label in [(CSV_DIR,"Behavioural/CSV"),(ECG_DIR,"ECG"),(RESP_DIR,"Respiration")]:
        print(f"  {'✓' if os.path.isdir(d) else '✗'}  {label}: {d}")
    print()

    cca, cca_mu, cca_sd, rf_hr, rf_rmssd, train_mu, train_sd = train_swell_models()
    a_vec = cca.x_weights_[:,0]

    resp_model = joblib.load(os.path.join(MODEL_DIR, "resp_dir_model.pkl"))
    resp_mu    = np.load(os.path.join(MODEL_DIR, "resp_mu.npy"))
    resp_sd    = np.load(os.path.join(MODEL_DIR, "resp_sd.npy"))
    print(f"  Resp model expects {len(resp_mu)} features ({resp_mu.shape})")
    print()

    # Inspect first CSV to show column structure
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    print(f"Found {len(csv_files)} CSV files")
    if csv_files:
        print("Column inspection of first participant:")
        inspect_csv(csv_files[0])
    print()

    all_results = []

    for csv_path in csv_files[:5]:   # pilot: 5 participants
        pid_str = os.path.basename(csv_path).split('_')[0]   # "001"
        pid_key = f"P{pid_str}"
        print(f"\n{'─'*50}\nParticipant {pid_key}")

        # HCI
        try:
            hci_df = extract_hci_from_csv(csv_path)
            if hci_df.empty:
                print("  No HCI windows extracted"); continue
        except Exception as e:
            print(f"  HCI ERROR: {e}"); import traceback; traceback.print_exc(); continue

        # ECG
        ecg_path = find_ecg_file(pid_str)
        ecg_df   = pd.DataFrame()
        if ecg_path:
            try:
                ecg_df = extract_ecg_features(ecg_path)
                print(f"  ECG: {len(ecg_df)} windows from {os.path.basename(ecg_path)}")
            except Exception as e:
                print(f"  ECG ERROR: {e}")
        else:
            print(f"  ECG: no file found for {pid_key}")

        # Respiration
        resp_path = find_resp_file(pid_str)
        resp_df   = pd.DataFrame()
        if resp_path:
            try:
                resp_df = extract_resp_features(resp_path)
                print(f"  Resp: {len(resp_df)} windows")
            except Exception as e:
                print(f"  Resp ERROR: {e}")
        else:
            print(f"  Resp: no file found for {pid_key}")

        # Apply proxy — per-participant z-score before CCA projection
        # SWELL-KW counts are per-minute (scale ~100-200 keystrokes/min)
        # SENSE-42 trial-level gives much smaller raw counts per window
        # → normalize per-participant to remove scale mismatch
        # This is CORAL-lite: aligns distribution, preserves direction
        X_hci = hci_df[HCI_COLS_SWELL].to_numpy(float)
        X_hci = np.nan_to_num(X_hci)

        pp_mu  = X_hci.mean(0)
        pp_sd  = X_hci.std(0) + 1e-9
        X_hci_z    = (X_hci - pp_mu) / pp_sd     # per-participant normalization
        u_scores   = X_hci_z @ a_vec              # CCA direction preserved

        X_delta    = np.diff(X_hci, axis=0, prepend=X_hci[[0]])
        pp_d_mu = X_delta.mean(0); pp_d_sd = X_delta.std(0) + 1e-9
        X_delta_z  = (X_delta - pp_d_mu) / pp_d_sd   # per-participant delta norm
        hr_prob    = rf_hr.predict_proba(X_delta_z)[:,1]
        rmssd_prob = rf_rmssd.predict_proba(X_delta_z)[:,1]

        # Resp model: 13 cols + HR + RMSSD + SCL = 16 features
        hr_col = np.zeros(len(hci_df)); rmssd_col = np.zeros(len(hci_df))
        if not ecg_df.empty:
            for i, row in hci_df.iterrows():
                m = ecg_df[(ecg_df.window_start >= row.window_start-2) &
                            (ecg_df.window_start <= row.window_start+2)]
                if len(m):
                    hr_col[i]    = m.iloc[0]["hr_mean"]
                    rmssd_col[i] = m.iloc[0]["hrv_rmssd"]
        X_resp = np.column_stack([
            hci_df[RESP_COLS_13].to_numpy(float),   # 13 cols
            hr_col.reshape(-1,1),                    # 1
            rmssd_col.reshape(-1,1),                 # 1
            np.zeros((len(hci_df),1))                # 1 (SCL placeholder)
        ])   # shape (N, 16) ← matches resp_mu shape (16,)
        X_resp_z   = (X_resp - resp_mu) / resp_sd
        resp_pred  = resp_model.predict(X_resp_z)

        # ── Align ECG/Resp to HCI windows ──────────────────────────
        # HCI clock (PsychoPy) starts ~100s after ECG clock (BioSemi).
        # We estimate the offset as: HCI_t0 - ECG_t0, then for each
        # HCI window find the nearest ECG window after offset correction.
        # Tolerance: half a window (15s) — any closer offset counts as match.

        hci_starts = hci_df["window_start"].to_numpy(float)

        # Pre-build aligned ECG arrays
        hr_real_arr    = np.full(len(hci_df), np.nan)
        rmssd_real_arr = np.full(len(hci_df), np.nan)
        if not ecg_df.empty and len(ecg_df) > 5:
            ecg_starts  = ecg_df["window_start"].to_numpy(float)
            ecg_hr      = ecg_df["hr_mean"].to_numpy(float)
            ecg_rmssd   = ecg_df["hrv_rmssd"].to_numpy(float)
            # Clock offset: HCI starts later than ECG by ~setup_time seconds
            t_offset = hci_starts[0] - ecg_starts[0]
            print(f"  Clock offset (HCI - ECG): {t_offset:.1f}s")
            for i, hci_t in enumerate(hci_starts):
                ecg_equiv = hci_t - t_offset   # convert to ECG clock
                nearest   = np.argmin(np.abs(ecg_starts - ecg_equiv))
                if np.abs(ecg_starts[nearest] - ecg_equiv) < WINDOW_S / 2:
                    hr_real_arr[i]    = ecg_hr[nearest]
                    rmssd_real_arr[i] = ecg_rmssd[nearest]
            matched = np.isfinite(hr_real_arr).sum()
            print(f"  ECG matched: {matched}/{len(hci_df)} HCI windows")

        # Pre-build aligned Resp array
        resp_real_arr = np.full(len(hci_df), np.nan)
        if not resp_df.empty and len(resp_df) > 5:
            resp_starts = resp_df["window_start"].to_numpy(float)
            resp_bpm    = resp_df["resp_bpm"].to_numpy(float)
            t_offset_r  = hci_starts[0] - resp_starts[0]
            for i, hci_t in enumerate(hci_starts):
                resp_equiv = hci_t - t_offset_r
                nearest    = np.argmin(np.abs(resp_starts - resp_equiv))
                if np.abs(resp_starts[nearest] - resp_equiv) < WINDOW_S / 2:
                    resp_real_arr[i] = resp_bpm[nearest]
            matched_r = np.isfinite(resp_real_arr).sum()
            print(f"  Resp matched: {matched_r}/{len(hci_df)} HCI windows")

        # Collect results
        for i, row in hci_df.iterrows():
            rec = {
                "participant":       pid_key,
                "window_start":      row["window_start"],
                "task_type":         row.get("task_type","unknown"),
                "cca_u":             float(u_scores[i]),
                "hr_rising_prob":    float(hr_prob[i]),
                "rmssd_rising_prob": float(rmssd_prob[i]),
                "resp_rising_pred":  int(resp_pred[i]),
            }
            if np.isfinite(hr_real_arr[i]):
                rec["hr_real"]    = float(hr_real_arr[i])
                rec["rmssd_real"] = float(rmssd_real_arr[i])
            if np.isfinite(resp_real_arr[i]):
                rec["resp_bpm_real"] = float(resp_real_arr[i])
            all_results.append(rec)

    if not all_results:
        print("\nNo results collected."); return

    df_out = pd.DataFrame(all_results)
    for pid, g in df_out.groupby("participant"):
        idx = g.index
        if "hr_real"       in df_out.columns:
            df_out.loc[idx,"hr_rising_real"]    = direction_label(g["hr_real"]).values
        if "rmssd_real"    in df_out.columns:
            df_out.loc[idx,"rmssd_rising_real"] = direction_label(g["rmssd_real"]).values
        if "resp_bpm_real" in df_out.columns:
            df_out.loc[idx,"resp_rising_real"]  = direction_label(g["resp_bpm_real"]).values

    print("\n" + "="*70)
    print("PHASE A RESULTS")
    print("="*70)

    def rho(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return spearmanr(a[ok], b[ok]).statistic if ok.sum() >= 10 else np.nan

    def acc(pred, real):
        ok = np.isfinite(real)
        return accuracy_score(real[ok].astype(int), pred[ok].astype(int)) if ok.sum() >= 10 else np.nan

    if "hr_rising_real" in df_out.columns:
        r = rho(df_out.hr_rising_prob.to_numpy(), df_out.hr_rising_real.fillna(np.nan).to_numpy())
        a = acc((df_out.hr_rising_prob>0.5).astype(int).to_numpy(),
                df_out.hr_rising_real.fillna(np.nan).to_numpy())
        print(f"HR direction:    rho={r:.3f}  acc={a:.3f}  (chance=0.50)")
        if not np.isnan(r):
            print(f"  {'✓' if r>0.30 else '~' if r>0.15 else '✗'}  "
                  f"{'Generalizes' if r>0.30 else 'Marginal' if r>0.15 else 'FAILS to generalize'}")
    else:
        print("HR direction:    no ECG to compare")

    if "rmssd_rising_real" in df_out.columns:
        r = rho(df_out.rmssd_rising_prob.to_numpy(), df_out.rmssd_rising_real.fillna(np.nan).to_numpy())
        a = acc((df_out.rmssd_rising_prob>0.5).astype(int).to_numpy(),
                df_out.rmssd_rising_real.fillna(np.nan).to_numpy())
        print(f"RMSSD direction: rho={r:.3f}  acc={a:.3f}  (chance=0.50)")

    if "resp_rising_real" in df_out.columns:
        a = acc(df_out.resp_rising_pred.to_numpy(),
                df_out.resp_rising_real.fillna(np.nan).to_numpy())
        print(f"Resp direction:  acc={a:.3f}  (chance=0.50, target>0.70)")

    print(f"\nCCA u: mean={df_out.cca_u.mean():.3f}  std={df_out.cca_u.std():.3f}")
    print(f"Windows: {len(df_out)}  |  Participants: {df_out.participant.nunique()}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print("Change csv_files[:5] → csv_files for full N=42 run")


if __name__ == "__main__":
    run_phase_a()