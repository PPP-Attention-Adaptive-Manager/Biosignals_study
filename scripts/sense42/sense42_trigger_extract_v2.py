"""
sense42_trigger_extract_v2.py
==============================
Patched trigger-aligned extraction. Supersedes sense42_trigger_extract.py.

FOUR BUGS FIXED
---------------
1. ECG returned 0 beats in ALL 16,792 epochs, including 2,717 longer than
   30 s. Cause: the script grabbed "EXG1", which DOES NOT EXIST in these
   files -- the externals are EXG2..EXG8. grab() returned None and every
   cardiac value became NaN.
   Fix: 3-lead ECG needs a BIPOLAR derivation, not a single-ended channel.
   Verified on P002: EXG2 - EXG3 with height=2.0 gives 9,322 peaks
   -> 70.0 bpm, matching the provided .fif (9,255 peaks -> 68.5 bpm).
   Note height=3.0 gives 62.1 bpm on the bipolar BDF signal vs 68.5 on
   the .fif, so the threshold differs between derivations. 2.0 is correct
   here.

2. Respiration returned 0 peaks at every parameter combination, despite
   the raw signal being real (std=0.0195).
   Cause: numerical instability. At 1024 Hz a 0.1 Hz cutoff is a
   normalised frequency of 0.0002, where a 3rd-order Butterworth has
   degenerate coefficients and filtfilt output is garbage.
   Fix: decimate to 32 Hz BEFORE filtering. Breathing is 0.1-0.5 Hz, so
   1024 Hz is ~2000x oversampled. Verified: 2,007 peaks -> 15.1 bpm, and
   all six tested band/prominence combinations agree within 0.9 bpm.

3. SCL was reported "complete 7394/7394" and was in fact meaningless.
   GSR1 and GSR2 are both railed at 262143 (= 2^18 - 1, the ADC maximum)
   with std exactly 0.0 for the entire 8,000 s recording. No GSR sensor
   was connected. BioSemi ActiveTwo writes its full channel complement
   regardless of what is plugged in, so the channels exist but carry no
   data. The README's modality list never claimed GSR:
       Behavioural / 32-ch EEG / Respiration belt / 3-lead ECG / Webcam
   SCL is therefore REMOVED rather than zero-filled. SENSE-42 has HR and
   RMSSD only, no electrodermal channel.

4. HCI could not be joined to EEG by timestamp at all.
   Field_Notes.txt: "P011 was paused for too many times (11) and this may
   affect the absolute time synchronisation between behavioural and
   physiological data". The experiment source shows why:
       def apply_pause(self, offset):
           self.clock.addTime(-1 * offset)     # REWINDS the PsychoPy clock
   The PsychoPy clock is rewound by each pause so the 2 h budget excludes
   breaks; the EEG recording keeps running. The two timelines therefore
   diverge CUMULATIVELY, so no global offset can ever be correct.
   Fix: EVENT-INDEX alignment. Both streams mark the same events, so the
   Nth occurrence in the CSV is matched to the Nth trigger of that code.
   This re-syncs at every event and pauses cannot accumulate.
   Verified on P002 -- counts match exactly:
       mail_content 31/31   mail_notification 31/31   notes_repeat 31/31
       browser_content 26/26   file_manager_dragging 620/620
       trash_bin_select 31/31
   window_close is the one exception (155 CSV / 174 EEG) and is excluded
   from index matching; the CSV is internally inconsistent there too
   (target_name 157, .started 155, scoped arrays sum to 181) because it
   is a nested routine that does not always log to the global column.

PARTICIPANT EXCLUSIONS (from Field_Notes.txt)
----------------------------------------------
    P001  no ECG recorded          -> EEG-only rows, cardiac NaN
    P005  two EEG segments (P005.bdf + P005_02.bdf) due to interruption
          -> EXCLUDED. Concatenating across an interruption would splice
             a discontinuity in exactly the stretch where the participant
             was already disengaged.
    P011  11 pauses                -> KEPT. Event-index alignment is
             immune to clock drift, but flag it in any writeup.

Run from: ~/biosignals_data/
Output:
  outputs/sense42_v2_events.csv      one row per task event, EEG+cardiac+resp+HCI
  outputs/sense42_v2_questions.csv   one row per questionnaire answer
"""
from __future__ import annotations
import os, sys, glob, gc, ast, warnings
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, butter, filtfilt, decimate
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

BASE    = os.path.expanduser("~/biosignals_data")
SENSE   = os.path.join(BASE, "data", "sense_42")
BDF_DIR = os.path.join(SENSE, "EEG_raw")
BDF_ZIP = os.path.join(SENSE, "EEG_raw.zip")
CSV_DIR = os.path.join(SENSE, "Behavioural", "CSV")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_EV = os.path.join(OUT_DIR, "sense42_v2_events.csv")
OUT_Q  = os.path.join(OUT_DIR, "sense42_v2_questions.csv")

KEEP_BDF   = False
EXCLUDE    = {"005"}          # split recording, see header
LOOKBACK_S = 30.0
MIN_EPOCH  = 2.0
MAX_EPOCH  = 120.0
RESP_FS    = 32.0             # decimation target

FRONTAL   = ['AF3', 'F3', 'Fz', 'F4', 'AF4']
POSTERIOR = ['P3', 'Pz', 'P4']
OCCIPITAL = ['O1', 'Oz', 'O2']
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

# trigger code -> CSV routine name. window_close (5) deliberately absent:
# counts disagree between streams, so index matching is unsafe for it.
TASK_TRIGGERS = {
     7: "mail_homescreen",          9: "mail_notification",
    11: "mail_content",            13: "file_manager_homescreen",
    15: "file_manager_dragging",   17: "file_manager_opening",
    19: "trash_bin_homescreen",    21: "trash_bin_select",
    23: "trash_bin_confirm",       25: "notes_homescreen",
    27: "notes_repeat",            29: "browser_homescreen",
    31: "browser_navigation",      33: "browser_content",
}
TASK_APP = {
    "mail_homescreen": "mail", "mail_notification": "mail", "mail_content": "mail",
    "file_manager_homescreen": "file_mgr", "file_manager_dragging": "file_mgr",
    "file_manager_opening": "file_mgr",
    "trash_bin_homescreen": "trash", "trash_bin_select": "trash",
    "trash_bin_confirm": "trash",
    "notes_homescreen": "notes", "notes_repeat": "notes",
    "browser_homescreen": "browser", "browser_navigation": "browser",
    "browser_content": "browser",
}
QUESTION_DIMS = ["sleepiness", "mental_demand", "temporal_demand",
                 "performance", "effort", "frustration", "attentiveness"]
# ^ order verified: trigger-decoded means match CSV-parsed means for all
#   seven dimensions to within 0.07 (e.g. sleepiness 5.579 vs 5.58).

SPECIAL = {'backspace','delete','lshift','rshift','lctrl','rctrl','lalt','ralt',
           'tab','escape','return','enter','caps_lock','lsuper','comma',
           'period','semicolon','slash','backslash','quote'}
DIRECT  = {'left','right','up','down','pageup','pagedown','home','end'}


# ══════════════════════════════════════════════════════════════════════
# Signal features
# ══════════════════════════════════════════════════════════════════════

def eeg_bandpowers(seg, sfreq, idx):
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

    ft = bp["theta"][idx["f"]].mean(); fa = bp["alpha"][idx["f"]].mean()
    fb = bp["beta"][idx["f"]].mean()
    pa = bp["alpha"][idx["p"]].mean() if idx["p"] else fa
    od = bp["delta"][idx["o"]].mean() if idx["o"] else bp["delta"].mean()
    return {
        "frontal_theta":       float(ft),
        "frontal_alpha":       float(fa),
        "theta_alpha_ratio":   float(ft / (fa + 1e-15)),
        "engagement_index":    float(fb / (fa + ft + 1e-15)),
        "posterior_alpha":     float(pa),
        "occipital_delta":     float(od),
        "broadband_amplitude": float(bp["broadband"].mean()),
    }


def cardiac_features(seg, sfreq):
    """Bipolar ECG segment -> HR, RMSSD. height=2.0 for this derivation."""
    if seg is None or len(seg) < int(sfreq * 5):
        return {"hr_mean": np.nan, "hrv_rmssd": np.nan, "n_beats": 0}
    b, a = butter(3, [0.5 / (sfreq / 2), 40.0 / (sfreq / 2)], btype="band")
    z = filtfilt(b, a, seg)
    z = (z - z.mean()) / (z.std() + 1e-9)
    peaks, _ = find_peaks(z, distance=int(0.35 * sfreq), height=2.0)
    if len(peaks) < 4:
        return {"hr_mean": np.nan, "hrv_rmssd": np.nan, "n_beats": len(peaks)}
    rr = np.diff(peaks) / sfreq
    rr = rr[(rr > 0.33) & (rr < 1.5)]
    if len(rr) < 3:
        return {"hr_mean": np.nan, "hrv_rmssd": np.nan, "n_beats": len(peaks)}
    return {
        "hr_mean":   float(60.0 / np.median(rr)),
        # RMSSD needs ~30 beats for stability; below 10 it is not reported
        "hrv_rmssd": float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000)
                     if len(rr) >= 10 else np.nan,
        "n_beats":   int(len(rr)),
    }


def resp_features(seg32):
    """Segment already decimated to RESP_FS. Filtering at 1024 Hz is unstable."""
    if seg32 is None or len(seg32) < int(RESP_FS * 8):
        return {"resp_bpm": np.nan, "resp_amp": np.nan}
    b, a = butter(3, [0.1 / (RESP_FS / 2), 0.5 / (RESP_FS / 2)], btype="band")
    x = filtfilt(b, a, seg32)
    if not np.all(np.isfinite(x)) or x.std() < 1e-12:
        return {"resp_bpm": np.nan, "resp_amp": np.nan}
    z = (x - x.mean()) / (x.std() + 1e-9)
    peaks, _ = find_peaks(z, distance=int(1.5 * RESP_FS), prominence=0.3)
    bpm = np.nan
    if len(peaks) >= 3:
        v = 60.0 / np.mean(np.diff(peaks) / RESP_FS)
        bpm = float(v) if 5 < v < 60 else np.nan
    return {"resp_bpm": bpm, "resp_amp": float(x.std())}


# ══════════════════════════════════════════════════════════════════════
# HCI per CSV row (event-index matched, never timestamp matched)
# ══════════════════════════════════════════════════════════════════════

def parse_list(v):
    try:
        r = ast.literal_eval(str(v))
        return r if isinstance(r, list) else None
    except Exception:
        return None


def hci_from_row(row, cols):
    """Keystroke and mouse aggregates from one CSV row's list-string cells."""
    keys = []
    for c in cols:
        if c.endswith(".keys") and pd.notna(row.get(c)):
            k = parse_list(row[c])
            if k:
                keys.extend([str(x) for x in k])
    total = len(keys)
    back  = sum(1 for k in keys if k in ("backspace", "delete"))
    space = sum(1 for k in keys if k == "space")
    dire  = sum(1 for k in keys if k in DIRECT)
    spec  = sum(1 for k in keys if k in SPECIAL)
    prnt  = sum(1 for k in keys if k not in SPECIAL and k not in DIRECT and len(k) == 1)

    dist = clicks = 0.0
    n_pts = 0
    for c in cols:
        if c.endswith("_mouse.x") and pd.notna(row.get(c)):
            xs = parse_list(row[c])
            ys = parse_list(row.get(c[:-2] + ".y"))
            lb = parse_list(row.get(c[:-2] + ".leftButton"))
            if not xs or not ys or len(xs) < 2:
                continue
            n = min(len(xs), len(ys))
            xa, ya = np.asarray(xs[:n], float), np.asarray(ys[:n], float)
            dist += float(np.sqrt(np.diff(xa) ** 2 + np.diff(ya) ** 2).sum())
            n_pts += n
            if lb and len(lb) >= n:
                la = np.asarray(lb[:n])
                clicks += int(((la[1:] == 1) & (la[:-1] == 0)).sum())

    return {
        "SnKeyStrokes": total, "SnChars": prnt, "SnSpecialKeys": spec,
        "SnDirectionKeys": dire, "SnErrorKeys": back, "SnSpaces": space,
        "CharactersRatio": prnt / max(total, 1),
        "ErrorKeyRatio":   back / max(total, 1),
        "SnLeftClicked": int(clicks), "SnMouseDistance": dist,
        "n_mouse_samples": n_pts,
    }


# ══════════════════════════════════════════════════════════════════════

def process(pid, done_ev, done_q):
    if f"P{pid}" in done_ev and f"P{pid}" in done_q:
        print("    cached"); return None, None

    csvs = glob.glob(os.path.join(CSV_DIR, f"{pid}_*.csv"))
    if not csvs:
        print("    no CSV"); return None, None
    bdf = os.path.join(BDF_DIR, f"P{pid}.bdf")
    tmp = False
    if not os.path.isfile(bdf):
        rc = os.system(f'unzip -q -o "{BDF_ZIP}" "EEG_raw/P{pid}.bdf" '
                       f'-d "{SENSE}" 2>/dev/null')
        tmp = (rc == 0 and os.path.isfile(bdf))
        if not tmp:
            print("    bdf unavailable"); return None, None

    try:
        raw = mne.io.read_raw_bdf(bdf, preload=True, verbose=False)
        sf = raw.info["sfreq"]
        ev = mne.find_events(raw, stim_channel="Status", verbose=False)
        if len(ev) == 0:
            print("    no triggers"); return None, None
        ev[:, 2] = ev[:, 2] & 0xFF

        def ix(names):
            return [raw.ch_names.index(c) for c in names if c in raw.ch_names]
        idx = {"f": ix(FRONTAL), "p": ix(POSTERIOR), "o": ix(OCCIPITAL)}
        if not idx["f"]:
            print("    no frontal channels"); return None, None

        eeg = raw.get_data(picks=mne.pick_types(raw.info, eeg=True, exclude=[]))

        # BUG 1 FIX: bipolar ECG. EXG1 does not exist in these files.
        ecg = None
        if "EXG2" in raw.ch_names and "EXG3" in raw.ch_names:
            ecg = (raw.get_data(picks=["EXG2"])[0]
                   - raw.get_data(picks=["EXG3"])[0])

        # BUG 2 FIX: decimate respiration once, up front.
        resp32, q = None, int(sf // RESP_FS)
        if "Resp" in raw.ch_names:
            resp32 = decimate(raw.get_data(picks=["Resp"])[0], q,
                              ftype="fir", zero_phase=True)

        print(f"    {len(ev)} triggers | ECG {'y' if ecg is not None else 'n'}"
              f" | Resp {'y' if resp32 is not None else 'n'}", flush=True)

        def feats(s0, s1):
            f = eeg_bandpowers(eeg[:, s0:s1], sf, idx)
            if f is None:
                return None
            f.update(cardiac_features(ecg[s0:s1] if ecg is not None else None, sf))
            f.update(resp_features(resp32[int(s0 // q):int(s1 // q)]
                                   if resp32 is not None else None))
            return f

        n_samp = eeg.shape[1]
        df = pd.read_csv(csvs[0], low_memory=False)
        cols = df.columns.tolist()

        # ── task events, matched BY INDEX not timestamp ─────────────
        ev_rows = []
        if f"P{pid}" not in done_ev:
            starts = ev[:, 0]
            # per-code running counter -> Nth trigger of that code
            seen = {}
            csv_idx = {}          # routine -> list of CSV row positions
            for code, name in TASK_TRIGGERS.items():
                sc = f"{name}.started"
                if sc in cols:
                    csv_idx[name] = df.index[df[sc].notna()].tolist()

            for i, (s, _, code) in enumerate(ev):
                if code not in TASK_TRIGGERS:
                    continue
                name = TASK_TRIGGERS[code]
                k = seen.get(code, 0); seen[code] = k + 1

                nxt = starts[i + 1] if i + 1 < len(ev) else n_samp
                dur = (nxt - s) / sf
                if dur < MIN_EPOCH:
                    continue
                s1 = min(int(s + min(dur, MAX_EPOCH) * sf), n_samp)
                f = feats(int(s), s1)
                if f is None:
                    continue

                # BUG 4 FIX: Nth trigger <-> Nth CSV occurrence
                rows = csv_idx.get(name, [])
                if k < len(rows):
                    f.update(hci_from_row(df.loc[rows[k]], cols))
                    f["hci_matched"] = True
                else:
                    f["hci_matched"] = False

                f.update({"participant": f"P{pid}", "task": name,
                          "app": TASK_APP.get(name, "unknown"),
                          "event_index": k, "trigger_code": int(code),
                          "onset_s": float(s / sf),
                          "duration_s": float(min(dur, MAX_EPOCH))})
                ev_rows.append(f)
            matched = sum(1 for r in ev_rows if r.get("hci_matched"))
            print(f"    events {len(ev_rows)} (HCI matched {matched})")

        # ── questionnaire epochs ────────────────────────────────────
        q_rows = []
        if f"P{pid}" not in done_q:
            for s, _, c in ev:
                if not (100 <= c < 100 + 10 * len(QUESTION_DIMS)):
                    continue
                qi, rating = (c - 100) // 10, (c - 100) % 10
                if qi >= len(QUESTION_DIMS):
                    continue
                f = feats(max(0, int(s - LOOKBACK_S * sf)), int(s))
                if f is None:
                    continue
                f.update({"participant": f"P{pid}", "onset_s": float(s / sf),
                          "question_index": int(qi),
                          "dimension": QUESTION_DIMS[qi], "rating": int(rating)})
                q_rows.append(f)
            print(f"    questions {len(q_rows)}")

        del eeg, raw
        gc.collect()
        return (pd.DataFrame(ev_rows) if ev_rows else None,
                pd.DataFrame(q_rows) if q_rows else None)
    finally:
        if tmp and not KEEP_BDF and os.path.isfile(bdf):
            os.remove(bdf)


def main():
    print("=" * 74)
    print("SENSE-42 TRIGGER EXTRACTION v2")
    print("=" * 74)
    print("Fixes: bipolar ECG (EXG2-EXG3, h=2.0) | resp decimated to 32 Hz")
    print("       SCL removed (GSR railed, no sensor) | event-index HCI join")
    print(f"Excluded: {sorted(EXCLUDE)} (P005 split recording)\n")

    edf = pd.read_csv(OUT_EV) if os.path.isfile(OUT_EV) else pd.DataFrame()
    qdf = pd.read_csv(OUT_Q)  if os.path.isfile(OUT_Q)  else pd.DataFrame()
    done_ev = set(edf.participant.unique()) if len(edf) else set()
    done_q  = set(qdf.participant.unique()) if len(qdf) else set()

    pids = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(os.path.join(CSV_DIR, "*.csv"))})
    pids = [p for p in pids if p not in EXCLUDE]
    print(f"Participants: {len(pids)}\n")

    for i, pid in enumerate(pids, 1):
        print(f"[{i:2d}/{len(pids)}] P{pid}")
        try:
            e, q = process(pid, done_ev, done_q)
        except Exception as exc:
            print(f"    ERROR: {exc}"); continue
        if e is not None and len(e):
            edf = pd.concat([edf, e], ignore_index=True) if len(edf) else e
            edf.to_csv(OUT_EV, index=False)
        if q is not None and len(q):
            qdf = pd.concat([qdf, q], ignore_index=True) if len(qdf) else q
            qdf.to_csv(OUT_Q, index=False)

    print("\n" + "=" * 74)
    print("DONE")
    print("=" * 74)
    if len(edf):
        print(f"\nEvents: {len(edf)} rows, {edf.participant.nunique()} participants")
        for c in ["frontal_theta", "hr_mean", "hrv_rmssd", "resp_bpm"]:
            if c in edf.columns:
                print(f"  {c:14s} {edf[c].notna().sum():6d}/{len(edf)}")
        if "hci_matched" in edf.columns:
            print(f"  hci_matched    {edf.hci_matched.sum():6d}/{len(edf)}")
    if len(qdf):
        print(f"\nQuestions: {len(qdf)} rows, {qdf.participant.nunique()} participants")
        for c in ["frontal_theta", "hr_mean", "resp_bpm"]:
            if c in qdf.columns:
                print(f"  {c:14s} {qdf[c].notna().sum():6d}/{len(qdf)}")
    print(f"\n  {OUT_EV}\n  {OUT_Q}")
    print("\nSanity check before analysing: hr_mean should now be non-null in")
    print("most epochs longer than 5 s, and resp_bpm in most longer than 8 s.")
    print("If either is still 0, stop and diagnose rather than interpreting.")


if __name__ == "__main__":
    main()
