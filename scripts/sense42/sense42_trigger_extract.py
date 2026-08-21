"""
sense42_trigger_extract.py
===========================
Rebuilds SENSE-42 feature extraction using the HARDWARE TRIGGERS embedded
in the raw .bdf files, replacing every previous alignment.

WHY EVERYTHING BEFORE THIS WAS WRONG
-------------------------------------
All earlier SENSE-42 scripts aligned EEG to HCI by assuming both streams
started at the same instant:

    t_offset  = hci_starts[0] - eeg_starts[0]      # 104.5 - 0.0
    eeg_equiv = hci_t - t_offset                   # HCI 104.5s -> EEG 0.0s

For P002 the trigger stream shows what actually happened:

    EEG_START_RECORDING (254)  sample    428  ->    0.42 s
    CALIB_BEGIN         (1)    sample   1442  ->    1.41 s
    CALIB_END           (2)    sample  96945  ->   94.67 s
    EXP_BEGIN           (3)    sample 103945  ->  101.51 s

and the HCI CSV's thisRow.t runs 104.5 - 7981.2 s. The two clocks are
effectively the SAME clock (offset ~3 s). So the correct mapping is
HCI 104.5s -> EEG ~104.5s, but the old code mapped it to EEG 0.0s --
a shift of about 104 seconds, landing in the pre-calibration period.

Every window was paired with EEG recorded ~1.7 minutes earlier. That
produces uncorrelated pairs, which is precisely the observed result:
a null whose scatter tightened from +/-0.04 to +/-0.01 as n grew from
3,583 to 230,463. That is what estimating zero more precisely looks
like, not evidence of absence.

WHAT THE RAW FILES ACTUALLY CONTAIN
------------------------------------
48 channels, 1024 Hz:
    32 scalp EEG
    EXG1-8    external -- 3-lead ECG lives here
    GSR1/GSR2 SKIN CONDUCTANCE  <- we assumed this did not exist and had
                                   been passing a zeros placeholder for SCL
    Resp      respiration belt (native, not the downsampled .wav)
    Plet      plethysmograph (blood volume pulse)
    Temp      skin temperature
    Status    trigger channel

So the full SWELL-KW input schema (HCI + HR + RMSSD + SCL) is available
in SENSE-42 after all.

TRIGGER CODES (from experiment/explorer_lastrun.py, SerialConnector)
--------------------------------------------------------------------
     1/2   CALIB begin/end            19/20  TRASH_BIN_HOMESCREEN
     3/4   EXP begin/end              21/22  TRASH_BIN_SELECT
     5/6   WINDOW_CLOSE               23/24  TRASH_BIN_CONFIRM
     7/8   MAIL_HOMESCREEN            25/26  NOTES_HOMESCREEN
     9/10  MAIL_NOTIFICATION          27/28  NOTES_REPEAT
    11/12  MAIL_CONTENT               29/30  BROWSER_HOMESCREEN
    13/14  FILE_MANAGER_HOMESCREEN    31/32  BROWSER_NAVIGATION
    15/16  FILE_MANAGER_DRAGGING      33/34  BROWSER_CONTENT
    17/18  FILE_MANAGER_OPENING       254/255 EEG start/stop recording

    Questions: 100 + 10*question_index + rating
               7 dimensions (index 0-6) x 26 questionnaires
               The RATING IS IN THE TRIGGER -- no alignment needed at all
               for the questionnaire analysis.

NOTE: the END triggers are commented out in the experiment source, so only
BEGIN markers are present. Epoch durations are derived from the next
trigger's onset.

Status channel carries high bits (code 65790 = 65536 + 254), so codes are
masked with & 0xFF.

OUTPUT (two tables, both trigger-aligned)
------------------------------------------
  outputs/sense42_trig_epochs.csv
      one row per task-event epoch (mail_content, notes_repeat, ...)
      EEG band powers + HR + SCL + resp, computed within the epoch

  outputs/sense42_trig_questions.csv
      one row per questionnaire answer
      rating decoded from the trigger, EEG/physio from the LOOKBACK
      seconds preceding it

MEMORY: one participant at a time; .bdf extracted, processed, deleted.
Resumable -- participants already in the output CSVs are skipped.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, sys, glob, gc, warnings
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, butter, filtfilt
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

BASE     = os.path.expanduser("~/biosignals_data")
SENSE    = os.path.join(BASE, "data", "sense_42")
BDF_DIR  = os.path.join(SENSE, "EEG_raw")
BDF_ZIP  = os.path.join(SENSE, "EEG_raw.zip")
CSV_DIR  = os.path.join(SENSE, "Behavioural", "CSV")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EPOCH = os.path.join(OUT_DIR, "sense42_trig_epochs.csv")
OUT_QUEST = os.path.join(OUT_DIR, "sense42_trig_questions.csv")

KEEP_BDF   = False
LOOKBACK_S = 30.0     # window before a questionnaire trigger
MIN_EPOCH  = 2.0      # ignore task epochs shorter than this
MAX_EPOCH  = 120.0    # clip absurdly long ones (missed END marker)

FRONTAL   = ['AF3', 'F3', 'Fz', 'F4', 'AF4']
POSTERIOR = ['P3', 'Pz', 'P4']
OCCIPITAL = ['O1', 'Oz', 'O2']
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

TASK_TRIGGERS = {
     5: "window_close",          7: "mail_homescreen",
     9: "mail_notification",    11: "mail_content",
    13: "file_manager_homescreen", 15: "file_manager_dragging",
    17: "file_manager_opening", 19: "trash_bin_homescreen",
    21: "trash_bin_select",     23: "trash_bin_confirm",
    25: "notes_homescreen",     27: "notes_repeat",
    29: "browser_homescreen",   31: "browser_navigation",
    33: "browser_content",
}
TASK_APP = {
    "mail_homescreen": "mail", "mail_notification": "mail",
    "mail_content": "mail",
    "file_manager_homescreen": "file_mgr", "file_manager_dragging": "file_mgr",
    "file_manager_opening": "file_mgr",
    "trash_bin_homescreen": "trash", "trash_bin_select": "trash",
    "trash_bin_confirm": "trash",
    "notes_homescreen": "notes", "notes_repeat": "notes",
    "browser_homescreen": "browser", "browser_navigation": "browser",
    "browser_content": "browser", "window_close": "system",
}
# question_index -> NASA-TLX dimension, in the order the sliders appear
QUESTION_DIMS = ["sleepiness", "mental_demand", "temporal_demand",
                 "performance", "effort", "frustration", "attentiveness"]

sys.path.insert(0, os.path.join(BASE, "scripts", "sense42"))
try:
    from sense42_extract_multiwindow import hci_events_from_csv
except Exception:
    from sense42_phase_a import extract_hci_from_csv
    hci_events_from_csv = None


# ══════════════════════════════════════════════════════════════════════
# Signal feature helpers -- all operate on an arbitrary [t0, t1] segment
# ══════════════════════════════════════════════════════════════════════

def eeg_bandpowers(data, sfreq, idx_map):
    """Welch band powers over one segment. data: (n_channels, n_samples)."""
    nper = min(data.shape[1], int(sfreq * 2))
    if nper < int(sfreq * 0.5):
        return None
    f, psd = welch(data, fs=sfreq, nperseg=nper, axis=1)
    bp = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp[name] = psd[:, m].mean(axis=1) if m.sum() else np.zeros(data.shape[0])
    m_bb = (f >= 1) & (f < 40)
    bp["broadband"] = psd[:, m_bb].mean(axis=1)

    i_f, i_p, i_o = idx_map["front"], idx_map["post"], idx_map["occ"]
    ft = bp["theta"][i_f].mean(); fa = bp["alpha"][i_f].mean()
    fb = bp["beta"][i_f].mean()
    pa = bp["alpha"][i_p].mean() if i_p else fa
    od = bp["delta"][i_o].mean() if i_o else bp["delta"].mean()
    return {
        "frontal_theta":       float(ft),
        "frontal_alpha":       float(fa),
        "theta_alpha_ratio":   float(ft / (fa + 1e-15)),
        "engagement_index":    float(fb / (fa + ft + 1e-15)),
        "posterior_alpha":     float(pa),
        "occipital_delta":     float(od),
        "broadband_amplitude": float(bp["broadband"].mean()),
    }


def cardiac_features(ecg_seg, sfreq):
    """HR and RMSSD from an ECG segment. RMSSD only if >=10 usable beats."""
    if len(ecg_seg) < int(sfreq * 5):
        return {"hr_mean": np.nan, "hrv_rmssd": np.nan, "n_beats": 0}
    b, a = butter(3, [0.5 / (sfreq / 2), 40.0 / (sfreq / 2)], btype="band")
    z = filtfilt(b, a, ecg_seg)
    z = (z - z.mean()) / (z.std() + 1e-9)
    peaks, _ = find_peaks(z, distance=int(0.35 * sfreq), height=3.0)
    if len(peaks) < 4:
        return {"hr_mean": np.nan, "hrv_rmssd": np.nan, "n_beats": len(peaks)}
    rr = np.diff(peaks) / sfreq
    rr = rr[(rr > 0.33) & (rr < 1.5)]
    if len(rr) < 3:
        return {"hr_mean": np.nan, "hrv_rmssd": np.nan, "n_beats": len(peaks)}
    return {
        "hr_mean":   float(60.0 / np.median(rr)),
        # RMSSD needs ~30 beats to be stable; below 10 it is not reported
        "hrv_rmssd": float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000)
                     if len(rr) >= 10 else np.nan,
        "n_beats":   int(len(rr)),
    }


def scl_features(gsr_seg, sfreq):
    """Tonic skin conductance level and its slope over the segment."""
    if len(gsr_seg) < int(sfreq):
        return {"scl_mean": np.nan, "scl_slope": np.nan}
    x = gsr_seg.astype(float)
    t = np.arange(len(x)) / sfreq
    return {"scl_mean":  float(np.mean(x)),
            "scl_slope": float(np.polyfit(t, x, 1)[0]) if len(x) > 10 else np.nan}


def resp_features(resp_seg, sfreq):
    """Respiration rate and amplitude over the segment."""
    if len(resp_seg) < int(sfreq * 4):
        return {"resp_bpm": np.nan, "resp_amp": np.nan}
    b, a = butter(3, [0.05 / (sfreq / 2), 1.0 / (sfreq / 2)], btype="band")
    x = filtfilt(b, a, resp_seg.astype(float))
    xz = (x - x.mean()) / (x.std() + 1e-9)
    peaks, _ = find_peaks(xz, distance=int(1.5 * sfreq), prominence=0.3)
    bpm = np.nan
    if len(peaks) >= 3:
        ipi = np.diff(peaks) / sfreq
        v = 60.0 / np.mean(ipi)
        bpm = float(v) if 5 < v < 60 else np.nan
    return {"resp_bpm": bpm, "resp_amp": float(np.std(x))}


# ══════════════════════════════════════════════════════════════════════

def process_participant(pid, done_ep, done_q):
    if f"P{pid}" in done_ep and f"P{pid}" in done_q:
        print("    already done"); return None, None

    bdf = os.path.join(BDF_DIR, f"P{pid}.bdf")
    tmp = False
    if not os.path.isfile(bdf):
        rc = os.system(f'unzip -q -o "{BDF_ZIP}" "EEG_raw/P{pid}.bdf" '
                       f'-d "{SENSE}" 2>/dev/null')
        tmp = (rc == 0 and os.path.isfile(bdf))
        if not tmp:
            print("    bdf not available"); return None, None

    try:
        raw = mne.io.read_raw_bdf(bdf, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]

        # ── triggers ────────────────────────────────────────────────
        try:
            ev = mne.find_events(raw, stim_channel="Status", verbose=False)
        except Exception as e:
            print(f"    no triggers: {e}"); return None, None
        if len(ev) == 0:
            print("    zero events"); return None, None
        ev[:, 2] = ev[:, 2] & 0xFF          # strip Status high bits
        print(f"    {len(ev)} triggers", end="")

        # ── channel groups ──────────────────────────────────────────
        def idx(names):
            return [raw.ch_names.index(c) for c in names if c in raw.ch_names]
        idx_map = {"front": idx(FRONTAL), "post": idx(POSTERIOR),
                   "occ": idx(OCCIPITAL)}
        if not idx_map["front"]:
            print("  no frontal channels"); return None, None

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        eeg_data  = raw.get_data(picks=eeg_picks)

        def grab(name):
            return raw.get_data(picks=[raw.ch_names.index(name)])[0] \
                   if name in raw.ch_names else None
        ecg  = grab("EXG1")
        gsr  = grab("GSR1")
        resp = grab("Resp")
        print(f" | ECG {'y' if ecg is not None else 'n'}"
              f" GSR {'y' if gsr is not None else 'n'}"
              f" Resp {'y' if resp is not None else 'n'}")

        def seg_features(s0, s1):
            out = {}
            bp = eeg_bandpowers(eeg_data[:, s0:s1], sfreq, idx_map)
            if bp is None:
                return None
            out.update(bp)
            out.update(cardiac_features(ecg[s0:s1], sfreq) if ecg is not None
                       else {"hr_mean": np.nan, "hrv_rmssd": np.nan, "n_beats": 0})
            out.update(scl_features(gsr[s0:s1], sfreq) if gsr is not None
                       else {"scl_mean": np.nan, "scl_slope": np.nan})
            out.update(resp_features(resp[s0:s1], sfreq) if resp is not None
                       else {"resp_bpm": np.nan, "resp_amp": np.nan})
            return out

        n_samp = eeg_data.shape[1]

        # ── task-event epochs ───────────────────────────────────────
        epoch_rows = []
        if f"P{pid}" not in done_ep:
            starts = ev[:, 0]
            for i, (s, _, code) in enumerate(ev):
                if code not in TASK_TRIGGERS:
                    continue
                nxt = starts[i + 1] if i + 1 < len(ev) else n_samp
                dur = (nxt - s) / sfreq
                if dur < MIN_EPOCH:
                    continue
                s1 = min(int(s + min(dur, MAX_EPOCH) * sfreq), n_samp)
                f = seg_features(int(s), s1)
                if f is None:
                    continue
                task = TASK_TRIGGERS[code]
                f.update({"participant": f"P{pid}", "trigger_code": int(code),
                          "task": task, "app": TASK_APP.get(task, "unknown"),
                          "onset_s": float(s / sfreq),
                          "duration_s": float(min(dur, MAX_EPOCH))})
                epoch_rows.append(f)
            print(f"    task epochs: {len(epoch_rows)}")

        # ── questionnaire epochs ────────────────────────────────────
        q_rows = []
        if f"P{pid}" not in done_q:
            qev = [(s, c) for s, _, c in ev if 100 <= c < 100 + 10 * len(QUESTION_DIMS)]
            for s, c in qev:
                qi, rating = (c - 100) // 10, (c - 100) % 10
                if qi >= len(QUESTION_DIMS):
                    continue
                s0 = max(0, int(s - LOOKBACK_S * sfreq))
                f = seg_features(s0, int(s))
                if f is None:
                    continue
                f.update({"participant": f"P{pid}", "onset_s": float(s / sfreq),
                          "question_index": int(qi),
                          "dimension": QUESTION_DIMS[qi],
                          "rating": int(rating)})
                q_rows.append(f)
            print(f"    questionnaire epochs: {len(q_rows)}")

        del eeg_data, raw
        gc.collect()
        return (pd.DataFrame(epoch_rows) if epoch_rows else None,
                pd.DataFrame(q_rows) if q_rows else None)

    finally:
        if tmp and not KEEP_BDF and os.path.isfile(bdf):
            os.remove(bdf)


def main():
    print("=" * 78)
    print("SENSE-42 TRIGGER-ALIGNED EXTRACTION")
    print("=" * 78)
    print("Replaces all offset-estimated alignment. Previous pipelines mapped")
    print("HCI ~104.5s onto EEG 0.0s -- a ~104 s shift into the pre-calibration")
    print("period. Trigger onsets are sample-accurate, so no estimation at all.\n")

    ep_df = pd.read_csv(OUT_EPOCH) if os.path.isfile(OUT_EPOCH) else pd.DataFrame()
    q_df  = pd.read_csv(OUT_QUEST) if os.path.isfile(OUT_QUEST) else pd.DataFrame()
    done_ep = set(ep_df["participant"].unique()) if len(ep_df) else set()
    done_q  = set(q_df["participant"].unique())  if len(q_df)  else set()
    if done_ep or done_q:
        print(f"Cached: {len(ep_df)} epochs / {len(q_df)} questionnaire rows\n")

    pids = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(os.path.join(CSV_DIR, "*.csv"))})
    print(f"Participants: {len(pids)}\n")

    for i, pid in enumerate(pids, 1):
        print(f"[{i:2d}/{len(pids)}] P{pid}")
        try:
            e, q = process_participant(pid, done_ep, done_q)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue
        if e is not None and len(e):
            ep_df = pd.concat([ep_df, e], ignore_index=True) if len(ep_df) else e
            ep_df.to_csv(OUT_EPOCH, index=False)
        if q is not None and len(q):
            q_df = pd.concat([q_df, q], ignore_index=True) if len(q_df) else q
            q_df.to_csv(OUT_QUEST, index=False)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    if len(ep_df):
        print(f"\nTask epochs: {len(ep_df)} rows, "
              f"{ep_df.participant.nunique()} participants")
        print(ep_df["app"].value_counts().to_string())
        for c in ["scl_mean", "hr_mean", "resp_bpm", "hrv_rmssd"]:
            if c in ep_df.columns:
                print(f"  {c:12s} present in {ep_df[c].notna().sum()}/{len(ep_df)}")
    if len(q_df):
        print(f"\nQuestionnaire epochs: {len(q_df)} rows, "
              f"{q_df.participant.nunique()} participants")
        print(q_df.groupby("dimension")["rating"]
                  .agg(["count", "mean", "std", "min", "max"]).to_string())
    print(f"\n  {OUT_EPOCH}")
    print(f"  {OUT_QUEST}")
    print("\nNext: analyse these two tables. Both are trigger-aligned, so the")
    print("~104 s misalignment that produced the earlier nulls is gone.")


if __name__ == "__main__":
    main()
