"""
sense42_hep_analysis.py
=========================
Heartbeat-Evoked Potential (HEP) analysis on SENSE-42 raw BDF.

WHY THIS IS A GENUINELY NEW TEST
------------------------------------
Every previous EEG analysis in this project (EXP1-6, block/sliding
physio, ICA-cleaned recheck, task-identity contrasts, trait-level
correlation) collapsed EEG to 30-second, 5-second, or session-level
scalars. HEP requires the OPPOSITE: sub-second, R-peak-locked epochs,
averaged across hundreds of individual heartbeats.

The literature: each heartbeat sends an afferent signal via baroreceptors
(aortic arch, carotid sinus) through the vagus/glossopharyngeal nerves to
the nucleus tractus solitarius, then to insula, anterior cingulate, and
somatosensory cortex. This produces a measurable ERP component roughly
200-600ms post-R-peak, typically over fronto-central and right-insular-
adjacent electrodes. Amplitude varies with interoceptive attention,
emotional state, and task demand -- this IS a literature-established,
directly testable mechanism, and we have exactly the data needed for it:
ECG and EEG on the same BioSemi clock, sample-accurate, in the raw BDF.

METHOD
--------
1. Detect R-peaks in bipolar ECG (EXG2-EXG3, height=2.0 -- the derivation
   verified earlier: 9,322 peaks -> 70.0 bpm on P002, matching the
   provided .fif).
2. Epoch EEG from -100ms to +600ms around each R-peak.
3. Baseline-correct each epoch against its own -100 to 0ms pre-R-peak window
   (standard HEP practice -- removes slow drift, isolates the heartbeat-
   locked component).
4. Reject epochs with amplitude > REJECT_UV (artifact rejection -- HEP is
   a small effect, ~1-3 uV, easily swamped by motion/muscle artifact).
5. Average across all clean epochs per participant, per task type.
6. Extract HEP amplitude in the literature-standard window (250-450ms
   post R-peak) at fronto-central electrodes.
7. Compare HEP amplitude across task types (paired, same design as the
   task-identity contrasts) and correlate against behavioral load markers.

USES RAW BDF, NOT ICA-CLEANED .set
--------------------------------------
The cleaned .set files had 4 ICA components removed, likely including
cardiac field artifact (CFA) components -- exactly the signal HEP wants
to measure. Using the cleaned files here would remove the effect we're
trying to detect. Raw BDF is correct for this specific analysis, unlike
every other EEG analysis in this project where CFA was the confound to
avoid, not the signal to keep.

This means the standard CFA control-feature logic does NOT apply the same
way here -- CFA and genuine HEP are both time-locked to the heartbeat and
cannot be fully separated by a broadband control feature. The result
should be interpreted as "cardiac-locked EEG response" without strong
claims about how much is neurogenic HEP vs residual CFA. What CAN be
tested is whether this cardiac-locked response DIFFERS by task type or
correlates with behavior -- if it does, that is task-modulated information
regardless of its precise physiological source, which is itself a
meaningful finding for the aux-target question.

CHANNELS
----------
Fronto-central cluster where HEP is classically largest:
    Fz, FC1, FC2, Cz  (adjust FRONTOCENTRAL list if these aren't present)

MEMORY
--------
One participant at a time. BDF extracted (~1.1GB), processed, deleted.
Resumable via the cached epoch-average CSV.

Run from: ~/biosignals_data/
Output:
    outputs/sense42_hep_amplitudes.csv     per participant x task HEP amplitude
    outputs/sense42_hep_results.json       contrast tests + behavior correlation
"""
from __future__ import annotations
import os, sys, glob, gc, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import ttest_rel, spearmanr
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
OUT_AMP  = os.path.join(OUT_DIR, "sense42_hep_amplitudes.csv")
OUT_JSON = os.path.join(OUT_DIR, "sense42_hep_results.json")
EV_CSV   = os.path.join(OUT_DIR, "sense42_v2_events.csv")   # for HCI join

KEEP_FILES   = False
PRE_MS       = 100     # baseline window start (ms before R-peak)
POST_MS      = 600     # epoch end (ms after R-peak)
HEP_WIN      = (250, 450)   # literature-standard HEP measurement window (ms)
REJECT_UV    = 100     # reject epochs with |amplitude| exceeding this
MIN_EPOCHS   = 50       # minimum clean epochs to trust an average
MIN_FOLDS    = 8

FRONTOCENTRAL = ['Fz', 'FC1', 'FC2', 'Cz']

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

CONTRASTS = [
    ("notes", "mail",     "sustained typing vs reading"),
    ("file_mgr", "mail",  "motor vs reading"),
    ("notes", "browser",  "composition vs navigation"),
]


# ══════════════════════════════════════════════════════════════════════
# R-peak detection (verified derivation)
# ══════════════════════════════════════════════════════════════════════

def detect_rpeaks(ecg, sf):
    b, a = butter(3, [0.5/(sf/2), 40.0/(sf/2)], btype="band")
    z = filtfilt(b, a, ecg)
    z = (z - z.mean()) / (z.std() + 1e-9)
    peaks, _ = find_peaks(z, distance=int(0.35 * sf), height=2.0)
    return peaks


# ══════════════════════════════════════════════════════════════════════
# HEP epoching
# ══════════════════════════════════════════════════════════════════════

def epoch_around_rpeaks(eeg, rpeaks, sf, task_labels_by_sample=None):
    """
    eeg: (n_channels, n_samples)
    rpeaks: sample indices of R-peaks
    Returns:
      epochs_by_app: dict app -> list of (n_channels, n_epoch_samples) clean epochs
      rr_by_app:     dict app -> list of R-R intervals (seconds) FOLLOWING
                     each epoched R-peak. This is the confound check: if an
                     app's median R-R falls below POST_MS, the HEP epoch for
                     that app systematically runs into the next heartbeat's
                     QRS complex, which can masquerade as a task effect.
    """
    pre  = int(PRE_MS  / 1000 * sf)
    post = int(POST_MS / 1000 * sf)
    n_samp = eeg.shape[1]

    rr_sec = np.diff(rpeaks) / sf   # rr_sec[i] = interval AFTER rpeaks[i]

    epochs_by_app = {}
    rr_by_app = {}
    for i, r in enumerate(rpeaks):
        s0, s1 = r - pre, r + post
        if s0 < 0 or s1 >= n_samp:
            continue
        ep = eeg[:, s0:s1].copy()

        # baseline correct: subtract mean of pre-R-peak window
        baseline = ep[:, :pre].mean(axis=1, keepdims=True)
        ep -= baseline

        # artifact rejection
        if np.max(np.abs(ep)) > REJECT_UV:
            continue

        app = "all"
        if task_labels_by_sample is not None:
            app = task_labels_by_sample(r)
            if app is None:
                continue

        epochs_by_app.setdefault(app, []).append(ep)
        if i < len(rr_sec):
            rr_by_app.setdefault(app, []).append(float(rr_sec[i]))

    return epochs_by_app, rr_by_app


# ══════════════════════════════════════════════════════════════════════
# Per-participant processing
# ══════════════════════════════════════════════════════════════════════

def process_participant(pid):
    print(f"  P{pid}", end="", flush=True)

    bdf = os.path.join(BDF_DIR, f"P{pid}.bdf")
    tmp = False
    if not os.path.isfile(bdf):
        rc = os.system(f'unzip -q -o "{BDF_ZIP}" "EEG_raw/P{pid}.bdf" '
                       f'-d "{SENSE}" 2>/dev/null')
        tmp = rc == 0 and os.path.isfile(bdf)
    if not os.path.isfile(bdf):
        print(" no bdf"); return None

    try:
        raw = mne.io.read_raw_bdf(bdf, preload=True, verbose=False)
        sf = raw.info["sfreq"]

        if "EXG2" not in raw.ch_names or "EXG3" not in raw.ch_names:
            print(" no ECG channels"); return None
        ecg = (raw.get_data(picks=["EXG2"])[0] - raw.get_data(picks=["EXG3"])[0])
        rpeaks = detect_rpeaks(ecg, sf)
        if len(rpeaks) < 200:
            print(f" too few R-peaks ({len(rpeaks)})"); return None
        print(f" {len(rpeaks)} R-peaks", end="")

        fc_idx = [raw.ch_names.index(c) for c in FRONTOCENTRAL if c in raw.ch_names]
        if not fc_idx:
            print(" no frontocentral channels"); return None

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        eeg = raw.get_data(picks=eeg_picks) * 1e6   # V -> uV

        # bandpass 0.5-40 Hz for HEP (standard practice, removes drift
        # without removing the slow HEP component itself)
        b, a = butter(3, [0.5/(sf/2), 40/(sf/2)], btype="band")
        eeg_filt = filtfilt(b, a, eeg, axis=1)

        # ── triggers for task labeling ──────────────────────────────
        ev = mne.find_events(raw, stim_channel="Status",
                             min_duration=1/sf, verbose=False)
        ev[:, 2] = ev[:, 2] & 0xFF
        task_events = sorted([(int(s), int(c)) for s, _, c in ev
                              if c in TASK_CODES])

        def label_at(sample):
            app = None
            for s, code in task_events:
                if s <= sample:
                    app = TASK_APP.get(TASK_CODES[code])
                else:
                    break
            return app

        epochs_by_app, rr_by_app = epoch_around_rpeaks(
            eeg_filt[fc_idx], rpeaks, sf, task_labels_by_sample=label_at)

        rows = []
        for app, epochs in epochs_by_app.items():
            if len(epochs) < MIN_EPOCHS:
                continue
            arr = np.array(epochs)          # (n_epochs, n_fc_ch, n_samples)
            avg = arr.mean(axis=0).mean(axis=0)   # grand average waveform

            t_axis = np.linspace(-PRE_MS, POST_MS, len(avg))
            hep_mask = (t_axis >= HEP_WIN[0]) & (t_axis < HEP_WIN[1])
            hep_amp = float(avg[hep_mask].mean())

            # CONFOUND-SAFE window: 200-350ms post R-peak. Guaranteed clear
            # of the next QRS complex even at heart rates up to 170bpm
            # (RR=353ms), unlike the literature-standard 250-450ms window
            # which assumes a resting-range HR. If a task-type contrast is
            # real HEP rather than motor-driven tachycardia contaminating
            # the epoch, it should replicate here too.
            safe_mask = (t_axis >= 200) & (t_axis < 350)
            hep_amp_safe = float(avg[safe_mask].mean())

            rr_vals = np.array(rr_by_app.get(app, []), dtype=float)
            median_rr_ms = float(np.median(rr_vals) * 1000) if len(rr_vals) else np.nan
            implied_hr = float(60000 / median_rr_ms) if median_rr_ms > 0 else np.nan

            rows.append({"participant": f"P{pid}", "app": app,
                        "n_epochs": len(epochs),
                        "hep_amplitude_uv": hep_amp,
                        "hep_amplitude_safe_uv": hep_amp_safe,
                        "median_rr_ms": median_rr_ms,
                        "implied_hr_bpm": implied_hr})
            rr_flag = " !RR<POST_MS!" if median_rr_ms > 0 and median_rr_ms < POST_MS else ""
            print(f" {app}:{len(epochs)}(RR{median_rr_ms:.0f}ms{rr_flag})", end="")
        print()

        del eeg, eeg_filt, raw
        gc.collect()
        return pd.DataFrame(rows) if rows else None

    except Exception as e:
        print(f"    ERROR: {e}")
        return None
    finally:
        if tmp and not KEEP_FILES and os.path.isfile(bdf):
            os.remove(bdf)


# ══════════════════════════════════════════════════════════════════════
# Contrast analysis
# ══════════════════════════════════════════════════════════════════════

def run_contrasts(df):
    print("\n" + "=" * 78)
    print("HEP AMPLITUDE CONTRASTS (paired across participants)")
    print(f"Measurement window: {HEP_WIN[0]}-{HEP_WIN[1]}ms post R-peak, "
          f"fronto-central electrodes")
    print("=" * 78)

    results = {}
    for app_a, app_b, label in CONTRASTS:
        a = df[df.app == app_a].set_index("participant")["hep_amplitude_uv"]
        b = df[df.app == app_b].set_index("participant")["hep_amplitude_uv"]
        common = a.index.intersection(b.index)
        if len(common) < MIN_FOLDS:
            print(f"\n{app_a} vs {app_b}: too few paired participants "
                  f"({len(common)})")
            continue
        va, vb = a.loc[common].to_numpy(), b.loc[common].to_numpy()
        diff = va - vb
        t, p = ttest_rel(va, vb)
        d = float(diff.mean() / (diff.std(ddof=1) + 1e-9))
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""

        print(f"\n{app_a} vs {app_b}  —  {label}  (n={len(common)})")
        print(f"  HEP amplitude: {va.mean():+.3f} uV vs {vb.mean():+.3f} uV")
        print(f"  diff={diff.mean():+.3f}  t={t:.2f}  p={p:.4f}{star}  d={d:+.2f}")

        # ── R-R interval confound check ─────────────────────────────
        rr_a = df[df.app==app_a].set_index("participant")["median_rr_ms"]
        rr_b = df[df.app==app_b].set_index("participant")["median_rr_ms"]
        common_rr = rr_a.index.intersection(rr_b.index)
        rr_flag = False
        if len(common_rr) >= MIN_FOLDS:
            rra = rr_a.loc[common_rr].to_numpy()
            rrb = rr_b.loc[common_rr].to_numpy()
            hr_a, hr_b = 60000/rra.mean(), 60000/rrb.mean()
            print(f"  R-R interval: {rra.mean():.0f}ms vs {rrb.mean():.0f}ms  "
                  f"(implied HR {hr_a:.0f} vs {hr_b:.0f} bpm)")
            if rra.mean() < POST_MS or rrb.mean() < POST_MS:
                rr_flag = True
                print(f"  *** WARNING: R-R interval below epoch length "
                      f"({POST_MS}ms) -- HEP window may run into next QRS ***")

        # ── confound-safe window cross-check ────────────────────────
        a_s = df[df.app==app_a].set_index("participant")["hep_amplitude_safe_uv"]
        b_s = df[df.app==app_b].set_index("participant")["hep_amplitude_safe_uv"]
        common_s = a_s.index.intersection(b_s.index)
        safe_confirms = None
        if len(common_s) >= MIN_FOLDS:
            va_s = a_s.loc[common_s].to_numpy()
            vb_s = b_s.loc[common_s].to_numpy()
            diff_s = va_s - vb_s
            t_s, p_s = ttest_rel(va_s, vb_s)
            d_s = float(diff_s.mean() / (diff_s.std(ddof=1) + 1e-9))
            safe_confirms = bool(p_s < 0.05 and np.sign(d_s) == np.sign(d))
            tag = "CONFIRMS" if safe_confirms else "DOES NOT CONFIRM"
            print(f"  [confound-safe 200-350ms] d={d_s:+.2f}  p={p_s:.4f}  "
                  f"-> {tag} the original-window result")

        results[f"{app_a}_vs_{app_b}"] = {
            "n": int(len(common)), "mean_a": float(va.mean()),
            "mean_b": float(vb.mean()), "diff": float(diff.mean()),
            "t": float(t), "p": float(p), "cohens_d": d,
            "rr_confound_flag": rr_flag,
            "safe_window_confirms": safe_confirms}

    return results


def run_behavior_correlation(df):
    """HEP amplitude vs HCI intensity, within task type, per participant."""
    if not os.path.isfile(EV_CSV):
        print("\n(skipping behavior correlation -- sense42_v2_events.csv not found)")
        return {}

    print("\n" + "=" * 78)
    print("HEP AMPLITUDE vs BEHAVIORAL INTENSITY (within task type)")
    print("=" * 78)

    ev = pd.read_csv(EV_CSV)
    hci_cols = [c for c in ["SnKeyStrokes", "SnMouseDistance", "SnErrorKeys"]
               if c in ev.columns]
    if not hci_cols:
        print("No HCI columns found in cached events."); return {}

    behavior = ev.groupby(["participant", "app"])[hci_cols].mean().reset_index()
    merged = df.merge(behavior, on=["participant", "app"], how="inner")

    results = {}
    for col in hci_cols:
        x = merged["hep_amplitude_uv"].to_numpy(float)
        y = merged[col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < MIN_FOLDS:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        print(f"  HEP amplitude vs {col:16s}  rho={rho:+.3f}  p={p:.4f}{star}"
              f"  (n={ok.sum()})")
        results[col] = {"rho": float(rho), "p": float(p), "n": int(ok.sum())}

    return results


# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("SENSE-42 HEARTBEAT-EVOKED POTENTIAL (HEP) ANALYSIS")
    print("=" * 78)
    print(f"""
R-peak-locked EEG epochs, {PRE_MS}ms pre to {POST_MS}ms post, baseline-
corrected, artifact-rejected at {REJECT_UV}uV. Fronto-central cluster:
{FRONTOCENTRAL}. Measurement window: {HEP_WIN[0]}-{HEP_WIN[1]}ms.

Uses RAW BDF (not ICA-cleaned) -- the cleaned files may have removed
cardiac-related components, which is exactly what HEP measures. See
docstring for why the usual CFA-control logic doesn't directly apply here.
""")

    if os.path.isfile(OUT_AMP):
        df = pd.read_csv(OUT_AMP)
        done = set(df.participant.unique())
        print(f"Cached: {len(df)} rows, {len(done)} participants\n")
    else:
        df, done = pd.DataFrame(), set()

    pids = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(os.path.join(CSV_DIR, "*.csv"))})
    pids = [p for p in pids if f"P{p}" not in done and p != "005"]
    print(f"To process: {len(pids)} participants\n")

    for i, pid in enumerate(pids, 1):
        print(f"[{i:2d}/{len(pids)}]", end="")
        try:
            r = process_participant(pid)
        except Exception as e:
            print(f"    ERROR: {e}"); continue
        if r is not None and len(r):
            df = pd.concat([df, r], ignore_index=True) if len(df) else r
            df.to_csv(OUT_AMP, index=False)

    if df.empty:
        print("\nNo data collected."); return

    print(f"\nTotal: {len(df)} participant x task rows, "
          f"{df.participant.nunique()} participants")
    print("\nMean epochs per task type:")
    print(df.groupby("app")["n_epochs"].agg(["count","mean"]).round(0).to_string())

    contrast_results = run_contrasts(df)
    behavior_results = run_behavior_correlation(df)

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    any_sig = any(r["p"] < 0.05 for r in contrast_results.values())
    if any_sig:
        print("\nAt least one task-type contrast reached significance.")
        print("Cardiac-locked EEG response differs by task -- worth")
        print("investigating further (larger N, more careful artifact")
        print("rejection, or genuine HEP vs residual CFA follow-up).")
    else:
        print("\nNo task-type contrast reached significance.")
        print("Combined with the trait-level null and the task-identity")
        print("contrast results, this closes another angle on EEG-behavior")
        print("coupling in SENSE-42's naturalistic design. As with the")
        print("other EEG experiments, this does not contradict HEP as a")
        print("real phenomenon (established elsewhere with dedicated")
        print("paradigms) -- it means task-type variation in a naturalistic")
        print("2-hour session isn't sufficient to modulate it detectably")
        print("at this sample size.")

    out = {"contrasts": contrast_results, "behavior_correlation": behavior_results,
          "n_participants": int(df.participant.nunique())}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
