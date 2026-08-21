"""
clare_ecg_clean.py
====================
Cleans and validates CLARE's ECG export before any R-peak detection or
downstream ECG->EEG chaining is trusted.

THE PROBLEM DIAGNOSED FROM THE RAW EXPORT
---------------------------------------------
    ecg_data_baseline_0.csv:   156,647 rows / 182.0s   -> ~861 Hz
    ecg_data_experiment_0.csv: 517,031 rows / 542.8s   -> ~952 Hz
Inconsistent apparent rate between files from the SAME device/modality,
neither a clean number (not 256, not 1024). Every other row is NaN, and
experiment_0 shows the SAME timestamp on two consecutive NaN rows --
this is a Shimmer/consensys-style multi-stream export artifact, not
missing data: the logger interleaves rows from multiple simultaneous
data streams into one file, so alternating rows belong to different
logical streams and only every Nth row is a real ECG sample.

WHAT THIS SCRIPT DOES, IN ORDER
------------------------------------
1. Load raw csv, report the raw NaN/duplicate-timestamp pattern
   explicitly (confirmed on THIS file, not assumed from one earlier
   inspection).
2. Drop all-NaN rows.
3. Deduplicate on Timestamp, keeping the first non-NaN occurrence --
   handles the repeated-timestamp NaN rows seen in experiment_0.
4. Recompute the ACTUAL sample rate from the median of clean, de-
   duplicated inter-sample timestamp diffs (not from row_count/duration,
   which is exactly what produced the inconsistent 861 vs 952 Hz
   estimates above -- that calculation was corrupted by the same
   duplicate/NaN rows this step removes).
5. Report how much of the original row count survives cleaning -- if a
   large fraction is dropped, that's flagged explicitly rather than
   silently accepted, since the goal is losing as LITTLE real
   information as possible, not just making the pipeline run.
6. R-peak detection on 'ECG LL-RA CAL' (the calibrated, mV-scale lead --
   chosen over RAW because CAL is already unit-corrected by the Shimmer
   firmware, unlike the arbitrary-ADC-count RAW columns).
7. Sanity check: resulting heart rate must land in 40-140 bpm. If it
   doesn't, cleaning or channel choice is still wrong and this stops
   here rather than feeding garbage into anything downstream.

WHY 'ECG LL-RA CAL' AND NOT THE OTHER TWO LEADS
----------------------------------------------------
Standard Einthoven-style naming: LL-RA (left leg to right arm, roughly
Lead II) typically gives the largest, cleanest QRS complex of the three
derivations. LA-RA (Lead I) and Vx-RL (a chest/precordial-style lead)
are also extracted and reported per-file so a different lead can be
picked per participant if LL-RA turns out noisy for someone specific --
electrode placement quality varies across 20 participants and picking
one lead blindly for everyone risks losing real participants
unnecessarily rather than losing real DATA.

Run from: ~/biosignals_data/
Input:  data/clare/doi-10.5683-sp3-h0aelt/ECG/<pid>/ecg_data_*.csv
Output: outputs/clare_ecg_clean_report.json
        (no cleaned CSVs written to disk yet -- this is a VALIDATION
        pass; once confirmed good, the actual R-peak/HRV extraction
        gets folded directly into the ECG->EEG chaining script rather
        than duplicating a full clean+resave step for 20 participants
        x 8 files each)
"""
from __future__ import annotations
import os, glob, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
CLARE_ROOT = os.path.join(BASE, "data", "clare", "doi-10.5683-sp3-h0aelt")
ECG_DIR = os.path.join(CLARE_ROOT, "ECG")
OUT_JSON = os.path.join(BASE, "outputs", "clare_ecg_clean_report.json")

ECG_LEADS = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]
PRIMARY_LEAD = "ECG LL-RA CAL"

MIN_HR, MAX_HR = 40, 140   # plausible resting-to-elevated adult range


def clean_ecg_file(path):
    """
    Returns (clean_df, report_dict). clean_df has columns:
    Timestamp + the 3 CAL leads, deduplicated, NaN-free, sorted.
    """
    raw = pd.read_csv(path)
    n_raw = len(raw)

    n_allnan = raw[ECG_LEADS].isna().all(axis=1).sum()
    n_dup_ts = raw["Timestamp"].duplicated().sum()

    df = raw.dropna(subset=ECG_LEADS, how="all").copy()
    df = df.sort_values("Timestamp")
    df = df.drop_duplicates(subset="Timestamp", keep="first")
    df = df.dropna(subset=[PRIMARY_LEAD])
    df = df.reset_index(drop=True)

    n_clean = len(df)
    pct_kept = 100 * n_clean / n_raw if n_raw else 0

    diffs = np.diff(df["Timestamp"].to_numpy())
    diffs = diffs[(diffs > 0) & (diffs < 1.0)]
    est_sf = float(1.0 / np.median(diffs)) if len(diffs) else np.nan

    report = {
        "file": os.path.basename(path),
        "n_raw_rows": int(n_raw),
        "n_allnan_rows": int(n_allnan),
        "n_duplicate_timestamps": int(n_dup_ts),
        "n_clean_rows": int(n_clean),
        "pct_kept": float(pct_kept),
        "estimated_sample_rate_hz": est_sf,
        "duration_s": float(df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0])
                     if n_clean > 1 else None,
    }
    return df, report


def detect_hr(df, sf, lead=PRIMARY_LEAD):
    """Bandpass + peak detection, same style validated on SENSE-42/Cog Lab."""
    sig = df[lead].to_numpy(float)
    if len(sig) < 10 or not np.isfinite(sf) or sf <= 0:
        return None, None, 0

    nyq = sf / 2
    high = min(40.0, nyq - 1)
    if high <= 0.5:
        return None, None, 0

    b, a = butter(3, [0.5/nyq, high/nyq], btype="band")
    z = filtfilt(b, a, sig)
    z = (z - z.mean()) / (z.std() + 1e-9)

    peaks = np.array([])
    for height in [2.5, 2.0, 1.5, 1.0]:
        peaks, _ = find_peaks(z, distance=int(0.35 * sf), height=height)
        if len(peaks) > 20:
            break
    if len(peaks) < 10:
        return None, None, len(peaks)

    ts = df["Timestamp"].to_numpy()
    rr = np.diff(ts[peaks])
    rr = rr[(rr > 0.33) & (rr < 1.5)]
    if len(rr) < 5:
        return None, None, len(peaks)

    hr = float(60.0 / np.median(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000) if len(rr) >= 10 else None
    return hr, rmssd, len(peaks)


def main():
    print("=" * 78)
    print("CLARE ECG CLEANING + VALIDATION")
    print("=" * 78)

    if not os.path.isdir(ECG_DIR):
        print(f"\nECG directory not found: {ECG_DIR}")
        print("Adjust CLARE_ROOT at the top of this script.")
        return

    pids = sorted(os.listdir(ECG_DIR))
    print(f"Participants found: {len(pids)}\n")

    all_reports = []

    for pid in pids:
        pdir = os.path.join(ECG_DIR, pid)
        files = sorted(glob.glob(os.path.join(pdir, "ecg_data_*.csv")))
        if not files:
            continue
        print(f"P{pid}:")

        for fpath in files:
            fname = os.path.basename(fpath)
            try:
                df, rep = clean_ecg_file(fpath)
            except Exception as e:
                print(f"    {fname}: ERROR during cleaning: {e}")
                continue

            sf = rep["estimated_sample_rate_hz"]
            flag = ""
            if rep["pct_kept"] < 70:
                flag = "  <- LOSING >30% of rows, check this file"

            dur = rep["duration_s"] or 0
            print(f"    {fname:30s} kept {rep['pct_kept']:5.1f}%  "
                  f"({rep['n_clean_rows']:6d}/{rep['n_raw_rows']:6d})  "
                  f"sf~{sf:6.1f}Hz  dur={dur:6.1f}s{flag}")

            hr, rmssd, n_peaks = detect_hr(df, sf)
            rep["participant"] = pid
            rep["detected_hr_bpm"] = hr
            rep["detected_rmssd_ms"] = rmssd
            rep["n_rpeaks"] = n_peaks

            if hr is None:
                print(f"      -> R-peak detection FAILED (n_peaks={n_peaks})")
            elif not (MIN_HR <= hr <= MAX_HR):
                print(f"      -> HR={hr:.1f} bpm OUT OF PLAUSIBLE RANGE "
                      f"({MIN_HR}-{MAX_HR}) -- do not trust this file")
            else:
                rmssd_disp = rmssd if rmssd is not None else float("nan")
                print(f"      -> HR={hr:.1f} bpm  RMSSD={rmssd_disp:.1f}ms  "
                      f"({n_peaks} peaks)  OK")

            all_reports.append(rep)
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    if not all_reports:
        print("\nNo files processed -- check ECG_DIR path.")
        return

    kept_pcts = [r["pct_kept"] for r in all_reports]
    valid_hr  = [r for r in all_reports if r["detected_hr_bpm"] is not None
                and MIN_HR <= r["detected_hr_bpm"] <= MAX_HR]
    sample_rates = [r["estimated_sample_rate_hz"] for r in all_reports
                    if r["estimated_sample_rate_hz"] and
                    np.isfinite(r["estimated_sample_rate_hz"])]

    print(f"\nFiles processed: {len(all_reports)}")
    print(f"Mean % rows kept after cleaning: {np.mean(kept_pcts):.1f}%")
    print(f"Min % rows kept: {np.min(kept_pcts):.1f}%  "
          f"(worst file -- check if this participant should be excluded)")
    if sample_rates:
        print(f"Sample rate estimates: mean={np.mean(sample_rates):.1f}Hz  "
              f"std={np.std(sample_rates):.1f}Hz  "
              f"({'CONSISTENT' if np.std(sample_rates) < 20 else 'INCONSISTENT -- investigate'})")
    print(f"\nFiles with valid HR in plausible range: "
          f"{len(valid_hr)}/{len(all_reports)} "
          f"({100*len(valid_hr)/len(all_reports):.0f}%)")
    if valid_hr:
        hrs = [r["detected_hr_bpm"] for r in valid_hr]
        print(f"HR distribution: mean={np.mean(hrs):.1f}  "
              f"min={np.min(hrs):.1f}  max={np.max(hrs):.1f} bpm")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    pct_valid = 100 * len(valid_hr) / len(all_reports) if all_reports else 0
    if pct_valid > 80 and np.mean(kept_pcts) > 85:
        print(f"""
  GOOD: {pct_valid:.0f}% of files produce plausible HR,
  {np.mean(kept_pcts):.0f}% of raw data survives cleaning on average.
  Safe to proceed to the ECG->EEG chaining test using this cleaning
  pipeline directly (no separate clean-and-resave step needed).
""")
    elif pct_valid > 50:
        print(f"""
  PARTIAL: {pct_valid:.0f}% of files usable. Some participants/sessions
  should be excluded (see per-file flags above) before the chaining
  test, rather than pooling everything and diluting the signal with
  broken files.
""")
    else:
        print(f"""
  POOR: only {pct_valid:.0f}% of files produce plausible HR. The lead
  choice or cleaning approach likely needs revisiting before this
  dataset is usable -- check which specific files/participants are
  failing and whether a different lead (LA-RA or Vx-RL) does better
  for them.
""")

    with open(OUT_JSON, "w") as f:
        json.dump(all_reports, f, indent=2, default=float)
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
