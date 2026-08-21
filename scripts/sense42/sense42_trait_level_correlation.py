"""
sense42_trait_level_correlation.py
=====================================
Replicates the Thayer & Lane (2000) neurovisceral integration design:
a BETWEEN-SUBJECTS correlation between resting/session-mean HRV and
frontal EEG, with NO per-participant z-scoring.

WHY THIS IS A NEW TEST, NOT A REPEAT
---------------------------------------
Every prior SENSE-42 analysis (EXP1-6, block/sliding physio, ICA-cleaned
recheck, task-identity contrasts) used per-participant z-scoring, which
by construction removes between-subject variance and keeps only
within-subject fluctuation.

Thayer & Lane's claim is explicitly a between-subjects claim: people
with chronically higher resting HRV tend to show better prefrontal
regulatory capacity, reflected in their baseline frontal EEG (higher
frontal alpha / lower frontal theta at rest, or a favorable
theta/alpha ratio, depending on the specific study).

We have never once run the analysis at the level this theory actually
makes its prediction. This script does exactly that: one number per
participant for RMSSD, one number per participant for each EEG feature,
correlated ACROSS the ~40 participants.

DATA SOURCE
-------------
Reads directly from the CACHED v2 extraction output --
outputs/sense42_v2_events.csv -- no new extraction, no BDF re-reads.
This is the cheap test: minutes to run, not hours.

Per participant:
    session_rmssd  = mean of all hrv_rmssd values across all events
                     (weighted toward longer/more reliable epochs
                      implicitly, since more heartbeats -> more valid
                      RMSSD estimates per event)
    session_hr     = mean of all hr_mean values
    session_theta  = mean of all frontal_theta values (log power)
    session_alpha  = mean of all frontal_alpha values (log power)
    session_theta_alpha = mean of theta_alpha_ratio

EXPECTED EFFECT SIZE
-----------------------
Published trait-level HRV-prefrontal correlations are typically in the
rho = 0.3-0.5 range. With ~40 participants this is exploratory --
confidence intervals will be wide -- but it is the correct test to run
before concluding trait-level coupling doesn't exist in this dataset.

CONTROLS
----------
occipital_delta and broadband_amplitude included as the same artifact
controls used throughout. If they correlate with RMSSD as strongly as
the cognitive features, that points to a shared non-neural confound
(e.g. participants with higher HR also fidget more, producing broadband
EMG contamination) rather than genuine prefrontal-vagal coupling.

Run from: ~/biosignals_data/
Output:   outputs/sense42_trait_level_results.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

BASE   = os.path.expanduser("~/biosignals_data")
EV_CSV = os.path.join(BASE, "outputs", "sense42_v2_events.csv")
OUT_J  = os.path.join(BASE, "outputs", "sense42_trait_level_results.json")

MIN_EVENTS_PER_SUBJECT = 5   # need at least this many valid events to
                             # trust a participant's session mean

EEG_FEATS = ["frontal_theta", "frontal_alpha", "theta_alpha_ratio",
             "engagement_index", "posterior_alpha"]
EEG_CTRL  = ["occipital_delta", "broadband_amplitude"]
CARDIAC   = ["hr_mean", "hrv_rmssd"]


def main():
    print("=" * 78)
    print("TRAIT-LEVEL CORRELATION -- Thayer & Lane replication")
    print("Between-subjects: session-mean RMSSD vs session-mean frontal EEG")
    print("NO per-participant z-scoring (deliberately -- see docstring)")
    print("=" * 78)

    if not os.path.isfile(EV_CSV):
        print(f"\nMissing {EV_CSV}")
        print("Run scripts/sense42/sense42_trigger_extract_v2.py first.")
        return

    ev = pd.read_csv(EV_CSV)

    # CRITICAL FIX: raw band power spans 8 orders of magnitude
    # (confirmed: 4.5e-13 to 8.3e-5), dominated by amplitude artifacts
    # (motion, muscle, electrode contact) rather than neural signal.
    # Log-transform BEFORE aggregating so rare extreme-amplitude windows
    # can't drag the session mean around. Same fix already validated
    # elsewhere in this project, missing here because this script read
    # cached raw power directly instead of recomputing from source.
    for _c in ["frontal_theta", "frontal_alpha", "posterior_alpha"]:
        if _c in ev.columns:
            _v = ev[_c].to_numpy(float)
            ev[_c] = np.where(_v > 0, np.log10(_v), np.nan)

    print(f"\nLoaded {len(ev)} events, {ev.participant.nunique()} participants")

    need_cols = CARDIAC + EEG_FEATS + EEG_CTRL
    missing = [c for c in need_cols if c not in ev.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return

    # ── Session-level aggregation: ONE number per participant ────────
    print("\nAggregating to session-level means (no z-scoring)...")
    sess = ev.groupby("participant").agg(
        n_events        = ("hrv_rmssd", "count"),
        n_rmssd_valid   = ("hrv_rmssd", lambda x: x.notna().sum()),
        n_theta_valid   = ("frontal_theta", lambda x: x.notna().sum()),
        session_rmssd   = ("hrv_rmssd", "mean"),
        session_hr      = ("hr_mean", "mean"),
        session_theta   = ("frontal_theta", "mean"),
        session_alpha   = ("frontal_alpha", "mean"),
        session_theta_alpha = ("theta_alpha_ratio", "mean"),
        session_engagement  = ("engagement_index", "mean"),
        session_post_alpha  = ("posterior_alpha", "mean"),
        session_occ_delta   = ("occipital_delta", "mean"),
        session_broadband    = ("broadband_amplitude", "mean"),
    ).reset_index()

    before = len(sess)
    sess = sess[(sess.n_rmssd_valid >= MIN_EVENTS_PER_SUBJECT) &
               (sess.n_theta_valid >= MIN_EVENTS_PER_SUBJECT)].copy()
    print(f"Participants with >= {MIN_EVENTS_PER_SUBJECT} valid RMSSD "
          f"and theta events: {len(sess)}/{before}")

    if len(sess) < 15:
        print("\nToo few participants with reliable session means for a "
              "between-subjects correlation to be meaningful.")
        print("Consider lowering MIN_EVENTS_PER_SUBJECT, but interpret")
        print("results cautiously if you do.")
        if len(sess) < 8:
            return

    print(f"\nSession-level summary (N={len(sess)} participants):")
    print(sess[["session_rmssd", "session_hr", "session_theta",
               "session_alpha", "session_theta_alpha"]].describe().round(3).to_string())

    # ── The core test: RMSSD vs each EEG feature, across participants ──
    print("\n" + "=" * 78)
    print("BETWEEN-SUBJECTS CORRELATIONS (Spearman, N=%d participants)" % len(sess))
    print("=" * 78)

    targets = {
        "session_theta":        "frontal_theta",
        "session_alpha":        "frontal_alpha",
        "session_theta_alpha":  "theta_alpha_ratio",
        "session_engagement":   "engagement_index",
        "session_post_alpha":   "posterior_alpha",
        "session_occ_delta":    "occipital_delta (CONTROL)",
        "session_broadband":    "broadband_amplitude (CONTROL)",
    }

    results = {}
    print(f"\n{'EEG feature':32s} {'rho (RMSSD)':>12s} {'p':>8s} "
          f"{'rho (HR)':>10s} {'p':>8s}")
    print("-" * 78)

    for col, label in targets.items():
        x = sess[col].to_numpy(float)
        rmssd = sess["session_rmssd"].to_numpy(float)
        hr    = sess["session_hr"].to_numpy(float)

        ok_r = np.isfinite(x) & np.isfinite(rmssd)
        ok_h = np.isfinite(x) & np.isfinite(hr)

        rho_r, p_r = (spearmanr(rmssd[ok_r], x[ok_r]) if ok_r.sum() >= 8
                      else (np.nan, np.nan))
        rho_h, p_h = (spearmanr(hr[ok_h], x[ok_h]) if ok_h.sum() >= 8
                      else (np.nan, np.nan))

        star_r = "***" if p_r < .001 else "**" if p_r < .01 else "*" if p_r < .05 else ""
        star_h = "***" if p_h < .001 else "**" if p_h < .01 else "*" if p_h < .05 else ""

        print(f"  {label:30s} {rho_r:+12.3f} {p_r:8.4f}{star_r:<3s} "
              f"{rho_h:+10.3f} {p_h:8.4f}{star_h}")

        results[col] = {
            "label": label,
            "rho_vs_rmssd": float(rho_r) if np.isfinite(rho_r) else None,
            "p_vs_rmssd":   float(p_r)   if np.isfinite(p_r)   else None,
            "rho_vs_hr":    float(rho_h) if np.isfinite(rho_h) else None,
            "p_vs_hr":      float(p_h)   if np.isfinite(p_h)   else None,
        }

    print("\n* p<.05  ** p<.01  *** p<.001")
    print("\nReference: published trait-level HRV-prefrontal correlations")
    print("are typically rho=0.3-0.5. With N=%d, treat this as exploratory." % len(sess))

    # ── Interpretation ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    cog_feats = ["session_theta", "session_alpha", "session_theta_alpha",
                 "session_engagement", "session_post_alpha"]
    ctrl_feats = ["session_occ_delta", "session_broadband"]

    cog_rhos = [abs(results[c]["rho_vs_rmssd"]) for c in cog_feats
               if results[c]["rho_vs_rmssd"] is not None]
    ctrl_rhos = [abs(results[c]["rho_vs_rmssd"]) for c in ctrl_feats
                if results[c]["rho_vs_rmssd"] is not None]

    cog_sig = [c for c in cog_feats if results[c]["p_vs_rmssd"] is not None
              and results[c]["p_vs_rmssd"] < 0.05]
    ctrl_sig = [c for c in ctrl_feats if results[c]["p_vs_rmssd"] is not None
               and results[c]["p_vs_rmssd"] < 0.05]

    mean_cog = np.mean(cog_rhos) if cog_rhos else np.nan
    mean_ctrl = np.mean(ctrl_rhos) if ctrl_rhos else np.nan

    print(f"\nMean |rho| vs RMSSD:  cognitive features = {mean_cog:.3f}   "
          f"control features = {mean_ctrl:.3f}")
    print(f"Significant (p<.05):  cognitive = {cog_sig}")
    print(f"                      control   = {ctrl_sig}")

    if cog_sig and not ctrl_sig:
        print("\n*** Cognitive features show significant trait-level coupling")
        print("    with RMSSD, controls do not. This is consistent with")
        print("    genuine neurovisceral integration (Thayer & Lane) rather")
        print("    than a shared artifact. Worth reporting even at N=%d,")
        print("    with appropriate caveats about sample size." % len(sess))
    elif cog_sig and ctrl_sig:
        print("\n~ Both cognitive and control features show significant")
        print("  correlation with RMSSD. This pattern (rather than clean")
        print("  cognitive-only signal) suggests a shared confound -- e.g.")
        print("  participants with different resting HR may also differ in")
        print("  overall movement/muscle tension, affecting broadband EEG.")
        print("  Interpret with caution; the control was not clean here.")
    elif not cog_sig:
        print("\nNo significant trait-level correlation detected at N=%d." % len(sess))
        print("This could mean: (a) no genuine effect in this population/task")
        print("context, (b) N too small to detect a rho=0.3-0.5 effect")
        print("reliably (power at N=%d for rho=0.35 is roughly 40-50%%),"  % len(sess))
        print("or (c) session-mean aggregation over a demanding task battery")
        print("isn't equivalent to a true resting-state measurement, which")
        print("is what the original trait-level literature typically uses.")

    out = {
        "n_participants": int(len(sess)),
        "min_events_threshold": MIN_EVENTS_PER_SUBJECT,
        "correlations": results,
        "mean_abs_rho_cognitive": float(mean_cog) if np.isfinite(mean_cog) else None,
        "mean_abs_rho_control":   float(mean_ctrl) if np.isfinite(mean_ctrl) else None,
        "significant_cognitive": cog_sig,
        "significant_control":   ctrl_sig,
    }
    with open(OUT_J, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {OUT_J}")


if __name__ == "__main__":
    main()
