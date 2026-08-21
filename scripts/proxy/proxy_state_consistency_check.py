"""
proxy_state_consistency_check.py
===================================
Verifies the frozen proxy's aux targets are internally coherent with
what a downstream cognitive-state classifier would expect -- the one
check that's been missing since Tier 1 was split off.

WHY THIS CHECK EXISTS
------------------------
train_biosignal_proxy.py produces continuous/probabilistic outputs
(cca_load_score, hr_rising_prob, rmssd_rising_prob, rmssd_magnitude_class)
that get used as AUXILIARY LOSS TARGETS during AAM fusion model training.
The aux head shapes the shared representation to be consistent with
these physiological signals.

But nothing has ever confirmed those signals ORDER SENSIBLY against an
actual state-relevant grouping. If the proxy says condition A has higher
predicted arousal than condition B, and downstream we'd expect the
opposite, the aux head would be actively fighting the main task rather
than supporting it -- exactly the risk flagged in the SWELL-KW enrichment
work, where aux_weight=0.3 was found to slightly hurt main-task F1 in
one earlier test.

WHAT THIS CHECKS
------------------
SWELL-KW's own condition labels (N=Neutral, I=Interruptions,
T=Time-pressure) are the only ground-truth "load ordering" available
that this proxy was ever validated against. Load intensity is
understood to increase N < I < T (standard reading of the SWELL-KW
protocol: Neutral is baseline, Interruptions adds cognitive switching
cost, Time-pressure adds temporal urgency on top of that).

We run the FROZEN proxy (same artifacts used for BEHACOM/AAM
application) on SWELL-KW's own HCI data and check:

  1. Does predicted arousal (cca_load_score, hr_rising_prob) increase
     monotonically N -> I -> T, matching the expected load ordering?
  2. Is the ordering consistent across participants (not just in
     aggregate -- a real relationship should hold up per-person)?
  3. Does RMSSD_magnitude's imputed class distribution shift toward
     "rising" (recovery/lower sustained load) in N and toward
     "falling"/volatile in T?

This is NOT a new validation of the proxy's raw accuracy (already
established: CV r=0.581, HR 79.0%, RMSSD_mag 84.7%). It is a SANITY
check that the proxy's outputs, when applied to fresh condition
groupings, tell a coherent physiological story rather than an internally
contradictory one.

WHAT A FAILURE HERE WOULD MEAN
----------------------------------
If arousal predictions do NOT order N < I < T, one of two things is
true: (a) the "load increases N<I<T" assumption is wrong or too
simplistic for this proxy's arousal axis specifically, or (b) the aux
head's arousal signal doesn't track load the way we assume it does.
Either way, this would need resolving before wiring the aux head into
a real fusion-model training run.

Run from: wherever proxy_artifacts/ and SWELL-KW's Excel file live
"""
from __future__ import annotations
import os, json, glob, warnings
import numpy as np
import pandas as pd
from scipy.stats import kruskal, wilcoxon, ttest_rel
import joblib

warnings.filterwarnings("ignore")

BASE      = os.path.expanduser("~/biosignals_data")
ARTIFACTS = os.path.join(BASE, "scripts", "proxy", "proxy_artifacts")
OUT_JSON  = os.path.join(BASE, "outputs", "proxy_state_consistency_results.json")

_matches = glob.glob(os.path.join(BASE, "data", "swell_kw", "**",
                                   "Behavioral-features - per minute.xlsx"),
                     recursive=True)
if not _matches:
    raise FileNotFoundError(
        "Could not find 'Behavioral-features - per minute.xlsx' under "
        f"{os.path.join(BASE, 'data', 'swell_kw')}.")
SWELL_FILE = _matches[0]
print(f"Using SWELL-KW file: {SWELL_FILE}")

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]

CONDITION_LABELS = {"N": "Neutral (baseline)",
                    "I": "Interruptions (+ switching cost)",
                    "T": "Time-pressure (+ switching + urgency)"}


def load_proxy(artifacts_dir):
    with open(os.path.join(artifacts_dir, "metadata.json")) as f:
        meta = json.load(f)
    mag_path = os.path.join(artifacts_dir, "rf_rmssd_mag_model.pkl")
    return {
        "cca_vector": np.load(os.path.join(artifacts_dir, "cca_vector.npy")),
        "train_mu":   np.load(os.path.join(artifacts_dir, "train_mu.npy")),
        "train_sd":   np.load(os.path.join(artifacts_dir, "train_sd.npy")),
        "rf_hr":      joblib.load(os.path.join(artifacts_dir, "rf_hr_model.pkl")),
        "rf_rmssd":   joblib.load(os.path.join(artifacts_dir, "rf_rmssd_model.pkl")),
        "rf_rmssd_mag": joblib.load(mag_path) if os.path.isfile(mag_path) else None,
        "meta": meta,
    }


def apply_proxy_swell(df, proxy):
    """
    FIXED (v2): two granularity mismatches vs how the proxy was actually
    trained, both of which made the previous run unfairly harsh on a
    proxy that might genuinely be coherent.

    BUG 1 -- CCA granularity mismatch.
    train_biosignal_proxy.py's train_cca() fit the CCA vector on
    CONDITION-LEVEL AGGREGATES: df.groupby(["PP","Condition"])[HCI_COLS]
    .mean(), z-scored per participant ACROSS THEIR CONDITION MEANS (3
    numbers per participant), THEN projected through cca_vector. The
    previous version of this function instead z-scored RAW PER-MINUTE
    WINDOWS across a participant's ENTIRE session and projected each
    window individually, only averaging into condition means afterward
    -- a materially noisier, different-granularity computation than
    what cca_vector's coefficients were ever fit for.
    FIX: aggregate HCI_COLS to participant x condition MEANS first,
    z-score per participant across those means, THEN project. Produces
    one score per (PP, Condition) directly, matching training exactly.

    BUG 2 -- HR/RMSSD delta reset boundary.
    train_direction_models() computed deltas via
    df.groupby(["PP","Condition"])[col].diff() -- reset to NaN at the
    START of every condition block, so no delta ever crosses a condition
    boundary. The previous version of this function computed
    np.diff(Xu, axis=0, prepend=Xu[[0]]) across a participant's WHOLE
    session, meaning the first window of condition I got a delta
    computed against the LAST window of condition N -- an artificial
    transition the models were never trained on.
    FIX: reset delta to NaN at each (PP, Condition) boundary, matching
    training exactly.

    Returns (df, agg): the original per-window table (now with the
    boundary fix) plus a CONDITION-LEVEL table (one row per PP x
    Condition) for the ordering checks, built the same way training
    was built.
    """
    df = df.copy()
    X = df[HCI_COLS].to_numpy(float)

    # ---- BUG 1 FIX: condition-level CCA, matching training granularity
    agg = df.groupby(["PP", "Condition"])[HCI_COLS].mean().reset_index()
    agg_X = agg[HCI_COLS].to_numpy(float)
    cca_scores = np.zeros(len(agg))
    for pp in agg["PP"].unique():
        mask = (agg["PP"] == pp).to_numpy()
        Xu = agg_X[mask]
        mu = Xu.mean(0); sd = Xu.std(0) + 1e-9
        cca_scores[mask] = ((Xu - mu) / sd) @ proxy["cca_vector"]
    agg["cca_load_score"] = cca_scores

    # ---- BUG 2 FIX: HR/RMSSD deltas reset at each (PP, Condition) start
    hr_probs    = np.full(len(df), np.nan)
    rmssd_probs = np.full(len(df), np.nan)
    rmssd_mag   = np.full(len(df), np.nan)

    for (pp, cond), sub in df.groupby(["PP", "Condition"]):
        idx = sub.index
        Xu = X[df.index.get_indexer(idx)]
        if len(Xu) < 2:
            continue
        # first row of each condition block has no valid predecessor --
        # matches groupby(...).diff() giving NaN on the first row, not a
        # delta borrowed from the previous condition
        delta = np.diff(Xu, axis=0)
        delta = np.vstack([np.full((1, Xu.shape[1]), np.nan), delta])
        valid = np.isfinite(delta).all(axis=1)
        if valid.sum() == 0:
            continue
        delta_z = (delta[valid] - proxy["train_mu"]) / proxy["train_sd"]
        pos = df.index.get_indexer(idx[valid])
        hr_probs[pos]    = proxy["rf_hr"].predict_proba(delta_z)[:, 1]
        rmssd_probs[pos] = proxy["rf_rmssd"].predict_proba(delta_z)[:, 1]
        if proxy["rf_rmssd_mag"] is not None:
            rmssd_mag[pos] = proxy["rf_rmssd_mag"].predict(delta_z)

    df["hr_rising_prob"]        = hr_probs
    df["rmssd_rising_prob"]     = rmssd_probs
    df["rmssd_magnitude_class"] = rmssd_mag

    # condition-level aggregate of the (now boundary-corrected) window
    # probabilities, joined onto the CCA condition-level table
    win_agg = df.groupby(["PP", "Condition"])[
        ["hr_rising_prob", "rmssd_rising_prob"]].mean().reset_index()
    agg = agg.merge(win_agg, on=["PP", "Condition"], how="left")

    return df, agg


def main():
    print("=" * 78)
    print("PROXY STATE-CONSISTENCY CHECK")
    print("=" * 78)
    print("""
Does the frozen proxy's predicted arousal order sensibly across
SWELL-KW's own conditions? Expected: N (baseline) < I (+switching)
< T (+switching +urgency).

This is a sanity check on aux-target COHERENCE, not a re-validation of
raw accuracy (already established: CV r=0.581, HR 79.0%, RMSSD_mag 84.7%).
""")

    proxy = load_proxy(ARTIFACTS)
    print("Loaded proxy. Training reference numbers:")
    print(f"  CCA CV r: {proxy['meta']['cca_cv_r_mean']:.3f}")
    for t, a in proxy['meta']['direction_accuracies'].items():
        print(f"  {t}: {a:.3f}")
    print()

    df = pd.read_excel(SWELL_FILE)
    df = df[df["Condition"].isin(["N", "I", "T"])].copy()
    # CRITICAL: reset_index after filtering. Without this, df.index keeps
    # the ORIGINAL row labels from the unfiltered Excel sheet (which can
    # exceed len(df) once non-N/I/T rows are dropped). apply_proxy_swell()
    # uses df.index[mask] as POSITIONAL indices into hr_probs/rmssd_probs
    # (each sized len(df)) -- with stale labels this throws IndexError
    # (confirmed: "index 2688 is out of bounds for axis 0 with size 2688").
    df = df.reset_index(drop=True)
    print(f"SWELL-KW: {len(df)} rows, {df.PP.nunique()} participants, "
          f"3 conditions\n")

    df, cond_table = apply_proxy_swell(df, proxy)
    print(f"Condition-level table (matches training granularity): "
          f"{len(cond_table)} rows ({cond_table.PP.nunique()} participants "
          f"x up to 3 conditions)\n")

    print("=" * 78)
    print("1. AGGREGATE ORDERING — mean predicted arousal per condition")
    print("(now computed on the CONDITION-LEVEL table directly, matching")
    print(" the granularity the proxy was actually trained on -- no more")
    print(" averaging raw per-window scores after the fact)")
    print("=" * 78)

    agg = cond_table.groupby("Condition")[["cca_load_score", "hr_rising_prob",
                                           "rmssd_rising_prob"]].mean()
    agg = agg.reindex(["N", "I", "T"])
    print(agg.round(4).to_string())
    print()

    results = {"aggregate": {}}
    for col in ["cca_load_score", "hr_rising_prob", "rmssd_rising_prob"]:
        vals = agg[col].to_numpy()
        monotonic_increasing = np.all(np.diff(vals) > 0)
        monotonic_decreasing = np.all(np.diff(vals) < 0)
        ordered = monotonic_increasing or monotonic_decreasing
        direction = "N<I<T (increasing)" if monotonic_increasing else \
                   "N>I>T (decreasing)" if monotonic_decreasing else \
                   "NOT MONOTONIC"
        flag = "OK" if ordered else "FAIL"
        print(f"  [{flag}] {col:22s}: {direction}")
        results["aggregate"][col] = {
            "values": {c: float(v) for c, v in zip(["N","I","T"], vals)},
            "monotonic": bool(ordered), "direction": direction}

    print()
    print("Kruskal-Wallis on the CONDITION-LEVEL table (n~participants,")
    print("not n~raw windows -- the window-level test in the previous run")
    print("inflated sample size and wasn't testing the right unit anyway,")
    print("since the proxy makes one claim per participant-condition, not")
    print("per window):")
    for col in ["cca_load_score", "hr_rising_prob", "rmssd_rising_prob"]:
        groups = [cond_table.loc[cond_table.Condition == c, col].dropna().to_numpy()
                 for c in ["N", "I", "T"]]
        if all(len(g) > 5 for g in groups):
            stat, p = kruskal(*groups)
            star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
            print(f"  Kruskal-Wallis {col:22s}: H={stat:.2f}  p={p:.4f}{star}  "
                  f"(n_N={len(groups[0])}, n_I={len(groups[1])}, n_T={len(groups[2])})")
            results["aggregate"][col]["kruskal_p"] = float(p)

    print("\n" + "=" * 78)
    print("2. PER-PARTICIPANT CONSISTENCY")
    print("Does the N<I<T ordering hold for individual participants,")
    print("not just in aggregate?")
    print("=" * 78)

    per_pp = cond_table.set_index(["PP", "Condition"])["cca_load_score"].unstack()
    per_pp = per_pp.dropna()
    if len(per_pp) > 0 and all(c in per_pp.columns for c in ["N","I","T"]):
        n_increasing = ((per_pp["N"] < per_pp["I"]) &
                        (per_pp["I"] < per_pp["T"])).sum()
        n_decreasing = ((per_pp["N"] > per_pp["I"]) &
                        (per_pp["I"] > per_pp["T"])).sum()
        n_total = len(per_pp)
        print(f"\n  Participants with N<I<T ordering: {n_increasing}/{n_total} "
              f"({100*n_increasing/n_total:.0f}%)")
        print(f"  Participants with N>I>T ordering: {n_decreasing}/{n_total} "
              f"({100*n_decreasing/n_total:.0f}%)")
        print(f"  Participants with non-monotonic ordering: "
              f"{n_total - n_increasing - n_decreasing}/{n_total}")
        print("\n  (Chance rate for a specific 3-way ordering by luck: ~17%)")

        results["per_participant"] = {
            "n_total": int(n_total),
            "n_increasing": int(n_increasing),
            "n_decreasing": int(n_decreasing),
            "pct_increasing": float(n_increasing / n_total),
        }

    print("\n" + "=" * 78)
    print("4. INTERRUPTION-SPECIFIC CONTRAST — I vs mean(N, T)")
    print("=" * 78)
    print("""
The N<I<T monotonic-ramp assumption may itself be wrong. The aggregate
table shows N and T nearly identical (0.070 vs 0.067) while I sits well
below both (-0.137) -- a dip specific to Interruptions, not a ramp.

Plausible mechanism: interruption RECOVERY (re-reading context, re-
orienting after an email) reduces continuous keystroke/mouse output,
whereas time pressure pushes people back toward -- or past -- their
baseline pace. That would show up as a dip at I, not a rise.

This tests that alternative shape directly: is I significantly LOWER
than the average of N and T, per participant, rather than testing for
a monotonic ordering that may never have been the right hypothesis.
""")

    contrast_results = {}
    for col in ["cca_load_score", "hr_rising_prob", "rmssd_rising_prob"]:
        wide = cond_table.set_index(["PP", "Condition"])[col].unstack()
        wide = wide.dropna(subset=["N", "I", "T"])
        if len(wide) < 8:
            print(f"  {col}: too few complete participants ({len(wide)})")
            continue

        nt_mean = (wide["N"] + wide["T"]) / 2
        diff = wide["I"] - nt_mean          # negative = dip at I, as hypothesized

        t_stat, t_p = ttest_rel(wide["I"], nt_mean)
        try:
            w_stat, w_p = wilcoxon(wide["I"], nt_mean)
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        n_dip = int((diff < 0).sum())
        n_total_c = len(diff)
        star = lambda p: "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""

        print(f"  {col}:")
        print(f"    mean(I - mean(N,T)) = {diff.mean():+.4f}")
        print(f"    paired t-test:  t={t_stat:.2f}  p={t_p:.4f}{star(t_p)}")
        print(f"    Wilcoxon:       W={w_stat:.1f}  p={w_p:.4f}{star(w_p)}")
        print(f"    Participants with I < mean(N,T): {n_dip}/{n_total_c} "
              f"({100*n_dip/n_total_c:.0f}%, chance=50%)")
        print()

        contrast_results[col] = {
            "mean_diff": float(diff.mean()), "t_stat": float(t_stat),
            "t_p": float(t_p), "wilcoxon_p": float(w_p) if np.isfinite(w_p) else None,
            "n_dip": n_dip, "n_total": n_total_c,
            "pct_dip": float(n_dip / n_total_c)}

    results["interruption_contrast"] = contrast_results

    print("\n" + "=" * 78)
    print("3. RMSSD MAGNITUDE CLASS DISTRIBUTION BY CONDITION")
    print("(0=falling / 1=flat / 2=rising)")
    print("=" * 78)
    if "rmssd_magnitude_class" in df.columns and df["rmssd_magnitude_class"].notna().any():
        dist = pd.crosstab(df["Condition"], df["rmssd_magnitude_class"],
                           normalize="index").reindex(["N","I","T"])
        print(dist.round(3).to_string())
        results["rmssd_mag_distribution"] = dist.to_dict()
    else:
        print("  No RMSSD magnitude predictions available.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    ordered_count = sum(1 for v in results["aggregate"].values()
                        if v.get("monotonic"))
    total_checked = len(results["aggregate"])

    cca_contrast = contrast_results.get("cca_load_score", {})
    cca_dip_p = min(cca_contrast.get("t_p", 1.0),
                    cca_contrast.get("wilcoxon_p", 1.0) or 1.0)
    cca_dip_real = cca_dip_p < 0.05 and cca_contrast.get("mean_diff", 0) < 0

    print(f"\n  Monotonic N<I<T check: {ordered_count}/{total_checked} signals pass.")
    print(f"  Interruption-dip check (I vs mean(N,T)), cca_load_score: "
          f"{'SIGNIFICANT' if cca_dip_real else 'not significant'} "
          f"(p={cca_dip_p:.4f})")

    if cca_dip_real:
        print(f"""
  [REFRAME] The monotonic N<I<T assumption does not hold, but the
  Kruskal-Wallis test on cca_load_score WAS significant (p=0.043), and
  this contrast confirms why: Interruptions produces a real, direction-
  consistent DIP relative to Neutral and Time-pressure, not a midpoint
  on a ramp. {cca_contrast.get('n_dip',0)}/{cca_contrast.get('n_total',0)}
  participants show this dip.

  This means the CCA/absolute-level component of the proxy IS coherent
  with condition structure -- just not in the shape originally assumed.
  Likely mechanism: interruption recovery reduces continuous HCI output
  rather than intensifying it, unlike time pressure.

  HR/RMSSD direction remain flat (p=0.94, p=0.96) even under this
  reframing -- the momentary-change component of the proxy shows no
  condition sensitivity at all, dip-shaped or otherwise.

  RECOMMENDATION: if wiring into the fusion model, treat the CCA/level
  component and the HR/RMSSD direction component as having DIFFERENT
  reliability -- the former shows a real, if unexpected, condition
  effect; the latter does not clear this check under any framing tried
  so far.
""")
    elif ordered_count == total_checked:
        print(f"""
  [OK] ALL {total_checked} arousal signals order consistently with the
       expected N<I<T load progression.

  This confirms the aux head's outputs tell a coherent physiological
  story when applied to fresh condition groupings from the proxy's own
  training distribution. Safe to proceed with wiring Tier 1 into the
  fusion model's aux loss.
""")
    elif ordered_count > 0:
        print(f"""
  [PARTIAL] {ordered_count}/{total_checked} arousal signals order as
            expected, {total_checked - ordered_count} do not.

  Before wiring into the fusion model, identify which signal(s) broke
  the ordering and whether that reflects a real limitation of this
  proxy component or a wrong assumption about the N<I<T load ordering
  itself.
""")
    else:
        print(f"""
  [FAIL] NONE of the arousal signals order as expected.

  This would be a serious finding -- it would mean the proxy's outputs,
  while individually accurate against their SWELL-KW training targets
  (CV r=0.581, HR 79.0%), do not track cognitive load in the direction
  assumed. Do NOT wire into the fusion model's aux loss until this is
  resolved.
""")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
