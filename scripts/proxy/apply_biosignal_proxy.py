"""
apply_biosignal_proxy.py
==========================
Applies the FROZEN biosignal proxy (trained on SWELL-KW) to any HCI
dataset that has been mapped onto the 18-column schema. Written for
BEHACOM but works on any correctly-formatted dataset.

WHAT THIS PRODUCES
--------------------
Per-window pseudo-physiological labels, usable two ways (both valid,
pick based on what you're doing):

  1. AUX-HEAD TARGETS (primary intended use)
     Feed these columns to a fusion model as auxiliary training targets.
     They shape the internal representation during training and are
     discarded at inference -- the model never needs real biosignals
     to run. See PROXY_HANDOFF.md for the full rationale.

  2. DATASET ENRICHMENT / VALIDATION
     Use the outputs to sanity-check whether BEHACOM's behavioral
     patterns look physiologically plausible -- e.g. the "silent
     overload" check below (low activity + high predicted arousal).
     This is exploratory, not a ground-truth label.

REQUIRED INPUT SCHEMA
------------------------
A CSV with one row per time window (per-minute, matching SWELL-KW's
native granularity) and these 18 columns. See PROXY_HANDOFF.md for
exact definitions and guidance mapping BEHACOM's native columns onto
this schema -- do that mapping BEFORE running this script.

    SnMouseAct, SnLeftClicked, SnRightClicked, SnDoubleClicked,
    SnWheel, SnDragged, SnMouseDistance, SnKeyStrokes, SnChars,
    SnSpecialKeys, SnDirectionKeys, SnErrorKeys, SnShortcutKeys,
    SnSpaces, SnAppChange, SnTabfocusChange, CharactersRatio, ErrorKeyRatio

Plus a participant/user identifier column (default name: "user_id" --
change USER_COL below if BEHACOM's column is named differently).

OUTPUT COLUMNS
----------------
    cca_load_score       continuous, roughly zero-centered. Higher =
                          more "SWELL-KW-condition-like" behavioral state.
    hr_rising_prob        P(HR increasing this window), 0-1
    rmssd_rising_prob      P(RMSSD increasing this window), 0-1
    rmssd_magnitude_class  0=falling / 1=flat / 2=rising, 3-class
    scl_rising_prob         FLAGGED -- see warning printed at runtime.
                          Included for completeness, NOT recommended
                          as an active aux target (chance-level source
                          model, see metadata.json in proxy_artifacts/).

VALIDATION CHECKLIST (run before trusting the output)
--------------------------------------------------------
  1. Coverage: every row should get a value for every column. If not,
     some of the 18 input columns have NaN -- check the BEHACOM mapping.
  2. cca_load_score should be roughly zero-centered per user (it's
     z-scored during projection) -- mean close to 0, std in a
     0.1-0.5 range is typical (compare to the SWELL-KW/BEHACOM
     reference run: std range 0.232-0.432 across users).
  3. hr_rising_prob should NOT be uniformly ~0.5 for every user --
     if it is, the input features are probably too flat/constant
     (check SnMouseAct and SnKeyStrokes aren't all zero).
  4. Look for the "silent overload" pattern as a sanity check: users
     with LOW SnMouseAct but HIGH hr_rising_prob are behaviorally
     quiet but physiologically predicted-aroused -- this pattern
     appeared in the original BEHACOM validation run (2 of 12 users)
     and is a plausible, not alarming, finding if it recurs.

Run from: wherever proxy_artifacts/ and your mapped BEHACOM CSV live
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]

USER_COL = "user_id"   # CHANGE THIS if your dataset uses a different name


def load_proxy(artifacts_dir):
    """Load the frozen proxy artifacts. Prints the metadata warnings."""
    with open(os.path.join(artifacts_dir, "metadata.json")) as f:
        meta = json.load(f)

    print("=" * 70)
    print("LOADED PROXY -- key numbers from training (SWELL-KW)")
    print("=" * 70)
    print(f"  CCA CV r:              {meta['cca_cv_r_mean']:.3f} "
          f"+/- {meta['cca_cv_r_std']:.3f}")
    for target, acc in meta["direction_accuracies"].items():
        flag = "  <- FLAGGED" if "SCL" in target else ""
        print(f"  {target:24s} {acc:.3f}{flag}")
    print(f"\n  WARNING (flagged target): {meta['flagged_target_warning']}")
    print()

    proxy = {
        "cca_vector": np.load(os.path.join(artifacts_dir, "cca_vector.npy")),
        "cca_mu":     np.load(os.path.join(artifacts_dir, "cca_mu.npy")),
        "cca_sd":     np.load(os.path.join(artifacts_dir, "cca_sd.npy")),
        "train_mu":   np.load(os.path.join(artifacts_dir, "train_mu.npy")),
        "train_sd":   np.load(os.path.join(artifacts_dir, "train_sd.npy")),
        "rf_hr":      joblib.load(os.path.join(artifacts_dir, "rf_hr_model.pkl")),
        "rf_rmssd":   joblib.load(os.path.join(artifacts_dir, "rf_rmssd_model.pkl")),
        "rf_scl":     joblib.load(os.path.join(artifacts_dir, "rf_scl_model.pkl")),
        "meta":       meta,
    }
    mag_path = os.path.join(artifacts_dir, "rf_rmssd_mag_model.pkl")
    if os.path.isfile(mag_path):
        proxy["rf_rmssd_mag"] = joblib.load(mag_path)
    return proxy


def validate_schema(df):
    missing = [c for c in HCI_COLS if c not in df.columns]
    if missing:
        print("ERROR: missing required columns:")
        for c in missing:
            print(f"    {c}")
        print("\nSee PROXY_HANDOFF.md for column definitions and mapping guidance.")
        sys.exit(1)
    if USER_COL not in df.columns:
        print(f"ERROR: user identifier column '{USER_COL}' not found.")
        print(f"Available columns: {list(df.columns)}")
        print(f"Set USER_COL at the top of this script to the correct name.")
        sys.exit(1)
    nan_counts = df[HCI_COLS].isna().sum()
    if nan_counts.sum() > 0:
        print("WARNING: NaN values found in input columns:")
        print(nan_counts[nan_counts > 0].to_string())
        print("These rows will be dropped before proxy application.\n")


def apply_proxy(df, proxy):
    """Apply the frozen proxy to every window, per-user z-scored."""
    df = df.dropna(subset=HCI_COLS).reset_index(drop=True)

    X = df[HCI_COLS].to_numpy(float)

    # ── CCA load score: per-user z-score, then project ────────────────
    # Per-user normalization (not the SWELL-KW-derived cca_mu/cca_sd
    # directly) because absolute scale differs across datasets -- this
    # is the same design used for the original BEHACOM validation.
    load_scores = np.zeros(len(df))
    for uid in df[USER_COL].unique():
        mask = df[USER_COL] == uid
        Xu = X[mask]
        mu = Xu.mean(0); sd = Xu.std(0) + 1e-9
        Xz = (Xu - mu) / sd
        load_scores[mask] = Xz @ proxy["cca_vector"]
    df["cca_load_score"] = load_scores

    # ── Direction classifiers: need delta features, per-user ──────────
    hr_probs, rmssd_probs, scl_probs = (np.full(len(df), np.nan) for _ in range(3))
    mag_classes = np.full(len(df), np.nan)

    for uid in df[USER_COL].unique():
        mask = df[USER_COL] == uid
        idx = df.index[mask]
        Xu = X[mask]
        if len(Xu) < 2:
            continue
        delta = np.diff(Xu, axis=0, prepend=Xu[[0]])
        delta_z = (delta - proxy["train_mu"]) / proxy["train_sd"]

        hr_probs[idx]    = proxy["rf_hr"].predict_proba(delta_z)[:, 1]
        rmssd_probs[idx] = proxy["rf_rmssd"].predict_proba(delta_z)[:, 1]
        scl_probs[idx]   = proxy["rf_scl"].predict_proba(delta_z)[:, 1]
        if "rf_rmssd_mag" in proxy:
            mag_classes[idx] = proxy["rf_rmssd_mag"].predict(delta_z)

    df["hr_rising_prob"]       = hr_probs
    df["rmssd_rising_prob"]    = rmssd_probs
    df["rmssd_magnitude_class"] = mag_classes
    df["scl_rising_prob_FLAGGED"] = scl_probs   # named to discourage blind use

    return df


def print_validation(df):
    print("=" * 70)
    print("VALIDATION CHECKLIST")
    print("=" * 70)

    n_users = df[USER_COL].nunique()
    print(f"\n1. Coverage: {len(df)} windows, {n_users} users")
    for c in ["cca_load_score","hr_rising_prob","rmssd_rising_prob"]:
        n = df[c].notna().sum()
        print(f"   {c:24s} {n}/{len(df)} ({100*n/len(df):.1f}%)")

    print(f"\n2. CCA load score distribution (per user):")
    by_user = df.groupby(USER_COL)["cca_load_score"].agg(["mean","std"])
    print(f"   mean of per-user means: {by_user['mean'].mean():+.4f} "
          f"(expect ~0, z-scored per user)")
    print(f"   per-user std range: {by_user['std'].min():.3f} - "
          f"{by_user['std'].max():.3f} "
          f"(reference BEHACOM run: 0.232-0.432)")

    print(f"\n3. hr_rising_prob spread check:")
    hr_std = df.groupby(USER_COL)["hr_rising_prob"].mean().std()
    print(f"   std across user means: {hr_std:.4f} "
          f"({'OK -- has variation' if hr_std > 0.02 else 'WARNING -- looks flat, check input features'})")

    print(f"\n4. Silent overload check (low activity + high predicted arousal):")
    act_col = "SnMouseAct" if "SnMouseAct" in df.columns else None
    if act_col:
        summary = df.groupby(USER_COL).agg(
            activity=(act_col, "mean"),
            hr_up=("hr_rising_prob", "mean")).reset_index()
        summary = summary.sort_values("activity")
        print(summary.to_string(index=False))
        low_act_high_hr = summary[(summary.activity < summary.activity.quantile(0.3)) &
                                  (summary.hr_up > summary.hr_up.quantile(0.7))]
        if len(low_act_high_hr):
            print(f"\n   {len(low_act_high_hr)} user(s) show the silent-overload "
                  f"pattern: {low_act_high_hr[USER_COL].tolist()}")
            print("   (Plausible finding if it recurs, not an error.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="path to your mapped HCI CSV (18-column schema)")
    parser.add_argument("--artifacts", default="proxy_artifacts",
                        help="path to the proxy_artifacts/ folder")
    parser.add_argument("--output", default="proxy_output.csv")
    args = parser.parse_args()

    proxy = load_proxy(args.artifacts)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}\n")
    validate_schema(df)

    df_out = apply_proxy(df, proxy)
    df_out.to_csv(args.output, index=False)

    print_validation(df_out)

    print(f"\nSaved: {args.output}")
    print("\nFor aux-head training: use cca_load_score, hr_rising_prob,")
    print("rmssd_rising_prob, rmssd_magnitude_class as active targets.")
    print("Do NOT use scl_rising_prob_FLAGGED as a live target without")
    print("re-reading the warning in proxy_artifacts/metadata.json.")


if __name__ == "__main__":
    main()
