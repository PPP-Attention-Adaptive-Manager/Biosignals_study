"""
train_biosignal_proxy.py
=========================
Trains the FROZEN biosignal proxy for use as an AAM auxiliary-loss target.

Run this on the machine that has SWELL-KW data. Produces a small set of
artifact files (~few hundred KB total) that get shipped to whoever applies
the proxy elsewhere (e.g. a collaborator with BEHACOM).

AUX-HEAD PHILOSOPHY
--------------------
These targets shape training only and are discarded at inference. That
means moderate-confidence signal is usable at low loss weight -- it
regularizes the shared representation without needing to be
deployment-grade on its own. But that logic only extends to REAL, tested
signal. A target that looks like signal until control-checked is worse
than no target, because it teaches a systematic bias rather than adding
harmless noise. See TIER SUMMARY below for what qualifies.

TIER SUMMARY (from the full cross-dataset investigation)
-----------------------------------------------------------
TIER 1 -- default active targets, higher aux weight (~0.3):
    CCA projection (HCI -> HR+RMSSD latent)     CV r = 0.581 (condition-
                                                 contrasted SWELL-KW)
    HR_rising           79.0% direction accuracy (chance 50%)
    RMSSD_rising        77.5%

TIER 1, FLAGGED -- trained and saved, EXCLUDED from default target set:
    RMSSD_magnitude     ~33% true-LOSO accuracy -- AT CHANCE (3-class).
        WARNING: previously cited as 84.7% / 69.9%. Both figures were
        artifacts of a NaN-handling bug: (NaN > 0).astype(float) silently
        evaluates to 0.0 instead of propagating NaN, so dropna() never
        removed rows with genuinely missing RMSSD deltas. Those NaN
        deltas then hit np.digitize(), which does not skip NaN -- it
        force-assigns every NaN to the LAST bin ("rising"), manufacturing
        a 74% class skew (should be ~33/33/33). A classifier exploiting
        that skew scored ~70% by leaning toward the majority class, not
        from real signal. Fixed here (NaN now correctly propagates,
        digitize only sees valid deltas) -- correctly measured accuracy
        is ~33%, exact chance. Diagnosed and confirmed via
        diagnose_rmssd_magnitude.py: two independent validation schemes
        (GroupKFold(10) and true per-participant LOSO) converge on
        32.9%/33.7% once the bug is fixed, and the 74%-skewed run
        reproduces 69.9-70.0% almost exactly, closing the loop. Model
        still trained/saved here in case a different window size or
        feature set recovers real signal later -- NOT a default-active
        target, and 84.7%/69.9% should never be cited again.
    SCL_rising          50.3% -- AT CHANCE. Do not use as a live target.
        WARNING: when added as a 4th CCA target alongside HR/RMSSD, CCA
        LOSO CV r DROPPED 0.581 -> 0.497 (delta -0.083) and SCL's own
        loading collapsed to +0.03 (near-zero) -- CCA discarded it as
        noise. Tonic SCL moves on a 1-3 minute timescale; the per-minute
        windows used here are too fast for it to carry signal. Kept and
        saved here (not deleted) in case future work uses longer
        aggregation windows where SCL's slow dynamics might resolve.
        DO NOT wire this into the aux head without re-validating at a
        window size matched to SCL's actual time constant.

NOT INCLUDED -- confirmed null or artifact across the SENSE-42 investigation,
do not add as an aux target at any weight:
    HCI -> EEG (any band, any window size, any model, raw or ICA-cleaned)
    HCI -> resp_bpm on naturalistic (non-condition-contrasted) data
    Task-identity -> EEG (9 of 10 pairwise contrasts failed artifact check
                          or were non-significant)
    See EEG section at the bottom of this docstring.

RESPIRATION -- folded into the ANS head, not an independent target:
    Cog Lab CatBoost resp model reaches 83.8% direction accuracy and is
    real (RSA mechanism). But adding resp as an independent 4th SWELL-KW
    CCA target also hurt the projection, for the same reason as SCL --
    it's mechanically redundant with HR via RSA, so it adds correlated
    noise rather than new information when forced into its own target
    slot. If respiration is available at inference-adjacent training data,
    use it as an auxiliary FEATURE feeding the same HR/RMSSD prediction,
    not as a fifth competing target.

OUTPUT ARTIFACTS
------------------
    proxy_artifacts/
        cca_vector.npy          CCA x-weights, component 1 (18,)
        cca_mu.npy, cca_sd.npy  per-participant normalization params (18,)
        rf_hr_model.pkl         HR direction classifier
        rf_rmssd_model.pkl      RMSSD direction classifier
        rf_rmssd_mag_model.pkl  RMSSD magnitude classifier (3-class)
        rf_scl_model.pkl        SCL direction classifier -- FLAGGED, see above
        train_mu.npy, train_sd.npy   normalization for the delta-features
                                      used by the direction/magnitude models
        metadata.json           schema, target performance, warnings --
                                 READ THIS FIRST when applying the proxy
                                 to a new dataset

Run from: wherever SWELL-KW's Excel file lives
"""
from __future__ import annotations
import os, json, warnings, glob
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
import joblib

warnings.filterwarnings("ignore")

# Robust path resolution: matches the ~/biosignals_data convention used by
# every other script in this project (BASE = os.path.expanduser(...)),
# and glob-searches for the xlsx since its exact nesting under
# data/swell_kw/ depends on how the original SWELL-KW zip unpacked.
BASE    = os.path.expanduser("~/biosignals_data")
OUT_DIR = os.path.join(BASE, "scripts", "proxy", "proxy_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

_matches = glob.glob(os.path.join(BASE, "data", "swell_kw", "**",
                                   "Behavioral-features - per minute.xlsx"),
                     recursive=True)
if not _matches:
    raise FileNotFoundError(
        "Could not find 'Behavioral-features - per minute.xlsx' under "
        f"{os.path.join(BASE, 'data', 'swell_kw')}. Check the SWELL-KW "
        "folder was extracted there, or set SWELL_FILE manually.")
SWELL_FILE = _matches[0]
print(f"Using SWELL-KW file: {SWELL_FILE}")

# The 18-column HCI schema every downstream dataset must map onto.
# This is the CONTRACT the apply script and any new dataset (BEHACOM
# included) must satisfy. See PROXY_HANDOFF.md for column definitions
# and guidance on mapping other datasets' native columns onto this schema.
HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]

PHYSIO_COLS_ACTIVE  = ["HR", "RMSSD"]          # Tier 1 default target set
PHYSIO_COLS_FLAGGED = ["SCL"]                  # trained, saved, NOT default-active


def train_cca(df):
    """
    CCA on condition-level aggregates (per participant x condition mean).
    This is the design that gave CV r=0.581 -- deliberately NOT per-minute
    windows, which is what made the SWELL-KW result different from every
    later naturalistic-dataset attempt. The condition contrast IS the
    signal source; collapsing to condition-level means is what exposes it.
    """
    agg = df.groupby(["PP","Condition"])[HCI_COLS + PHYSIO_COLS_ACTIVE + PHYSIO_COLS_FLAGGED].mean().reset_index()
    all_physio = PHYSIO_COLS_ACTIVE  # CCA trained WITHOUT SCL by default
    for col in HCI_COLS + all_physio:
        agg[col+"_z"] = agg.groupby("PP")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))
    clean = agg[[c+"_z" for c in HCI_COLS] + [c+"_z" for c in all_physio]].dropna()

    Xcca = clean[[c+"_z" for c in HCI_COLS]].to_numpy(float)
    Ycca = clean[[c+"_z" for c in all_physio]].to_numpy(float)
    cca_mu = Xcca.mean(0); cca_sd = Xcca.std(0) + 1e-9

    cca = CCA(n_components=1, max_iter=2000)
    cca.fit((Xcca - cca_mu) / cca_sd, Ycca)

    # cross-validate to report the honest number in metadata
    from sklearn.model_selection import ShuffleSplit
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
    cv_rs = []
    Xz = (Xcca - cca_mu) / cca_sd
    for tr, te in cv.split(Xz):
        c = CCA(n_components=1, max_iter=2000).fit(Xz[tr], Ycca[tr])
        Xt, Yt = c.transform(Xz[te], Ycca[te])
        if np.std(Xt) > 1e-9 and np.std(Yt) > 1e-9:
            cv_rs.append(np.corrcoef(Xt[:,0], Yt[:,0])[0,1])

    print(f"CCA LOSO-style CV r: {np.mean(cv_rs):.3f} +/- {np.std(cv_rs):.3f}  ({len(cv_rs)} folds)")

    return cca, cca_mu, cca_sd, float(np.mean(cv_rs)), float(np.std(cv_rs))


def train_direction_models(df):
    """
    RF direction classifiers on delta features (per-minute change), the
    framing that produced 79-80% accuracy vs the ~50% ceiling of
    absolute-value regression throughout this project.

    BUG FIXED HERE (found via diagnose_rmssd_magnitude.py):
    `(df_s[col+"_delta"] > 0).astype(float)` silently turns NaN deltas
    into 0.0 ("falling"), NOT NaN -- (NaN > 0) evaluates to False in
    numpy/pandas. This meant `dropna(subset=[..._rising])` was a no-op
    for genuinely missing deltas (e.g. RMSSD rows flagged 999/missing
    in the raw sheet). For RMSSD_magnitude specifically, those leftover
    NaN deltas then hit np.digitize(), which does NOT skip NaN -- it
    silently assigns every NaN to the LAST bin. Confirmed exactly:
    2688-1062=1626 rows should have been excluded and weren't; the
    resulting "rising" class ballooned to 1980 = 354+1626, an exact
    match. A classifier exploiting that manufactured 74% class skew
    scored ~70% by just leaning toward "rising" -- not real signal.
    Correctly balanced (33/33/33), true per-participant LOSO accuracy
    for RMSSD_magnitude is ~33%, i.e. exact chance. See
    RMSSD_magnitude's exclusion below.

    FIX: _rising now explicitly propagates NaN via np.where instead of
    a raw boolean comparison, so dropna() actually removes the rows it
    was always supposed to.

    ALSO FIXED: GroupKFold(min(10, n)) replaced with true per-
    participant LOSO (n_splits = n_participants), matching every other
    "LOSO-style" figure elsewhere in this project instead of silently
    using a coarser, different validation scheme under the same name.
    """
    df_s = df.sort_values(["PP","Condition"]).copy()

    targets = {}
    for col in PHYSIO_COLS_ACTIVE + PHYSIO_COLS_FLAGGED:
        df_s[col+"_delta"] = df_s.groupby(["PP","Condition"])[col].diff()
        # FIX: NaN delta -> NaN rising, not False->0.0. np.where keeps
        # the comparison result where delta is valid, NaN otherwise.
        df_s[col+"_rising"] = np.where(
            df_s[col+"_delta"].isna(), np.nan,
            (df_s[col+"_delta"] > 0).astype(float))

    for col in HCI_COLS:
        df_s[col+"_delta"] = df_s.groupby(["PP","Condition"])[col].diff()

    HCI_DELTA = [c+"_delta" for c in HCI_COLS]

    # BUG 2 (found on this run's first attempt): the previous version
    # filtered on ALL THREE physio targets' _rising columns being
    # non-null SIMULTANEOUSLY (dropna(subset=["HR_rising","RMSSD_rising",
    # "SCL_rising"])). Before the NaN-propagation fix above, that filter
    # was a silent no-op (bug 1), so this coupling was never actually
    # exercised. Once bug 1 was fixed, this shared filter started doing
    # real work -- and SCL has known-poor raw sensor coverage (gappy,
    # unreliable throughout this project). Requiring SCL's delta to be
    # valid before HR_rising/RMSSD_rising could even see a row gutted
    # their training data for no principled reason, collapsed some LOSO
    # folds to a handful of unstable test rows, and dropped HR_rising
    # 79.9%->57.6% and RMSSD_rising 78.1%->45.6% (BELOW chance) --
    # despite neither target's own underlying relationship having
    # changed. FIX: each target now filters ONLY on its own delta being
    # valid, computed and z-scored independently, matching how SWELL-KW
    # direction accuracy was validated everywhere else in this project.

    models = {}
    accuracies = {}
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score

    train_mu = train_sd = None   # set from the first (HR) target below,
                                  # reused for CCA-application consistency

    for col in PHYSIO_COLS_ACTIVE + PHYSIO_COLS_FLAGGED:
        df_col = df_s.dropna(subset=[col+"_rising"])
        X_col = np.nan_to_num(df_col[HCI_DELTA].to_numpy(float))
        mu_col = X_col.mean(0); sd_col = X_col.std(0) + 1e-9
        Xz_col = (X_col - mu_col) / sd_col
        if train_mu is None:
            train_mu, train_sd = mu_col, sd_col   # saved for apply_biosignal_proxy.py

        y = df_col[col+"_rising"].to_numpy(int)
        groups = df_col["PP"].to_numpy()
        n_participants = len(np.unique(groups))
        print(f"  {col}: {len(df_col)} rows, {n_participants} participants "
              f"after per-target dropna")

        gkf = GroupKFold(n_splits=n_participants)   # true per-participant LOSO
        accs = []
        for tr, te in gkf.split(Xz_col, y, groups):
            m = RandomForestClassifier(200, min_samples_leaf=5,
                                       class_weight="balanced",
                                       random_state=0, n_jobs=-1)
            m.fit(Xz_col[tr], y[tr])
            accs.append(accuracy_score(y[te], m.predict(Xz_col[te])))
        acc = float(np.mean(accs))
        accuracies[col+"_rising"] = acc
        flag = "" if col in PHYSIO_COLS_ACTIVE else "  <- FLAGGED, see docstring"
        print(f"  {col}_rising true-LOSO accuracy: {acc:.3f}{flag}")

        # final model on all data for deployment
        m_final = RandomForestClassifier(200, min_samples_leaf=5,
                                         class_weight="balanced",
                                         random_state=0, n_jobs=-1)
        m_final.fit(Xz_col, y)
        models[col] = m_final

    # RMSSD magnitude uses its own independent filter too, same reasoning
    df_clean = df_s.dropna(subset=["RMSSD_rising"])
    X = np.nan_to_num(df_clean[HCI_DELTA].to_numpy(float))
    Xz = (X - train_mu) / train_sd
    groups_all = df_clean["PP"].to_numpy()

    # RMSSD magnitude (3-class): EXCLUDED as of this fix.
    # Correctly filtered (NaN deltas properly dropped, not leaked into
    # np.digitize) and correctly balanced (33/33/33 via tertile split
    # on the clean data), true per-participant LOSO accuracy is ~33% --
    # exact chance for a balanced 3-class problem. The previously-cited
    # 69.9% and 84.7% figures were both artifacts of the NaN-handling
    # bug described in this function's docstring: missing RMSSD deltas
    # got silently forced into the "rising" bin by np.digitize, and a
    # classifier exploiting that manufactured 74% class skew scored
    # ~70% by predicting the majority class. Model still trained/saved
    # here (same policy as SCL) using the CORRECTLY filtered data, in
    # case a future window size or feature set genuinely recovers
    # signal -- but it is NOT a default-active target and should not be
    # cited as 84.7%, 69.9%, or any other figure without re-verifying.
    if "RMSSD" in df_clean.columns:
        rmssd_delta = df_clean["RMSSD_delta"].to_numpy(float)
        # rmssd_delta should now be NaN-free (dropna on RMSSD_rising,
        # itself NaN-correct after the fix above) -- but guard anyway
        valid = np.isfinite(rmssd_delta)
        thresholds = np.nanpercentile(rmssd_delta[valid], [33.3, 66.7])
        y_mag_valid = np.digitize(rmssd_delta[valid], thresholds)
        Xz_mag = Xz[valid]
        groups_mag = groups_all[valid]
        n_p_mag = len(np.unique(groups_mag))

        gkf = GroupKFold(n_splits=n_p_mag)   # true per-participant LOSO
        accs = []
        for tr, te in gkf.split(Xz_mag, y_mag_valid, groups_mag):
            m = RandomForestClassifier(200, min_samples_leaf=5,
                                       class_weight="balanced",
                                       random_state=0, n_jobs=-1)
            m.fit(Xz_mag[tr], y_mag_valid[tr])
            accs.append(accuracy_score(y_mag_valid[te], m.predict(Xz_mag[te])))
        acc = float(np.mean(accs))
        accuracies["RMSSD_magnitude"] = acc
        class_dist = np.bincount(y_mag_valid) / len(y_mag_valid)
        print(f"  RMSSD_magnitude (3-class) true-LOSO accuracy: {acc:.3f}  "
              f"<- FLAGGED/EXCLUDED, see docstring (class balance: "
              f"{class_dist.round(3)}, should be ~0.333 each)")
        m_mag = RandomForestClassifier(200, min_samples_leaf=5,
                                       class_weight="balanced",
                                       random_state=0, n_jobs=-1)
        m_mag.fit(Xz_mag, y_mag_valid)
        models["RMSSD_magnitude"] = m_mag

    return models, train_mu, train_sd, accuracies


def main():
    print("=" * 70)
    print("TRAINING FROZEN BIOSIGNAL PROXY -- SWELL-KW")
    print("=" * 70)

    df = pd.read_excel(SWELL_FILE)
    df = df[df["Condition"].isin(["N","I","T"])].copy()
    for c in ["HR","RMSSD","SCL"]:
        df[c] = df[c].replace(999, np.nan)
    print(f"Loaded {len(df)} rows, {df.PP.nunique()} participants, "
          f"3 conditions (N/I/T)\n")

    print("Training CCA (condition-level, HR+RMSSD only -- SCL excluded")
    print("from CCA target set, see docstring warning)...")
    cca, cca_mu, cca_sd, cv_r, cv_sd = train_cca(df)
    print()

    print("Training direction/magnitude classifiers...")
    models, train_mu, train_sd, accuracies = train_direction_models(df)
    print()

    # ── save everything ─────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "cca_vector.npy"), cca.x_weights_[:, 0])
    np.save(os.path.join(OUT_DIR, "cca_mu.npy"), cca_mu)
    np.save(os.path.join(OUT_DIR, "cca_sd.npy"), cca_sd)
    np.save(os.path.join(OUT_DIR, "train_mu.npy"), train_mu)
    np.save(os.path.join(OUT_DIR, "train_sd.npy"), train_sd)

    joblib.dump(models["HR"],    os.path.join(OUT_DIR, "rf_hr_model.pkl"))
    joblib.dump(models["RMSSD"], os.path.join(OUT_DIR, "rf_rmssd_model.pkl"))
    if "RMSSD_magnitude" in models:
        joblib.dump(models["RMSSD_magnitude"],
                   os.path.join(OUT_DIR, "rf_rmssd_mag_model.pkl"))
    joblib.dump(models["SCL"],   os.path.join(OUT_DIR, "rf_scl_model.pkl"))

    metadata = {
        "hci_schema": HCI_COLS,
        "default_active_targets": PHYSIO_COLS_ACTIVE,
        "flagged_excluded_targets": PHYSIO_COLS_FLAGGED + ["RMSSD_magnitude"],
        "flagged_target_warnings": {
            "SCL_rising": (
                "50.3% accuracy, at chance. Adding SCL as a 4th CCA target "
                "dropped CV r from 0.581 to 0.497. Model saved for future "
                "revisit at longer aggregation windows (SCL moves on a "
                "1-3 min timescale) but must NOT be wired into the aux "
                "head as-is."
            ),
            "RMSSD_magnitude": (
                "~33% true-LOSO accuracy, exact chance for a balanced "
                "3-class problem. Previously cited as 84.7%/69.9% -- both "
                "were artifacts of a NaN-handling bug where missing RMSSD "
                "deltas leaked past dropna() and got force-assigned to "
                "the 'rising' bin by np.digitize(), manufacturing a 74% "
                "class skew that a classifier then exploited. Fixed and "
                "re-measured; do not cite 84.7% or 69.9% again. Model "
                "saved in case a different window/feature set recovers "
                "real signal, but NOT a default-active target."
            ),
        },
        "cca_cv_r_mean": cv_r,
        "cca_cv_r_std": cv_sd,
        "direction_accuracies": accuracies,
        "reference_swellkw_numbers": {
            "cca_cv_r": 0.581, "hr_direction": 0.790,
            "rmssd_direction": 0.775,
            "rmssd_magnitude_AT_CHANCE_see_warning": 0.33,
            "scl_direction_AT_CHANCE": 0.503,
        },
        "not_included_confirmed_null": [
            "HCI -> EEG (any config, see SENSE-42 investigation)",
            "HCI -> resp_bpm on naturalistic non-contrasted data",
            "Task-identity -> EEG contrasts",
            "RMSSD_magnitude (see flagged_target_warnings -- confirmed "
            "chance-level once a NaN-handling bug is fixed)",
        ],
        "training_design_note": (
            "CCA trained on CONDITION-LEVEL aggregates (participant x "
            "condition mean), not per-minute windows. This design choice "
            "is what produces CV r=0.581 -- collapsing to condition-level "
            "means is what exposes the induced-load signal. Direction "
            "classifiers use per-minute DELTA features with true per-"
            "participant LOSO (not GroupKFold(10), which was a coarser, "
            "different validation scheme silently mislabeled as "
            "'LOSO-style' in an earlier version of this script)."
        ),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print(f"Saved to {OUT_DIR}/")
    print("=" * 70)
    print("\nShip the whole proxy_artifacts/ folder plus apply_biosignal_proxy.py")
    print("and PROXY_HANDOFF.md to anyone applying this to a new dataset.")


if __name__ == "__main__":
    main()
