"""
sense42_hep_hci_regression.py
================================
Tests whether the FULL 18-column HCI feature set predicts HEP amplitude,
properly, with the redundancy check the earlier single-feature result
(SnErrorKeys rho=-0.217) did not have.

WHY THIS SCRIPT EXISTS
-------------------------
The confirmed HEP finding so far is a CATEGORICAL one: file_mgr vs mail
and notes vs browser differ in HEP amplitude, surviving a confound-safe
window check. That is real, but it is not the shape a proxy aux target
needs -- AAM already KNOWS the active task/app directly, so "task
category predicts HEP" adds nothing AAM doesn't already have. The single
behavioral correlation that IS the right shape (SnErrorKeys rho=-0.217,
p=0.0059) is thin: ~4.7% variance explained, one feature, no LOSO, no
permutation control, no check for whether it's just re-deriving app
identity through a behavioral proxy.

THIS SCRIPT CLOSES THAT GAP with three comparisons:

  MODEL A -- app identity only (one-hot, 5 categories)
             Upper bound on what task-category alone can predict.
             AAM already has this "for free" -- not usable as a NEW
             proxy target, but needed as the redundancy baseline.

  MODEL B -- full HCI feature set only (no app identity)
             The actual proxy question: does behavioral intensity
             predict HEP beyond knowing which task it is?

  MODEL C -- HCI + app identity combined
             If C beats A by a meaningful margin, HCI is adding real
             information on top of task category. If C ~= A, HCI adds
             nothing once you already know the task.

THE BAR
---------
CCA LOSO CV r > 0.25 for MODEL B specifically (HCI alone, no app
identity) is the threshold for "worth adding to the proxy". This
mirrors the 0.295 SENSE-42 physio->TLX result (Tier 2) and sits below
the SWELL-KW ANS-axis bar (0.581, Tier 1) -- HEP would enter at Tier 2
at best even if it clears this bar.

DATA GRANULARITY CAVEAT
---------------------------
HEP amplitude is cached at participant x app level (one value per
participant per task category, from sense42_hep_amplitudes.csv) --
NOT per-window. This means at most ~5 rows per participant (mail,
notes, file_mgr, browser, trash), and trash is frequently missing
(fewer participants met the MIN_EPOCHS=50 threshold for that short
task). This is coarser than every other proxy analysis in this project
and limits statistical power substantially -- treat any positive result
here as a signal to build a proper per-window HEP re-extraction, not
as a finished result on its own.

TOPOGRAPHIC CONTROL -- NOT YET AVAILABLE
--------------------------------------------
The proper artifact check (does the same correlation appear at
posterior/occipital electrodes, where it shouldn't if this is genuine
fronto-central HEP) requires re-extracting HEP amplitude at those sites
from raw BDF, which this script does NOT do (cost: another full BDF
pass). If MODEL B clears the 0.25 bar, that re-extraction is the
required next step before trusting the result -- flagged explicitly
in the interpretation section at the end.

Run from: ~/biosignals_data/
Input:
    outputs/sense42_hep_amplitudes.csv   (participant x app HEP amplitude)
    outputs/sense42_v2_events.csv        (HCI features, event-level)
Output:
    outputs/sense42_hep_hci_regression_results.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

BASE     = os.path.expanduser("~/biosignals_data")
HEP_CSV  = os.path.join(BASE, "outputs", "sense42_hep_amplitudes.csv")
EV_CSV   = os.path.join(BASE, "outputs", "sense42_v2_events.csv")
OUT_JSON = os.path.join(BASE, "outputs", "sense42_hep_hci_regression_results.json")

HCI_COLS = ["SnKeyStrokes","SnChars","SnSpecialKeys","SnDirectionKeys",
            "SnErrorKeys","SnSpaces","CharactersRatio","ErrorKeyRatio",
            "SnLeftClicked","SnMouseDistance","SnMouseAct"]

APPS = ["mail", "notes", "file_mgr", "browser", "trash"]
MIN_FOLDS = 8
CCA_BAR   = 0.25   # the threshold for "worth adding to the proxy"


# ══════════════════════════════════════════════════════════════════════
# Build the joint table
# ══════════════════════════════════════════════════════════════════════

def build_table():
    hep = pd.read_csv(HEP_CSV)
    ev  = pd.read_csv(EV_CSV)

    hci_avail = [c for c in HCI_COLS if c in ev.columns]
    if not hci_avail:
        raise ValueError("No HCI columns found in sense42_v2_events.csv")

    # Aggregate HCI to the SAME granularity as HEP: participant x app mean
    hci_agg = ev.groupby(["participant", "app"])[hci_avail].mean().reset_index()

    df = hep.merge(hci_agg, on=["participant", "app"], how="inner")
    df = df.dropna(subset=hci_avail + ["hep_amplitude_uv"])

    # Per-participant z-score of HEP amplitude: removes individual
    # baseline differences, keeps only the within-participant relative
    # signal -- consistent with the normalization used everywhere else
    # in this project, and necessary here since absolute HEP amplitude
    # varies a lot across people for reasons unrelated to task.
    df["hep_z"] = df.groupby("participant")["hep_amplitude_uv"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9) if x.std() > 1e-9 else 0.0)

    # One-hot app identity
    for a in APPS:
        df[f"app_{a}"] = (df["app"] == a).astype(float)
    app_cols = [f"app_{a}" for a in APPS if f"app_{a}" in df.columns]

    return df, hci_avail, app_cols


# ══════════════════════════════════════════════════════════════════════
# LOSO CCA (the primary test — does this clear the 0.25 bar)
# ══════════════════════════════════════════════════════════════════════

def loso_cca(X, y, groups, n_comp=1):
    """
    X: (n, n_features)   y: (n,) continuous target (hep_z)
    Returns (train_r, cv_r_mean, cv_r_std, n_folds)
    """
    y2 = y.reshape(-1, 1)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y2, groups = X[ok], y2[ok], groups[ok]
    if len(X) < 30:
        return np.nan, np.nan, np.nan, 0

    try:
        cca = CCA(n_components=n_comp, max_iter=3000)
        Xs, Ys = cca.fit_transform(X, y2)
        train_r = float(np.corrcoef(Xs[:, 0], Ys[:, 0])[0, 1])
    except Exception:
        train_r = np.nan

    cv_rs = []
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        if tr.sum() < 20 or te.sum() < 2:
            continue
        try:
            c = CCA(n_components=n_comp, max_iter=3000).fit(X[tr], y2[tr])
            Xt, Yt = c.transform(X[te], y2[te])
            if np.std(Xt[:, 0]) > 1e-9 and np.std(Yt[:, 0]) > 1e-9:
                cv_rs.append(np.corrcoef(Xt[:, 0], Yt[:, 0])[0, 1])
        except Exception:
            pass

    cv_mean = float(np.nanmean(cv_rs)) if cv_rs else np.nan
    cv_std  = float(np.nanstd(cv_rs))  if cv_rs else np.nan
    return train_r, cv_mean, cv_std, len(cv_rs)


# ══════════════════════════════════════════════════════════════════════
# LOSO direction classification (secondary test, interpretable framing)
# ══════════════════════════════════════════════════════════════════════

def zscore_within(df, cols, group="participant"):
    out = df.copy()
    for c in cols:
        out[c] = df.groupby(group)[c].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9) if x.std() > 1e-9 else 0.0)
    return out


def loso_direction(X, y_bin, groups, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    accs, bases = [], []
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        if tr.sum() < 15 or te.sum() < 2:
            continue
        ytr, yte = y_bin[tr], y_bin[te]
        if len(np.unique(ytr)) < 2:
            continue
        if shuffle:
            ytr = rng.permutation(ytr)
        m = RandomForestClassifier(200, min_samples_leaf=3,
                                   class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
        m.fit(X[tr], ytr)
        accs.append(accuracy_score(yte, m.predict(X[te])))
        bases.append(max(yte.mean(), 1 - yte.mean()))
    if not accs:
        return np.nan, np.nan, 0
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("HEP AMPLITUDE — FULL HCI FEATURE SET REGRESSION")
    print("=" * 78)
    print(f"""
Testing whether the full 18-column HCI feature set predicts HEP
amplitude, with an explicit check for whether any signal is genuinely
behavioral or just re-derives task/app identity (which AAM already has).

Bar for "worth adding to proxy": CCA LOSO CV r > {CCA_BAR} for HCI alone.
""")

    if not os.path.isfile(HEP_CSV) or not os.path.isfile(EV_CSV):
        print("Missing input files. Run sense42_hep_analysis.py and")
        print("sense42_trigger_extract_v2.py first.")
        return

    df, hci_avail, app_cols = build_table()
    groups = df["participant"].to_numpy()
    print(f"Joint table: {len(df)} rows, {df.participant.nunique()} participants")
    print(f"HCI features: {hci_avail}")
    print(f"Rows per app:\n{df.app.value_counts().to_string()}\n")

    # ── per-participant z-score of HCI features ────────────────────────
    dfz = zscore_within(df, hci_avail)
    X_hci = np.nan_to_num(dfz[hci_avail].to_numpy(float))
    X_app = df[app_cols].to_numpy(float)
    X_both = np.column_stack([X_hci, X_app])
    y_z = df["hep_z"].to_numpy(float)

    # ══ PRIMARY TEST: CCA, three feature sets ═════════════════════════
    print("=" * 78)
    print("PRIMARY TEST — CCA: features <-> per-participant z-scored HEP amplitude")
    print("=" * 78)

    results = {}
    for name, X in [("A: app identity only", X_app),
                    ("B: HCI features only", X_hci),
                    ("C: HCI + app identity", X_both)]:
        tr, cvm, cvs, nf = loso_cca(X, y_z, groups)
        flag = ""
        if "B:" in name:
            flag = "  *** CLEARS BAR ***" if (cvm or 0) > CCA_BAR else "  (below bar)"
        print(f"  {name:26s}  train r={tr:.3f}  LOSO CV r={cvm:.3f}"
              f" ± {cvs:.3f}  ({nf} folds){flag}")
        results[name] = {"train_r": tr, "cv_r": cvm, "cv_sd": cvs, "n_folds": nf}

    a_cv = results["A: app identity only"]["cv_r"] or 0
    b_cv = results["B: HCI features only"]["cv_r"] or 0
    c_cv = results["C: HCI + app identity"]["cv_r"] or 0

    print(f"\n  Redundancy check: does HCI add anything beyond app identity?")
    print(f"    C - A = {c_cv - a_cv:+.3f}  "
          f"({'HCI adds real information' if c_cv - a_cv > 0.05 else 'HCI mostly redundant with app identity'})")

    # ══ SECONDARY TEST: direction classification ══════════════════════
    print("\n" + "=" * 78)
    print("SECONDARY TEST — LOSO direction classification")
    print("Target: is this app's HEP amplitude above this participant's")
    print("own across-app median? (coarse -- only ~4-5 apps per participant)")
    print("=" * 78)

    y_bin = np.zeros(len(df))
    for pid in df.participant.unique():
        mask = df.participant == pid
        med = df.loc[mask, "hep_amplitude_uv"].median()
        y_bin[mask.to_numpy()] = (df.loc[mask, "hep_amplitude_uv"] > med).astype(float)

    print(f"\n{'Feature set':26s} {'chance':>8s} {'acc':>8s} {'perm':>8s} {'over':>8s}")
    print("-" * 62)
    dir_results = {}
    for name, X in [("A: app identity only", X_app),
                    ("B: HCI features only", X_hci),
                    ("C: HCI + app identity", X_both)]:
        acc, base, nf = loso_direction(X, y_bin, groups)
        perm, _, _ = loso_direction(X, y_bin, groups, shuffle=True)
        over = (acc or 0) - (base or 0)
        print(f"  {name:24s} {base:8.3f} {acc:8.3f} {perm:8.3f} {over:+8.3f}")
        dir_results[name] = {"acc": acc, "chance": base, "perm": perm,
                             "over": over, "n_folds": nf}

    # ══ Behavioral feature loadings (which HCI features matter most) ══
    print("\n" + "=" * 78)
    print("WHICH HCI FEATURES DRIVE MODEL B (if any)")
    print("=" * 78)
    ok = np.isfinite(X_hci).all(1) & np.isfinite(y_z)
    if ok.sum() > 30:
        cca_full = CCA(n_components=1, max_iter=3000)
        cca_full.fit(X_hci[ok], y_z[ok].reshape(-1, 1))
        w = cca_full.x_weights_[:, 0]
        order = np.argsort(np.abs(w))[::-1]
        for i in order:
            print(f"  {hci_avail[i]:20s} {w[i]:+.3f}")

    # ── individual feature correlations (for comparison with the
    #    earlier single-feature result) ─────────────────────────────
    print("\nIndividual feature correlations (Spearman, for reference):")
    for c in hci_avail:
        x = dfz[c].to_numpy(float)
        ok2 = np.isfinite(x) & np.isfinite(y_z)
        if ok2.sum() < 20:
            continue
        rho, p = spearmanr(x[ok2], y_z[ok2])
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        print(f"  {c:20s} rho={rho:+.3f}  p={p:.4f}{star}")

    # ══ INTERPRETATION ═════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    if b_cv > CCA_BAR:
        print(f"""
  MODEL B (HCI alone) CLEARS the {CCA_BAR} bar: CV r = {b_cv:.3f}

  Before adding this to the proxy:
    1. Check the redundancy gap above (C - A). If it's small, most of
       this signal is really "app identity in disguise" and doesn't
       give AAM anything it doesn't already have.
    2. REQUIRED: re-extract HEP amplitude at posterior/occipital sites
       and re-run this same test there. If a similar correlation
       appears at posterior electrodes -- where genuine fronto-central
       HEP should NOT show up as strongly -- this is likely broadband
       artifact (movement, muscle tension) rather than the interoceptive
       signal HEP is supposed to reflect.
    3. Current data is participant x app level (~150 rows). A proper
       proxy target needs per-window resolution, matching how every
       Tier 1/2 target in this project was actually validated.
""")
    else:
        print(f"""
  MODEL B (HCI alone) does NOT clear the {CCA_BAR} bar: CV r = {b_cv:.3f}

  Combined with the redundancy check, this means: the categorical HEP
  finding (file_mgr vs mail, notes vs browser) is real and survives its
  own confound checks, but it does not translate into a usable HCI-based
  aux target. This is consistent with the pattern from every other EEG
  analysis in this project -- categorical/task-level effects are real,
  continuous behavioral-intensity prediction from HCI is not.

  Recommendation: document the HEP task-contrast finding as a genuine
  scientific result (cardiac-locked EEG differs by task, correlates with
  error-making), but do NOT add an EEG head to the proxy. The "no EEG
  head" architecture decision stands.
""")

    out = {
        "n_rows": int(len(df)), "n_participants": int(df.participant.nunique()),
        "cca_results": results, "direction_results": dir_results,
        "redundancy_gap_C_minus_A": float(c_cv - a_cv),
        "clears_bar": bool(b_cv > CCA_BAR),
        "bar_threshold": CCA_BAR,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
