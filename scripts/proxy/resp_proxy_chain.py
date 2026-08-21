"""
resp_proxy_chain.py
=====================
Tests HCI -> [frozen SWELL-KW proxy] -> predicted arousal -> resp_bpm,
a fully sensor-free chain with no real ECG/skin-conductance anywhere.

THE THREE MODELS COMPARED
----------------------------
V1   (floor, already known):     HCI only
                                  -> resp direction 0.709 / magnitude 0.480
V2   (ceiling, already known):   HCI + REAL HR + RMSSD + SCL (CatBoost)
                                  -> resp direction 0.838 / magnitude 0.684
V1.5 (this script):              HCI + PREDICTED hr_rising_prob,
                                  rmssd_rising_prob, cca_load_score
                                  -> resp direction ?

THE BUG THIS VERSION FIXES
-------------------------------
The original direction_labels() compared each window to the single
immediately-preceding window. Diagnostic on Y_remaining's resp_bpm
found this captures overwhelming SESSION-LEVEL DRIFT, not short-term
fluctuation: mean "rising fraction" across all 16 Cog Lab subjects was
0.153 (i.e. ~85% of windows were "falling"), individually ranging
0.017-0.254 -- EVERY subject showed the same one-directional drift,
consistent with physiological settling/habituation over a single
uninterrupted session (elevated arousal at the start decaying toward a
resting rate), not real up-and-down variability.

This produced chance=0.847 instead of the expected ~0.50, and even V2
(HCI + REAL HR/RMSSD/SCL, which should reproduce ~83.8% direction
accuracy from the original resp gate test) failed to beat that inflated
chance -- proof the TARGET construction was broken, not the features.

FIX: per-subject linear detrending before computing direction labels.
Removes the session-level drift, isolates the residual window-to-window
variability that HCI/arousal features could plausibly track. Diagnostic
printed before/after detrending so the fix is verifiable, not assumed.

WHY THIS IS A LEGITIMATE NEW TEST, NOT A REPEAT OF THE FAILED CCA ATTEMPT
------------------------------------------------------------------------
The earlier failed attempt added resp as a 4th CCA TARGET competing in
the SAME shared subspace as HR/RMSSD -- parallel competition, and resp
lost because it's mechanically redundant with HR via RSA (CV r dropped
0.581->0.497). This is SEQUENTIAL CHAINING instead: Stage 1 (HCI->arousal,
frozen and validated) feeds Stage 2 (arousal+HCI->resp) as an
independent downstream model.

Run from: ~/biosignals_data/
Output:   outputs/resp_proxy_chain_results.json
"""
from __future__ import annotations
import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
PROXY_DIR = os.path.join(BASE, "scripts", "proxy", "proxy_artifacts")
OUT_JSON  = os.path.join(BASE, "outputs", "resp_proxy_chain_results.json")

HCI_16 = [
    "SnKeyStrokes","SnChars","SnSpecialKeys","SnDirectionKeys","SnErrorKeys",
    "SnShortcutKeys","SnSpaces","CharactersRatio","ErrorKeyRatio",
    "SnLeftClicked","SnRightClicked","SnDoubleClicked","SnWheel","SnDragged",
    "SnMouseDistance","SnMouseAct",
]
ALWAYS_NAN = {"SnRightClicked", "SnDoubleClicked", "SnDragged"}
HCI_13 = [c for c in HCI_16 if c not in ALWAYS_NAN]
HCI_18 = HCI_16 + ["SnAppChange", "SnTabfocusChange"]

MIN_FOLDS = 8


def load_cog_lab_cache():
    d = os.path.join(BASE, "data", "cache", "proxy_cache_swellstyle")
    files = sorted(glob.glob(os.path.join(d, "S*.npz")))
    print(f"Loading {len(files)} subject files from {d}")

    rows = []
    for fpath in files:
        sid = os.path.basename(fpath).replace(".npz", "")
        s = np.load(fpath, allow_pickle=True)
        swell_names     = list(s["swell_names"])
        remaining_names = list(s["remaining_names"])
        input_names     = list(s["input_names"])

        df_s = pd.DataFrame(s["X_counts"], columns=swell_names)
        df_b = pd.DataFrame(s["X_input_biosig"], columns=input_names)
        df_r = pd.DataFrame(s["Y_remaining"], columns=remaining_names)
        df_sub = pd.concat([df_s, df_b, df_r], axis=1)
        df_sub["subject"] = sid
        df_sub["window_start"] = s["starts"]
        rows.append(df_sub)

    df = pd.concat(rows, ignore_index=True)
    print(f"Loaded {len(df)} total windows across {df.subject.nunique()} subjects")
    return df


def load_proxy():
    with open(os.path.join(PROXY_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return {
        "cca_vector": np.load(os.path.join(PROXY_DIR, "cca_vector.npy")),
        "train_mu":   np.load(os.path.join(PROXY_DIR, "train_mu.npy")),
        "train_sd":   np.load(os.path.join(PROXY_DIR, "train_sd.npy")),
        "rf_hr":      joblib.load(os.path.join(PROXY_DIR, "rf_hr_model.pkl")),
        "rf_rmssd":   joblib.load(os.path.join(PROXY_DIR, "rf_rmssd_model.pkl")),
        "meta": meta,
    }


def apply_proxy(df, proxy, subject_col):
    for c in HCI_18:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)
    X = df[HCI_18].to_numpy(float)

    load_scores = np.zeros(len(df))
    hr_probs    = np.full(len(df), np.nan)
    rmssd_probs = np.full(len(df), np.nan)

    for sid in df[subject_col].unique():
        mask = (df[subject_col] == sid).to_numpy()
        idx  = df.index[mask]
        Xu   = X[mask]
        if len(Xu) < 2:
            continue
        mu = Xu.mean(0); sd = Xu.std(0) + 1e-9
        load_scores[idx] = ((Xu - mu) / sd) @ proxy["cca_vector"]

        delta   = np.diff(Xu, axis=0, prepend=Xu[[0]])
        delta_z = (delta - proxy["train_mu"]) / proxy["train_sd"]
        hr_probs[idx]    = proxy["rf_hr"].predict_proba(delta_z)[:, 1]
        rmssd_probs[idx] = proxy["rf_rmssd"].predict_proba(delta_z)[:, 1]

    df = df.copy()
    df["proxy_cca_load"]          = load_scores
    df["proxy_hr_rising_prob"]    = hr_probs
    df["proxy_rmssd_rising_prob"] = rmssd_probs
    return df


def build_models():
    m = {}
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = lambda seed: CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=seed,
            verbose=0, allow_writing_files=False)
    except ImportError:
        print("CatBoost not installed -- falling back to RF")
    from sklearn.ensemble import RandomForestClassifier
    m["RF"] = lambda seed: RandomForestClassifier(
        200, min_samples_leaf=5, class_weight="balanced",
        random_state=seed, n_jobs=-1)
    return m


def loso_eval(X, y, groups, model_fn):
    from sklearn.metrics import accuracy_score
    accs, bases = [], []
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum() < 20 or ote.sum() < 4:
            continue
        ytr, yte = y[tr][otr].astype(int), y[te][ote].astype(int)
        if len(np.unique(ytr)) < 2:
            continue
        m = model_fn(0)
        m.fit(X[tr][otr], ytr)
        accs.append(accuracy_score(yte, m.predict(X[te][ote])))
        bases.append(max(yte.mean(), 1 - yte.mean()))
    if len(accs) < MIN_FOLDS:
        return np.nan, np.nan, len(accs)
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


def detrend_per_subject(series, groups):
    """
    THE FIX. Removes each subject's linear session-level drift before
    direction labeling. resp_bpm drifts monotonically across nearly the
    whole session for ~all subjects (diagnosed: mean rising fraction
    0.153 across 16 subjects, range 0.017-0.254 -- every single subject
    showed the same one-directional drift). Window-to-window direction
    on the RAW signal was overwhelmingly capturing this drift, not
    genuine short-term fluctuation. Detrending isolates the residual
    variability that HCI/arousal features could plausibly track.
    """
    s = series.to_numpy(float)
    g = groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups)
    out = np.full(len(s), np.nan)
    for u in np.unique(g):
        idx = np.where(g == u)[0]
        y = s[idx]
        valid = np.isfinite(y)
        if valid.sum() < 5:
            continue
        x = np.arange(len(y))
        coeffs = np.polyfit(x[valid], y[valid], 1)
        trend = np.polyval(coeffs, x)
        out[idx] = y - trend
    return pd.Series(out, index=series.index)


def direction_labels(series, groups):
    out = np.full(len(series), np.nan)
    s, g = series.to_numpy(float), groups.to_numpy()
    for u in np.unique(g):
        idx = np.where(g == u)[0]
        for k in range(1, len(idx)):
            i, j = idx[k-1], idx[k]
            if np.isfinite(s[i]) and np.isfinite(s[j]):
                out[j] = float(s[j] > s[i])
    return out


def zscore_within(X, groups):
    Xz = np.zeros_like(X, dtype=float)
    for u in np.unique(groups):
        m = groups == u
        Xz[m] = (X[m] - np.nanmean(X[m], 0)) / (np.nanstd(X[m], 0) + 1e-9)
    return Xz


def main():
    print("=" * 78)
    print("RESPIRATION PROXY CHAIN — HCI -> predicted arousal -> resp_bpm")
    print("=" * 78)
    print("""
Reference:
  V1 (HCI only):                    direction 0.709  magnitude 0.480
  V2 (HCI + REAL HR/RMSSD/SCL):     direction 0.838  magnitude 0.684  <- ceiling
  V1.5 (HCI + PREDICTED arousal):   ?  <- this script

DETRENDING FIX APPLIED: direction labels now computed on the per-subject
linearly-detrended resp_bpm signal, not the raw session-drift-dominated
signal that gave chance=0.847 previously.
""")

    df = load_cog_lab_cache()
    print(f"\nColumns: {list(df.columns)}\n")

    subj_col, resp_col, hr_col, rmssd_col, scl_col = (
        "subject", "resp_bpm", "hr_mean", "hrv_rmssd", "eda_scl")
    print(f"Using: subject={subj_col}  resp={resp_col}  "
          f"HR={hr_col}  RMSSD={rmssd_col}  SCL={scl_col}")

    hci_present = [c for c in HCI_13 if c in df.columns]
    print(f"HCI features ({len(hci_present)}/13): {hci_present}")

    proxy = load_proxy()
    df = apply_proxy(df, proxy, subj_col)
    print(f"\nProxy applied. Coverage: "
          f"{df['proxy_hr_rising_prob'].notna().sum()}/{len(df)}")

    groups = df[subj_col].to_numpy()

    # ── verify the fix: rising fraction before vs after detrending ────
    print("\n" + "=" * 78)
    print("VERIFYING THE DETREND FIX")
    print("=" * 78)
    raw_dir = direction_labels(df[resp_col], df[subj_col])
    resp_detrended = detrend_per_subject(df[resp_col], df[subj_col])
    detrend_dir = direction_labels(resp_detrended, df[subj_col])

    raw_rising = np.nanmean(raw_dir)
    detrend_rising = np.nanmean(detrend_dir)
    print(f"  Rising fraction BEFORE detrending: {raw_rising:.3f}  "
          f"(chance would be {max(raw_rising,1-raw_rising):.3f})")
    print(f"  Rising fraction AFTER  detrending: {detrend_rising:.3f}  "
          f"(chance would be {max(detrend_rising,1-detrend_rising):.3f})")
    if abs(detrend_rising - 0.5) < abs(raw_rising - 0.5):
        print("  -> Detrending moved the POOLED average closer to 0.50, but that")
        print("     doesn't guarantee it fixed things PER-SUBJECT -- checking below.")
    else:
        print("  -> WARNING: detrending did not reduce the skew as expected.")

    # per-subject diagnostic: the RESULTS table's chance is an UNWEIGHTED
    # average of each held-out subject's OWN majority-class rate -- not
    # the same number as the pooled global rising fraction printed above.
    # If a handful of subjects still have strongly skewed residuals (e.g.
    # a truly nonlinear/exponential settling curve that a straight-line
    # detrend under/over-corrects), they can pull the per-subject average
    # chance much higher than the pooled figure suggests, even after the
    # pooled number looks improved.
    print("\n  Per-subject rising fraction AFTER detrending (chance = max(p,1-p)):")
    per_subj_chance = []
    for sid in df[subj_col].unique():
        m = (df[subj_col] == sid).to_numpy()
        vals = detrend_dir[m]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            continue
        p = vals.mean()
        chance_s = max(p, 1 - p)
        per_subj_chance.append(chance_s)
        flag = "  <- still heavily skewed" if chance_s > 0.70 else ""
        print(f"    {sid:6s}  rising={p:.3f}  chance={chance_s:.3f}{flag}")
    print(f"\n  Mean per-subject chance: {np.mean(per_subj_chance):.3f}  "
          f"(this is what the results table actually uses, NOT the")
    print(f"   pooled {max(detrend_rising,1-detrend_rising):.3f} printed above)")
    n_bad = sum(1 for c in per_subj_chance if c > 0.70)
    if n_bad > 0:
        print(f"\n  {n_bad}/{len(per_subj_chance)} subjects still have chance>0.70")
        print(f"  after linear detrending -- likely NONLINEAR settling curves that")
        print(f"  a straight-line fit under-corrects. A rolling-median baseline")
        print(f"  (instead of one global linear fit per subject) would handle this")
        print(f"  better if results below are still inconclusive.")

    y_resp = direction_labels(resp_detrended, df[subj_col])

    feat_sets = {
        "V1 (HCI only)": hci_present,
        "V1.5 (HCI + predicted arousal)": hci_present + [
            "proxy_cca_load", "proxy_hr_rising_prob", "proxy_rmssd_rising_prob"],
        "V2 (HCI + REAL HR/RMSSD/SCL)": hci_present + [
            c for c in [hr_col, rmssd_col, scl_col] if c],
    }

    models = build_models()
    results = {}
    print("\n" + "=" * 78)
    print("RESULTS (detrended target)")
    print("=" * 78)
    print(f"\n{'Feature set':32s} {'model':10s} {'chance':>8s} {'acc':>8s} {'over':>8s}")
    print("-" * 72)

    for name, cols in feat_sets.items():
        avail = [c for c in cols if c in df.columns]
        X = np.nan_to_num(zscore_within(df[avail].to_numpy(float), groups))
        results[name] = {}
        for mname, mfn in models.items():
            acc, base, nf = loso_eval(X, y_resp, groups, mfn)
            over = (acc or 0) - (base or 0)
            flag = " ***" if over > 0.05 else ""
            print(f"  {name:30s} {mname:10s} {base:8.3f} {acc:8.3f} "
                  f"{over:+8.3f}{flag}")
            results[name][mname] = {"acc": acc, "chance": base,
                                    "over": over, "n_folds": nf}

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    v1  = max((r["acc"] or 0) for r in results.get("V1 (HCI only)", {}).values())
    v15 = max((r["acc"] or 0) for r in results.get("V1.5 (HCI + predicted arousal)", {}).values())
    v2  = max((r["acc"] or 0) for r in results.get("V2 (HCI + REAL HR/RMSSD/SCL)", {}).values())

    print(f"\n  V1   (HCI only):                best acc = {v1:.3f}")
    print(f"  V1.5 (HCI + predicted arousal): best acc = {v15:.3f}")
    print(f"  V2   (HCI + real physio):       best acc = {v2:.3f}")
    v2_chance = max((r["chance"] or 0) for r in results.get("V2 (HCI + REAL HR/RMSSD/SCL)", {}).values())
    print(f"\n  Sanity check: V2 should approach the original resp gate test's")
    print(f"  ~83.8% direction accuracy now that chance is properly ~0.50")
    print(f"  (V2 chance this run: {v2_chance:.3f})")

    if v15 > v1 + 0.03:
        gap_closed = (v15 - v1) / (v2 - v1) if v2 > v1 else 0
        print(f"\n  V1.5 beats V1 by {v15-v1:+.3f}, closing "
              f"{100*gap_closed:.0f}% of the gap to the real-sensor ceiling.")
        print("  This is a genuine fully sensor-free respiration signal,")
        print("  chained entirely through HCI and the frozen proxy.")
    else:
        print(f"\n  V1.5 does not clearly beat V1 ({v15-v1:+.3f}).")
        print("  The proxy's predicted arousal doesn't carry enough signal")
        print("  to help resp prediction beyond what HCI alone gives.")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
