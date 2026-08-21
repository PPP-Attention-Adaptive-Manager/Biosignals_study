"""
sense42_v2_question_analysis.py
================================
First analysis on trigger-aligned SENSE-42 data.

WHY THIS ONE IS DIFFERENT FROM EVERY EARLIER SENSE-42 ANALYSIS
---------------------------------------------------------------
Every previous result was computed on data misaligned by ~104 s (and
cumulatively worse for participants who paused, since apply_pause()
rewinds the PsychoPy clock while the EEG keeps running). Those nulls
measured nothing.

Here the label IS the trigger: the rating is encoded in the trigger value
itself (100 + 10*question_index + rating), stamped into the EEG at the
exact sample the participant submitted it. Signal and label come from one
file on one clock. No alignment step exists to get wrong.

Coverage: 7,233 rows, 40 participants
    frontal_theta 100.0%   resp_bpm 99.9%   hr_mean 88.7%

STRUCTURE AND ONE IMPORTANT SUBTLETY
-------------------------------------
Rows = 40 participants x ~26 questionnaires x 7 dimensions.

The seven dimensions of a single questionnaire are answered seconds
apart, so their 30 s lookback windows OVERLAP ALMOST COMPLETELY. The
seven rows of one questionnaire therefore carry near-identical features
but different ratings.

That is not a defect if each dimension is analysed separately -- it is a
built-in control. If near-identical physiology predicts mental_demand but
not performance, the discrimination is real rather than an artifact of
the window. If it predicts all seven equally, that is a red flag for a
common-cause confound (time-on-task, say) rather than genuine construct
specificity.

METHOD -- RANK CORRELATION, NOT MEDIAN SPLIT
---------------------------------------------
Earlier work binarised ratings at each participant's median. Because
ratings are small integers, ties fell into class 0 and produced ~80/20
imbalance; RF then scored 0.794 against a 0.798 majority baseline, i.e.
below chance, while the script printed "chance=0.50" and made it look
like a win.

This analysis avoids that entirely:
  - within-participant Spearman rho between each feature and the rating
  - then a one-sample t-test on those rho values across participants
Rank correlation uses the full ordinal scale, needs no threshold, and has
no class-balance failure mode.

For the prediction step the empirical majority rate is always reported
alongside accuracy, and a permutation control (labels shuffled within
participant) is run for every cell.

Run from: ~/biosignals_data/
Output:   outputs/sense42_v2_question_results.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

BASE  = os.path.expanduser("~/biosignals_data")
QCSV  = os.path.join(BASE, "outputs", "sense42_v2_questions.csv")
OUT   = os.path.join(BASE, "outputs", "sense42_v2_question_results.json")

EEG_FEATS = ["frontal_theta", "frontal_alpha", "theta_alpha_ratio",
             "engagement_index", "posterior_alpha"]
EEG_CTRL  = ["occipital_delta", "broadband_amplitude"]   # no cognitive reading
PHY_FEATS = ["hr_mean", "resp_bpm", "resp_amp"]

DIMS = ["mental_demand", "temporal_demand", "effort", "frustration",
        "performance", "attentiveness", "sleepiness"]
# load-related dims; performance is reverse-coded, sleepiness is fatigue
LOAD_DIMS = ["mental_demand", "temporal_demand", "effort", "frustration"]


def zscore_within(df, cols, group="participant"):
    """Per-participant z-score: removes individual baselines so the model
    sees within-person variation rather than who the person is."""
    out = df.copy()
    for c in cols:
        out[c] = df.groupby(group)[c].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return out


# ══════════════════════════════════════════════════════════════════════
# 1. Within-participant rank correlations
# ══════════════════════════════════════════════════════════════════════

def correlation_analysis(q, feats):
    print("\n" + "=" * 82)
    print("1. WITHIN-PARTICIPANT RANK CORRELATION (feature vs rating)")
    print("=" * 82)
    print("Spearman rho computed inside each participant, then a one-sample")
    print("t-test on those rho values across participants. No thresholding,")
    print("no class balance to get wrong.\n")

    hdr = f"{'feature':22s}"
    for d in DIMS:
        hdr += f"{d[:9]:>11s}"
    print(hdr)
    print("-" * 82)

    results = {}
    for feat in feats:
        line = f"{feat:22s}"
        for dim in DIMS:
            sub = q[q.dimension == dim]
            rhos = []
            for pid, g in sub.groupby("participant"):
                x = g[feat].to_numpy(float)
                y = g["rating"].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 10 or np.std(y[ok]) < 1e-9 or np.std(x[ok]) < 1e-9:
                    continue
                r = stats.spearmanr(x[ok], y[ok]).statistic
                if np.isfinite(r):
                    rhos.append(r)
            if len(rhos) < 15:
                line += f"{'--':>11s}"
                continue
            rhos = np.array(rhos)
            t, p = stats.ttest_1samp(rhos, 0.0)
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            line += f"{rhos.mean():+8.3f}{star:<3s}"
            results.setdefault(feat, {})[dim] = {
                "mean_rho": float(rhos.mean()), "t": float(t),
                "p": float(p), "n_participants": int(len(rhos))}
        print(line)

    print("\n* p<.05   ** p<.01   *** p<.001   (n = participants contributing)")
    return results


# ══════════════════════════════════════════════════════════════════════
# 2. CCA on the pivoted table
# ══════════════════════════════════════════════════════════════════════

def cca_analysis(q, feats):
    print("\n" + "=" * 82)
    print("2. CCA : physiological state  <->  NASA-TLX profile")
    print("=" * 82)
    print("Pivoted to one row per questionnaire: features averaged across the")
    print("seven dimension windows (they overlap anyway), ratings as 7 columns.")
    print("Directly comparable to SWELL-KW, which reached CV r = 0.581.\n")

    piv = q.pivot_table(index=["participant", "onset_s"],
                        columns="dimension", values="rating").reset_index()
    fmean = (q.groupby(["participant", "onset_s"])[feats]
               .mean().reset_index())
    m = piv.merge(fmean, on=["participant", "onset_s"])
    m = m.dropna(subset=feats + DIMS)
    print(f"Questionnaire events with complete data: {len(m)} "
          f"({m.participant.nunique()} participants)")
    if len(m) < 200:
        print("Too few complete rows for CCA."); return None

    mz = zscore_within(m, feats + DIMS)
    X = np.nan_to_num(mz[feats].to_numpy(float))
    Y = np.nan_to_num(mz[DIMS].to_numpy(float))
    groups = m.participant.to_numpy()

    n_comp = min(3, X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_comp, max_iter=3000)
    Xs, Ys = cca.fit_transform(X, Y)
    train_r = [float(np.corrcoef(Xs[:, i], Ys[:, i])[0, 1]) for i in range(n_comp)]

    # LOSO cross-validation -- the honest generalisation estimate
    cv = [[] for _ in range(n_comp)]
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        if tr.sum() < 100 or te.sum() < 5:
            continue
        try:
            c = CCA(n_components=n_comp, max_iter=3000).fit(X[tr], Y[tr])
            Xt, Yt = c.transform(X[te], Y[te])
            for i in range(n_comp):
                if np.std(Xt[:, i]) > 1e-9 and np.std(Yt[:, i]) > 1e-9:
                    cv[i].append(np.corrcoef(Xt[:, i], Yt[:, i])[0, 1])
        except Exception:
            pass

    cv_m = [float(np.nanmean(c)) if c else np.nan for c in cv]
    cv_s = [float(np.nanstd(c)) if c else np.nan for c in cv]
    for i in range(n_comp):
        print(f"  Component {i+1}: train r={train_r[i]:.3f}   "
              f"LOSO CV r={cv_m[i]:.3f} +/- {cv_s[i]:.3f}")

    xw, yw = cca.x_weights_[:, 0], cca.y_weights_[:, 0]
    print("\n  Component 1 physiological loadings:")
    for k in np.argsort(np.abs(xw))[::-1]:
        print(f"    {feats[k]:22s} {xw[k]:+.3f}")
    print("  Component 1 TLX loadings:")
    for k in np.argsort(np.abs(yw))[::-1]:
        print(f"    {DIMS[k]:22s} {yw[k]:+.3f}")

    print(f"\n  SWELL-KW reference: CV r = 0.581 (HCI <-> HR/RMSSD/SCL,")
    print( "  condition-contrasted design, N=75 condition-level rows)")
    r1 = cv_m[0]
    if np.isfinite(r1):
        if r1 > 0.30:
            print("  -> CV r > 0.30. Physiology tracks the self-report profile.")
        elif r1 > 0.15:
            print("  -> Weak but non-zero coupling.")
        else:
            print("  -> At zero. No reliable physiological signature of the")
            print("     self-reported load profile at this granularity.")
    return {"train_r": train_r, "cv_r_mean": cv_m, "cv_r_std": cv_s,
            "n_rows": int(len(m)),
            "x_loadings": {feats[k]: float(xw[k]) for k in range(len(feats))},
            "y_loadings": {DIMS[k]: float(yw[k]) for k in range(len(DIMS))}}


# ══════════════════════════════════════════════════════════════════════
# 3. LOSO prediction with honest baselines
# ══════════════════════════════════════════════════════════════════════

def loso_predict(q, feats):
    print("\n" + "=" * 82)
    print("3. LOSO PREDICTION -- high vs low rating (tertile split)")
    print("=" * 82)
    print("Top vs bottom within-participant tertile, middle discarded. Tertiles")
    print("avoid the tie-driven imbalance that broke the earlier median split.")
    print("'chance' is the empirical majority rate of each test fold, never 0.50.")
    print("'perm' shuffles labels within the training fold.\n")

    print(f"{'dimension':18s} {'chance':>8s} {'acc':>8s} {'perm':>8s} "
          f"{'over':>8s}  n_folds")
    print("-" * 66)

    out = {}
    for dim in DIMS:
        sub = q[q.dimension == dim].copy()
        sub = sub.dropna(subset=feats)
        if len(sub) < 300:
            print(f"{dim:18s}  too few rows"); continue

        keep, lab = [], []
        for pid, g in sub.groupby("participant"):
            r = g["rating"].to_numpy(float)
            lo, hi = np.percentile(r, [33.3, 66.7])
            if hi <= lo:
                continue          # no spread for this participant
            for i, v in zip(g.index, r):
                if v <= lo:
                    keep.append(i); lab.append(0)
                elif v >= hi:
                    keep.append(i); lab.append(1)
        if len(keep) < 200:
            print(f"{dim:18s}  too few after tertile split"); continue

        d = sub.loc[keep].copy()
        d["_y"] = lab
        dz = zscore_within(d, feats)
        X = np.nan_to_num(dz[feats].to_numpy(float))
        y = d["_y"].to_numpy(int)
        g = d["participant"].to_numpy()

        accs, bases, perms = [], [], []
        rng = np.random.default_rng(0)
        for held in np.unique(g):
            tr, te = g != held, g == held
            if tr.sum() < 100 or te.sum() < 8:
                continue
            if len(np.unique(y[tr])) < 2:
                continue
            mdl = RandomForestClassifier(200, min_samples_leaf=5,
                                         class_weight="balanced",
                                         random_state=0, n_jobs=-1)
            mdl.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], mdl.predict(X[te])))
            bases.append(max(y[te].mean(), 1 - y[te].mean()))
            mp = RandomForestClassifier(200, min_samples_leaf=5,
                                        class_weight="balanced",
                                        random_state=0, n_jobs=-1)
            mp.fit(X[tr], rng.permutation(y[tr]))
            perms.append(accuracy_score(y[te], mp.predict(X[te])))
        if len(accs) < 10:
            print(f"{dim:18s}  too few folds"); continue

        a, b, p = np.mean(accs), np.mean(bases), np.mean(perms)
        star = "*" if a - b > 0.03 else " "
        print(f"{dim:18s} {b:8.3f} {a:8.3f}{star} {p:8.3f} "
              f"{a-b:+8.3f}  {len(accs)}")
        out[dim] = {"chance": float(b), "acc": float(a), "perm": float(p),
                    "over_chance": float(a - b), "n_folds": len(accs)}
    return out


# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 82)
    print("SENSE-42 v2 : QUESTIONNAIRE ANALYSIS (trigger-aligned)")
    print("=" * 82)

    if not os.path.isfile(QCSV):
        print(f"Missing {QCSV}"); return
    q = pd.read_csv(QCSV)
    print(f"Rows: {len(q)}   participants: {q.participant.nunique()}")

    feats = [f for f in EEG_FEATS + PHY_FEATS if f in q.columns]
    ctrl  = [f for f in EEG_CTRL if f in q.columns]
    print(f"Features: {feats}")
    print(f"Controls: {ctrl}  (no cognitive interpretation -- artifact probes)")

    print("\nRating distributions:")
    for d in DIMS:
        s = q.loc[q.dimension == d, "rating"]
        if len(s):
            print(f"  {d:18s} n={len(s):5d}  mean={s.mean():.2f}  "
                  f"sd={s.std():.2f}  range={s.min():.0f}-{s.max():.0f}")

    # sanity: how similar are features across the 7 rows of one questionnaire?
    grp = q.groupby(["participant", "onset_s"])
    print(f"\nDistinct questionnaire timestamps: {grp.ngroups}")
    print("(each dimension has its own trigger, so windows overlap but are")
    print(" not identical -- see header note on why that is a useful control)")

    res = {
        "correlations":      correlation_analysis(q, feats),
        "correlations_ctrl": correlation_analysis(q, ctrl),
        "cca":               cca_analysis(q, feats),
        "prediction":        loso_predict(q, feats),
    }

    print("\n" + "=" * 82)
    print("READING THESE RESULTS")
    print("=" * 82)
    print("""
Genuine signal looks like:
  - correlations significant for LOAD dims (mental_demand, effort,
    temporal_demand, frustration) but NOT for the control features
  - prediction accuracy clearly above its own chance column, with the
    permutation column well below it
  - CCA CV r above ~0.15

Artifact looks like:
  - control features (occipital_delta, broadband_amplitude) correlate as
    strongly as the cognitive ones
  - all seven dimensions behave identically, which points to a common
    cause such as time-on-task rather than construct-specific coupling

Null looks like:
  - everything at its own baseline, permutation matching the real score

Whatever the outcome, this is the first SENSE-42 analysis on correctly
aligned data. If it is null, the null is finally interpretable.
""")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
