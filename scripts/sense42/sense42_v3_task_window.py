"""
sense42_v3_task_window.py
==========================
Tests the dissociation hypothesis by comparing two epoching schemes.

THE PROBLEM WITH THE PREVIOUS ANALYSIS
---------------------------------------
v2 epoched EEG from (trigger - 30 s) to trigger. But the 30 s before a
question trigger is when the participant is ANSWERING THE QUESTIONNAIRE,
not doing the task. Response times ran 1.7-34.6 s per question and there
are seven questions per block, so a block spans ~35-140 s. For question
index 6 the entire lookback sits inside the questionnaire period; even
for index 0 most of it does.

So v2 measured the EEG of reading sliders and clicking, not the EEG of
the task the rating refers to.

WHAT v2 FOUND, AND WHY IT FITS
-------------------------------
    sleepiness        theta_alpha_ratio -0.113**   resp_bpm -0.128**
    mental_demand     flat
    effort            flat
    temporal_demand   flat
    frustration       flat

Sleepiness is a STATE: if you are drowsy during the task you are still
drowsy 30 s later filling in the slider, so alpha stays high and
breathing stays slow. The window catches it because it persists.

Mental demand is RETROSPECTIVE: "how demanding WAS the task" refers to
something that ended minutes ago. By slider time the load is over.

That is a coherent account of why exactly the state-like dimension showed
signal and every task-referring dimension was flat.

WHAT THIS SCRIPT DOES
----------------------
Builds TASK-WINDOW features: for each questionnaire block, aggregate the
task events occurring between the END of the previous block and the START
of this one -- the actual work the rating refers to.

No re-extraction needed. The v2 events table already holds per-task-event
EEG/HR/resp with onsets; this script aggregates it into task windows and
joins to the ratings.

Then compares the two schemes side by side:

    QUESTIONNAIRE WINDOW (v2)  30 s before the trigger  -> slider period
    TASK WINDOW        (v3)  previous block -> this block -> actual task

PREDICTION IF THE DISSOCIATION IS REAL
    sleepiness  significant in BOTH  (state persists)
    load dims   significant in TASK window only, flat in questionnaire window

If load dims stay flat in both, the retrospective-rating explanation is
wrong and the honest reading is that load has no recoverable EEG signature
here at all.

OTHER FIXES
-----------
1. EEG power log-transformed. MNE returns volts, so band power is ~1e-10
   to 1e-12 V^2/Hz. The v2 guard `np.std(x) < 1e-9` silently skipped every
   participant for frontal_theta, frontal_alpha, posterior_alpha and
   occipital_delta -- four features dropped without warning. Power is
   log-normal anyway, so log10 is the correct transform, not a workaround.

2. Question blocks clustered properly. v2 pivoted on (participant,
   onset_s), but each dimension has its OWN trigger at its own time --
   7233 rows, 7233 distinct timestamps. Every pivoted row got 1 rating and
   6 NaNs and dropna() removed all of them, so the CCA ran on zero rows.
   Here consecutive question triggers within BLOCK_GAP_S are grouped into
   one block.

Run from: ~/biosignals_data/
Output:   outputs/sense42_v3_results.json
          outputs/sense42_v3_taskwindows.csv
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

BASE   = os.path.expanduser("~/biosignals_data")
EV_CSV = os.path.join(BASE, "outputs", "sense42_v2_events.csv")
Q_CSV  = os.path.join(BASE, "outputs", "sense42_v2_questions.csv")
OUT_TW = os.path.join(BASE, "outputs", "sense42_v3_taskwindows.csv")
OUT_J  = os.path.join(BASE, "outputs", "sense42_v3_results.json")

BLOCK_GAP_S = 120.0     # gap that separates two questionnaire blocks
MIN_TASK_EV = 3         # task events needed for a usable task window

# log-transformed (raw power, log-normal, spans orders of magnitude)
POWER_FEATS = ["frontal_theta", "frontal_alpha", "posterior_alpha"]
POWER_CTRL  = ["occipital_delta", "broadband_amplitude"]
# already dimensionless ratios -- do NOT log these
RATIO_FEATS = ["theta_alpha_ratio", "engagement_index"]
PHY_FEATS   = ["hr_mean", "resp_bpm", "resp_amp"]

DIMS = ["mental_demand", "temporal_demand", "effort", "frustration",
        "performance", "attentiveness", "sleepiness"]
LOAD_DIMS  = ["mental_demand", "temporal_demand", "effort", "frustration"]
STATE_DIMS = ["sleepiness", "attentiveness"]


def log_power(df, cols):
    """log10 of band power. Power is log-normal and spans ~3 orders of
    magnitude in volts^2/Hz; untransformed it also underflows naive
    variance guards, which is what silently dropped these features in v2."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            v = out[c].to_numpy(float)
            out[c] = np.log10(np.where(v > 0, v, np.nan))
    return out


def zscore_within(df, cols, group="participant"):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = df.groupby(group)[c].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return out


# ══════════════════════════════════════════════════════════════════════
# Build question blocks and task windows
# ══════════════════════════════════════════════════════════════════════

def assign_blocks(q):
    """Group consecutive question triggers into questionnaire blocks."""
    q = q.sort_values(["participant", "onset_s"]).copy()
    blk, out = 0, []
    prev_p, prev_t = None, None
    for p, t in zip(q.participant, q.onset_s):
        if prev_p != p or (t - prev_t) > BLOCK_GAP_S:
            blk += 1
        out.append(blk)
        prev_p, prev_t = p, t
    q["block"] = out
    return q


def build_task_windows(ev, q):
    """
    For each questionnaire block, aggregate task events between the END of
    the previous block and the START of this one -- the period the rating
    actually refers to.
    """
    binfo = (q.groupby(["participant", "block"])
               .agg(blk_start=("onset_s", "min"),
                    blk_end=("onset_s", "max")).reset_index()
               .sort_values(["participant", "blk_start"]))

    feats = [c for c in POWER_FEATS + POWER_CTRL + RATIO_FEATS + PHY_FEATS
             if c in ev.columns]
    hci = [c for c in ["SnKeyStrokes", "SnErrorKeys", "SnMouseDistance",
                       "SnLeftClicked", "ErrorKeyRatio", "CharactersRatio"]
           if c in ev.columns]

    rows = []
    for pid, g in binfo.groupby("participant"):
        ev_p = ev[ev.participant == pid].sort_values("onset_s")
        prev_end = 0.0
        for _, b in g.iterrows():
            lo, hi = prev_end, b.blk_start
            prev_end = b.blk_end
            seg = ev_p[(ev_p.onset_s >= lo) & (ev_p.onset_s < hi)]
            if len(seg) < MIN_TASK_EV:
                continue
            rec = {"participant": pid, "block": int(b.block),
                   "task_lo": float(lo), "task_hi": float(hi),
                   "n_task_events": int(len(seg)),
                   "task_span_s": float(hi - lo)}
            for c in feats + hci:
                v = seg[c].to_numpy(float)
                v = v[np.isfinite(v)]
                rec[c] = float(np.mean(v)) if len(v) else np.nan
            # variability matters too: unstable performance signals load
            for c in ["frontal_theta", "hr_mean", "SnKeyStrokes"]:
                if c in seg.columns:
                    v = seg[c].to_numpy(float); v = v[np.isfinite(v)]
                    rec[c + "_sd"] = float(np.std(v)) if len(v) > 2 else np.nan
            rows.append(rec)

    tw = pd.DataFrame(rows)
    if tw.empty:
        return tw
    ratings = q.pivot_table(index=["participant", "block"],
                            columns="dimension", values="rating").reset_index()
    return tw.merge(ratings, on=["participant", "block"], how="inner")


# ══════════════════════════════════════════════════════════════════════
# Analyses
# ══════════════════════════════════════════════════════════════════════

def correlations(df, feats, label, long_format=False):
    """Within-participant Spearman rho, then one-sample t-test across
    participants. No thresholding, so no class-balance failure mode."""
    print(f"\n{'=' * 88}")
    print(f"RANK CORRELATIONS -- {label}")
    print("=" * 88)
    hdr = f"{'feature':24s}"
    for d in DIMS:
        hdr += f"{d[:9]:>11s}"
    print(hdr); print("-" * 88)

    res = {}
    for f in feats:
        if f not in df.columns:
            continue
        line = f"{f:24s}"
        for d in DIMS:
            if long_format:
                sub = df[df.dimension == d]
                ycol = "rating"
            else:
                sub, ycol = df, d
            if ycol not in sub.columns:
                line += f"{'--':>11s}"; continue
            rhos = []
            for pid, g in sub.groupby("participant"):
                x = g[f].to_numpy(float); y = g[ycol].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                # guards are on COUNT and on rank-variability, never on the
                # raw scale -- that is what dropped four features in v2
                if ok.sum() < 8:
                    continue
                if len(np.unique(x[ok])) < 3 or len(np.unique(y[ok])) < 2:
                    continue
                r = stats.spearmanr(x[ok], y[ok]).statistic
                if np.isfinite(r):
                    rhos.append(r)
            if len(rhos) < 12:
                line += f"{'--':>11s}"; continue
            rhos = np.array(rhos)
            t, p = stats.ttest_1samp(rhos, 0.0)
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            line += f"{rhos.mean():+8.3f}{star:<3s}"
            res.setdefault(f, {})[d] = {"mean_rho": float(rhos.mean()),
                                        "p": float(p), "n": int(len(rhos))}
        print(line)
    print("\n* p<.05   ** p<.01   *** p<.001")
    return res


def run_cca(tw, feats):
    print(f"\n{'=' * 88}")
    print("CCA -- task-window physiology  <->  NASA-TLX profile")
    print("=" * 88)
    d = tw.dropna(subset=[c for c in feats if c in tw.columns] + DIMS)
    print(f"Complete blocks: {len(d)} ({d.participant.nunique()} participants)")
    if len(d) < 150:
        print("Too few for CCA."); return None

    fe = [c for c in feats if c in d.columns]
    dz = zscore_within(d, fe + DIMS)
    X = np.nan_to_num(dz[fe].to_numpy(float))
    Y = np.nan_to_num(dz[DIMS].to_numpy(float))
    g = d.participant.to_numpy()

    n_comp = min(3, X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_comp, max_iter=3000)
    Xs, Ys = cca.fit_transform(X, Y)
    tr = [float(np.corrcoef(Xs[:, i], Ys[:, i])[0, 1]) for i in range(n_comp)]

    cv = [[] for _ in range(n_comp)]
    for held in np.unique(g):
        a, b = g != held, g == held
        if a.sum() < 80 or b.sum() < 4:
            continue
        try:
            c = CCA(n_components=n_comp, max_iter=3000).fit(X[a], Y[a])
            Xt, Yt = c.transform(X[b], Y[b])
            for i in range(n_comp):
                if np.std(Xt[:, i]) > 1e-9 and np.std(Yt[:, i]) > 1e-9:
                    cv[i].append(np.corrcoef(Xt[:, i], Yt[:, i])[0, 1])
        except Exception:
            pass
    cvm = [float(np.nanmean(c)) if c else np.nan for c in cv]
    cvs = [float(np.nanstd(c)) if c else np.nan for c in cv]
    for i in range(n_comp):
        print(f"  Component {i+1}: train r={tr[i]:.3f}   "
              f"LOSO CV r={cvm[i]:.3f} +/- {cvs[i]:.3f}")

    xw, yw = cca.x_weights_[:, 0], cca.y_weights_[:, 0]
    print("\n  Component 1 physiology:")
    for k in np.argsort(np.abs(xw))[::-1][:6]:
        print(f"    {fe[k]:24s} {xw[k]:+.3f}")
    print("  Component 1 TLX:")
    for k in np.argsort(np.abs(yw))[::-1]:
        print(f"    {DIMS[k]:24s} {yw[k]:+.3f}")
    print("\n  SWELL-KW reference: CV r = 0.581 (condition-contrasted design)")
    return {"train_r": tr, "cv_r": cvm, "cv_sd": cvs, "n": int(len(d))}


def predict(tw, feats):
    print(f"\n{'=' * 88}")
    print("LOSO PREDICTION -- top vs bottom tertile of each rating")
    print("=" * 88)
    print("'chance' is the empirical majority rate of the test fold, never")
    print("0.50. 'perm' shuffles training labels within participant.\n")
    print(f"{'dimension':18s} {'chance':>8s} {'acc':>8s} {'perm':>8s} "
          f"{'over':>8s}  folds")
    print("-" * 62)

    fe = [c for c in feats if c in tw.columns]
    out = {}
    for dim in DIMS:
        if dim not in tw.columns:
            continue
        sub = tw.dropna(subset=fe + [dim]).copy()
        if len(sub) < 200:
            print(f"{dim:18s}  too few rows"); continue
        keep, lab = [], []
        for pid, g in sub.groupby("participant"):
            r = g[dim].to_numpy(float)
            lo, hi = np.percentile(r, [33.3, 66.7])
            if hi <= lo:
                continue
            for i, v in zip(g.index, r):
                if v <= lo: keep.append(i); lab.append(0)
                elif v >= hi: keep.append(i); lab.append(1)
        if len(keep) < 120:
            print(f"{dim:18s}  too few after split"); continue
        d = sub.loc[keep].copy(); d["_y"] = lab
        dz = zscore_within(d, fe)
        X = np.nan_to_num(dz[fe].to_numpy(float))
        y = d["_y"].to_numpy(int); g = d.participant.to_numpy()

        accs, bases, perms = [], [], []
        rng = np.random.default_rng(0)
        for held in np.unique(g):
            a, b = g != held, g == held
            if a.sum() < 60 or b.sum() < 4 or len(np.unique(y[a])) < 2:
                continue
            m = RandomForestClassifier(200, min_samples_leaf=5,
                                       class_weight="balanced",
                                       random_state=0, n_jobs=-1).fit(X[a], y[a])
            accs.append(accuracy_score(y[b], m.predict(X[b])))
            bases.append(max(y[b].mean(), 1 - y[b].mean()))
            mp = RandomForestClassifier(200, min_samples_leaf=5,
                                        class_weight="balanced",
                                        random_state=0, n_jobs=-1
                                        ).fit(X[a], rng.permutation(y[a]))
            perms.append(accuracy_score(y[b], mp.predict(X[b])))
        if len(accs) < 8:
            print(f"{dim:18s}  too few folds"); continue
        A, B, P = np.mean(accs), np.mean(bases), np.mean(perms)
        print(f"{dim:18s} {B:8.3f} {A:8.3f}{'*' if A-B > .03 else ' '} "
              f"{P:8.3f} {A-B:+8.3f}  {len(accs)}")
        out[dim] = {"chance": float(B), "acc": float(A), "perm": float(P),
                    "over": float(A - B), "folds": len(accs)}
    return out


# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 88)
    print("SENSE-42 v3 : TASK WINDOW vs QUESTIONNAIRE WINDOW")
    print("=" * 88)

    for p in (EV_CSV, Q_CSV):
        if not os.path.isfile(p):
            print(f"Missing {p}"); return
    ev = pd.read_csv(EV_CSV)
    q  = pd.read_csv(Q_CSV)
    print(f"Events: {len(ev)} rows | Questions: {len(q)} rows | "
          f"{q.participant.nunique()} participants")

    ev = log_power(ev, POWER_FEATS + POWER_CTRL)
    q  = log_power(q,  POWER_FEATS + POWER_CTRL)
    print("EEG power log10-transformed (fixes the v2 variance-guard dropout)")

    q = assign_blocks(q)
    nb = q.groupby("participant").block.nunique()
    print(f"Question blocks: {q.block.nunique()} total, "
          f"{nb.mean():.1f} per participant (expect ~26)")

    tw = build_task_windows(ev, q)
    if tw.empty:
        print("No task windows built."); return
    tw.to_csv(OUT_TW, index=False)
    print(f"Task windows: {len(tw)} "
          f"({tw.participant.nunique()} participants), saved to {OUT_TW}")
    print(f"  median task events per window: {tw.n_task_events.median():.0f}")
    print(f"  median span: {tw.task_span_s.median():.0f} s")

    feats = [c for c in POWER_FEATS + RATIO_FEATS + PHY_FEATS if c in tw.columns]
    ctrl  = [c for c in POWER_CTRL if c in tw.columns]
    extra = [c for c in tw.columns if c.endswith("_sd")]

    res = {}
    res["quest_window"] = correlations(q, POWER_FEATS + RATIO_FEATS + PHY_FEATS,
                                       "QUESTIONNAIRE WINDOW (30 s before "
                                       "trigger = slider period)",
                                       long_format=True)
    res["task_window"]  = correlations(tw, feats + extra,
                                       "TASK WINDOW (period the rating "
                                       "refers to)")
    res["task_control"] = correlations(tw, ctrl,
                                       "TASK WINDOW -- CONTROL FEATURES")
    res["cca"]          = run_cca(tw, feats)
    res["prediction"]   = predict(tw, feats)

    print(f"\n{'=' * 88}")
    print("THE DISSOCIATION TEST")
    print("=" * 88)
    print("""
Compare the two correlation tables above.

  Dissociation confirmed
      sleepiness significant in BOTH windows  (a state persists across
      the task and into the slider period), while load dimensions become
      significant only in the TASK window. That would mean load has a real
      EEG signature and v2 simply measured the wrong 30 seconds.

  Retrospective-rating explanation wrong
      load dimensions flat in BOTH windows. Then the honest reading is
      that load has no recoverable EEG signature in this dataset, and the
      sleepiness effect stands alone as the one real finding.

  Artifact
      control features (occipital_delta, broadband_amplitude) correlate as
      strongly as the cognitive ones, or all seven dimensions move
      together -- which points to a common cause such as time-on-task.

Multiple comparisons: roughly 10 features x 7 dimensions x 2 windows, so
~5 hits at p<.05 are expected from noise alone. Weight p<.01 results, and
weight consistency across related features far more than any single cell.
""")

    with open(OUT_J, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"Saved: {OUT_J}")


if __name__ == "__main__":
    main()
