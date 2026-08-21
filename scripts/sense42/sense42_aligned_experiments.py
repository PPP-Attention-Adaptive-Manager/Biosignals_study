"""
sense42_aligned_experiments.py
================================
Runs every directional experiment on correctly aligned SENSE-42 data.

ALL PREVIOUS SENSE-42 RESULTS WERE INVALID.
The ~104s misalignment mapped experiment windows onto pre-calibration EEG.
This script reads sense42_v2_events.csv — trigger-aligned, hci_matched=100%.

SIX EXPERIMENTS
---------------
EXP 1:  HCI → HR + RMSSD + resp_bpm          (SWELL-KW replication)
        CCA + RF direction. Reference: SWELL-KW CV r=0.581, HR dir 79.0%

EXP 2:  HCI → EEG band powers                 (Phase B, corrected)
        CCA + RF direction on log-power.

EXP 3:  HR + RMSSD + resp → EEG              (cardiac alone → cortical)
        Isolates autonomic-cortical coupling without HCI dilution.
        In the 30s-window gate, cardiac features predicted CONTROL targets
        better than cognitive ones in 5/5 blocks. Check whether that
        artifact pattern survives at the event-epoch level.

EXP 4:  HCI + HR + RMSSD → EEG               (corrected gate V2)
        The gate test we ran before on misaligned data.

EXP 5:  HCI + HR + RMSSD + resp → EEG        (full feature gate V3)
        All available modalities → cortical.

EXP 6:  HCI + HR + RMSSD → HR + RMSSD + resp  (cross-modal prediction)
        Using EEG + cardiac to predict the other physio stream.

FOR EACH EXPERIMENT:
  - CCA with LOSO leave-one-participant-out cross-validation
  - RF and CatBoost direction classifiers (rising/falling)
  - Per-participant z-scoring (removes individual baselines)
  - Empirical majority baseline (never 0.50)
  - Permutation control (labels shuffled in training fold)
  - Per-task-type breakdown for the best experiment

DESIGN DECISIONS
----------------
EEG power: log10-transformed. MNE returns V²/Hz (~1e-10 to 1e-12).
A variance guard on raw values silently dropped four features in v2.

Epoch filtering: short epochs lack physiology.
  EEG:   all epochs (≥2s, always present)
  HR:    epochs ≥ 5s  (need ≥4 beats for median HR)
  RMSSD: epochs ≥ 20s (need ≥10 consecutive RR intervals)
  resp:  epochs ≥ 8s  (need ≥3 breath peaks)

Direction labels: computed across consecutive events within participant,
sorted by onset_s. Only valid between adjacent events where both have
the metric and the gap is <CONSEC_GAP_S (no huge task-type jumps).

Run from: ~/biosignals_data/
Input:   outputs/sense42_v2_events.csv   (trigger-aligned, v2 extraction)
Output:  outputs/sense42_aligned_results.json
"""
from __future__ import annotations
import os, json, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

BASE   = os.path.expanduser("~/biosignals_data")
IN_CSV = os.path.join(BASE, "outputs", "sense42_v2_events.csv")
OUT_J  = os.path.join(BASE, "outputs", "sense42_aligned_results.json")

# ── feature groups ────────────────────────────────────────────────────
HCI_COLS = ["SnKeyStrokes","SnChars","SnSpecialKeys","SnDirectionKeys",
            "SnErrorKeys","SnSpaces","CharactersRatio","ErrorKeyRatio",
            "SnLeftClicked","SnMouseDistance","SnMouseAct"]

EEG_POWER = ["frontal_theta","frontal_alpha","posterior_alpha"]   # log10
EEG_RATIO = ["theta_alpha_ratio","engagement_index"]              # dimensionless
EEG_CTRL  = ["occipital_delta","broadband_amplitude"]             # artifact probes
EEG_ALL   = EEG_POWER + EEG_RATIO
EEG_CTRL_LOG = ["occipital_delta"]   # log, broadband stays linear

CARDIAC   = ["hr_mean","hrv_rmssd"]
RESP      = ["resp_bpm","resp_amp"]
PHYSIO    = CARDIAC + RESP

# minimum epoch duration for each modality
DUR_EEG   = 2.0
DUR_HR    = 5.0
DUR_RMSSD = 20.0
DUR_RESP  = 8.0

# maximum gap between consecutive events to accept as "adjacent"
CONSEC_GAP_S = 300.0   # 5 min; bigger gaps = different task contexts

MIN_TRAIN = 60
MIN_TEST  = 8
MIN_FOLDS = 8


# ══════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════

def load_and_prep():
    if not os.path.isfile(IN_CSV):
        raise FileNotFoundError(
            f"Missing {IN_CSV}\n"
            "Run scripts/sense42/sense42_trigger_extract_v2.py first.")
    ev = pd.read_csv(IN_CSV)
    print(f"Loaded {len(ev)} events, {ev.participant.nunique()} participants")

    # Check HCI presence
    hci_found = [c for c in HCI_COLS if c in ev.columns]
    if not hci_found:
        raise ValueError(
            "No HCI columns found. The uploaded sense42_trig_epochs.csv is the "
            "OLD v1 file (no HCI). Use sense42_v2_events.csv from the v2 extraction.")
    print(f"HCI columns: {len(hci_found)}/{len(HCI_COLS)} present")

    # Log10-transform EEG power (raw values ~1e-10 to 1e-12 V²/Hz)
    for c in EEG_POWER + EEG_CTRL_LOG:
        if c in ev.columns:
            v = ev[c].to_numpy(float)
            ev[c] = np.where(v > 0, np.log10(v), np.nan)
    print("EEG power log10-transformed")

    # Apply duration filters — set physiology to NaN below threshold
    if "duration_s" in ev.columns:
        d = ev["duration_s"].to_numpy(float)
        for c in CARDIAC:
            if c in ev.columns:
                ev.loc[d < DUR_HR, c] = np.nan
        if "hrv_rmssd" in ev.columns:
            ev.loc[d < DUR_RMSSD, "hrv_rmssd"] = np.nan
        for c in RESP:
            if c in ev.columns:
                ev.loc[d < DUR_RESP, c] = np.nan
        print(f"Duration filters applied:"
              f" HR≥{DUR_HR}s RMSSD≥{DUR_RMSSD}s resp≥{DUR_RESP}s")

    # Coverage report
    print("\nCoverage after filters:")
    for c in hci_found[:3] + EEG_ALL[:3] + PHYSIO:
        if c in ev.columns:
            n = ev[c].notna().sum()
            print(f"  {c:22s} {n:6d}/{len(ev)} ({100*n/len(ev):.1f}%)")
    return ev


def zscore_within(df, cols, group="participant"):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = df.groupby(group)[c].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return out


def direction_labels(series, groups, onsets, max_gap=CONSEC_GAP_S):
    """
    1 = rising vs PREVIOUS event in same participant.
    NaN at first event or if gap > max_gap (different session context).
    """
    out = np.full(len(series), np.nan)
    s, g, t = series.to_numpy(float), groups.to_numpy(), onsets.to_numpy(float)
    for u in np.unique(g):
        idx = np.where(g == u)[0]
        for k in range(1, len(idx)):
            i, j = idx[k-1], idx[k]
            if t[j] - t[i] > max_gap:
                continue
            if np.isfinite(s[i]) and np.isfinite(s[j]):
                out[j] = float(s[j] > s[i])
    return out


# ══════════════════════════════════════════════════════════════════════
# Statistical tests
# ══════════════════════════════════════════════════════════════════════

def build_models():
    m = {"RF": lambda s: RandomForestClassifier(
            200, min_samples_leaf=5, class_weight="balanced",
            random_state=s, n_jobs=-1)}
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = lambda s: CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=s,
            verbose=0, allow_writing_files=False)
    except ImportError:
        print("  CatBoost not installed (pip install catboost)")
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = lambda s: XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=s, n_jobs=-1, verbosity=0)
    except ImportError:
        print("  XGBoost not installed (pip install xgboost)")
    return m


def loso_dir(X, y, groups, model_fn, shuffle=False):
    rng = np.random.default_rng(42)
    accs, bases = [], []
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum() < MIN_TRAIN or ote.sum() < MIN_TEST:
            continue
        ytr, yte = y[tr][otr].astype(int), y[te][ote].astype(int)
        if len(np.unique(ytr)) < 2:
            continue
        if shuffle:
            ytr = rng.permutation(ytr)
        try:
            m = model_fn(0)
            m.fit(X[tr][otr], ytr)
            accs.append(accuracy_score(yte, m.predict(X[te][ote])))
            bases.append(max(yte.mean(), 1 - yte.mean()))
        except Exception:
            continue
    if len(accs) < MIN_FOLDS:
        return np.nan, np.nan, len(accs)
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


def loso_cca(X, Y, groups, n_comp=2):
    """LOSO CCA. Returns (train_r, cv_r_mean, cv_r_std, n_folds)."""
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y, groups = X[ok], Y[ok], groups[ok]
    if len(X) < 200:
        return [np.nan]*n_comp, [np.nan]*n_comp, [np.nan]*n_comp, 0
    try:
        cca = CCA(n_components=n_comp, max_iter=3000)
        Xs, Ys = cca.fit_transform(X, Y)
        tr = [float(np.corrcoef(Xs[:,i], Ys[:,i])[0,1]) for i in range(n_comp)]
    except Exception:
        return [np.nan]*n_comp, [np.nan]*n_comp, [np.nan]*n_comp, 0

    cv = [[] for _ in range(n_comp)]
    for held in np.unique(groups):
        a, b = groups != held, groups == held
        if a.sum() < 100 or b.sum() < 5:
            continue
        try:
            c = CCA(n_components=n_comp, max_iter=3000).fit(X[a], Y[a])
            Xt, Yt = c.transform(X[b], Y[b])
            for i in range(n_comp):
                if np.std(Xt[:,i]) > 1e-9 and np.std(Yt[:,i]) > 1e-9:
                    cv[i].append(np.corrcoef(Xt[:,i], Yt[:,i])[0,1])
        except Exception:
            pass

    cvm = [float(np.nanmean(c)) if c else np.nan for c in cv]
    cvs = [float(np.nanstd(c))  if c else np.nan for c in cv]
    return tr, cvm, cvs, len(cv[0]) if cv[0] else 0


# ══════════════════════════════════════════════════════════════════════
# Experiment runner
# ══════════════════════════════════════════════════════════════════════

def run_experiment(name, ev, X_cols, Y_cols, models,
                   per_task=False, label=""):
    """
    Run CCA + direction classifiers for one (X→Y) pair.
    X_cols, Y_cols: column names in ev.
    """
    avail_X = [c for c in X_cols if c in ev.columns]
    avail_Y = [c for c in Y_cols if c in ev.columns]
    if not avail_X or not avail_Y:
        return {"error": f"missing cols: X={[c for c in X_cols if c not in ev.columns]} "
                         f"Y={[c for c in Y_cols if c not in ev.columns]}"}

    groups = ev["participant"].to_numpy()
    onsets = ev["onset_s"].to_numpy(float)

    # per-participant z-score (removes individual baselines)
    evz = zscore_within(ev, avail_X + avail_Y)
    X   = np.nan_to_num(evz[avail_X].to_numpy(float))
    Y_raw = evz[avail_Y].to_numpy(float)

    result = {"n_windows": int(len(ev)), "n_participants": int(len(np.unique(groups)))}

    # ── CCA ──────────────────────────────────────────────────────────
    n_comp = min(2, len(avail_X), len(avail_Y))
    tr, cvm, cvs, nf = loso_cca(X, Y_raw, groups, n_comp)
    result["cca"] = {"train_r": tr, "cv_r": cvm, "cv_sd": cvs, "n_folds": nf}
    print(f"\n  CCA: train r={tr[0]:.3f}  LOSO CV r={cvm[0]:.3f}±{cvs[0]:.3f}"
          f"  ({nf} folds)")
    if not np.isnan(cvm[0]):
        if cvm[0] > 0.30:
            print(f"  *** ABOVE 0.30 — reliable coupling detected ***")
        elif cvm[0] > 0.15:
            print(f"  ~ marginal coupling (0.15-0.30)")
        else:
            print(f"  null (< 0.15)")

    # ── Direction classifiers ─────────────────────────────────────────
    dir_results = {}
    for target in avail_Y:
        y = direction_labels(ev[target], ev["participant"], ev["onset_s"])
        dir_results[target] = {}
        for mname, mfn in models.items():
            acc, base, nf2 = loso_dir(X, y, groups, mfn)
            perm, _, _    = loso_dir(X, y, groups, mfn, shuffle=True)
            over = acc - base if np.isfinite(acc) and np.isfinite(base) else np.nan
            dir_results[target][mname] = {
                "acc": acc, "chance": base, "perm": perm, "over": over, "folds": nf2}
        # print summary for this target
        vals = [(mn, r["acc"], r["chance"], r["over"])
                for mn, r in dir_results[target].items()
                if np.isfinite(r["acc"])]
        if vals:
            best = max(vals, key=lambda x: x[3] if np.isfinite(x[3]) else -99)
            flag = " ***" if (best[3] or 0) > 0.03 else ""
            print(f"  dir {target:20s}  best={best[0]} "
                  f"acc={best[1]:.3f} chance={best[2]:.3f} "
                  f"over={best[3]:+.3f}{flag}")
    result["direction"] = dir_results

    # ── CFA control check (if EEG targets) ───────────────────────────
    ctrl_cols = [c for c in EEG_CTRL if c in ev.columns]
    if ctrl_cols and any(c in avail_Y for c in EEG_ALL):
        cog_over  = np.nanmean([dir_results[t][list(models.keys())[0]]["over"]
                                for t in avail_Y if t in dir_results
                                and t not in EEG_CTRL
                                and np.isfinite(dir_results[t][list(models.keys())[0]]["over"])])
        evzc = zscore_within(ev, avail_X + ctrl_cols)
        Yc = evzc[ctrl_cols].to_numpy(float)
        _, cvm_c, _, _ = loso_cca(X, Yc, groups, min(2, len(avail_X), len(ctrl_cols)))
        print(f"\n  CFA control: cognitive over-chance={cog_over:+.3f}"
              f"  control CCA CV r={cvm_c[0]:.3f}")
        if cvm_c[0] > cog_over:
            print("  WARNING: controls stronger than cognitive — possible artifact")
        result["cfa_check"] = {"cog_mean_over": float(cog_over),
                               "ctrl_cca_cv_r": cvm_c[0]}

    # ── Per-task breakdown ────────────────────────────────────────────
    if per_task and "app" in ev.columns:
        task_res = {}
        for task in ev["app"].value_counts().index:
            sub = ev[ev.app == task].reset_index(drop=True)
            if len(sub) < 150:
                continue
            gst = sub["participant"].to_numpy()
            subz = zscore_within(sub, avail_X + avail_Y)
            Xt = np.nan_to_num(subz[avail_X].to_numpy(float))
            Yt = subz[avail_Y].to_numpy(float)
            _, cvm_t, cvs_t, nft = loso_cca(Xt, Yt, gst, min(1, len(avail_X), len(avail_Y)))
            task_res[task] = {"cca_cv_r": cvm_t[0], "n": int(len(sub))}
            print(f"    task={task:10s} n={len(sub):5d}  "
                  f"CCA CV r={cvm_t[0]:.3f}")
        result["per_task"] = task_res
    return result


# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("SENSE-42 ALIGNED EXPERIMENTS (trigger-aligned v2 events)")
    print("=" * 80)
    ev = load_and_prep()
    models = build_models()
    print(f"Models: {list(models.keys())}\n")

    hci_avail  = [c for c in HCI_COLS   if c in ev.columns]
    eeg_avail  = [c for c in EEG_ALL    if c in ev.columns]
    card_avail = [c for c in CARDIAC    if c in ev.columns]
    resp_avail = [c for c in RESP       if c in ev.columns]
    all_phys   = [c for c in PHYSIO     if c in ev.columns]

    print(f"HCI features:  {len(hci_avail)}")
    print(f"EEG features:  {len(eeg_avail)}")
    print(f"Cardiac:       {len(card_avail)}")
    print(f"Respiration:   {len(resp_avail)}")

    results = {}

    # ── EXP 1: HCI → HR + RMSSD + resp ──────────────────────────────
    print("\n" + "="*80)
    print("EXP 1: HCI → HR + RMSSD + resp_bpm  (SWELL-KW REPLICATION)")
    print("SWELL-KW reference: CCA CV r=0.581  HR direction 79.0%")
    print("="*80)
    # Use only epochs long enough for cardiac
    ev1 = ev[ev["duration_s"] >= DUR_HR].copy().reset_index(drop=True)
    print(f"After duration filter (≥{DUR_HR}s): {len(ev1)} events "
          f"({ev1.participant.nunique()} participants)")
    results["exp1_hci_physio"] = run_experiment(
        "HCI→physio", ev1, hci_avail, all_phys, models, per_task=True)

    # ── EXP 2: HCI → EEG ─────────────────────────────────────────────
    print("\n" + "="*80)
    print("EXP 2: HCI → EEG band powers  (Phase B corrected)")
    print("All epochs usable (EEG always present). Log10-transformed.")
    print("="*80)
    results["exp2_hci_eeg"] = run_experiment(
        "HCI→EEG", ev, hci_avail, eeg_avail, models, per_task=True)

    # ── EXP 3: cardiac alone → EEG ───────────────────────────────────
    print("\n" + "="*80)
    print("EXP 3: HR + RMSSD + resp → EEG  (isolates autonomic-cortical)")
    print("Previous 30s gate: cardiac predicted CONTROLS > cognitive (5/5 blocks)")
    print("Check: was that CFA artifact or dilution by HCI features?")
    print("="*80)
    ev3 = ev[ev["duration_s"] >= DUR_HR].copy().reset_index(drop=True)
    results["exp3_cardiac_eeg"] = run_experiment(
        "cardiac→EEG", ev3, all_phys, eeg_avail, models, per_task=False)

    # ── EXP 4: HCI + cardiac → EEG ───────────────────────────────────
    print("\n" + "="*80)
    print("EXP 4: HCI + HR + RMSSD → EEG  (corrected gate V2)")
    print("="*80)
    ev4 = ev[ev["duration_s"] >= DUR_HR].copy().reset_index(drop=True)
    results["exp4_hci_cardiac_eeg"] = run_experiment(
        "HCI+cardiac→EEG", ev4, hci_avail + card_avail, eeg_avail, models)

    # ── EXP 5: HCI + all physio → EEG ────────────────────────────────
    print("\n" + "="*80)
    print("EXP 5: HCI + HR + RMSSD + resp → EEG  (full feature gate V3)")
    print("="*80)
    ev5 = ev[ev["duration_s"] >= DUR_RESP].copy().reset_index(drop=True)
    print(f"After resp filter (≥{DUR_RESP}s): {len(ev5)} events")
    results["exp5_full_eeg"] = run_experiment(
        "full→EEG", ev5, hci_avail + all_phys, eeg_avail, models)

    # ── EXP 6: HCI → EEG per task type (most granular) ───────────────
    print("\n" + "="*80)
    print("EXP 6: HCI → EEG per task type separately")
    print("Pooling mail/notes/file_mgr/browser mixes incompatible brain states")
    print("="*80)
    per_task_cca = {}
    for task in ev["app"].value_counts().index:
        sub = ev[ev.app == task].reset_index(drop=True)
        if len(sub) < 200 or sub.participant.nunique() < MIN_FOLDS:
            continue
        gst  = sub["participant"].to_numpy()
        subz = zscore_within(sub, hci_avail + eeg_avail)
        X_t  = np.nan_to_num(subz[hci_avail].to_numpy(float))
        Y_t  = subz[eeg_avail].to_numpy(float)
        ok   = np.isfinite(X_t).all(1) & np.isfinite(Y_t).all(1)
        if ok.sum() < 100:
            continue
        _, cvm, cvs, nf = loso_cca(X_t[ok], Y_t[ok], gst[ok])
        per_task_cca[task] = {
            "cca_cv_r": cvm[0], "cca_cv_sd": cvs[0],
            "n": int(ok.sum()), "n_participants": int(sub.participant.nunique())}
        flag = " ***" if (cvm[0] or 0) > 0.30 else \
               " *"   if (cvm[0] or 0) > 0.15 else ""
        print(f"  {task:12s} n={ok.sum():5d}  "
              f"CV r={cvm[0]:.3f}±{cvs[0]:.3f}{flag}")
    results["exp6_hci_eeg_pertask"] = per_task_cca

    # ── SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUMMARY — CCA CV r across all experiments")
    print("="*80)
    print(f"  Reference SWELL-KW (HCI→physio, condition-contrasted): "
          f"CV r = 0.581\n")

    exp_labels = {
        "exp1_hci_physio":    "EXP1 HCI→physio (SWELL-KW replication)",
        "exp2_hci_eeg":       "EXP2 HCI→EEG",
        "exp3_cardiac_eeg":   "EXP3 cardiac→EEG (autonomic-cortical)",
        "exp4_hci_cardiac_eeg":"EXP4 HCI+cardiac→EEG (gate V2)",
        "exp5_full_eeg":      "EXP5 HCI+all physio→EEG (gate V3)",
    }
    for key, label in exp_labels.items():
        if key not in results:
            continue
        r = results[key]
        if "error" in r:
            print(f"  {label}: ERROR — {r['error']}")
            continue
        cv = r.get("cca", {}).get("cv_r", [np.nan])[0]
        n  = r.get("n_windows", "?")
        flag = " *** ABOVE 0.30" if (cv or 0) > 0.30 else \
               " *   above 0.15" if (cv or 0) > 0.15 else ""
        print(f"  {label}\n    CV r={cv:.3f}  n={n}{flag}")

    print("\nPer-task HCI→EEG (EXP6):")
    for task, r in per_task_cca.items():
        flag = " ***" if (r["cca_cv_r"] or 0) > 0.30 else \
               " *"   if (r["cca_cv_r"] or 0) > 0.15 else ""
        print(f"  {task:12s} CV r={r['cca_cv_r']:.3f}{flag}")

    print("""
READING THE RESULTS:

  > 0.30  Real coupling. Replicates SWELL-KW finding on aligned data.
  0.15-0.30  Marginal but worth further investigation.
  < 0.15  No coupling at this epoch resolution / feature set.

  EXP1 is the most important. If HCI predicts HR+RMSSD on SENSE-42
  (CV r > 0.30), the SWELL-KW proxy generalizes cross-dataset.

  EXP2 vs EXP3 isolates WHETHER the coupling is behavioral or autonomic:
    EXP2 flat, EXP3 non-zero → cardiac alone drives EEG (possible artifact)
    EXP2 non-zero, EXP3 flat → behavioral signal drives EEG
    Both non-zero → independent contributions

  CFA check: if control features (occipital_delta, broadband) score as
  high as cognitive targets in EXP3/4/5, the cardiac→EEG link is artifact.
""")

    with open(OUT_J, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {OUT_J}")


if __name__ == "__main__":
    main()
