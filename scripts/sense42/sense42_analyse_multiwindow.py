"""
sense42_analyse_multiwindow.py
===============================
Runs the full experiment grid on the 1s / 2s / 4s features.

    3 window sizes  x  3 models  x  3 feature sets  x  7 EEG targets
    plus per-task-type splits, permutation tests, and CFA controls.

This supersedes every earlier SENSE-42 analysis, all of which used 30s
windows -- a length dictated by RMSSD's ~30-beat requirement and wrongly
applied to EEG, where frontal theta bursts last 1-2 seconds.

FEATURE SETS:
    CARDIAC  : hr_inst, rr_local, hr_slope, resp_amp,
               resp_phase_sin, resp_phase_cos, resp_slope
    HCI      : 12 behavioural counts
    COMBINED : both

  Note these differ from the 30s runs: hrv_rmssd and resp_bpm are gone
  (not computable at these widths) and are replaced by instantaneous
  equivalents. So this is not a like-for-like comparison with the 30s
  numbers -- it is a different, better-posed question.

WHY resp_phase MATTERS HERE AND COULD NOT AT 30s:
  RSA means heart rate rises on inhalation and falls on exhalation, on a
  sub-second timescale. At 30s windows the respiratory phase is averaged
  away entirely. At 1-4s it is preserved, sin/cos encoded so the model
  treats it as circular. If cardiac-cortical coupling exists anywhere in
  this dataset, respiratory phase is one of the few features fine-grained
  enough to expose it.

SAMPLE SIZE:
  1s windows give ~8000 rows per participant (~300k total). Fitting
  CatBoost on 300k rows x 27 LOSO folds x 7 targets x 3 feature sets is
  not practical, so training data is subsampled per fold (MAX_TRAIN).
  Random subsampling of an i.i.d.-within-fold training set does not bias
  the accuracy estimate; it only widens its confidence interval slightly.
  Test folds are never subsampled.

CHANCE LEVEL:
  Always the empirical majority-class rate of the test fold, never 0.50.
  A model that always predicts the majority class scores that rate for
  free. This is the check that exposed the questionnaire RF result
  (0.794 accuracy against a 0.798 majority baseline -- below free).

PERMUTATION TEST:
  Labels shuffled within the training fold. If the shuffled score matches
  the real one, the model learned nothing and the accuracy is an artifact
  of class balance.

CFA CONTROL:
  ECG shares an amplifier with EEG in SENSE-42, so heartbeats leave a
  broadband trace in scalp EEG. occipital_delta and broadband_amplitude
  have no cognitive interpretation and act as artifact detectors. Real
  coupling is band-specific and frontal; cardiac artifact is broadband
  and diffuse. If controls gain as much as cognitive targets -> artifact.

  In the 30s run, CARDIAC features predicted CONTROL targets better than
  cognitive ones in 5 of 5 blocks. Watch whether that ordering persists.

  CAVEAT AT 1s: occipital_delta is unreliable at 1s resolution (1 Hz
  frequency bins, delta needs 2-3 cycles of a 1 Hz oscillation which does
  not fit in 1s). At 1s, treat broadband_amplitude as the primary control.

Run from: ~/biosignals_data/
Output:   outputs/sense42_multiwindow_results.json
"""
from __future__ import annotations
import os, json, warnings, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

BASE    = os.path.expanduser("~/biosignals_data")
OUT_DIR = os.path.join(BASE, "outputs")
OUT_JSON = os.path.join(OUT_DIR, "sense42_multiwindow_results.json")

WINDOWS = [1, 2, 4]

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnSpaces","SnAppChange",
    "CharactersRatio","ErrorKeyRatio",
]
CARDIAC_COLS = ["hr_inst", "rr_local", "hr_slope",
                "resp_amp", "resp_phase_sin", "resp_phase_cos", "resp_slope"]

EEG_COGNITIVE = ["frontal_theta", "frontal_alpha", "theta_alpha_ratio",
                 "engagement_index", "posterior_alpha"]
EEG_CONTROL   = ["occipital_delta", "broadband_amplitude"]
EEG_TARGETS   = EEG_COGNITIVE + EEG_CONTROL

FEATURE_SETS = {
    "CARDIAC":  CARDIAC_COLS,
    "HCI":      HCI_COLS,
    "COMBINED": HCI_COLS + CARDIAC_COLS,
}

MIN_TRAIN, MIN_TEST, MIN_FOLDS = 100, 20, 5
MAX_TRAIN = 40000        # per-fold training cap, keeps runtime sane
STAR = 0.03


def build_models():
    m = {"RF": lambda s: RandomForestClassifier(
             150, min_samples_leaf=10, class_weight="balanced",
             random_state=s, n_jobs=-1)}
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = lambda s: XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=s, n_jobs=-1, verbosity=0)
    except ImportError:
        print("  XGBoost missing (pip install xgboost)")
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = lambda s: CatBoostClassifier(
            iterations=150, depth=5, learning_rate=0.08,
            auto_class_weights="Balanced", random_seed=s,
            verbose=0, allow_writing_files=False)
    except ImportError:
        print("  CatBoost missing (pip install catboost)")
    return m


def zscore_by_group(X, g):
    Xz = np.zeros_like(X, dtype=float)
    for u in np.unique(g):
        m = g == u
        Xz[m] = (X[m] - np.nanmean(X[m], 0)) / (np.nanstd(X[m], 0) + 1e-9)
    return Xz


def direction_labels(y, g):
    out = np.full(len(y), np.nan)
    for u in np.unique(g):
        i = np.where(g == u)[0]
        if len(i) < 2:
            continue
        out[i[1:]] = (np.diff(y[i]) > 0).astype(float)
    return out


def loso_eval(X, y, g, model_fn, seed=0, shuffle=False):
    """LOSO. Returns (mean_acc, mean_majority_baseline, n_folds)."""
    rng = np.random.default_rng(seed)
    accs, bases = [], []
    for held in np.unique(g):
        tr, te = g != held, g == held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum() < MIN_TRAIN or ote.sum() < MIN_TEST:
            continue
        Xtr, ytr = X[tr][otr], y[tr][otr].astype(int)
        Xte, yte = X[te][ote], y[te][ote].astype(int)
        if len(np.unique(ytr)) < 2:
            continue
        if len(Xtr) > MAX_TRAIN:                 # subsample training only
            sel = rng.choice(len(Xtr), MAX_TRAIN, replace=False)
            Xtr, ytr = Xtr[sel], ytr[sel]
        if shuffle:
            ytr = rng.permutation(ytr)
        try:
            mdl = model_fn(seed)
            mdl.fit(Xtr, ytr)
            accs.append(accuracy_score(yte, mdl.predict(Xte)))
            bases.append(max(yte.mean(), 1 - yte.mean()))
        except Exception:
            continue
    if len(accs) < MIN_FOLDS:
        return np.nan, np.nan, len(accs)
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


def evaluate(d, label, models, counter, win, run_perm=True):
    g = d["participant"].to_numpy()
    npp = len(np.unique(g))
    print(f"\n{'=' * 88}")
    print(f"{label}   ({len(d)} windows, {npp} participants)")
    print(f"{'=' * 88}")
    if len(d) < 300 or npp < MIN_FOLDS:
        print("  insufficient data -- skipped")
        return None

    Xs = {n: zscore_by_group(np.nan_to_num(d[c].to_numpy(float)), g)
          for n, c in FEATURE_SETS.items()}
    block = {}

    for mname, mfn in models.items():
        t0 = time.time()
        print(f"\n  --- {mname} ---")
        hdr = f"  {'EEG target':21s} {'chance':>7s}"
        for fs in FEATURE_SETS:
            hdr += f" {fs:>10s}"
        hdr += f" {'perm':>7s}"
        print(hdr)
        print("  " + "-" * 86)

        for tgt in EEG_TARGETS:
            y = direction_labels(d[tgt].to_numpy(float), g)
            row, base = {}, np.nan
            line = f"  {tgt:21s}"
            for fs in FEATURE_SETS:
                acc, b, nf = loso_eval(Xs[fs], y, g, mfn)
                row[fs] = {"acc": acc, "chance": b, "folds": nf}
                if np.isfinite(b):
                    base = b
                if np.isfinite(acc):
                    counter["total"] += 1
                    if acc - b > STAR:
                        counter["starred"] += 1
            line += f" {base:7.3f}" if np.isfinite(base) else f" {'--':>7s}"
            for fs in FEATURE_SETS:
                a = row[fs]["acc"]
                line += f" {'--':>10s}" if np.isnan(a) else \
                        f" {a:9.3f}{'*' if a - row[fs]['chance'] > STAR else ' '}"

            perm = np.nan
            if run_perm:
                cands = [f for f in FEATURE_SETS if np.isfinite(row[f]["acc"])]
                if cands:
                    best = max(cands, key=lambda f: row[f]["acc"] - row[f]["chance"])
                    perm, _, _ = loso_eval(Xs[best], y, g, mfn, shuffle=True)
                    row["_perm_set"] = best
            row["perm"] = perm
            line += f" {perm:7.3f}" if np.isfinite(perm) else f" {'--':>7s}"
            if tgt in EEG_CONTROL:
                line += "  <- CONTROL"
                if win == 1 and tgt == "occipital_delta":
                    line += " (unreliable at 1s)"
            print(line)
            block.setdefault(mname, {})[tgt] = row

        for fs in FEATURE_SETS:
            def mo(keys):
                v = [block[mname][t][fs]["acc"] - block[mname][t][fs]["chance"]
                     for t in keys if t in block[mname]
                     and np.isfinite(block[mname][t][fs]["acc"])]
                return float(np.mean(v)) if v else np.nan
            mc, mk = mo(EEG_COGNITIVE), mo(EEG_CONTROL)
            if not (np.isfinite(mc) and np.isfinite(mk)):
                continue
            v = ("null" if mc < 0.02 and mk < 0.02 else
                 "ARTIFACT (control matches cognitive)" if mk >= mc - 0.01 else
                 "band-specific signal")
            flag = "  [control > cognitive: CFA signature]" if mk > mc else ""
            print(f"    {fs:9s}  cognitive {mc:+.3f}   control {mk:+.3f}   -> {v}{flag}")
        print(f"    ({time.time() - t0:.0f}s)")
    return block


def main():
    print("=" * 88)
    print("SENSE-42 MULTI-WINDOW ANALYSIS  (1s / 2s / 4s)")
    print("=" * 88)
    models = build_models()
    print(f"Models: {list(models.keys())}")

    all_results, counter = {}, {"total": 0, "starred": 0}

    for win in WINDOWS:
        path = os.path.join(OUT_DIR, f"sense42_feat_{win}s.csv")
        if not os.path.isfile(path):
            print(f"\nMissing {path} -- run sense42_extract_multiwindow.py first")
            continue

        print("\n\n" + "#" * 88)
        print(f"#  WINDOW = {win}s")
        if win == 1:
            print("#  CAVEAT: 1 Hz frequency resolution. occipital_delta is")
            print("#  unreliable; theta spans only 4 bins. Interpret with care.")
        print("#" * 88)

        df = pd.read_csv(path)
        need = HCI_COLS + CARDIAC_COLS + EEG_TARGETS
        miss = [c for c in need if c not in df.columns]
        if miss:
            print(f"  missing columns: {miss}")
            continue
        d = df.dropna(subset=need).reset_index(drop=True)
        print(f"\nRows: {len(df)} -> complete: {len(d)} "
              f"({d.participant.nunique()} participants)")
        if len(d) < 300:
            print("  too few complete rows -- skipped")
            continue

        wr = {"pooled": evaluate(d, f"[{win}s] POOLED", models, counter, win)}

        if "task_type" in d.columns:
            for task in d["task_type"].value_counts().index:
                if str(task) in ("unknown", "nan"):
                    continue
                sub = d[d.task_type == task].reset_index(drop=True)
                if len(sub) < 300:
                    continue
                r = evaluate(sub, f"[{win}s] TASK = {task}", models,
                             counter, win, run_perm=False)
                if r:
                    wr[f"task_{task}"] = r
        all_results[f"{win}s"] = wr

    # ── summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 88)
    print("SUMMARY -- mean (accuracy - own chance) on COGNITIVE targets, pooled")
    print("=" * 88)
    print(f"\n{'window':>8s} {'model':10s} " +
          " ".join(f"{fs:>11s}" for fs in FEATURE_SETS))
    best = None
    for wk, wr in all_results.items():
        pooled = wr.get("pooled") or {}
        for mname in models:
            if mname not in pooled:
                continue
            line = f"{wk:>8s} {mname:10s} "
            for fs in FEATURE_SETS:
                v = [pooled[mname][t][fs]["acc"] - pooled[mname][t][fs]["chance"]
                     for t in EEG_COGNITIVE if t in pooled[mname]
                     and np.isfinite(pooled[mname][t][fs]["acc"])]
                if v:
                    mv = float(np.mean(v))
                    line += f" {mv:+11.3f}"
                    if best is None or mv > best[0]:
                        best = (mv, wk, mname, fs)
                else:
                    line += f" {'--':>11s}"
            print(line)

    print(f"\nTests run: {counter['total']}   "
          f"above +{STAR:.2f}: {counter['starred']}")
    if counter["total"]:
        print(f"Rate: {counter['starred'] / counter['total']:.1%} -- at this "
              f"number of tests some crossings are expected from noise.")
        print("A crossing counts only if its permutation score is clearly")
        print("lower AND the control targets do not gain alongside it.")

    if best:
        mv, wk, mname, fs = best
        print(f"\nBest cell: {wk} / {mname} / {fs}  ->  {mv:+.3f}")
        if mv < 0.02:
            print("  Still at baseline. Shortening the window from 30s to")
            print("  1-4s did NOT reveal autonomic-cortical coupling. Combined")
            print("  with the model comparison, this closes both the 'wrong")
            print("  timescale' and 'weak learner' objections.")
        else:
            print("  Above baseline. Check its permutation score and control")
            print("  targets in the block above before treating it as real.")

    print("\nComparison anchors:")
    print("  30s windows, RF:      cognitive -0.025 to -0.011  (null)")
    print("  Cog Lab resp_bpm:     RF 0.795 / XGB 0.829 / CatBoost 0.838")
    print("    -- there signal existed and a better model found more of it.")
    print("  SWELL-KW CCA:         CV r = 0.581 (condition-contrasted design)")

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
