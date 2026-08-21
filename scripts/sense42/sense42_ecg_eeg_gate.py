"""
sense42_ecg_eeg_gate.py
========================
Isolates the autonomic -> cortical question that the first gate test
could not answer, and adds the per-task-type split that was missing.

THREE FEATURE SETS (the point of this script):
    ECG-ONLY : [hr_mean, hrv_rmssd, resp_bpm]              (3 features)
    HCI-ONLY : [12 behavioural counts]                     (12 features)
    COMBINED : both                                        (15 features)

WHY ECG-ONLY MATTERS:
  In the previous gate, 12 HCI features sat alongside 3 autonomic ones
  inside the same random forest. Feature dilution is real -- a forest
  with 12 noisy-but-varied predictors can bury 3 informative ones.
  Testing ECG alone removes that objection entirely. If autonomic
  signals carry cortical information, 3 clean features with nothing
  to hide behind is where it shows up.

WHY THE THEORY SAYS THIS SHOULD WORK:
  Thayer & Lane (2000) neurovisceral integration -- prefrontal cortex
  and vagal tone share an inhibitory network.
  Heartbeat-evoked potentials -- each beat produces a cortical response,
  a direct physiological channel from heart to scalp.
  Critchley & Harrison (2013) -- visceral afferents modulate cortical
  processing.
  NOTE: do NOT cite Lacey (1967) here. Lacey is about cardiac vs
  electrodermal directional fractionation, not cortical-autonomic
  coupling. A null result here runs AGAINST theory, not with it, and
  should be framed as "coupling does not survive 30s aggregation and
  cross-participant generalisation" -- not as "no coupling exists".

WHY NOT CHAIN HCI -> ECG -> EEG:
  Chaining cannot beat the direct path. Information is lost at each
  step, never created. If HCI->EEG is flat, routing through an
  imperfect ECG prediction compounds two lossy transforms.
  The useful question is whether ECG->EEG works AT ALL, measured
  directly against real ECG. That is what this script tests.

PER-TASK-TYPE SPLIT:
  Pooling mail reading, file dragging, note typing and browsing
  averages incompatible brain states. Each is analysed separately.

CFA CONTROL (retained, and more important here):
  ECG and EEG shared one amplifier in SENSE-42, so heartbeats leave a
  broadband trace in scalp EEG. With ECG-only inputs, cardiac artifact
  is the MOST likely explanation for any positive result. Two control
  targets with no cognitive interpretation are predicted alongside:
      occipital_delta      (O1/Oz/O2, 1-4 Hz)
      broadband_amplitude  (all channels, 1-40 Hz)
  Real coupling is band-specific and frontal. CFA is broadband and
  diffuse. If controls gain as much as cognitive targets -> artifact.

PERMUTATION TEST:
  Labels shuffled within participant. Any "accuracy" that survives
  shuffling is an artifact of class imbalance, not learning. This is
  the check that caught the questionnaire RF result.

Reads the cache built by sense42_eeg_gate.py -- no re-extraction.
Run from: ~/biosignals_data/
Output:   outputs/sense42_ecg_eeg_results.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

BASE     = os.path.expanduser("~/biosignals_data")
FEAT_CSV = os.path.join(BASE, "outputs", "sense42_gate_features.csv")
OUT_JSON = os.path.join(BASE, "outputs", "sense42_ecg_eeg_results.json")

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnSpaces","SnAppChange",
    "CharactersRatio","ErrorKeyRatio",
]
ECG_COLS = ["hr_mean", "hrv_rmssd", "resp_bpm"]

EEG_COGNITIVE = ["frontal_theta", "frontal_alpha", "theta_alpha_ratio",
                 "engagement_index", "posterior_alpha"]
EEG_CONTROL   = ["occipital_delta", "broadband_amplitude"]
EEG_TARGETS   = EEG_COGNITIVE + EEG_CONTROL

FEATURE_SETS = {
    "ECG-only": ECG_COLS,
    "HCI-only": HCI_COLS,
    "COMBINED": HCI_COLS + ECG_COLS,
}

MIN_TRAIN = 60
MIN_TEST  = 8
MIN_FOLDS = 5


# ── helpers ───────────────────────────────────────────────────────────

def zscore_by_group(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-participant z-score. Removes individual baseline differences
    so the model learns within-person variation, not who is who."""
    Xz = np.zeros_like(X, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        Xz[m] = (X[m] - np.nanmean(X[m], 0)) / (np.nanstd(X[m], 0) + 1e-9)
    return Xz


def direction_labels(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """1 = rising vs previous window, 0 = falling. NaN at each block start.
    Direction framing rather than absolute value -- the reframe that
    rescued the SWELL-KW analysis."""
    out = np.full(len(y), np.nan)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        out[idx[1:]] = (np.diff(y[idx]) > 0).astype(float)
    return out


def loso_eval(X, y, groups, seed=0, shuffle=False):
    """
    Leave-one-subject-out evaluation.
    Returns (mean_accuracy, mean_majority_baseline, n_folds).

    The majority baseline is the honest chance level -- NOT 0.50.
    With discrete labels and imbalanced classes, a model that always
    predicts the majority class scores the majority rate. Any accuracy
    at or below that is learning nothing.
    """
    rng = np.random.default_rng(seed)
    accs, bases = [], []
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum() < MIN_TRAIN or ote.sum() < MIN_TEST:
            continue
        ytr = y[tr][otr].astype(int)
        yte = y[te][ote].astype(int)
        if len(np.unique(ytr)) < 2:
            continue
        if shuffle:
            ytr = rng.permutation(ytr)
        m = RandomForestClassifier(200, min_samples_leaf=5,
                                   class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
        m.fit(X[tr][otr], ytr)
        accs.append(accuracy_score(yte, m.predict(X[te][ote])))
        bases.append(max(yte.mean(), 1 - yte.mean()))
    if len(accs) < MIN_FOLDS:
        return np.nan, np.nan, len(accs)
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


def evaluate_block(d: pd.DataFrame, label: str, run_perm: bool = True):
    """Run all feature sets x all EEG targets on one slice of data."""
    groups = d["participant"].to_numpy()
    n_pp   = len(np.unique(groups))

    print(f"\n{'='*78}")
    print(f"{label}   ({len(d)} windows, {n_pp} participants)")
    print(f"{'='*78}")

    if len(d) < 200 or n_pp < MIN_FOLDS:
        print("  insufficient data -- skipped")
        return None

    Xs = {name: zscore_by_group(np.nan_to_num(d[cols].to_numpy(float)), groups)
          for name, cols in FEATURE_SETS.items()}

    hdr = f"{'EEG target':21s} {'chance':>7s}"
    for name in FEATURE_SETS:
        hdr += f" {name:>10s}"
    hdr += f" {'perm':>7s}"
    print(hdr)
    print("-" * 78)

    block = {}
    for tgt in EEG_TARGETS:
        y = direction_labels(d[tgt].to_numpy(float), groups)

        row  = {}
        base = np.nan
        line = f"{tgt:21s}"
        for name in FEATURE_SETS:
            acc, b, nf = loso_eval(Xs[name], y, groups)
            row[name] = {"acc": acc, "chance": b, "folds": nf}
            if np.isfinite(b):
                base = b
        line += f" {base:7.3f}" if np.isfinite(base) else f" {'--':>7s}"

        for name in FEATURE_SETS:
            acc = row[name]["acc"]
            if np.isnan(acc):
                line += f" {'--':>10s}"
            else:
                over = acc - row[name]["chance"]
                star = "*" if over > 0.03 else " "
                line += f" {acc:9.3f}{star}"

        # permutation check on the best-performing set
        perm = np.nan
        if run_perm:
            best = max(
                (n for n in FEATURE_SETS if np.isfinite(row[n]["acc"])),
                key=lambda n: row[n]["acc"] - row[n]["chance"],
                default=None)
            if best:
                perm, _, _ = loso_eval(Xs[best], y, groups, shuffle=True)
                row["_perm_set"] = best
        line += f" {perm:7.3f}" if np.isfinite(perm) else f" {'--':>7s}"
        row["perm"] = perm

        if tgt in EEG_CONTROL:
            line += "  <- CONTROL"
        print(line)
        block[tgt] = row

    print("\n(* = more than 0.03 above its own majority baseline)")
    print("(perm = same features, labels shuffled within participant.")
    print(" If perm matches the real score, nothing was learned.)")

    # ── CFA verdict for this block ───────────────────────────────────
    def mean_over(names, keyset):
        vals = [block[t][n]["acc"] - block[t][n]["chance"]
                for t in keyset for n in names
                if t in block and np.isfinite(block[t][n]["acc"])]
        return float(np.mean(vals)) if vals else np.nan

    for name in FEATURE_SETS:
        mc = mean_over([name], EEG_COGNITIVE)
        mk = mean_over([name], EEG_CONTROL)
        if not (np.isfinite(mc) and np.isfinite(mk)):
            continue
        if mc < 0.02 and mk < 0.02:
            v = "null"
        elif mk >= mc - 0.01:
            v = "ARTIFACT (control gains match cognitive)"
        else:
            v = "band-specific signal"
        print(f"  {name:9s}  cognitive {mc:+.3f}   control {mk:+.3f}   -> {v}")

    return block


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("SENSE-42 : ECG-only vs HCI-only vs COMBINED  ->  EEG")
    print("=" * 78)

    if not os.path.isfile(FEAT_CSV):
        print(f"Missing {FEAT_CSV}")
        print("Run scripts/sense42/sense42_eeg_gate.py first to build the cache.")
        return

    df = pd.read_csv(FEAT_CSV)
    print(f"Cache: {len(df)} windows, {df.participant.nunique()} participants")

    need = HCI_COLS + ECG_COLS + EEG_TARGETS
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return

    d = df.dropna(subset=need).reset_index(drop=True)
    print(f"Complete rows (HCI + ECG + EEG all present): "
          f"{len(d)}  |  {d.participant.nunique()} participants")

    if "task_type" in d.columns:
        print("\nWindows per task type:")
        for t, n in d["task_type"].value_counts().items():
            pp = d[d.task_type == t].participant.nunique()
            print(f"    {str(t):12s} {n:5d} windows  ({pp} participants)")

    results = {"pooled": evaluate_block(d, "POOLED (all task types)")}

    # ── per task type ────────────────────────────────────────────────
    if "task_type" in d.columns:
        print("\n\n" + "#" * 78)
        print("# PER TASK TYPE")
        print("# Pooling mail reading, file dragging, typing and browsing")
        print("# averages incompatible brain states. Split them.")
        print("#" * 78)

        for task in d["task_type"].value_counts().index:
            if str(task) in ("unknown", "nan"):
                continue
            sub = d[d.task_type == task].reset_index(drop=True)
            r = evaluate_block(sub, f"TASK = {task}", run_perm=False)
            if r:
                results[f"task_{task}"] = r

    # ── summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    pooled = results.get("pooled")
    if pooled:
        print("\nPooled -- mean (accuracy - own chance), cognitive targets:")
        for name in FEATURE_SETS:
            vals = [pooled[t][name]["acc"] - pooled[t][name]["chance"]
                    for t in EEG_COGNITIVE
                    if t in pooled and np.isfinite(pooled[t][name]["acc"])]
            if vals:
                print(f"    {name:9s} {np.mean(vals):+.3f}")

    print("\nDoes ECG-only beat HCI-only anywhere?")
    found = False
    for block_name, block in results.items():
        if not block:
            continue
        for tgt in EEG_COGNITIVE:
            if tgt not in block:
                continue
            e, h = block[tgt]["ECG-only"], block[tgt]["HCI-only"]
            if not (np.isfinite(e["acc"]) and np.isfinite(h["acc"])):
                continue
            eo, ho = e["acc"] - e["chance"], h["acc"] - h["chance"]
            if eo > 0.03 and eo > ho + 0.02:
                print(f"    {block_name:22s} {tgt:20s} "
                      f"ECG {eo:+.3f} vs HCI {ho:+.3f}")
                found = True
    if not found:
        print("    No. ECG-only never clearly beats HCI-only on a cognitive")
        print("    target. Feature dilution was not the explanation for the")
        print("    previous null -- the autonomic signal is not there at this")
        print("    timescale, with or without HCI features alongside it.")

    print("\nInterpretation guide:")
    print("  Real coupling  -> cognitive targets gain, controls do not,")
    print("                    permutation score drops well below the real one.")
    print("  Artifact       -> controls gain as much as cognitive targets.")
    print("  Null           -> everything sits at its own majority baseline.")
    print("\nIf null: frame as 'coupling does not survive 30s aggregation and")
    print("cross-participant generalisation', NOT as 'no coupling exists'.")
    print("Theory (Thayer & Lane 2000; heartbeat-evoked potentials) predicts")
    print("coupling at finer timescales. Do not cite Lacey (1967) for this --")
    print("that paper concerns cardiac vs electrodermal fractionation.")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
