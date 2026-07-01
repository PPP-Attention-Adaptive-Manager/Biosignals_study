"""
gate_dynamics_v1_vs_v2.py
===========================
Re-applies the dynamics-framing technique that worked on SWELL-KW
(Experiments 4 & 6.1: predict DIRECTION/MAGNITUDE of biosignal change,
not the absolute value) to the Cog Lab gate test.

Two separate questions, both worth answering on their own:

  Q1 (vs CHANCE) — does the dynamics framing find ANY signal at all for
     these 10 targets, regardless of which inputs are used? This is the
     same question SWELL Experiment 4 answered for HR/RMSSD/SCL
     (chance = 50% direction, chance = 33% magnitude).

  Q2 (V1 vs V2) — given whatever signal exists, does adding HR/RMSSD/SCL
     DELTAS as extra inputs improve it beyond mouse+keyboard count
     deltas alone? Same V1-vs-V2 contrast as the absolute-value gate,
     just on a different target representation.

METHODOLOGICAL NOTE — window overlap:
Cog Lab windows use window=30s, stride=15s (50% overlap). A delta
between consecutive windows therefore represents a ~15s step, not a
full window-length step like SWELL-KW's non-overlapping 1-minute
windows. Still a meaningful dynamic, just a finer/different timescale
than the SWELL work — flagged here so it isn't silently assumed
identical to that setup.

DECISION RULE for Q2, stated before results (same discipline as the
absolute-value gate):
  V2 PASSES Q2 if mean accuracy lift over V1 > 0.05 across all
  target x framing combos, AND at least 3 combos show lift > 0.08.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

CACHE_DIR = os.path.expanduser("~/biosignals_data/proxy_cache_swellstyle")
COG_LAB_DIR = os.path.expanduser("~/biosignals_data/cog_lab")
EXCLUDE = {"S2", "S17"}
ALWAYS_NAN_COLS = {"SnRightClicked", "SnDoubleClicked", "SnDragged"}

REMAINING_TARGETS = [
    "acc_movement", "acc_jerk",
    "eda_tonic_slope", "eda_phasic_count",
    "resp_bpm",
    "eeg_theta_alpha", "eeg_engagement", "eeg_alpha_asym",
    "fnirs_hbo_slope_L", "fnirs_hbo_slope_R",
]

subjects = sorted(
    d for d in os.listdir(COG_LAB_DIR)
    if d.startswith("S") and d[1:].isdigit() and d not in EXCLUDE
)

z0 = np.load(os.path.join(CACHE_DIR, f"{subjects[0]}.npz"), allow_pickle=True)
swell_names = list(z0["swell_names"])
keep_idx = [i for i, n in enumerate(swell_names) if n not in ALWAYS_NAN_COLS]

# ---------------------------------------------------------------- build deltas
data = {}   # sid -> (dXc, dXb, dY)  all shape (n_windows-1, k)
for sid in subjects:
    z = np.load(os.path.join(CACHE_DIR, f"{sid}.npz"), allow_pickle=True)
    starts = z["starts"]
    order = np.argsort(starts)          # defensive — should already be sorted
    Xc = z["X_counts"][order][:, keep_idx].astype(float)
    Xb = z["X_input_biosig"][order].astype(float)
    Y  = z["Y_remaining"][order].astype(float)

    dXc = np.diff(Xc, axis=0)
    dXb = np.diff(Xb, axis=0)
    dY  = np.diff(Y, axis=0)
    data[sid] = (dXc, dXb, dY)

def zscore_global(arrs):
    M = np.vstack(arrs)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(M, axis=0, keepdims=True)
        sd = np.nanstd(M, axis=0, keepdims=True) + 1e-9
    return mu, sd

mu_c, sd_c = zscore_global([data[s][0] for s in subjects])
mu_b, sd_b = zscore_global([data[s][1] for s in subjects])

# ---------------------------------------------------------------- label builders
def direction_labels(dY_col):
    """Binary: 1 = rising, 0 = falling. NaN where dY is NaN."""
    lab = np.full(len(dY_col), np.nan)
    ok = np.isfinite(dY_col)
    lab[ok] = (dY_col[ok] > 0).astype(float)
    return lab

def magnitude_labels(dY_col):
    """
    3-class: 0=fall, 1=flat, 2=rise.
    Threshold = 0.5 x THIS SUBJECT'S OWN delta std (matches SWELL's
    per-person magnitude framing — removes individual variability
    differences rather than using one global threshold).
    """
    lab = np.full(len(dY_col), np.nan)
    ok = np.isfinite(dY_col)
    if ok.sum() < 5:
        return lab
    sd = np.nanstd(dY_col[ok])
    lab[ok] = 1
    lab[ok & (dY_col > 0.5 * sd)] = 2
    lab[ok & (dY_col < -0.5 * sd)] = 0
    return lab

# ---------------------------------------------------------------- LOSO classifier harness
def loso_classify(X_per_subj, y_per_subj, subs_list):
    accs = []
    for held in subs_list:
        train_subs = [s for s in subs_list if s != held]
        Xtr_parts = [X_per_subj[s] for s in train_subs if len(X_per_subj[s]) > 0]
        ytr_parts = [y_per_subj[s] for s in train_subs if len(y_per_subj[s]) > 0]
        if not Xtr_parts or len(X_per_subj[held]) == 0:
            continue
        Xtr = np.vstack(Xtr_parts)
        ytr = np.concatenate(ytr_parts)
        Xte = X_per_subj[held]
        yte = y_per_subj[held]
        if len(Xte) == 0 or len(np.unique(ytr)) < 2:
            continue
        m = RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced",
                                    random_state=0, n_jobs=-1)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        accs.append(accuracy_score(yte, pred))
    return np.array(accs)


def run_gate_for_target(j):
    dXc_z = {s: (data[s][0] - mu_c) / sd_c for s in subjects}
    dXb_z = {s: (data[s][1] - mu_b) / sd_b for s in subjects}

    results = {}
    for label_name, label_fn, chance in [
        ("direction", direction_labels, 0.50),
        ("magnitude", magnitude_labels, 1 / 3),
    ]:
        Xc_v1, Xfull_v2, y_lab = {}, {}, {}
        for s in subjects:
            dY_col = data[s][2][:, j]
            lab = label_fn(dY_col)
            ok = np.isfinite(lab)
            Xc_v1[s]    = dXc_z[s][ok]
            Xfull_v2[s] = np.concatenate([dXc_z[s][ok], dXb_z[s][ok]], axis=1)
            y_lab[s]    = lab[ok]

        acc_v1 = loso_classify(Xc_v1, y_lab, subjects)
        acc_v2 = loso_classify(Xfull_v2, y_lab, subjects)

        results[label_name] = dict(
            chance=chance,
            v1_mean=np.nanmean(acc_v1) if len(acc_v1) else np.nan,
            v2_mean=np.nanmean(acc_v2) if len(acc_v2) else np.nan,
        )
    return results


print(f"Running dynamics gate across {len(subjects)} subjects, "
      f"{len(REMAINING_TARGETS)} targets, direction + magnitude framings...\n")
print(f"{'target':20s} {'frame':10s} {'chance':>7s} {'V1_acc':>7s} {'V2_acc':>7s} {'lift':>9s}")
print("-" * 70)

all_lifts = []
all_v1_vs_chance = []
for j, name in enumerate(REMAINING_TARGETS):
    res = run_gate_for_target(j)
    for frame in ("direction", "magnitude"):
        r = res[frame]
        lift = r["v2_mean"] - r["v1_mean"]
        all_lifts.append(lift)
        all_v1_vs_chance.append(r["v1_mean"] - r["chance"])
        print(f"{name:20s} {frame:10s} {r['chance']:7.3f} {r['v1_mean']:7.3f} "
              f"{r['v2_mean']:7.3f} {lift:+9.3f}")

print()
print("=" * 70)
print("Q1 -- does the DYNAMICS framing find signal at all (V1 vs chance)?")
print("=" * 70)
n_above_chance = sum(1 for d in all_v1_vs_chance if d > 0.05)
print(f"  Target x frame combos clearing chance by >0.05: {n_above_chance} / {len(all_v1_vs_chance)}")
print(f"  Mean (V1_acc - chance) across all combos: {np.nanmean(all_v1_vs_chance):+.3f}")

print()
print("=" * 70)
print("Q2 -- does adding HR/RMSSD/SCL deltas help beyond count deltas (V2 vs V1)?")
print("=" * 70)
n_strong_lift = sum(1 for d in all_lifts if d > 0.08)
mean_lift = np.nanmean(all_lifts)
print(f"  Mean lift (V2-V1) across all combos: {mean_lift:+.3f}")
print(f"  Combos with lift > 0.08: {n_strong_lift} / {len(all_lifts)}")
print()
if mean_lift > 0.05 and n_strong_lift >= 3:
    print("GATE (dynamics): PASS -- biosignal deltas add real value beyond behavior.")
else:
    print("GATE (dynamics): FAIL -- same direction as the absolute-value gate:")
    print("  HR/RMSSD/SCL deltas do not meaningfully improve prediction of")
    print("  EEG/fNIRS/ACC/RIP dynamics either, beyond behavior alone.")
    print()
    print("  Note: even if Q2 fails, check Q1 above separately -- if V1 alone")
    print("  clears chance meaningfully (e.g. via acc_jerk/eeg_engagement's")
    print("  known activity confound), that's still informative on its own,")
    print("  just not an argument for adding the biosignal inputs.")
