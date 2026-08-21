"""
gate_dynamics_v2.py
======================
Patches the ORIGINAL gate_dynamics_v1_vs_v2.py to measure chance
EMPIRICALLY, per LOSO fold, instead of trusting the hardcoded
chance=0.50 (direction) / chance=1/3 (magnitude) it used before.

WHY THIS MATTERS
------------------
The original script reported resp_bpm direction V1=0.709, V2=0.802
against a HARDCODED chance=0.50. But resp_bpm's raw window-to-window
delta was later found to be heavily skewed (~85% falling across nearly
every Cog Lab subject -- physiological settling over a single
uninterrupted session, confirmed via direct diagnostic and reproduced
independently in resp_proxy_chain.py). If that same skew is present
here, the TRUE empirical chance could be much higher than 0.50, and the
reported "win" could be at or below real chance -- the exact bug we
already caught and partially fixed elsewhere, just never checked in
THIS script because it never measured its own assumption.

This version changes nothing about the features, models, or windows --
identical dXc/dXb/dY construction, identical RF, identical LOSO
structure. The ONLY change: chance is now computed as the empirical
per-fold majority-class rate (mean(yte), 1-mean(yte) for direction;
per-class bincount max for magnitude), same convention used everywhere
else in this project, instead of the hardcoded constant.

Run from: ~/biosignals_data/
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

CACHE_DIR = os.path.expanduser("~/biosignals_data/data/cache/proxy_cache_swellstyle")
COG_LAB_DIR = os.path.expanduser("~/biosignals_data/data/cog_lab")
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

data = {}
for sid in subjects:
    z = np.load(os.path.join(CACHE_DIR, f"{sid}.npz"), allow_pickle=True)
    starts = z["starts"]
    order = np.argsort(starts)
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


def direction_labels(dY_col):
    lab = np.full(len(dY_col), np.nan)
    ok = np.isfinite(dY_col)
    lab[ok] = (dY_col[ok] > 0).astype(float)
    return lab


def magnitude_labels(dY_col):
    lab = np.full(len(dY_col), np.nan)
    ok = np.isfinite(dY_col)
    if ok.sum() < 5:
        return lab
    sd = np.nanstd(dY_col[ok])
    lab[ok] = 1
    lab[ok & (dY_col > 0.5 * sd)] = 2
    lab[ok & (dY_col < -0.5 * sd)] = 0
    return lab


def loso_classify_with_chance(X_per_subj, y_per_subj, subs_list, n_classes):
    """
    Same classification logic as the original, PLUS empirical per-fold
    chance -- the piece that was missing before.
    """
    accs, chances = [], []
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
        counts = np.bincount(yte.astype(int), minlength=n_classes)
        chances.append(counts.max() / counts.sum())
    return np.array(accs), np.array(chances)


def run_gate_for_target(j):
    dXc_z = {s: (data[s][0] - mu_c) / sd_c for s in subjects}
    dXb_z = {s: (data[s][1] - mu_b) / sd_b for s in subjects}

    results = {}
    for label_name, label_fn, n_classes in [
        ("direction", direction_labels, 2),
        ("magnitude", magnitude_labels, 3),
    ]:
        Xc_v1, Xfull_v2, y_lab = {}, {}, {}
        for s in subjects:
            dY_col = data[s][2][:, j]
            lab = label_fn(dY_col)
            ok = np.isfinite(lab)
            Xc_v1[s]    = dXc_z[s][ok]
            Xfull_v2[s] = np.concatenate([dXc_z[s][ok], dXb_z[s][ok]], axis=1)
            y_lab[s]    = lab[ok]

        acc_v1, chance_v1 = loso_classify_with_chance(Xc_v1, y_lab, subjects, n_classes)
        acc_v2, chance_v2 = loso_classify_with_chance(Xfull_v2, y_lab, subjects, n_classes)

        results[label_name] = dict(
            chance_hardcoded=0.50 if label_name == "direction" else 1/3,
            chance_empirical_v1=np.nanmean(chance_v1) if len(chance_v1) else np.nan,
            chance_empirical_v2=np.nanmean(chance_v2) if len(chance_v2) else np.nan,
            v1_mean=np.nanmean(acc_v1) if len(acc_v1) else np.nan,
            v2_mean=np.nanmean(acc_v2) if len(acc_v2) else np.nan,
        )
    return results


print(f"Re-running the ORIGINAL dynamics gate with EMPIRICAL chance instead")
print(f"of the hardcoded 0.50/0.333 the original script used.\n")
print(f"{'target':20s} {'frame':10s} {'hardcd':>7s} {'real_ch':>8s} "
      f"{'V1_acc':>7s} {'over(hc)':>9s} {'over(real)':>11s}")
print("-" * 88)

flagged_suspect = []
for j, name in enumerate(REMAINING_TARGETS):
    res = run_gate_for_target(j)
    for frame in ("direction", "magnitude"):
        r = res[frame]
        over_hardcoded = r["v1_mean"] - r["chance_hardcoded"]
        over_real = r["v1_mean"] - r["chance_empirical_v1"]
        flag = ""
        if over_hardcoded > 0.05 and over_real < 0.02:
            flag = "  <- WAS FALSE POSITIVE (real chance too high)"
            flagged_suspect.append(f"{name}/{frame}")
        print(f"{name:20s} {frame:10s} {r['chance_hardcoded']:7.3f} "
              f"{r['chance_empirical_v1']:8.3f} {r['v1_mean']:7.3f} "
              f"{over_hardcoded:+9.3f} {over_real:+11.3f}{flag}")

print("\n" + "=" * 88)
print("VERDICT")
print("=" * 88)
if flagged_suspect:
    print(f"\n{len(flagged_suspect)} target/frame combos looked like wins against")
    print(f"the hardcoded chance but do NOT clear the REAL empirical chance:")
    for f in flagged_suspect:
        print(f"    {f}")
    print("\nThese were false positives from the same class-imbalance blind spot")
    print("already diagnosed and fixed elsewhere in this project.")
else:
    print("\nNo combo flipped from apparent-win to real-null. The original")
    print("hardcoded-chance results happen to be robust to this specific check")
    print("(their empirical chance was close enough to 0.50/0.333 that it")
    print("didn't matter here) -- but this should still be the standard going")
    print("forward rather than assumed safe next time.")
