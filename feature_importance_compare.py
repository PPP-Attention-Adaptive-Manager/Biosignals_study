"""
feature_importance_compare.py
================================
Tests whether predicting resp_bpm from HCI is circular (same redundant
"general motion intensity" signal as acc_jerk/eeg_engagement) or
genuinely distinct (different feature reliance — e.g. pause/burst
structure rather than raw speed/count).

Method: fit RF on X_counts -> each of the three targets separately,
compare feature importance vectors. High similarity = same underlying
signal driving all three (redundant, circular). Low similarity =
resp_bpm captures something the other two don't.

Not a LOSO generalization claim — this is purely a "what is the model
leaning on" diagnostic, run on all available data.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

CACHE_DIR = os.path.expanduser("~/biosignals_data/proxy_cache_swellstyle")
COG_LAB_DIR = os.path.expanduser("~/biosignals_data/cog_lab")
EXCLUDE = {"S2", "S17"}
ALWAYS_NAN_COLS = {"SnRightClicked", "SnDoubleClicked", "SnDragged"}

# Full target list includes the 3 we removed as V2 inputs earlier (hr_mean,
# hrv_rmssd, eda_scl) plus the 10 remaining — acc_jerk and eeg_engagement
# are in REMAINING_TARGETS already; resp_bpm too. Indices below match
# the REMAINING_TARGETS order used when Y_remaining was built.
REMAINING_TARGETS = [
    "acc_movement", "acc_jerk",
    "eda_tonic_slope", "eda_phasic_count",
    "resp_bpm",
    "eeg_theta_alpha", "eeg_engagement", "eeg_alpha_asym",
    "fnirs_hbo_slope_L", "fnirs_hbo_slope_R",
]
COMPARE = ["acc_jerk", "eeg_engagement", "resp_bpm"]
compare_idx = [REMAINING_TARGETS.index(t) for t in COMPARE]

subjects = sorted(
    d for d in os.listdir(COG_LAB_DIR)
    if d.startswith("S") and d[1:].isdigit() and d not in EXCLUDE
)

z0 = np.load(os.path.join(CACHE_DIR, f"{subjects[0]}.npz"), allow_pickle=True)
swell_names = list(z0["swell_names"])
keep_idx = [i for i, n in enumerate(swell_names) if n not in ALWAYS_NAN_COLS]
kept_names = [swell_names[i] for i in keep_idx]

X_all, Y_all = [], []
for sid in subjects:
    z = np.load(os.path.join(CACHE_DIR, f"{sid}.npz"), allow_pickle=True)
    X_all.append(z["X_counts"][:, keep_idx].astype(float))
    Y_all.append(z["Y_remaining"].astype(float))
X_all = np.vstack(X_all)
Y_all = np.vstack(Y_all)

print(f"Pooled rows: {len(X_all)}  |  Features: {kept_names}\n")

importances = {}
for j, name in zip(compare_idx, COMPARE):
    y = Y_all[:, j]
    ok = np.isfinite(y) & np.isfinite(X_all).all(axis=1)
    m = RandomForestRegressor(300, min_samples_leaf=5, max_depth=10,
                               n_jobs=-1, random_state=0)
    m.fit(X_all[ok], y[ok])
    importances[name] = m.feature_importances_
    print(f"{name} (n={ok.sum()}):")
    order = np.argsort(importances[name])[::-1]
    for k in order[:5]:
        print(f"    {kept_names[k]:20s} {importances[name][k]:.3f}")
    print()

print("=" * 60)
print("SIMILARITY BETWEEN FEATURE-IMPORTANCE PROFILES")
print("=" * 60)
print("(Spearman rank correlation — high = same features matter for both,")
print(" low/negative = different features driving each)\n")
pairs = [("acc_jerk", "eeg_engagement"),
         ("acc_jerk", "resp_bpm"),
         ("eeg_engagement", "resp_bpm")]
for a, b in pairs:
    rho, _ = spearmanr(importances[a], importances[b])
    print(f"  {a:15s} vs {b:15s}  rho = {rho:+.3f}")

print()
print("INTERPRETATION GUIDE:")
print("  acc_jerk vs eeg_engagement should be HIGH — both are already")
print("  confirmed motion confounds, expect them to lean on the same")
print("  speed/count features. This is the baseline 'redundant' case.")
print()
print("  If resp_bpm's rho vs BOTH of the above is similarly high:")
print("    -> resp_bpm-from-HCI is ALSO just the same exertion signal,")
print("       redundant with HCI's own native activity features.")
print()
print("  If resp_bpm's rho vs both is noticeably LOWER:")
print("    -> resp_bpm relies on a different feature pattern (e.g. pause")
print("       structure, idle ratio) than raw motion intensity — worth")
print("       treating as a separate, possibly genuine signal, not the")
print("       same confound in different clothes.")
