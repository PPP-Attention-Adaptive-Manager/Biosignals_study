"""
gate_v1_vs_v2.py
==================
THE GATE TEST. This is the actual question the whole Phase 0 mission
was built to answer:

  Does adding HR + RMSSD + SCL as INPUTS help predict the remaining
  10 biosignal targets (EEG/fNIRS/RIP/ACC), beyond what mouse+keyboard
  counts alone already give you?

  V1 = SWELL-style counts only              -> 10 remaining targets
  V2 = counts + (hr_mean, hrv_rmssd, eda_scl) -> 10 remaining targets

Both predict the SAME targets, on the SAME LOSO folds, so the R²
difference is a clean, direct measure of how much the 3 extra inputs
add.

3 of the 16 count columns (SnRightClicked, SnDoubleClicked, SnDragged)
are always-NaN for Cog Lab (confirmed: not logged event types) and are
DROPPED here before training — they carry zero information by
construction and would only break Ridge (which can't handle NaN at all).

GATE DECISION RULE (stated before looking at results, to avoid
post-hoc rationalization):
  PASS if V2 beats V1 by mean ΔR² > 0.10 across the 10 targets, AND
  at least 3 individual targets show ΔR² > 0.15.
  Otherwise: FAIL — document as a negative result, do not proceed to
  the SWELL-KW transfer step.
"""

import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

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

# ---------------------------------------------------------------- load + clean
def zscore(M):
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(M, axis=0, keepdims=True)
        sd = np.nanstd(M, axis=0, keepdims=True) + 1e-9
    return (M - mu) / sd

z0 = np.load(os.path.join(CACHE_DIR, f"{subjects[0]}.npz"), allow_pickle=True)
swell_names = list(z0["swell_names"])
keep_idx = [i for i, n in enumerate(swell_names) if n not in ALWAYS_NAN_COLS]
kept_names = [swell_names[i] for i in keep_idx]
print(f"Dropping always-NaN columns: {sorted(ALWAYS_NAN_COLS)}")
print(f"Keeping {len(kept_names)} count features: {kept_names}\n")

data = {}
for sid in subjects:
    z = np.load(os.path.join(CACHE_DIR, f"{sid}.npz"), allow_pickle=True)
    X_counts = z["X_counts"][:, keep_idx].astype(float)
    X_biosig = z["X_input_biosig"].astype(float)
    Y = z["Y_remaining"].astype(float)
    data[sid] = (zscore(X_counts), zscore(X_biosig), zscore(Y))

# ---------------------------------------------------------------- LOSO harness
def loso_compare(data, subs_list, n_targets):
    F = len(subs_list)
    r = {k: np.full((F, n_targets), np.nan) for k in ("v0", "v1_counts", "v1_full",
                                                        "v3_counts", "v3_full")}
    for fi, held in enumerate(subs_list):
        Xc_tr = np.vstack([data[s][0] for s in subs_list if s != held])
        Xb_tr = np.vstack([data[s][1] for s in subs_list if s != held])
        Ytr   = np.vstack([data[s][2] for s in subs_list if s != held])
        Xc_te, Xb_te, Yte = data[held]

        Xfull_tr = np.concatenate([Xc_tr, Xb_tr], axis=1)
        Xfull_te = np.concatenate([Xc_te, Xb_te], axis=1)

        c_ok_tr = np.isfinite(Xc_tr).all(axis=1)
        c_ok_te = np.isfinite(Xc_te).all(axis=1)
        f_ok_tr = np.isfinite(Xfull_tr).all(axis=1)
        f_ok_te = np.isfinite(Xfull_te).all(axis=1)

        for j in range(n_targets):
            tr_c = c_ok_tr & np.isfinite(Ytr[:, j])
            te_c = c_ok_te & np.isfinite(Yte[:, j])
            tr_f = f_ok_tr & np.isfinite(Ytr[:, j])
            te_f = f_ok_te & np.isfinite(Yte[:, j])
            if tr_c.sum() < 20 or te_c.sum() < 3:
                continue

            yt = Yte[te_c, j]
            sst = ((yt - yt.mean()) ** 2).sum() + 1e-12
            r["v0"][fi, j] = 1 - ((yt - Ytr[tr_c, j].mean()) ** 2).sum() / sst

            def sc(model, Xfit, Xpred, tr_mask, te_mask, y_col):
                yt_local = Yte[te_mask, y_col]
                sst_local = ((yt_local - yt_local.mean()) ** 2).sum() + 1e-12
                model.fit(Xfit[tr_mask], Ytr[tr_mask, y_col])
                pred = model.predict(Xpred[te_mask])
                return 1 - ((yt_local - pred) ** 2).sum() / sst_local

            r["v1_counts"][fi, j] = sc(RidgeCV([0.1, 1, 10, 100]),
                                        Xc_tr, Xc_te, tr_c, te_c, j)
            r["v3_counts"][fi, j] = sc(RandomForestRegressor(
                300, min_samples_leaf=5, max_depth=10, n_jobs=-1, random_state=0),
                                        Xc_tr, Xc_te, tr_c, te_c, j)

            if tr_f.sum() < 20 or te_f.sum() < 3:
                continue
            r["v1_full"][fi, j] = sc(RidgeCV([0.1, 1, 10, 100]),
                                      Xfull_tr, Xfull_te, tr_f, te_f, j)
            r["v3_full"][fi, j] = sc(RandomForestRegressor(
                300, min_samples_leaf=5, max_depth=10, n_jobs=-1, random_state=0),
                                      Xfull_tr, Xfull_te, tr_f, te_f, j)
    return r


print("Running LOSO across", len(subjects), "subjects on", len(REMAINING_TARGETS), "targets...")
print("(this trains Ridge + RF, twice each, for every fold x target — may take a moment)\n")

results = loso_compare(data, list(data.keys()), len(REMAINING_TARGETS))

# ---------------------------------------------------------------- report
print("=" * 90)
print("GATE RESULTS — V1 (counts only) vs V2 (counts + HR/RMSSD/SCL)")
print("=" * 90)
header = f"{'target':20s} {'V0':>6s} {'V1_cnt':>7s} {'V1_full':>8s} {'ΔRidge':>8s}  {'V3_cnt':>7s} {'V3_full':>8s} {'ΔRF':>7s}"
print(header)
print("-" * len(header))

delta_ridge_all, delta_rf_all = [], []
n_strong = 0
for j, name in enumerate(REMAINING_TARGETS):
    v0 = np.nanmean(results["v0"][:, j])
    v1c = np.nanmean(results["v1_counts"][:, j])
    v1f = np.nanmean(results["v1_full"][:, j])
    v3c = np.nanmean(results["v3_counts"][:, j])
    v3f = np.nanmean(results["v3_full"][:, j])
    d_ridge = v1f - v1c
    d_rf = v3f - v3c
    delta_ridge_all.append(d_ridge)
    delta_rf_all.append(d_rf)
    if max(d_ridge, d_rf) > 0.15:
        n_strong += 1
    print(f"{name:20s} {v0:6.3f} {v1c:7.3f} {v1f:8.3f} {d_ridge:+8.3f}  "
          f"{v3c:7.3f} {v3f:8.3f} {d_rf:+7.3f}")

mean_d_ridge = np.nanmean(delta_ridge_all)
mean_d_rf = np.nanmean(delta_rf_all)

print()
print("=" * 90)
print("GATE DECISION")
print("=" * 90)
print(f"Mean ΔR² (Ridge, full - counts-only): {mean_d_ridge:+.3f}")
print(f"Mean ΔR² (RF,    full - counts-only): {mean_d_rf:+.3f}")
print(f"Targets with ΔR² > 0.15 (either model): {n_strong} / {len(REMAINING_TARGETS)}")
print()

best_mean_delta = max(mean_d_ridge, mean_d_rf)
if best_mean_delta > 0.10 and n_strong >= 3:
    print("GATE: PASS")
    print("  Adding HR+RMSSD+SCL as inputs meaningfully improves prediction of")
    print("  the remaining biosignals from behavior. Proceed to Phase 1:")
    print("  apply this V2 model's logic to SWELL-KW's existing counts +")
    print("  HR/RMSSD/SCL to generate imputed EEG/fNIRS/RIP/ACC columns —")
    print("  but remember: SWELL-KW has nothing to verify those imputations")
    print("  against, so treat them as soft/auxiliary signal only, never")
    print("  ground truth (see aam_physio_grounding_guide.md framing).")
else:
    print("GATE: FAIL")
    print("  Adding HR+RMSSD+SCL as inputs does not meaningfully improve")
    print("  prediction of EEG/fNIRS/RIP/ACC from Cog Lab behavior+biosignals.")
    print("  Do NOT proceed to the SWELL-KW transfer step — there is nothing")
    print("  validated to transfer. Document this as a negative result:")
    print('  "Cheap physiological inputs (HR, HRV, SCL) do not unlock')
    print('   prediction of richer biosignals (EEG, fNIRS, respiration,')
    print('   accelerometer) from behavior at this sample size (N=16)."')
