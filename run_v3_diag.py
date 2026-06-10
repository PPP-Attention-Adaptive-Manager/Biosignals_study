"""
biosignal_proxy/run_v3_diag.py
===============================
Two decisive checks layered on the V0/V1 gate:

  V3  — Random Forest per target (same LOSO harness). Does nonlinearity rescue
        the autonomic targets (HR / HRV / resp) that ridge left near zero?

  ACT — "activity-only" confound baseline: ridge using ONLY gross activity-magnitude
        features (mouse speed, click rate, keystroke rate, idle ratio). If a target's
        activity-only R2 ~= its full R2, that target is explained by "how much the
        person was moving", i.e. a trivial activity confound — NOT a physiological
        mapping. This is the test for acc_jerk and eeg_engagement.

  COEF — top ridge coefficients per target (fit on all data, interpretability only)
         so you can see WHICH behaviors drive the weak-but-nonzero targets.

Reuses the loaders / normalization from run_baseline.py.

Run:
    pip install scikit-learn --quiet
    python run_v3_diag.py
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor

import run_baseline as RB
import hci_features as H
import biosignals_targets as B

# gross "how much are they doing stuff" features — the activity-confound probe
ACTIVITY = ["ms_speed_mean", "ms_click_rate", "kb_rate", "ms_idle_ratio"]


def loso_models(data, subs, n_targets, xnames):
    act_idx = np.array([xnames.index(a) for a in ACTIVITY if a in xnames])
    out = {k: np.full((len(subs), n_targets), np.nan) for k in ("v1", "v3", "act")}

    for fi, held in enumerate(subs):
        Xtr = np.vstack([data[s][0] for s in subs if s != held])
        Ytr = np.vstack([data[s][1] for s in subs if s != held])
        Xte, Yte = data[held]
        xtr_ok = np.isfinite(Xtr).all(axis=1)
        xte_ok = np.isfinite(Xte).all(axis=1)

        for j in range(n_targets):
            tr = xtr_ok & np.isfinite(Ytr[:, j])
            te = xte_ok & np.isfinite(Yte[:, j])
            if tr.sum() < 30 or te.sum() < 5:
                continue
            yt = Yte[te, j]
            sst = ((yt - yt.mean()) ** 2).sum() + 1e-12

            def score(model, cols):
                model.fit(Xtr[tr][:, cols], Ytr[tr, j])
                p = model.predict(Xte[te][:, cols])
                return 1 - ((yt - p) ** 2).sum() / sst

            allc = np.arange(Xtr.shape[1])
            out["v1"][fi, j]  = score(RidgeCV(alphas=[0.1, 1, 10, 100]), allc)
            out["v3"][fi, j]  = score(RandomForestRegressor(
                                    n_estimators=300, min_samples_leaf=5,
                                    max_depth=10, n_jobs=-1, random_state=0), allc)
            out["act"][fi, j] = score(RidgeCV(alphas=[0.1, 1, 10, 100]), act_idx)
    return out


def top_coefficients(data, subs, xnames, ynames, k=3):
    """Fit one ridge per target on ALL data (interpretability only) -> top-|coef| features."""
    X = np.vstack([data[s][0] for s in subs])
    Y = np.vstack([data[s][1] for s in subs])
    xok = np.isfinite(X).all(axis=1)
    lines = []
    for j, name in enumerate(ynames):
        m = xok & np.isfinite(Y[:, j])
        if m.sum() < 30:
            lines.append(f"{name:20s}  (insufficient data)"); continue
        r = Ridge(alpha=10.0).fit(X[m], Y[m, j])
        order = np.argsort(-np.abs(r.coef_))[:k]
        feats = ", ".join(f"{xnames[i]}({r.coef_[i]:+.2f})" for i in order)
        lines.append(f"{name:20s}  {feats}")
    return lines


def main():
    subs = RB.list_subjects()
    print(f"subjects ({len(subs)}): {subs}\n")
    data = {}
    for s in subs:
        X, Y = RB.load_subject(s)
        data[s] = (H.per_user_zscore(X), RB.zscore_self(Y))

    yn = B.TARGET_NAMES
    out = loso_models(data, subs, len(yn), H.FEATURE_NAMES)

    print(f"{'target':20s} {'V1(lin)':>8s} {'V3(RF)':>8s} {'ACT-only':>9s}  verdict")
    print("-" * 70)
    for j, name in enumerate(yn):
        v1 = np.nanmean(out["v1"][:, j])
        v3 = np.nanmean(out["v3"][:, j])
        ac = np.nanmean(out["act"][:, j])
        # verdict heuristics
        if max(v1, v3) < 0.1:
            verdict = "flat (no signal)"
        elif ac > 0.6 * max(v1, v3):
            verdict = "ACTIVITY CONFOUND"
        elif v3 > v1 + 0.1:
            verdict = "nonlinear signal"
        else:
            verdict = "weak signal"
        print(f"{name:20s} {v1:8.3f} {v3:8.3f} {ac:9.3f}  {verdict}")
    print("-" * 70)
    print("\nReading it:")
    print("  ACTIVITY CONFOUND -> target is just 'how much they moved', discard as validation")
    print("  nonlinear signal  -> RF beats ridge; real but nonlinear relationship")
    print("  weak / flat       -> HCI doesn't carry this biosignal\n")

    print("Top ridge coefficients per target (which behaviors drive each):")
    for line in top_coefficients(data, subs, H.FEATURE_NAMES, yn):
        print("  " + line)


if __name__ == "__main__":
    main()