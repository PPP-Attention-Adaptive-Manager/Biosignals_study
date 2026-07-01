"""
decorative_biosig_proxy.py
=============================
Bundled proxy: mouse/keyboard counts -> all 10 "remaining" biosignal
targets from the Cog Lab gate work. Two tiers, by explicit design:

TIER 1 — REAL TRAINED MODELS (acc_jerk, eeg_engagement, resp_bpm)
  These three are the only targets with real predictability from HCI
  counts (R² 0.12-0.72 depending on target). Models here are genuine,
  fitted RandomForestRegressors.

  CAVEAT BAKED INTO THIS FILE, NOT JUST DOCUMENTED ELSEWHERE:
  feature_importance_compare.py showed all three rely on the SAME
  underlying HCI features (SnKeyStrokes, SnMouseDistance, SnMouseAct,
  CharactersRatio) — Spearman rho between their importance profiles:
    acc_jerk vs eeg_engagement   : 0.758
    acc_jerk vs resp_bpm         : 0.720
    eeg_engagement vs resp_bpm   : 0.907  <- most entangled pair
  This means these three outputs are NOT three independent signals —
  they are one shared "typing/clicking intensity" dimension, expressed
  through three different sensor channels. Predicting them adds
  reconstruction noise on top of information AAM's own pre-embedders
  (ms_speed_mean, kb_rate, SnMouseDistance, etc.) already have directly
  and exactly, with zero reconstruction error.

  USE CASE THIS IS APPROPRIATE FOR: illustrative/decorative display
  (e.g. a demo dashboard showing "estimated physical activity level"),
  NOT as a cognitive-load feature feeding the AAM fusion model. Using
  these as fusion-model inputs would just be feeding the model a noisy
  copy of features it already has, dressed up as if from new sensors.

TIER 2 — FLAT DECORATIVE PLACEHOLDERS (the other 7 targets)
  acc_movement, eda_tonic_slope, eda_phasic_count, eeg_theta_alpha,
  eeg_alpha_asym, fnirs_hbo_slope_L, fnirs_hbo_slope_R showed no real
  signal under either absolute-value or dynamics framing (both gates).
  No model is fitted for these — they return the population mean,
  explicitly flagged as non-predictive. Building a "model" for a flat
  target would just be an elaborate way of computing the same constant.
"""

import os
import numpy as np
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
TIER1_TARGETS = ["acc_jerk", "eeg_engagement", "resp_bpm"]
TIER2_TARGETS = [t for t in REMAINING_TARGETS if t not in TIER1_TARGETS]


class BiosignalProxy:
    """
    .predict(counts) -> dict of {target_name: (value, tier, note)}
    counts: array matching the 13 kept SWELL-style count features.
    """

    def __init__(self, feature_names, models, means):
        self.feature_names = feature_names
        self.models = models      # {target: fitted RF}, Tier 1 only
        self.means = means        # {target: float}, Tier 2 only

    def predict(self, counts: np.ndarray) -> dict:
        counts = np.asarray(counts, dtype=float).reshape(1, -1)
        out = {}
        for t in TIER1_TARGETS:
            val = float(self.models[t].predict(counts)[0])
            out[t] = (val, "TIER1_real_but_redundant",
                      "trained model, shares feature basis with the other "
                      "two TIER1 targets — see module docstring")
        for t in TIER2_TARGETS:
            out[t] = (self.means[t], "TIER2_flat_decorative",
                      "population mean, no real predictability found")
        return out


def build_proxy() -> BiosignalProxy:
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

    models, means = {}, {}
    for t in TIER1_TARGETS:
        j = REMAINING_TARGETS.index(t)
        y = Y_all[:, j]
        ok = np.isfinite(y) & np.isfinite(X_all).all(axis=1)
        m = RandomForestRegressor(300, min_samples_leaf=5, max_depth=10,
                                   n_jobs=-1, random_state=0)
        m.fit(X_all[ok], y[ok])
        models[t] = m

    for t in TIER2_TARGETS:
        j = REMAINING_TARGETS.index(t)
        y = Y_all[:, j]
        means[t] = float(np.nanmean(y))

    return BiosignalProxy(kept_names, models, means), X_all, Y_all, subjects


if __name__ == "__main__":
    proxy, X_all, Y_all, subjects = build_proxy()

    print("=" * 70)
    print("DEMO — proxy predictions on a few sample windows")
    print("=" * 70)
    print(f"Features used: {proxy.feature_names}\n")

    # show 5 sample windows from the pooled data, predicted vs actual
    sample_rows = np.linspace(0, len(X_all) - 1, 5).astype(int)
    for row in sample_rows:
        x = X_all[row]
        preds = proxy.predict(x)
        print(f"--- window {row} ---")
        for t in REMAINING_TARGETS:
            val, tier, note = preds[t]
            actual_idx = REMAINING_TARGETS.index(t)
            actual = Y_all[row, actual_idx]
            tag = "REAL" if tier.startswith("TIER1") else "FLAT"
            print(f"  [{tag}] {t:18s} predicted={val:8.3f}  actual={actual:8.3f}")
        print()

    print("=" * 70)
    print("REMINDER (baked into this file's docstring too):")
    print("  TIER1 targets (acc_jerk, eeg_engagement, resp_bpm) share the")
    print("  same feature basis — Spearman rho 0.72-0.91 between them.")
    print("  Appropriate for illustrative/demo display. NOT recommended")
    print("  as AAM fusion-model inputs — they would add reconstruction")
    print("  noise on top of features AAM's pre-embedders already extract")
    print("  directly and exactly (ms_speed_mean, kb_rate, SnMouseDistance).")
    print("=" * 70)
