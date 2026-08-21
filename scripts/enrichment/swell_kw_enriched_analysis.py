"""
swell_kw_enriched_analysis.py
===============================
Replicates the original SWELL-KW direction/magnitude/CCA experiments
(from analyse_cogload.ipynb Sections 4, 6, 9) but now with 4
physiological targets instead of 3:

  ORIGINAL:   HR, RMSSD, SCL          (all native SWELL-KW measurements)
  ENRICHED:   HR, RMSSD, SCL, resp    (resp imputed from Cog Lab CatBoost model)

The comparison between original-3 and enriched-4 results is the
scientifically interesting number — it tells you whether resp adds a
genuinely independent physiological dimension, or just redundant noise
on top of what HR/RMSSD already captured.

Run from: ~/biosignals_data/
Requires: swell_kw_enriched.csv (produced by resp_model_comparison.py)
Output:   prints results + saves swell_enriched_results.json
"""
from __future__ import annotations
import sys, os as _os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "features"))
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import ShuffleSplit
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ENRICHED_FILE = os.path.expanduser(
    "~/biosignals_data/outputs/swell_kw_enriched.csv")
OUT_JSON = os.path.expanduser(
    "~/biosignals_data/outputs/swell_enriched_results.json")
PLOT_DIR = os.path.expanduser("~/biosignals_data/plots")
os.makedirs(PLOT_DIR, exist_ok=True)

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]
PHY_3 = ["HR","RMSSD","SCL"]
PHY_4 = ["HR","RMSSD","SCL","resp"]     # resp = reconstructed breathing rate proxy
COND_ORDER = ["N","I","T"]
COND_NAMES = {"N":"Neutral","I":"Interruptions","T":"Time pressure"}

# ── load ──────────────────────────────────────────────────────────
print("Loading enriched SWELL-KW...")
df = pd.read_csv(ENRICHED_FILE, low_memory=False)
df = df[df["Condition"].isin(COND_ORDER)].copy()

# resp column: the imputed breathing rate direction is already binary (0/1)
# BUT for CCA and correlation work we need a continuous value.
# We reconstruct a soft continuous proxy from the direction label:
#   use resp_rising (0/1) as a rough ordinal stand-in where raw resp_bpm
#   isn't available. For CCA this is imperfect but informative.
# For direction/magnitude classification the label columns are used directly.
if "resp_rising" in df.columns:
    df["resp"] = df["resp_rising"].astype(float)   # 0/1 proxy for CCA
else:
    raise ValueError("resp_rising column not found — run resp_model_comparison.py first")

# replace 999 sentinel with NaN just in case any slipped through
for c in PHY_3:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace(999, np.nan)

print(f"  Rows: {len(df)}  |  Participants: {df['PP'].nunique()}")
print(f"  resp_rising available: {df['resp_rising'].notna().sum()} / {len(df)} rows")
print()

# ── LOSO helper ───────────────────────────────────────────────────
def loso_classify(X, y, groups, n_estimators=200):
    subs = sorted(set(groups))
    accs, f1s = [], []
    for held in subs:
        tr = groups != held; te = groups == held
        if tr.sum() < 20 or te.sum() < 3:
            continue
        Xtr, ytr = X[tr], y[tr]; Xte, yte = X[te], y[te]
        ok_tr = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
        ok_te = np.isfinite(Xte).all(1) & np.isfinite(yte)
        if ok_tr.sum() < 15 or ok_te.sum() < 3:
            continue
        mu = Xtr[ok_tr].mean(0); sd = Xtr[ok_tr].std(0) + 1e-9
        m  = RandomForestClassifier(n_estimators, min_samples_leaf=5,
                                     class_weight="balanced",
                                     random_state=0, n_jobs=-1)
        m.fit((Xtr[ok_tr]-mu)/sd, ytr[ok_tr].astype(int))
        pred = m.predict((Xte[ok_te]-mu)/sd)
        yt   = yte[ok_te].astype(int)
        accs.append(accuracy_score(yt, pred))
        f1s.append(f1_score(yt, pred, average="macro", zero_division=0))
    return np.array(accs), np.array(f1s)

groups = df["PP"].to_numpy()
X_hci  = np.nan_to_num(df[HCI_COLS].to_numpy(dtype=float))

# ── EXPERIMENT A — Direction prediction ──────────────────────────
print("="*65)
print("EXPERIMENT A — Direction prediction (chance = 0.50)")
print("="*65)
print(f"{'Target':12s} {'3-target acc':>13s} {'4-target acc':>13s} {'Δ':>8s}  note")
print("-"*60)

dir_results = {}
for tgt, col in [("HR",    "HR_rising"),
                  ("RMSSD", "RMSSD_rising"),
                  ("SCL",   "SCL_rising"),
                  ("resp",  "resp_rising")]:
    if col not in df.columns:
        print(f"{tgt:12s}  column missing"); continue
    y = df[col].to_numpy(float)
    accs, f1s = loso_classify(X_hci, y, groups)
    note = "native" if tgt != "resp" else "IMPUTED — treat with caution"
    dir_results[tgt] = {"acc": float(accs.mean()), "f1": float(f1s.mean()),
                          "note": note}
    print(f"{tgt:12s} {accs.mean():13.3f}   {'—':>13s}   {'—':>8s}  {note}")

print()
print("Replication check (original 3-target results from gate_dynamics):")
print("  HR_rising   0.790  RMSSD_rising  0.775  SCL_rising  ~0.503")
print()

# ── EXPERIMENT B — Magnitude prediction ──────────────────────────
print("="*65)
print("EXPERIMENT B — Magnitude prediction (chance = 0.333)")
print("="*65)
print(f"{'Target':12s} {'acc':>8s} {'F1':>8s}  note")
print("-"*50)

mag_results = {}
for tgt, col in [("HR",    "HR_magnitude"),
                  ("RMSSD", "RMSSD_magnitude"),
                  ("SCL",   "SCL_magnitude"),
                  ("resp",  "resp_magnitude")]:
    if col not in df.columns:
        print(f"{tgt:12s}  column missing"); continue
    y = df[col].to_numpy(float)
    accs, f1s = loso_classify(X_hci, y, groups)
    note = "native" if tgt != "resp" else "IMPUTED"
    mag_results[tgt] = {"acc": float(accs.mean()), "f1": float(f1s.mean()),
                          "note": note}
    print(f"{tgt:12s} {accs.mean():8.3f} {f1s.mean():8.3f}  {note}")

print()
print("Replication check (original results from analyse_cogload.ipynb):")
print("  HR_magnitude  0.711  RMSSD_magnitude  0.847  SCL  ~0.500")
print()

# ── EXPERIMENT C — CCA on 3 targets (replication) ────────────────
print("="*65)
print("EXPERIMENT C1 — CCA with original 3 targets (replication)")
print("="*65)

def run_cca(df, phy_targets, label="3-target"):
    agg = df.groupby(["PP","Condition"])[HCI_COLS + phy_targets].mean().reset_index()
    agg = agg[agg["Condition"].isin(COND_ORDER)]
    for col in HCI_COLS + phy_targets:
        agg[col+"_z"] = agg.groupby("PP")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))

    hci_z = [c+"_z" for c in HCI_COLS]
    phy_z = [c+"_z" for c in phy_targets]
    clean = agg[hci_z + phy_z].dropna()
    if len(clean) < 20:
        print(f"  {label}: not enough clean rows ({len(clean)})")
        return None

    Xcca = clean[hci_z].to_numpy(float)
    Ycca = clean[phy_z].to_numpy(float)
    n_comp = min(3, len(phy_targets))
    cca = CCA(n_components=n_comp, max_iter=2000)
    Xc, Yc = cca.fit_transform(Xcca, Ycca)
    canons = [float(np.corrcoef(Xc[:,i], Yc[:,i])[0,1]) for i in range(n_comp)]

    # cross-validate
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
    cv_r = [[] for _ in range(n_comp)]
    for tr_idx, te_idx in cv.split(Xcca):
        cca_cv = CCA(n_components=n_comp, max_iter=2000)
        cca_cv.fit(Xcca[tr_idx], Ycca[tr_idx])
        Xte_c, Yte_c = cca_cv.transform(Xcca[te_idx], Ycca[te_idx])
        for i in range(n_comp):
            cv_r[i].append(np.corrcoef(Xte_c[:,i], Yte_c[:,i])[0,1])

    print(f"\n  {label} ({len(clean)} condition-level rows):")
    for i in range(n_comp):
        cv_mean = float(np.nanmean(cv_r[i]))
        cv_std  = float(np.nanstd(cv_r[i]))
        print(f"    Component {i+1}: train r={canons[i]:.3f}  "
              f"CV r={cv_mean:.3f} ± {cv_std:.3f}")

    # top HCI weights for component 1
    hw = cca.x_weights_[:,0]
    pw = cca.y_weights_[:,0]
    top5 = np.argsort(np.abs(hw))[::-1][:5]
    print(f"    Component 1 HCI drivers: "
          + ", ".join(f"{HCI_COLS[k]}({hw[k]:+.2f})" for k in top5))
    print(f"    Component 1 physio:      "
          + ", ".join(f"{phy_targets[k]}({pw[k]:+.2f})"
                      for k in np.argsort(np.abs(pw))[::-1]))

    # scatter plot component 1
    cond_lbl = clean["Condition"].to_numpy() if "Condition" in clean.columns else agg.loc[clean.index,"Condition"].to_numpy()
    colors = {"N":"#4dac26","I":"#d01c8b","T":"#f1a340"}
    fig, ax = plt.subplots(figsize=(5,4))
    for cond in COND_ORDER:
        m = agg.loc[clean.index,"Condition"].to_numpy() == cond
        ax.scatter(Xc[m,0], Yc[m,0], label=COND_NAMES[cond],
                   s=60, alpha=0.85, color=colors[cond])
    ax.set_xlabel("HCI canonical variate 1")
    ax.set_ylabel("Physio canonical variate 1")
    ax.set_title(f"CCA component 1 — {label} (r={canons[0]:.2f})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR,
                          f"cca_{label.replace(' ','_').replace('-','')}.png")
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"    Plot saved: {fname}")

    return {
        "n_rows": len(clean),
        "train_r": canons,
        "cv_r_mean": [float(np.nanmean(r)) for r in cv_r],
        "cv_r_std":  [float(np.nanstd(r))  for r in cv_r],
    }

cca_3 = run_cca(df, PHY_3, "3-target (HR+RMSSD+SCL)")

print()
print("="*65)
print("EXPERIMENT C2 — CCA with 4 targets (HR+RMSSD+SCL+resp)")
print("="*65)
print("NOTE: resp column is imputed (0/1 direction label used as proxy).")
print("A higher CCA r here could reflect genuine new information from")
print("respiration, OR circular overlap since resp was predicted from")
print("the same HCI features. The Δ between C1 and C2 is what matters.")

cca_4 = run_cca(df, PHY_4, "4-target (HR+RMSSD+SCL+resp)")

# ── SUMMARY ──────────────────────────────────────────────────────
print()
print("="*65)
print("SUMMARY — 3-target vs 4-target comparison")
print("="*65)

print("\nDirection accuracy (LOSO, chance=0.50):")
for tgt, r in dir_results.items():
    print(f"  {tgt:10s} {r['acc']:.3f}  [{r['note']}]")

print("\nMagnitude accuracy (LOSO, chance=0.333):")
for tgt, r in mag_results.items():
    print(f"  {tgt:10s} {r['acc']:.3f}  [{r['note']}]")

if cca_3 and cca_4:
    d_train = cca_4["train_r"][0] - cca_3["train_r"][0]
    d_cv    = cca_4["cv_r_mean"][0] - cca_3["cv_r_mean"][0]
    print(f"\nCCA component 1 delta (4-target minus 3-target):")
    print(f"  train r: {cca_3['train_r'][0]:.3f} → {cca_4['train_r'][0]:.3f}  "
          f"Δ={d_train:+.3f}")
    print(f"  CV r:    {cca_3['cv_r_mean'][0]:.3f} → {cca_4['cv_r_mean'][0]:.3f}  "
          f"Δ={d_cv:+.3f}")
    print()
    if d_cv > 0.05:
        print("Δ > 0.05: resp adds a genuinely independent physiological")
        print("dimension beyond what HR/RMSSD/SCL already captured.")
    elif d_cv > 0:
        print("Δ small but positive: resp adds marginal information.")
        print("Likely partial overlap with the HR/RMSSD dimension (RSA).")
    else:
        print("Δ ≤ 0: resp does not add independent information beyond")
        print("HR/RMSSD/SCL. CCA discarded the resp dimension as redundant.")
        print("This is the expected result given the RSA mechanism — resp")
        print("and HR share the same cardiac-respiratory coupling signal.")

# save all results
results = {
    "direction": dir_results,
    "magnitude": mag_results,
    "cca_3target": cca_3,
    "cca_4target": cca_4,
}
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to: {OUT_JSON}")
print("Plots saved to:", PLOT_DIR)
