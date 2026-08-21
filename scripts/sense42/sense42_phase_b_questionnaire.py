"""
sense42_phase_b_questionnaire.py
=================================
Phase B (corrected) — Build the new proxy from SENSE-42 using
NASA-TLX questionnaire ratings as the cognitive load ground truth.

WHY THIS INSTEAD OF EEG BAND POWERS:
  The first Phase B attempt correlated HCI with EEG band powers at
  30-second windows and found CV r = 0.096 (no coupling). Two reasons:
    1. 30s is far too coarse for EEG — cognitive load markers operate
       at 1-4s scale; averaging over 30s flattens the signal
    2. Averaging across task types mixes brain states that differ
       fundamentally (mail reading vs file dragging vs typing)
  The questionnaire ratings, by contrast, ARE integrated impressions
  of the preceding minutes — 30s HCI aggregation matches them naturally.

STRUCTURE (directly analogous to the SWELL-KW analysis that worked):
  SWELL-KW: 25 participants × 3 conditions        =   75 rows, CV r=0.58
  SENSE-42: 42 participants × 26 questionnaires   = 1092 rows, CV r=?

  X = mean HCI counts in the LOOKBACK window before each questionnaire
  Y = NASA-TLX ratings at that questionnaire

QUESTIONNAIRE DIMENSIONS (7 available, all 1-10 sliders):
  mental_demand   — how mentally demanding was the task
  temporal_demand — how hurried or rushed was the pace
  effort          — how hard did you work
  frustration     — how insecure/discouraged/irritated/stressed
  performance     — how successful were you (REVERSED: high = good)
  attentiveness   — how focused were you
  sleepiness      — how sleepy are you

Run from: ~/biosignals_data/
Output:
  outputs/sense42_questionnaire_rows.csv  — the 1092-row analysis table
  outputs/sense42_questionnaire_results.json
  models/sense42_tlx_cca_vector.npy       — NEW PROXY: HCI → cognitive load
  models/sense42_tlx_cca_norm.npy
"""
from __future__ import annotations
import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, r2_score

warnings.filterwarnings("ignore")

BASE     = os.path.expanduser("~/biosignals_data")
CSV_DIR  = os.path.join(BASE, "data", "sense_42", "Behavioural", "CSV")
OUT_ROWS = os.path.join(BASE, "outputs", "sense42_questionnaire_rows.csv")
OUT_JSON = os.path.join(BASE, "outputs", "sense42_questionnaire_results.json")
MDL_DIR  = os.path.join(BASE, "models")
os.makedirs(MDL_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)

WINDOW_S   = 30.0     # HCI aggregation window
LOOKBACK_S = 180.0    # how far back from questionnaire to aggregate HCI (3 min)

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]

# questionnaire short name → substring to find the column
TLX_DIMS = {
    "mental_demand":   "mental_demand",
    "temporal_demand": "temporal_demand",
    "effort":          "effort:",
    "frustration":     "frustration:",
    "performance":     "performance:",
    "attentiveness":   "attentiveness:",
    "sleepiness":      "sleepiness:",
}

# Primary cognitive load targets (performance is reversed, handled below)
LOAD_TARGETS = ["mental_demand", "temporal_demand", "effort", "frustration"]

sys.path.insert(0, os.path.join(BASE, "scripts", "sense42"))
from sense42_phase_a import extract_hci_from_csv


# ── Find questionnaire columns and extract ratings + timestamps ───────
def extract_questionnaires(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame: q_time, mental_demand, temporal_demand, effort,
                       frustration, performance, attentiveness, sleepiness
    One row per questionnaire event.
    """
    cols = df.columns.tolist()
    t_col = 'thisRow.t'
    if t_col not in cols:
        return pd.DataFrame()

    # map dim → actual column name
    dim_cols = {}
    for dim, substr in TLX_DIMS.items():
        matches = [c for c in cols if substr in c and c.endswith('slider.rating')]
        if matches:
            dim_cols[dim] = matches[0]

    if 'mental_demand' not in dim_cols:
        return pd.DataFrame()

    # Rows where mental_demand has a rating = a questionnaire event
    anchor = dim_cols['mental_demand']
    q_mask = df[anchor].notna()

    out = pd.DataFrame({
        'q_time': pd.to_numeric(df.loc[q_mask, t_col], errors='coerce')
    })
    for dim, col in dim_cols.items():
        out[dim] = pd.to_numeric(df.loc[q_mask, col], errors='coerce').values

    return out.dropna(subset=['q_time']).reset_index(drop=True)


# ── Aggregate HCI features in the lookback window before a questionnaire ──
def aggregate_hci_before(hci_df: pd.DataFrame, q_time: float,
                          lookback: float = LOOKBACK_S) -> dict | None:
    """
    Mean of each HCI count over the windows falling in
    [q_time - lookback, q_time].
    Returns None if fewer than 2 windows available.
    """
    lo = q_time - lookback
    mask = (hci_df['window_start'] >= lo) & (hci_df['window_start'] < q_time)
    sub  = hci_df[mask]
    if len(sub) < 2:
        return None
    agg = {c: float(sub[c].mean()) for c in HCI_COLS if c in sub.columns}
    agg['n_windows'] = int(len(sub))
    # also capture variability — load may show as inconsistency
    agg['SnKeyStrokes_std']    = float(sub['SnKeyStrokes'].std())
    agg['SnMouseDistance_std'] = float(sub['SnMouseDistance'].std())
    agg['SnAppChange_std']     = float(sub['SnAppChange'].std())
    return agg


# ── Build the full analysis table ─────────────────────────────────────
def build_table() -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    print(f"Building questionnaire table from {len(csv_files)} participants\n")

    rows = []
    for i, csv_path in enumerate(csv_files):
        pid = os.path.basename(csv_path).split('_')[0]
        print(f"[{i+1:2d}/{len(csv_files)}] P{pid}", end=" ")

        try:
            df_raw = pd.read_csv(csv_path, low_memory=False)
            q_df   = extract_questionnaires(df_raw)
            if q_df.empty:
                print("— no questionnaires found"); continue

            hci_df = extract_hci_from_csv(csv_path)
            if hci_df.empty:
                print("— HCI extraction failed"); continue

            n_ok = 0
            for _, q in q_df.iterrows():
                agg = aggregate_hci_before(hci_df, float(q['q_time']))
                if agg is None:
                    continue
                rec = {'participant': f"P{pid}", 'q_time': float(q['q_time'])}
                rec.update(agg)
                for dim in TLX_DIMS:
                    if dim in q_df.columns:
                        rec[dim] = float(q[dim])
                rows.append(rec)
                n_ok += 1
            print(f"— {n_ok}/{len(q_df)} questionnaires with HCI context")

        except Exception as e:
            print(f"— ERROR: {e}")

    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(OUT_ROWS, index=False)
        print(f"\nSaved {len(table)} rows to {OUT_ROWS}")
    return table


# ── Analysis ──────────────────────────────────────────────────────────
def run_analysis(table: pd.DataFrame):
    print("\n" + "="*72)
    print("SENSE-42 QUESTIONNAIRE ANALYSIS")
    print("="*72)
    print(f"Rows: {len(table)}  |  Participants: {table.participant.nunique()}")
    print(f"Questionnaires per participant: "
          f"{len(table) / table.participant.nunique():.1f} avg\n")

    # Show rating distributions
    print("Rating distributions (1-10 scale):")
    for dim in TLX_DIMS:
        if dim in table.columns:
            v = table[dim].dropna()
            print(f"  {dim:16s}: mean={v.mean():.2f}  std={v.std():.2f}  "
                  f"range={v.min():.0f}-{v.max():.0f}")
    print()

    groups = table['participant'].to_numpy()

    # feature list — counts + variability features
    FEATS = [c for c in HCI_COLS if c in table.columns] + \
            [c for c in ['SnKeyStrokes_std','SnMouseDistance_std','SnAppChange_std']
             if c in table.columns]

    X = table[FEATS].to_numpy(float)
    X = np.nan_to_num(X)

    # ── per-participant z-score (removes individual baseline) ─────────
    Xz = np.zeros_like(X)
    for pid in np.unique(groups):
        m = groups == pid
        Xz[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)

    # ══ EXPERIMENT 1 — Direct correlations ═══════════════════════════
    print("="*72)
    print("EXPERIMENT 1 — Per-feature correlations with cognitive load")
    print("="*72)
    print(f"{'Feature':22s} " + "  ".join(f"{d[:9]:>9s}" for d in LOAD_TARGETS))
    print("-"*72)

    corr_results = {}
    for fi, feat in enumerate(FEATS):
        line = f"{feat:22s} "
        row_corrs = {}
        for dim in LOAD_TARGETS:
            if dim not in table.columns:
                line += f"{'—':>9s}  "; continue
            y = table[dim].to_numpy(float)
            # per-participant z-score the target too
            yz = np.zeros_like(y)
            for pid in np.unique(groups):
                m = groups == pid
                yz[m] = (y[m] - y[m].mean()) / (y[m].std() + 1e-9)
            ok = np.isfinite(Xz[:,fi]) & np.isfinite(yz)
            if ok.sum() < 30:
                line += f"{'—':>9s}  "; continue
            r = spearmanr(Xz[ok,fi], yz[ok]).statistic
            row_corrs[dim] = float(r)
            marker = "*" if abs(r) > 0.15 else " "
            line += f"{r:+8.3f}{marker} "
        corr_results[feat] = row_corrs
        print(line)
    print("\n(* = |rho| > 0.15)")

    # ══ EXPERIMENT 2 — CCA: HCI ↔ NASA-TLX ═══════════════════════════
    print("\n" + "="*72)
    print("EXPERIMENT 2 — CCA: HCI behavior ↔ NASA-TLX cognitive load")
    print("="*72)

    tlx_avail = [d for d in LOAD_TARGETS if d in table.columns]
    Y = table[tlx_avail].to_numpy(float)
    Yz = np.zeros_like(Y)
    for pid in np.unique(groups):
        m = groups == pid
        Yz[m] = (Y[m] - Y[m].mean(0)) / (Y[m].std(0) + 1e-9)

    ok = np.isfinite(Xz).all(1) & np.isfinite(Yz).all(1)
    Xc_in, Yc_in = Xz[ok], Yz[ok]
    print(f"Clean rows: {ok.sum()} / {len(table)}")
    print(f"HCI features: {len(FEATS)}  |  TLX targets: {tlx_avail}\n")

    n_comp = min(3, len(tlx_avail))
    cca = CCA(n_components=n_comp, max_iter=3000)
    Xs, Ys = cca.fit_transform(Xc_in, Yc_in)
    train_r = [float(np.corrcoef(Xs[:,i], Ys[:,i])[0,1]) for i in range(n_comp)]

    # CV with participant-aware splits (LOSO-style grouped)
    uniq_pids = np.unique(groups[ok])
    cv_r = [[] for _ in range(n_comp)]
    for held in uniq_pids:
        tr = groups[ok] != held
        te = groups[ok] == held
        if tr.sum() < 50 or te.sum() < 5: continue
        try:
            cca_cv = CCA(n_components=n_comp, max_iter=3000)
            cca_cv.fit(Xc_in[tr], Yc_in[tr])
            Xte, Yte = cca_cv.transform(Xc_in[te], Yc_in[te])
            for i in range(n_comp):
                if np.std(Xte[:,i]) > 1e-9 and np.std(Yte[:,i]) > 1e-9:
                    cv_r[i].append(np.corrcoef(Xte[:,i], Yte[:,i])[0,1])
        except Exception:
            pass

    cv_mean = [float(np.nanmean(r)) if r else np.nan for r in cv_r]
    cv_std  = [float(np.nanstd(r))  if r else np.nan for r in cv_r]

    for i in range(n_comp):
        print(f"Component {i+1}: train r={train_r[i]:.3f}  "
              f"LOSO CV r={cv_mean[i]:.3f} ± {cv_std[i]:.3f}")

    # loadings
    hw = cca.x_weights_[:,0]
    top = np.argsort(np.abs(hw))[::-1][:6]
    print(f"\nComponent 1 HCI drivers:")
    for k in top:
        print(f"  {FEATS[k]:24s} {hw[k]:+.3f}")
    yw = cca.y_weights_[:,0]
    print(f"Component 1 TLX loadings:")
    for k, d in enumerate(tlx_avail):
        print(f"  {d:24s} {yw[k]:+.3f}")

    print()
    r1 = cv_mean[0]
    if r1 > 0.30:
        print("✓  CV r > 0.30 — HCI predicts subjective cognitive load")
        print("   This IS the new proxy. Save and use for AAM.")
    elif r1 > 0.15:
        print("~  CV r 0.15-0.30 — weak but real coupling")
        print("   Usable as soft auxiliary signal, flag uncertainty")
    else:
        print("✗  CV r < 0.15 — no reliable behavioral signature of load")

    # Save the CCA vector — this is the new proxy
    np.save(os.path.join(MDL_DIR, "sense42_tlx_cca_vector.npy"), cca.x_weights_[:,0])
    np.save(os.path.join(MDL_DIR, "sense42_tlx_cca_norm.npy"),
            np.array([Xc_in.mean(0), Xc_in.std(0)+1e-9]))
    with open(os.path.join(MDL_DIR, "sense42_tlx_features.json"), "w") as f:
        json.dump(FEATS, f, indent=2)
    print(f"\nProxy saved: models/sense42_tlx_cca_vector.npy")

    # ══ EXPERIMENT 3 — RF: predict high vs low load ══════════════════
    print("\n" + "="*72)
    print("EXPERIMENT 3 — RF binary classification (LOSO)")
    print("="*72)
    print("Target: is this questionnaire above the participant's median rating?\n")

    rf_results = {}
    for dim in tlx_avail + (['attentiveness'] if 'attentiveness' in table.columns else []):
        y_raw = table[dim].to_numpy(float)
        # binary: above own median
        y_bin = np.zeros(len(y_raw))
        for pid in np.unique(groups):
            m = groups == pid
            med = np.nanmedian(y_raw[m])
            y_bin[m] = (y_raw[m] > med).astype(float)

        accs = []
        for held in np.unique(groups):
            tr = groups != held; te = groups == held
            ok_tr = np.isfinite(Xz[tr]).all(1) & np.isfinite(y_bin[tr])
            ok_te = np.isfinite(Xz[te]).all(1) & np.isfinite(y_bin[te])
            if ok_tr.sum() < 40 or ok_te.sum() < 4: continue
            if len(np.unique(y_bin[tr][ok_tr])) < 2: continue
            m_rf = RandomForestClassifier(300, min_samples_leaf=3,
                                           class_weight="balanced",
                                           random_state=0, n_jobs=-1)
            m_rf.fit(Xz[tr][ok_tr], y_bin[tr][ok_tr].astype(int))
            pred = m_rf.predict(Xz[te][ok_te])
            accs.append(accuracy_score(y_bin[te][ok_te].astype(int), pred))
        acc = float(np.mean(accs)) if accs else np.nan
        rf_results[dim] = acc
        flag = "✓" if acc > 0.58 else "~" if acc > 0.54 else "✗"
        print(f"  {flag}  {dim:18s}: {acc:.3f}  (chance=0.50, n_folds={len(accs)})")

    # ══ SUMMARY ══════════════════════════════════════════════════════
    print("\n" + "="*72)
    print("SUMMARY — SENSE-42 questionnaire proxy")
    print("="*72)
    print(f"\nData: {len(table)} questionnaire events, "
          f"{table.participant.nunique()} participants")
    print(f"\nCCA (HCI ↔ NASA-TLX):")
    print(f"  Component 1 train r = {train_r[0]:.3f}")
    print(f"  Component 1 CV r    = {cv_mean[0]:.3f} ± {cv_std[0]:.3f}")
    print(f"\nRF binary classification (above/below own median):")
    for d, a in rf_results.items():
        print(f"  {d:18s}: {a:.3f}")

    print(f"\nCOMPARISON TO SWELL-KW:")
    print(f"  SWELL-KW CCA (HCI ↔ HR/RMSSD/SCL): CV r = 0.581, N=75 rows")
    print(f"  SENSE-42 CCA (HCI ↔ NASA-TLX):     CV r = {cv_mean[0]:.3f}, "
          f"N={ok.sum()} rows")
    print(f"\n  Note: SWELL-KW target was PHYSIOLOGY (objective)")
    print(f"        SENSE-42 target is SELF-REPORT (subjective)")
    print(f"        Both are valid cognitive load proxies but measure")
    print(f"        different constructs — physiological arousal vs")
    print(f"        perceived effort. Low correlation between them is")
    print(f"        expected and documented (Matthews et al. 2015).")

    results = {
        "n_rows": int(len(table)),
        "n_participants": int(table.participant.nunique()),
        "cca_train_r": train_r,
        "cca_cv_r_mean": cv_mean,
        "cca_cv_r_std": cv_std,
        "cca_hci_loadings": {FEATS[k]: float(hw[k]) for k in range(len(FEATS))},
        "cca_tlx_loadings": {tlx_avail[k]: float(yw[k]) for k in range(len(tlx_avail))},
        "rf_accuracy": rf_results,
        "feature_correlations": corr_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUT_JSON}")


if __name__ == "__main__":
    if os.path.isfile(OUT_ROWS):
        print(f"Loading cached table from {OUT_ROWS}")
        table = pd.read_csv(OUT_ROWS)
        print(f"  {len(table)} rows, {table.participant.nunique()} participants")
        print("  (delete this file to rebuild from CSVs)\n")
    else:
        table = build_table()

    if table.empty:
        print("No data — check CSV_DIR path")
    else:
        run_analysis(table)
