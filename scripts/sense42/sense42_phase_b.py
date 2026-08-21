"""
sense42_phase_b.py
===================
Phase B — Build the new proxy directly from SENSE-42.

Instead of transferring SWELL-KW models (which Phase A showed fails),
we train directly on SENSE-42 where HCI and EEG are synchronized.

What this computes per 30-second window:
  HCI side  : 18 SWELL-style counts (from CSV list-based extraction)
  EEG side  : frontal theta, alpha, theta/alpha ratio, engagement index
              from channels AF3, F3, Fz, F4, AF4

Then runs:
  1. CCA : HCI ↔ EEG band powers  →  CV r = the key number
  2. RF  : HCI → theta_rising, engagement_rising (direction classifiers)

Channels confirmed: AF3, F3, Fz, F4, AF4 (frontal)
                    P3, Pz, P4 (posterior alpha)
Sample rate: 100 Hz, Duration ~7993s, 32 channels

Run from: ~/biosignals_data/
Requires:
  - data/sense_42/EEG_cleaned/P00N_100Hz_downsampled.set
    (extract one at a time: unzip data/sense_42/EEG_cleaned.zip
     "EEG_cleaned/P00N_100Hz_downsampled.set" -d data/sense_42/)
  - data/sense_42/Behavioural/CSV/*.csv  (already extracted)
  - scripts/sense42/sense42_phase_a.py   (for HCI extraction function)

Process: extract one participant, cache features, delete .set, next.
Output:
  outputs/sense42_phase_b_windows.csv  — per-window HCI + EEG features
  outputs/sense42_phase_b_results.json — CCA + RF results
  models/sense42_cca_vector.npy        — new frozen CCA projection
  models/sense42_cca_mu_sd.npy         — normalization params
"""
from __future__ import annotations
import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
import mne, joblib

warnings.filterwarnings("ignore")

BASE     = os.path.expanduser("~/biosignals_data")
EEG_DIR  = os.path.join(BASE, "data", "sense_42", "EEG_cleaned")
CSV_DIR  = os.path.join(BASE, "data", "sense_42", "Behavioural", "CSV")
EEG_ZIP  = os.path.join(BASE, "data", "sense_42", "EEG_cleaned.zip")
OUT_WIN  = os.path.join(BASE, "outputs", "sense42_phase_b_windows.csv")
OUT_JSON = os.path.join(BASE, "outputs", "sense42_phase_b_results.json")
MDL_DIR  = os.path.join(BASE, "models")
os.makedirs(MDL_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)

WINDOW_S  = 30.0
SFREQ_EEG = 100.0

# Frontal channels for theta/alpha/engagement
FRONTAL   = ['AF3', 'F3', 'Fz', 'F4', 'AF4']
POSTERIOR = ['P3', 'Pz', 'P4']          # posterior alpha (attention focus)

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]


# ── Import HCI extractor from phase_a ────────────────────────────────
sys.path.insert(0, os.path.join(BASE, "scripts", "sense42"))
from sense42_phase_a import extract_hci_from_csv


# ── EEG band power extraction ─────────────────────────────────────────
def compute_band_powers(epoch_data: np.ndarray, sfreq: float) -> dict:
    """
    Compute per-channel band powers using Welch method.
    epoch_data shape: (n_channels, n_samples)
    Returns dict of band power scalars.
    """
    f, psd = welch(epoch_data, fs=sfreq, nperseg=int(sfreq * 2), axis=1)
    # freq masks
    theta_m = (f >= 4)  & (f < 8)
    alpha_m = (f >= 8)  & (f < 13)
    beta_m  = (f >= 13) & (f < 30)

    theta = psd[:, theta_m].mean(axis=1)   # (n_channels,)
    alpha = psd[:, alpha_m].mean(axis=1)
    beta  = psd[:, beta_m ].mean(axis=1)

    return {"theta": theta, "alpha": alpha, "beta": beta}


def extract_eeg_features(set_path: str, ch_names: list) -> pd.DataFrame:
    """
    Load cleaned EEG .set, extract band powers per 30s window.
    Returns DataFrame: window_start, frontal_theta, frontal_alpha,
                       theta_alpha_ratio, engagement_index, posterior_alpha
    """
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    data  = raw.get_data()        # (32, n_samples)
    times = raw.times

    # channel indices
    def ch_idx(names):
        return [raw.ch_names.index(c) for c in names if c in raw.ch_names]

    frontal_idx   = ch_idx(FRONTAL)
    posterior_idx = ch_idx(POSTERIOR)

    if not frontal_idx:
        print(f"  WARNING: no frontal channels found in {os.path.basename(set_path)}")
        return pd.DataFrame()

    rows = []
    w = times[0]
    while w + WINDOW_S <= times[-1]:
        w_end   = w + WINDOW_S
        mask    = (times >= w) & (times < w_end)
        epoch   = data[:, mask]

        if epoch.shape[1] < int(sfreq * 2):
            w += WINDOW_S; continue

        bp = compute_band_powers(epoch, sfreq)
        f_theta = bp["theta"][frontal_idx].mean()
        f_alpha = bp["alpha"][frontal_idx].mean()
        f_beta  = bp["beta"][frontal_idx].mean()
        p_alpha = bp["alpha"][posterior_idx].mean() if posterior_idx else f_alpha

        theta_alpha = float(f_theta / (f_alpha + 1e-12))
        engagement  = float(f_beta  / (f_alpha + f_theta + 1e-12))

        rows.append({
            "window_start":     w,
            "frontal_theta":    float(f_theta),
            "frontal_alpha":    float(f_alpha),
            "theta_alpha_ratio":theta_alpha,
            "engagement_index": engagement,
            "posterior_alpha":  float(p_alpha),
        })
        w += WINDOW_S

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Per-participant extraction ─────────────────────────────────────────
def process_participant(pid_str: str) -> pd.DataFrame | None:
    """
    Extract HCI + EEG features for one participant.
    Extracts EEG from zip if not already present, processes, then deletes.
    Returns merged DataFrame or None if extraction fails.
    """
    set_path = os.path.join(EEG_DIR, f"P{pid_str}_100Hz_downsampled.set")
    csv_glob = glob.glob(os.path.join(CSV_DIR, f"{pid_str}_*.csv"))

    if not csv_glob:
        print(f"  P{pid_str}: no CSV found"); return None

    # Extract EEG from zip if not already present
    extracted_here = False
    if not os.path.isfile(set_path):
        fname = f"EEG_cleaned/P{pid_str}_100Hz_downsampled.set"
        ret   = os.system(f'unzip -q "{EEG_ZIP}" "{fname}" -d '
                          f'"{os.path.dirname(EEG_DIR)}" 2>/dev/null')
        if ret != 0 or not os.path.isfile(set_path):
            print(f"  P{pid_str}: could not extract EEG"); return None
        extracted_here = True

    print(f"  P{pid_str}: extracting HCI...")
    hci_df = extract_hci_from_csv(csv_glob[0])
    if hci_df.empty:
        print(f"  P{pid_str}: HCI extraction failed")
        if extracted_here: os.remove(set_path)
        return None

    print(f"  P{pid_str}: extracting EEG band powers...")
    eeg_df = extract_eeg_features(set_path, FRONTAL + POSTERIOR)
    print(f"  P{pid_str}: EEG {len(eeg_df)} windows")

    # Delete extracted .set to free space (unless it was pre-existing)
    if extracted_here and os.path.isfile(set_path):
        os.remove(set_path)
        print(f"  P{pid_str}: .set deleted to free space")

    if eeg_df.empty:
        return None

    # Align by clock offset (same approach as Phase A)
    hci_starts = hci_df["window_start"].to_numpy(float)
    eeg_starts = eeg_df["window_start"].to_numpy(float)
    t_offset   = hci_starts[0] - eeg_starts[0]

    EEG_COLS = ["frontal_theta","frontal_alpha","theta_alpha_ratio",
                "engagement_index","posterior_alpha"]
    eeg_arr = {c: np.full(len(hci_df), np.nan) for c in EEG_COLS}

    for i, hci_t in enumerate(hci_starts):
        eeg_equiv = hci_t - t_offset
        nearest   = np.argmin(np.abs(eeg_starts - eeg_equiv))
        if np.abs(eeg_starts[nearest] - eeg_equiv) < WINDOW_S / 2:
            for c in EEG_COLS:
                eeg_arr[c][i] = eeg_df.iloc[nearest][c]

    matched = np.isfinite(eeg_arr["frontal_theta"]).sum()
    print(f"  P{pid_str}: {matched}/{len(hci_df)} windows matched")

    merged = hci_df.copy()
    merged["participant"] = f"P{pid_str}"
    for c in EEG_COLS:
        merged[c] = eeg_arr[c]

    return merged


# ── Main pipeline ─────────────────────────────────────────────────────
def run_phase_b():
    print("\n" + "="*70)
    print("SENSE-42 PHASE B — Build new proxy from HCI + EEG")
    print("="*70)

    # Find all CSV files to get participant list
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    pids = [os.path.basename(f).split('_')[0] for f in csv_files]
    print(f"Participants to process: {len(pids)}")
    print(f"EEG .set files currently extracted: "
          f"{len(glob.glob(os.path.join(EEG_DIR, '*.set')))}")
    print(f"Free space check: run 'df -h ~' if unsure\n")

    # Load cached windows if already computed
    all_dfs = []
    if os.path.isfile(OUT_WIN):
        print(f"Loading cached windows from {OUT_WIN}")
        all_dfs = [pd.read_csv(OUT_WIN)]
        cached_pids = set(all_dfs[0]["participant"].unique())
        pids = [p for p in pids if f"P{p}" not in cached_pids]
        print(f"Cached: {cached_pids}  |  Remaining: {len(pids)}")

    # Process participants one at a time
    for pid in pids:
        print(f"\n{'─'*50}")
        df = process_participant(pid)
        if df is not None and not df.empty:
            all_dfs.append(df)
            # Save incrementally
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(OUT_WIN, index=False)
            print(f"  Saved {len(combined)} total windows")

    if not all_dfs:
        print("No data collected — check paths"); return

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["frontal_theta","frontal_alpha",
                                    "engagement_index"])

    print(f"\n{'='*70}")
    print(f"DATASET: {len(df_all)} windows, "
          f"{df_all['participant'].nunique()} participants")
    print(f"EEG features: frontal_theta mean={df_all.frontal_theta.mean():.4f}, "
          f"alpha mean={df_all.frontal_alpha.mean():.4f}")

    # ── CCA: HCI ↔ EEG band powers ───────────────────────────────────
    print("\n" + "="*70)
    print("EXPERIMENT 1 — CCA: HCI counts ↔ EEG band powers")
    print("="*70)

    EEG_TARGETS = ["frontal_theta","frontal_alpha",
                   "theta_alpha_ratio","engagement_index"]

    # per-participant z-score HCI and EEG
    groups = df_all["participant"].to_numpy()
    X_hci  = df_all[HCI_COLS].to_numpy(float)
    Y_eeg  = df_all[EEG_TARGETS].to_numpy(float)

    X_z = np.zeros_like(X_hci)
    Y_z = np.zeros_like(Y_eeg)
    for pid in np.unique(groups):
        mask = groups == pid
        for j in range(X_hci.shape[1]):
            mu = np.nanmean(X_hci[mask, j])
            sd = np.nanstd(X_hci[mask, j]) + 1e-9
            X_z[mask, j] = (X_hci[mask, j] - mu) / sd
        for j in range(Y_eeg.shape[1]):
            mu = np.nanmean(Y_eeg[mask, j])
            sd = np.nanstd(Y_eeg[mask, j]) + 1e-9
            Y_z[mask, j] = (Y_eeg[mask, j] - mu) / sd

    ok = (np.isfinite(X_z).all(1) & np.isfinite(Y_z).all(1))
    X_clean = X_z[ok]; Y_clean = Y_z[ok]
    print(f"Clean windows for CCA: {ok.sum()}")

    # Fit CCA
    cca = CCA(n_components=2, max_iter=2000)
    Xc, Yc = cca.fit_transform(X_clean, Y_clean)
    train_r = [float(np.corrcoef(Xc[:,i], Yc[:,i])[0,1]) for i in range(2)]
    print(f"Train r: comp1={train_r[0]:.3f}  comp2={train_r[1]:.3f}")

    # Cross-validation (50-fold ShuffleSplit)
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
    cv_r = [[], []]
    for tr_idx, te_idx in cv.split(X_clean):
        cca_cv = CCA(n_components=2, max_iter=2000)
        cca_cv.fit(X_clean[tr_idx], Y_clean[tr_idx])
        Xte, Yte = cca_cv.transform(X_clean[te_idx], Y_clean[te_idx])
        for i in range(2):
            cv_r[i].append(np.corrcoef(Xte[:,i], Yte[:,i])[0,1])

    cv_mean = [float(np.nanmean(r)) for r in cv_r]
    cv_std  = [float(np.nanstd(r))  for r in cv_r]
    print(f"CV r:   comp1={cv_mean[0]:.3f}±{cv_std[0]:.3f}  "
          f"comp2={cv_mean[1]:.3f}±{cv_std[1]:.3f}")

    # HCI loadings comp1
    hw = cca.x_weights_[:, 0]
    top = np.argsort(np.abs(hw))[::-1][:5]
    print(f"HCI drivers: " +
          ", ".join(f"{HCI_COLS[k]}({hw[k]:+.2f})" for k in top))
    # EEG loadings comp1
    yw = cca.y_weights_[:, 0]
    print(f"EEG loadings: " +
          ", ".join(f"{EEG_TARGETS[k]}({yw[k]:+.2f})" for k in range(len(EEG_TARGETS))))

    print()
    if cv_mean[0] > 0.30:
        print("✓  rho > 0.30 — HCI and EEG share a cognitive-load dimension in SENSE-42")
        print("   This becomes the new proxy layer (cortical axis)")
    elif cv_mean[0] > 0.15:
        print("~  Marginal coupling detected (0.15 < rho < 0.30)")
        print("   Usable as weak signal but interpret cautiously")
    else:
        print("✗  rho < 0.15 — no reliable HCI-EEG coupling at 30s window scale")
        print("   Consider shorter windows or different EEG features")

    # Save CCA model
    np.save(os.path.join(MDL_DIR, "sense42_cca_vector.npy"),
            cca.x_weights_[:, 0])
    norm_params = np.array([X_clean.mean(0), X_clean.std(0) + 1e-9])
    np.save(os.path.join(MDL_DIR, "sense42_cca_mu_sd.npy"), norm_params)
    print(f"\nCCA vector saved to models/sense42_cca_vector.npy")

    # ── RF direction classifiers ──────────────────────────────────────
    print("\n" + "="*70)
    print("EXPERIMENT 2 — RF direction classifiers (chance = 0.50)")
    print("="*70)

    df_all["theta_rising"] = (df_all.groupby("participant")["frontal_theta"]
                              .diff() > 0).astype(float)
    df_all["engagement_rising"] = (df_all.groupby("participant")["engagement_index"]
                                   .diff() > 0).astype(float)
    df_all["alpha_falling"] = (df_all.groupby("participant")["frontal_alpha"]
                               .diff() < 0).astype(float)

    results_rf = {}
    for target in ["theta_rising", "engagement_rising", "alpha_falling"]:
        accs = []
        for pid in np.unique(groups):
            tr = groups != pid; te = groups == pid
            y_tr = df_all.loc[tr, target].to_numpy(float)
            y_te = df_all.loc[te, target].to_numpy(float)
            X_tr = X_z[tr]; X_te = X_z[te]
            ok_tr = np.isfinite(X_tr).all(1) & np.isfinite(y_tr)
            ok_te = np.isfinite(X_te).all(1) & np.isfinite(y_te)
            if ok_tr.sum() < 20 or ok_te.sum() < 5: continue
            m = RandomForestClassifier(200, min_samples_leaf=5,
                                        class_weight="balanced",
                                        random_state=0, n_jobs=-1)
            m.fit(X_tr[ok_tr], y_tr[ok_tr].astype(int))
            pred = m.predict(X_te[ok_te])
            accs.append(accuracy_score(y_te[ok_te].astype(int), pred))
        mean_acc = float(np.mean(accs)) if accs else np.nan
        results_rf[target] = mean_acc
        flag = "✓" if mean_acc > 0.55 else "~" if mean_acc > 0.52 else "✗"
        print(f"  {flag}  {target:22s}: {mean_acc:.3f}  "
              f"({'above chance' if mean_acc>0.52 else 'at chance'})")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE B SUMMARY")
    print("="*70)
    print(f"\nCCA Component 1:")
    print(f"  Train r = {train_r[0]:.3f}")
    print(f"  CV r    = {cv_mean[0]:.3f} ± {cv_std[0]:.3f}")
    print(f"\nRF direction classifiers (LOSO N={df_all.participant.nunique()}):")
    for t, a in results_rf.items():
        print(f"  {t:22s}: {a:.3f}  (chance=0.50)")
    print()
    print("INTERPRETATION:")
    if cv_mean[0] > 0.30:
        print("  The SENSE-42 HCI-EEG CCA is the new proxy layer.")
        print("  Combined with SWELL-KW ANS proxy, the full proxy now covers:")
        print("    ANS axis   (HR, RMSSD) ← SWELL-KW CCA vector")
        print("    Cortical axis (theta, engagement) ← SENSE-42 CCA vector")
        print("  Together these can distinguish all 5 AAM states.")
    else:
        print("  CCA did not find reliable coupling. Options:")
        print("  1. Try shorter windows (10s instead of 30s)")
        print("  2. Use task-specific epochs only (mail_content, notes_repeat)")
        print("  3. Add ICA artifact correction to the EEG preprocessing")

    # Save results
    results = {
        "cca_train_r": train_r,
        "cca_cv_r_mean": cv_mean,
        "cca_cv_r_std": cv_std,
        "cca_hci_loadings": {HCI_COLS[k]: float(hw[k]) for k in range(len(HCI_COLS))},
        "cca_eeg_loadings": {EEG_TARGETS[k]: float(yw[k]) for k in range(len(EEG_TARGETS))},
        "rf_direction": results_rf,
        "n_participants": int(df_all.participant.nunique()),
        "n_windows": int(len(df_all)),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUT_JSON}")


if __name__ == "__main__":
    run_phase_b()
