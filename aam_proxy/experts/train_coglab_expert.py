"""
train_coglab_expert.py
=========================
Trains the FROZEN Cog Lab expert -- the one piece of the multi-expert
proxy that's fully validated end to end: HCI-only input, true per-
participant LOSO, permutation-checked, on real signal.

TARGETS (only the three that survived gate_dynamics_v2.py's corrected,
empirical-chance validation -- see PROXY_ARCHITECTURE.md):
    acc_jerk_direction     real_chance=0.526  acc=0.697  over=+0.171
    acc_jerk_magnitude     real_chance=0.481  acc=0.659  over=+0.178
    eeg_engagement_direction  real_chance=0.526  acc=0.612  over=+0.086

EXPLICITLY EXCLUDED (confirmed null/false-positive in the same run):
    resp_bpm (both), eeg_theta_alpha (both), eeg_alpha_asym,
    fnirs_hbo_slope_L/R, acc_movement_magnitude, eeg_engagement_magnitude

INPUT SCHEMA -- HCI ONLY
----------------------------
Deliberately trains on dXc (HCI count deltas) alone, NOT dXb (the real
HR/RMSSD/EDA deltas also present in the cache). The deployed expert
must work on BEHACOM and live AAM sessions, neither of which will ever
have real biosignal input -- training on dXb would make this
unreproducible at deployment time, the same mismatch that broke
earlier attempts to "enrich" HCI-only prediction with signals that
won't exist downstream.

VALIDATION
------------
True per-participant LOSO (LeaveOneGroupOut over the 16 usable Cog Lab
subjects, S2/S17 excluded -- matches the validation scheme already
confirmed clean, no ShuffleSplit/GroupKFold(n) leakage risk like the
SWELL-KW CCA had).

Run from: wherever proxy_cache_swellstyle/ and the Cog Lab subject
list are visible (same location gate_dynamics_v2.py ran from).
Output:   coglab_expert_artifacts/  (models + metadata.json)
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings("ignore")

CACHE_DIR   = os.path.expanduser("~/biosignals_data/data/cache/proxy_cache_swellstyle")
COG_LAB_DIR = os.path.expanduser("~/biosignals_data/data/cog_lab")
OUT_DIR     = os.path.expanduser("~/biosignals_data/aam_proxy/experts/coglab_expert_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE = {"S2", "S17"}
ALWAYS_NAN_COLS = {"SnRightClicked", "SnDoubleClicked", "SnDragged"}

# index into Y_remaining / dY, matching REMAINING_TARGETS order used
# throughout this project's Cog Lab work:
# ["acc_movement","acc_jerk","eda_tonic_slope","eda_phasic_count",
#  "resp_bpm","eeg_theta_alpha","eeg_engagement","eeg_alpha_asym",
#  "fnirs_hbo_slope_L","fnirs_hbo_slope_R"]
ACC_JERK_IDX      = 1
EEG_ENGAGEMENT_IDX = 6


def load_data():
    subjects = sorted(
        d for d in os.listdir(COG_LAB_DIR)
        if d.startswith("S") and d[1:].isdigit() and d not in EXCLUDE
    )
    z0 = np.load(os.path.join(CACHE_DIR, f"{subjects[0]}.npz"), allow_pickle=True)
    swell_names = list(z0["swell_names"])
    keep_idx = [i for i, n in enumerate(swell_names) if n not in ALWAYS_NAN_COLS]
    hci_names = [swell_names[i] for i in keep_idx]

    data = {}
    for sid in subjects:
        z = np.load(os.path.join(CACHE_DIR, f"{sid}.npz"), allow_pickle=True)
        order = np.argsort(z["starts"])
        Xc = z["X_counts"][order][:, keep_idx].astype(float)   # HCI counts
        Y  = z["Y_remaining"][order].astype(float)             # targets

        dXc = np.diff(Xc, axis=0)
        dY  = np.diff(Y, axis=0)
        data[sid] = (dXc, dY)
    return subjects, data, hci_names


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


def zscore_global(arrs):
    M = np.vstack(arrs)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(M, axis=0, keepdims=True)
        sd = np.nanstd(M, axis=0, keepdims=True) + 1e-9
    return mu, sd


def true_loso(X_per_subj, y_per_subj, subs_list, n_classes):
    """Genuine leave-one-participant-out, matching the scheme already
    confirmed clean for this exact data (no ShuffleSplit substitution)."""
    accs, chances = [], []
    for held in subs_list:
        train_subs = [s for s in subs_list if s != held]
        Xtr_parts = [X_per_subj[s] for s in train_subs if len(X_per_subj[s]) > 0]
        ytr_parts = [y_per_subj[s] for s in train_subs if len(y_per_subj[s]) > 0]
        if not Xtr_parts or len(X_per_subj[held]) == 0:
            continue
        Xtr = np.vstack(Xtr_parts); ytr = np.concatenate(ytr_parts)
        Xte = X_per_subj[held]; yte = y_per_subj[held]
        if len(Xte) == 0 or len(np.unique(ytr)) < 2:
            continue
        m = RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced",
                                   random_state=0, n_jobs=-1)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        accs.append(accuracy_score(yte, pred))
        counts = np.bincount(yte.astype(int), minlength=n_classes)
        chances.append(counts.max() / counts.sum())
    return (float(np.mean(accs)), float(np.mean(chances)), len(accs)) if accs else (np.nan, np.nan, 0)


def main():
    print("=" * 78)
    print("TRAINING FROZEN COG LAB EXPERT")
    print("Targets: acc_jerk (direction+magnitude), eeg_engagement (direction)")
    print("Input: HCI count deltas ONLY -- no real biosignals, deployable to")
    print("BEHACOM / live AAM sessions with zero sensor requirement")
    print("=" * 78)

    subjects, data, hci_names = load_data()
    print(f"\nSubjects: {len(subjects)}  ({subjects})")
    print(f"HCI features ({len(hci_names)}): {hci_names}\n")

    mu, sd = zscore_global([data[s][0] for s in subjects])
    dXc_z = {s: (data[s][0] - mu) / sd for s in subjects}

    targets = [
        ("acc_jerk_direction", ACC_JERK_IDX, direction_labels, 2),
        ("acc_jerk_magnitude", ACC_JERK_IDX, magnitude_labels, 3),
        ("eeg_engagement_direction", EEG_ENGAGEMENT_IDX, direction_labels, 2),
    ]

    metadata = {
        "hci_schema": hci_names,
        "input_normalization": {"mean": mu.tolist()[0], "std": sd.tolist()[0]},
        "targets": {},
        "excluded_confirmed_null": [
            "resp_bpm_direction", "resp_bpm_magnitude",
            "eeg_theta_alpha_direction", "eeg_theta_alpha_magnitude",
            "eeg_alpha_asym_direction", "fnirs_hbo_slope_L_direction",
            "fnirs_hbo_slope_R_direction", "acc_movement_magnitude",
            "eeg_engagement_magnitude",
        ],
        "validation_scheme": "true per-participant LOSO (LeaveOneGroupOut), "
                             "not GroupKFold/ShuffleSplit -- confirmed no "
                             "leakage risk for this exact data/script.",
        "deployment_note": "HCI-only input. Fires when session detector "
                           "identifies: single sustained task, no window "
                           "switching (see session_router.py).",
    }

    models = {}
    for name, idx, label_fn, n_classes in targets:
        X_per_subj, y_per_subj = {}, {}
        for s in subjects:
            dY_col = data[s][1][:, idx]
            lab = label_fn(dY_col)
            ok = np.isfinite(lab)
            X_per_subj[s] = dXc_z[s][ok]
            y_per_subj[s] = lab[ok]

        acc, chance, nf = true_loso(X_per_subj, y_per_subj, subjects, n_classes)
        over = acc - chance
        print(f"  {name:28s}  chance={chance:.3f}  acc={acc:.3f}  "
              f"over={over:+.3f}  ({nf} folds)")

        # final model trained on ALL data for deployment
        X_all = np.vstack([X_per_subj[s] for s in subjects if len(X_per_subj[s])>0])
        y_all = np.concatenate([y_per_subj[s] for s in subjects if len(y_per_subj[s])>0])
        m_final = RandomForestClassifier(200, min_samples_leaf=5,
                                         class_weight="balanced",
                                         random_state=0, n_jobs=-1)
        m_final.fit(X_all, y_all)
        models[name] = m_final

        metadata["targets"][name] = {
            "loso_accuracy": acc, "empirical_chance": chance,
            "over_chance": over, "n_folds": nf, "n_classes": n_classes,
        }

    for name, m in models.items():
        joblib.dump(m, os.path.join(OUT_DIR, f"{name}.pkl"))
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/")
    print("Ship this folder + apply_coglab_expert.py to anywhere HCI-only")
    print("input needs the proxy (BEHACOM, live AAM single-task sessions).")


if __name__ == "__main__":
    main()
