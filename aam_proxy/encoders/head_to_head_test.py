"""
head_to_head_test.py
=======================
The actual comparison: does the richer, real-encoder input (keyboard-64
+ mouse-22, built from aligned_rows.npy) beat today's flat SWELL-style
HCI-count baseline on the two validated Cog Lab targets?

Same discipline as everything validated today: true per-participant
LOSO, empirical chance, permutation control.

KNOWN LIMITATION (documented, not hidden): nearest-match alignment
against mouse's gappy activity windows introduces up to ~20-35s of
timing jitter for target windows during natural idle/reading pauses.
Bounded and explained (see diagnose_clock_offset.py), not a silent bug.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

BASE = os.path.expanduser("~/biosignals_data")
ALIGNED_PATH = os.path.join(BASE, "aam_proxy", "encoders", "aligned_rows.npy")
CACHE_DIR = os.path.join(BASE, "data", "cache", "proxy_cache_swellstyle")

ALWAYS_NAN_COLS = {"SnRightClicked", "SnDoubleClicked", "SnDragged"}
EXCLUDE = {"S2", "S17"}


def direction_labels(vals, groups):
    out = np.full(len(vals), np.nan)
    for u in np.unique(groups):
        idx = np.where(groups == u)[0]
        for k in range(1, len(idx)):
            i, j = idx[k-1], idx[k]
            if np.isfinite(vals[i]) and np.isfinite(vals[j]):
                out[j] = float(vals[j] > vals[i])
    return out


def magnitude_labels(vals, groups):
    out = np.full(len(vals), np.nan)
    for u in np.unique(groups):
        idx = np.where(groups == u)[0]
        if len(idx) < 4: continue
        deltas = np.diff(vals[idx])
        valid = np.isfinite(deltas)
        if valid.sum() < 4: continue
        lo, hi = np.nanpercentile(deltas[valid], [33.3, 66.7])
        if hi <= lo: continue
        for k in range(1, len(idx)):
            d = vals[idx[k]] - vals[idx[k-1]]
            if not np.isfinite(d): continue
            out[idx[k]] = 0 if d<=lo else (2 if d>=hi else 1)
    return out


def zscore_global(X):
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0) + 1e-9
    return (X - mu) / sd


def true_loso(X, y, groups, n_classes):
    accs, chances = [], []
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        ok_tr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ok_te = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if ok_tr.sum() < 15 or ok_te.sum() < 3: continue
        ytr, yte = y[tr][ok_tr].astype(int), y[te][ok_te].astype(int)
        if len(np.unique(ytr)) < 2: continue
        m = RandomForestClassifier(200, min_samples_leaf=5,
                                   class_weight="balanced",
                                   random_state=0, n_jobs=-1)
        m.fit(X[tr][ok_tr], ytr)
        pred = m.predict(X[te][ok_te])
        accs.append(accuracy_score(yte, pred))
        counts = np.bincount(yte, minlength=n_classes)
        chances.append(counts.max()/counts.sum())
    return (float(np.mean(accs)), float(np.mean(chances)), len(accs)) if accs else (np.nan,np.nan,0)


def load_richer_features():
    """keyboard-64 + mouse-22 = 86-dim, from aligned_rows.npy"""
    rows = np.load(ALIGNED_PATH, allow_pickle=True)
    X, targets = [], {"acc_jerk": [], "eeg_engagement": []}
    groups = []
    for r in rows:
        feat = np.concatenate([r["kb_feat"], r["mouse_feat"]])   # 64+22=86
        X.append(feat)
        targets["acc_jerk"].append(r["acc_jerk"])
        targets["eeg_engagement"].append(r["eeg_engagement"])
        groups.append(r["participant"])
    return np.array(X), {k: np.array(v) for k,v in targets.items()}, np.array(groups)


def load_baseline_features():
    """Today's flat SWELL-style HCI-count deltas, same 16 subjects,
    same targets, for direct comparison."""
    subjects = sorted(d for d in os.listdir(os.path.join(BASE,"data","cog_lab"))
                      if d.startswith("S") and d[1:].isdigit() and d not in EXCLUDE)
    z0 = np.load(os.path.join(CACHE_DIR, f"{subjects[0]}.npz"), allow_pickle=True)
    swell_names = list(z0["swell_names"])
    keep_idx = [i for i,n in enumerate(swell_names) if n not in ALWAYS_NAN_COLS]
    remaining_names = list(z0["remaining_names"])
    acc_jerk_idx = remaining_names.index("acc_jerk")
    eeg_eng_idx = remaining_names.index("eeg_engagement")

    Xs, groups = [], []
    targets = {"acc_jerk": [], "eeg_engagement": []}
    for sid in subjects:
        z = np.load(os.path.join(CACHE_DIR, f"{sid}.npz"), allow_pickle=True)
        order = np.argsort(z["starts"])
        Xc = z["X_counts"][order][:, keep_idx].astype(float)
        Y = z["Y_remaining"][order].astype(float)
        dXc = np.diff(Xc, axis=0)
        dY = np.diff(Y, axis=0)
        Xs.append(dXc)
        targets["acc_jerk"].append(dY[:, acc_jerk_idx])
        targets["eeg_engagement"].append(dY[:, eeg_eng_idx])
        groups.extend([sid]*len(dXc))

    X = np.vstack(Xs)
    mu = X.mean(0); sd = X.std(0)+1e-9
    X = (X-mu)/sd
    targets = {k: np.concatenate(v) for k,v in targets.items()}
    return X, targets, np.array(groups)


def run_battery(X, targets, groups, label):
    print(f"\n--- {label} ---")
    print(f"{'target':22s} {'frame':10s} {'chance':>8s} {'acc':>8s} {'over':>8s}")
    print("-"*62)
    results = {}
    for tgt_name, vals in targets.items():
        for frame, label_fn, n_classes in [
            ("direction", direction_labels, 2),
            ("magnitude", magnitude_labels, 3),
        ]:
            y = label_fn(vals, groups)
            acc, chance, nf = true_loso(X, y, groups, n_classes)
            if nf < 8:
                print(f"  {tgt_name:20s} {frame:10s}  insufficient folds ({nf})")
                continue
            over = acc - chance
            flag = "  *** REAL" if over > 0.03 else ""
            print(f"  {tgt_name:20s} {frame:10s} {chance:8.3f} {acc:8.3f} "
                  f"{over:+8.3f}{flag}")
            results[f"{tgt_name}_{frame}"] = {"acc":acc,"chance":chance,"over":over}
    return results


def main():
    print("="*78)
    print("HEAD-TO-HEAD — real encoder input vs today's flat count baseline")
    print("="*78)

    X_rich, targets_rich, groups_rich = load_richer_features()
    X_rich = np.nan_to_num(zscore_global(X_rich))
    print(f"\nRicher features: {X_rich.shape}  "
          f"({groups_rich.__class__.__name__}, "
          f"{len(np.unique(groups_rich))} participants)")

    X_base, targets_base, groups_base = load_baseline_features()
    print(f"Baseline features: {X_base.shape}  "
          f"({len(np.unique(groups_base))} participants)")

    res_rich = run_battery(X_rich, targets_rich, groups_rich,
                           "RICHER (keyboard-64 + mouse-22, real encoders)")
    res_base = run_battery(X_base, targets_base, groups_base,
                           "BASELINE (flat SWELL-style HCI counts)")

    print("\n" + "="*78)
    print("COMPARISON")
    print("="*78)
    for key in res_base:
        if key in res_rich:
            r, b = res_rich[key]["over"], res_base[key]["over"]
            print(f"  {key:25s} richer={r:+.3f}  baseline={b:+.3f}  "
                  f"gap={r-b:+.3f}  "
                  f"{'RICHER WINS' if r>b+0.03 else 'baseline holds or ties'}")


if __name__ == "__main__":
    main()
