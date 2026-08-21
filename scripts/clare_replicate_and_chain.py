"""
clare_replicate_and_chain.py
===============================
TWO STAGES.

STAGE 1 -- REPLICATE THE PAPER'S OWN BENCHMARK
-------------------------------------------------
The paper (Bhatti et al.) never tested modality-predicting-modality.
Their ONLY task, every time: binary "high" (score>=5) vs "low" (score<5)
cognitive load classification, from 10-second-segment handcrafted
features, evaluated LOSO. Every modality subset gets the same task.

Their LOSO reference numbers (accuracy/F1, best model = Transformer,
not available to us -- we use RF/XGBoost/CatBoost as the closest
classical-ML comparison, matching their RF/XGBoost/LGBM columns):
    ECG              66.15 / 54.20
    EDA              63.73 / 54.43
    EEG              67.77 / 57.03   <- their best single modality
    ECG, EDA         68.99 / 62.32
    ECG, EEG         68.74 / 60.72
    EDA, EEG         70.45 / 65.00
    ECG, EDA, EEG    70.90 / 66.84   <- closest to our full non-Gaze set

We don't have Gaze processed, so our ceiling comparison is their
ECG+EDA+EEG row, not their true "all modalities" row (72.70/69.46).

Replicating this first validates that our ECG/EDA/EEG feature
extraction, at their exact 10-second window matching their label
frequency, produces numbers in a comparable range -- if we're wildly
off from their published numbers, that's a pipeline problem to find
BEFORE trusting anything built on top of it.

STAGE 2 -- THE PROXY-RELEVANT QUESTION THEY NEVER ASKED
-------------------------------------------------------------
  2a. Does EEG add real value to the ACTUAL task (cognitive load
      classification) beyond ECG+EDA alone? Replicates their own
      ECG,EDA (68.99/62.32) vs ECG,EDA,EEG (70.90/66.84) ablation --
      if EEG helps their real task even a little, that means EEG
      carries genuine task-relevant information, even if that
      information is NOT recoverable FROM ECG/EDA (a different question,
      answered separately in 2b).

  2b. Does the FULLER ECG+EDA feature set (real HRV + EDA features,
      not just HR+RMSSD) predict EEG any better than our earlier
      2-scalar test did? Same design as clare_expanded_test.py but
      with much richer predictors -- if predictor richness was the
      limiting factor before, this is where it would show up.

  2c. If 2b shows real signal, chain it into the cognitive-load
      classifier and check for improvement over ECG+EDA alone. Not run
      automatically -- only triggers if 2b finds something.

EDA CLEANING
--------------
Never processed before in this project. Uses the same interleaved-
NaN-row cleaning pattern that worked for ECG (Shimmer export artifact,
not missing data).

Run from: ~/biosignals_data/
Output:   outputs/clare_replicate_results.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch, iirnotch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
CLARE_ROOT = os.path.join(BASE, "data", "clare", "doi-10.5683-sp3-h0aelt")
ECG_DIR = os.path.join(CLARE_ROOT, "ECG")
EDA_DIR = os.path.join(CLARE_ROOT, "EDA")
EEG_DIR = os.path.join(CLARE_ROOT, "EEG")
LABEL_DIR = os.path.join(CLARE_ROOT, "Labels")
OUT_JSON = os.path.join(BASE, "outputs", "clare_replicate_results.json")

ECG_LEADS = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]
PRIMARY_LEAD = "ECG LL-RA CAL"
EDA_COL = "GSR Conductance CAL"
ECG_SF, EDA_SF, EEG_SF = 512.0, 128.0, 256.0

FRONTAL  = ["AF7", "AF8"]
TEMPORAL = ["TP9", "TP10"]

WINDOW_S = 10.0     # matches the paper's label frequency exactly
MIN_FOLDS = 8
STAR_BAR = 0.03

PAPER_REFERENCE = {
    "ECG":            {"acc": 66.15, "f1": 54.20},
    "EDA":            {"acc": 63.73, "f1": 54.43},
    "EEG":            {"acc": 67.77, "f1": 57.03},
    "ECG,EDA":        {"acc": 68.99, "f1": 62.32},
    "ECG,EEG":        {"acc": 68.74, "f1": 60.72},
    "EDA,EEG":        {"acc": 70.45, "f1": 65.00},
    "ECG,EDA,EEG":    {"acc": 70.90, "f1": 66.84},
}


def clean_stream(path, value_cols, primary_col):
    raw = pd.read_csv(path)
    df = raw.dropna(subset=value_cols, how="all").copy()
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp", keep="first")
    df = df.dropna(subset=[primary_col]).reset_index(drop=True)
    return df


def ecg_features_10s(ts, sig, w0, w1, sf=ECG_SF):
    mask = (ts >= w0) & (ts < w1)
    if mask.sum() < sf * 3:
        return None
    seg = sig[mask]
    b, a = butter(3, [5/(sf/2), 15/(sf/2)], btype="band")
    z = filtfilt(b, a, seg)
    z = (z - z.mean()) / (z.std() + 1e-9)
    peaks = np.array([])
    for h in [2.5, 2.0, 1.5, 1.0]:
        peaks, _ = find_peaks(z, distance=int(0.35*sf), height=h)
        if len(peaks) > 3:
            break
    if len(peaks) < 3:
        return {"ecg_hr_mean": np.nan, "ecg_rmssd": np.nan,
               "ecg_sdrr": np.nan, "ecg_mean_rr": np.nan}
    seg_ts = ts[mask]
    rr = np.diff(seg_ts[peaks])
    rr = rr[(rr > 0.33) & (rr < 1.5)]
    if len(rr) < 2:
        return {"ecg_hr_mean": np.nan, "ecg_rmssd": np.nan,
               "ecg_sdrr": np.nan, "ecg_mean_rr": np.nan}
    return {
        "ecg_hr_mean":  float(60.0/np.median(rr)),
        "ecg_mean_rr":  float(np.mean(rr)),
        "ecg_sdrr":     float(np.std(rr)) if len(rr) >= 2 else np.nan,
        "ecg_rmssd":    float(np.sqrt(np.mean(np.diff(rr)**2))*1000) if len(rr)>=3 else np.nan,
    }


def eda_features_10s(ts, sig, w0, w1, sf=EDA_SF):
    mask = (ts >= w0) & (ts < w1)
    if mask.sum() < sf * 3:
        return None
    seg = sig[mask]
    if not np.all(np.isfinite(seg)) or len(seg) < 5:
        return {"eda_tonic_mean": np.nan, "eda_tonic_slope": np.nan,
               "eda_scr_count": np.nan, "eda_std": np.nan}
    tonic_mean = float(np.mean(seg))
    tonic_slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])
    try:
        b, a = butter(2, 0.05/(sf/2), btype="high")
        phasic = filtfilt(b, a, seg)
        pk, _ = find_peaks(phasic, height=np.std(phasic)*0.5,
                           distance=int(1*sf))
        scr_count = len(pk)
    except Exception:
        scr_count = np.nan
    return {"eda_tonic_mean": tonic_mean, "eda_tonic_slope": tonic_slope,
           "eda_scr_count": float(scr_count), "eda_std": float(np.std(seg))}


def eeg_features_10s(ts, data, chans, w0, w1, sf=EEG_SF):
    mask = (ts >= w0) & (ts < w1)
    if mask.sum() < sf * 3:
        return None
    seg = data[:, mask]
    b, a = butter(3, [1/(sf/2), min(40,sf/2-1)/(sf/2)], btype="band")
    seg_f = filtfilt(b, a, seg, axis=1)
    if sf/2 > 60:
        b_n, a_n = iirnotch(60.0, 30.0, sf)
        seg_f = filtfilt(b_n, a_n, seg_f, axis=1)

    nper = min(seg_f.shape[1], int(sf * 2))
    if nper < sf:
        return None
    f, psd = welch(seg_f, fs=sf, nperseg=nper, axis=1)
    total = psd[:, (f>=4)&(f<40)].mean(axis=1) + 1e-15
    theta = psd[:, (f>=4)&(f<8)].mean(axis=1)  / total
    alpha = psd[:, (f>=8)&(f<12)].mean(axis=1) / total
    beta  = psd[:, (f>=12)&(f<31)].mean(axis=1)/ total

    def safelog(x):
        return float(np.log10(x)) if x > 0 else np.nan

    out = {}
    for site, names in [("frontal", FRONTAL), ("temporal", TEMPORAL)]:
        idx = [chans.index(c) for c in names if c in chans]
        if not idx: continue
        t_, a_, b_ = theta[idx].mean(), alpha[idx].mean(), beta[idx].mean()
        out[f"log_{site}_theta"] = safelog(t_)
        out[f"log_{site}_alpha"] = safelog(a_)
        out[f"log_{site}_beta"]  = safelog(b_)
        out[f"{site}_theta_alpha_ratio"] = float(t_/(a_+1e-15))
        out[f"{site}_engagement_index"]  = float(b_/(a_+t_+1e-15))
    return out


def process_participant(pid):
    label_path = os.path.join(LABEL_DIR, f"{pid}.csv")
    if not os.path.isfile(label_path):
        return []
    labels = pd.read_csv(label_path)

    rows = []
    for level in range(4):
        lcol = f"level_{level}"
        if lcol not in labels.columns:
            continue

        ecg_path = os.path.join(ECG_DIR, pid, f"ecg_data_experiment_{level}.csv")
        eda_path = os.path.join(EDA_DIR, pid, f"eda_data_experiment_{level}.csv")
        eeg_path = os.path.join(EEG_DIR, pid, f"eeg_data_exp_{level}.csv")
        if not all(os.path.isfile(p) for p in (ecg_path, eda_path, eeg_path)):
            continue

        try:
            ecg_df = clean_stream(ecg_path, ECG_LEADS, PRIMARY_LEAD)
            ecg_ts = ecg_df["Timestamp"].to_numpy()
            ecg_sig = ecg_df[PRIMARY_LEAD].to_numpy(float)

            eda_df = clean_stream(eda_path, [EDA_COL], EDA_COL)
            eda_ts = eda_df["Timestamp"].to_numpy()
            eda_sig = eda_df[EDA_COL].to_numpy(float)

            eeg_df = pd.read_csv(eeg_path).dropna()
            chans = [c for c in FRONTAL + TEMPORAL if c in eeg_df.columns]
            if not all(c in chans for c in FRONTAL):
                continue
            eeg_ts = eeg_df["Timestamp"].to_numpy()
            eeg_data = eeg_df[chans].to_numpy(float).T
        except Exception:
            continue

        ratings = labels[lcol].to_numpy(float)
        for seg_idx in range(len(ratings)):
            if not np.isfinite(ratings[seg_idx]):
                continue
            w0, w1 = seg_idx * WINDOW_S, (seg_idx + 1) * WINDOW_S

            ecg_f = ecg_features_10s(ecg_ts, ecg_sig, w0, w1)
            eda_f = eda_features_10s(eda_ts, eda_sig, w0, w1)
            eeg_f = eeg_features_10s(eeg_ts, eeg_data, chans, w0, w1)
            if ecg_f is None or eda_f is None or eeg_f is None:
                continue

            row = {"participant": pid, "level": level, "seg_idx": seg_idx,
                  "rating": ratings[seg_idx],
                  "label_high": float(ratings[seg_idx] >= 5)}
            row.update(ecg_f); row.update(eda_f); row.update(eeg_f)
            rows.append(row)
    return rows


def build_models():
    m = {}
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = lambda seed: CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=seed,
            verbose=0, allow_writing_files=False)
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        # scale_pos_weight computed per-fold inside loso_classify/loso_direction
        # (needs the training fold's actual class ratio, not a fixed constant --
        # RF and CatBoost balance automatically via class_weight/auto_class_weights,
        # XGBoost has neither by default, which let it silently exploit label
        # imbalance and win every "best model" selection in Stage 1/2a)
        m["XGBoost"] = lambda seed, spw=1.0: XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0,
            scale_pos_weight=spw)
    except ImportError:
        pass
    m["RF"] = lambda seed: RandomForestClassifier(
        150, min_samples_leaf=3, class_weight="balanced",
        random_state=seed, n_jobs=-1)
    return m


def zscore_within(X, g):
    Xz = np.zeros_like(X, dtype=float)
    for u in np.unique(g):
        m = g == u
        Xz[m] = (X[m]-X[m].mean(0))/(X[m].std(0)+1e-9)
    return Xz


def loso_classify(X, y, groups, model_fn, seed=0):
    """
    Now also returns the empirical majority-class chance baseline per
    fold (same convention as loso_direction and every other LOSO
    function in this project) -- missing here originally, which let
    Stage 1/2a report inflated accuracy/F1 without any check against
    the label's own class imbalance. Also passes a per-fold
    scale_pos_weight to model_fn so XGBoost balances classes the same
    way RF/CatBoost already do.
    """
    accs, f1s, bases = [], [], []
    for held in np.unique(groups):
        tr, te = groups!=held, groups==held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum()<15 or ote.sum()<3: continue
        ytr, yte = y[tr][otr].astype(int), y[te][ote].astype(int)
        if len(np.unique(ytr))<2: continue
        try:
            n_pos = (ytr==1).sum(); n_neg = (ytr==0).sum()
            spw = float(n_neg/max(n_pos,1))
            try:
                m = model_fn(seed, spw)      # XGBoost: accepts spw
            except TypeError:
                m = model_fn(seed)           # RF/CatBoost: balance internally
            m.fit(X[tr][otr], ytr)
            pred = m.predict(X[te][ote])
            accs.append(accuracy_score(yte, pred))
            f1s.append(f1_score(yte, pred, zero_division=0))
            counts = np.bincount(yte, minlength=2)
            bases.append(counts.max()/counts.sum())
        except Exception:
            continue
    if len(accs) < MIN_FOLDS:
        return np.nan, np.nan, np.nan, len(accs)
    return (float(np.mean(accs))*100, float(np.mean(f1s))*100,
           float(np.mean(bases))*100, len(accs))


def direction_labels(series, groups):
    out = np.full(len(series), np.nan)
    s, g = series.to_numpy(float), groups
    for u in np.unique(g):
        idx = np.where(g==u)[0]
        for k in range(1, len(idx)):
            i, j = idx[k-1], idx[k]
            if np.isfinite(s[i]) and np.isfinite(s[j]):
                out[j] = float(s[j] > s[i])
    return out


def loso_direction(X, y, groups, model_fn, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    accs, bases = [], []
    for held in np.unique(groups):
        tr, te = groups!=held, groups==held
        otr = np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])
        ote = np.isfinite(X[te]).all(1) & np.isfinite(y[te])
        if otr.sum()<15 or ote.sum()<3: continue
        ytr, yte = y[tr][otr].astype(int), y[te][ote].astype(int)
        if len(np.unique(ytr))<2: continue
        if shuffle: ytr = rng.permutation(ytr)
        try:
            m = model_fn(seed)
            m.fit(X[tr][otr], ytr)
            accs.append(accuracy_score(yte, m.predict(X[te][ote])))
            bases.append(max(yte.mean(), 1-yte.mean()))
        except Exception:
            continue
    if len(accs) < MIN_FOLDS:
        return np.nan, np.nan, len(accs)
    return float(np.mean(accs)), float(np.mean(bases)), len(accs)


ECG_FEATS = ["ecg_hr_mean","ecg_mean_rr","ecg_sdrr","ecg_rmssd"]
EDA_FEATS = ["eda_tonic_mean","eda_tonic_slope","eda_scr_count","eda_std"]
EEG_FEATS = ["log_frontal_theta","log_frontal_alpha","log_frontal_beta",
            "frontal_theta_alpha_ratio","frontal_engagement_index"]


def stage1_replicate(df):
    print("\n" + "=" * 88)
    print("STAGE 1 -- REPLICATING THE PAPER'S OWN BENCHMARK")
    print("Binary high(>=5)/low(<5) cognitive load, LOSO, 10s segments")
    print("=" * 88)

    groups = df["participant"].to_numpy()
    y = df["label_high"].to_numpy(float)
    models = build_models()

    modality_sets = {
        "ECG": ECG_FEATS, "EDA": EDA_FEATS, "EEG": EEG_FEATS,
        "ECG,EDA": ECG_FEATS+EDA_FEATS,
        "ECG,EEG": ECG_FEATS+EEG_FEATS,
        "EDA,EEG": EDA_FEATS+EEG_FEATS,
        "ECG,EDA,EEG": ECG_FEATS+EDA_FEATS+EEG_FEATS,
    }

    print(f"\n{'Modality':14s} {'model':10s} {'chance':>8s} {'acc':>8s} "
          f"{'F1':>8s} {'over':>8s}   {'paper acc':>10s} {'paper F1':>9s}")
    print("-" * 90)

    results = {}
    for name, cols in modality_sets.items():
        avail = [c for c in cols if c in df.columns]
        X = zscore_within(df[avail].to_numpy(float), groups)
        best = None
        for mname, mfn in models.items():
            acc, f1, chance, nf = loso_classify(X, y, groups, mfn)
            if not np.isfinite(acc): continue
            over = acc - chance   # select by margin over chance, NOT raw
            # accuracy -- raw-accuracy selection is exactly what let an
            # unbalanced XGBoost win every slot by exploiting label skew
            if best is None or over > best[4]:
                best = (mname, acc, f1, chance, over, nf)
        if best is None:
            print(f"  {name:12s}  insufficient data"); continue
        mname, acc, f1, chance, over, nf = best
        ref = PAPER_REFERENCE.get(name, {})
        flag = "" if over > 3 else "  <- near/at chance"
        print(f"  {name:12s} {mname:10s} {chance:8.2f} {acc:8.2f} {f1:8.2f} "
              f"{over:+8.2f}   {ref.get('acc',0):10.2f} {ref.get('f1',0):9.2f}{flag}")
        results[name] = {"model": mname, "acc": acc, "f1": f1, "chance": chance,
                         "over_chance": over, "n_folds": nf,
                         "paper_acc": ref.get("acc"), "paper_f1": ref.get("f1")}

    print("\nWe lack Gaze -- our ceiling is ECG,EDA,EEG (paper: 70.90/66.84),")
    print("not the paper's true all-modality row (72.70/69.46).")
    return results


def stage2a_eeg_value(df):
    print("\n" + "=" * 88)
    print("STAGE 2a -- does EEG add value to the REAL task beyond ECG+EDA?")
    print("Replicates paper's own ablation: ECG,EDA (68.99/62.32) vs")
    print("ECG,EDA,EEG (70.90/66.84)")
    print("=" * 88)
    groups = df["participant"].to_numpy()
    y = df["label_high"].to_numpy(float)
    models = build_models()

    for name, cols in [("ECG,EDA", ECG_FEATS+EDA_FEATS),
                       ("ECG,EDA,EEG", ECG_FEATS+EDA_FEATS+EEG_FEATS)]:
        avail = [c for c in cols if c in df.columns]
        X = zscore_within(df[avail].to_numpy(float), groups)
        best = None
        for mname, mfn in models.items():
            acc, f1, chance, nf = loso_classify(X, y, groups, mfn)
            if not np.isfinite(acc): continue
            over = acc - chance
            if best is None or over > best[4]:
                best = (mname, acc, f1, chance, over)
        if best:
            print(f"  {name:14s} {best[0]:10s} chance={best[3]:.2f}  "
                  f"acc={best[1]:.2f}  F1={best[2]:.2f}  over={best[4]:+.2f}")


def stage2b_richer_predictors(df):
    print("\n" + "=" * 88)
    print("STAGE 2b -- FULLER ECG+EDA feature set -> EEG (direction, permutation-checked)")
    print("Was 2-scalar (HR,RMSSD) predictor richness the limiting factor before?")
    print("=" * 88)
    groups = df["participant"].to_numpy()
    models = build_models()
    pred_cols = [c for c in ECG_FEATS+EDA_FEATS if c in df.columns]
    X = zscore_within(df[pred_cols].to_numpy(float), groups)

    print(f"\nPredictors ({len(pred_cols)}): {pred_cols}")
    print(f"\n{'EEG feature':28s} {'model':10s} {'chance':>8s} {'acc':>8s} "
          f"{'perm':>8s} {'over':>8s}")
    print("-" * 76)

    any_real = False
    for feat in EEG_FEATS:
        if feat not in df.columns: continue
        y = direction_labels(df[feat], groups)
        best = None
        for mname, mfn in models.items():
            acc, base, nf = loso_direction(X, y, groups, mfn)
            if not np.isfinite(acc) or nf < MIN_FOLDS: continue
            over = acc - base
            if best is None or over > best[3]:
                best = (mname, acc, base, over, nf)
        if best is None:
            print(f"  {feat:26s}  insufficient folds"); continue
        mname, acc, base, over, nf = best
        perm, _, _ = loso_direction(X, y, groups, models[mname], shuffle=True)
        perm_over = perm - base
        real_vs_perm = over - perm_over
        flag = ""
        if over > STAR_BAR and real_vs_perm > STAR_BAR:
            flag = "  *** REAL"; any_real = True
        print(f"  {feat:26s} {mname:10s} {base:8.3f} {acc:8.3f} "
              f"{perm:8.3f} {over:+8.3f}{flag}")

    return any_real


def main():
    print("=" * 88)
    print("CLARE: replicate paper benchmark, then test the proxy-relevant question")
    print("=" * 88)

    if not os.path.isdir(ECG_DIR):
        print(f"ECG dir not found: {ECG_DIR}"); return

    pids = sorted(os.listdir(ECG_DIR))
    all_rows = []
    for pid in pids:
        rows = process_participant(pid)
        print(f"  P{pid}: {len(rows)} 10s segments")
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo usable segments."); return
    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} segments, {df.participant.nunique()} participants")
    print(f"Label balance: high={df.label_high.mean():.2f}  "
          f"low={1-df.label_high.mean():.2f}")

    results = {}
    results["stage1"] = stage1_replicate(df)
    stage2a_eeg_value(df)
    any_real = stage2b_richer_predictors(df)

    print("\n" + "=" * 88)
    print("STAGE 2c -- chaining")
    print("=" * 88)
    if any_real:
        print("\nStage 2b found signal -- chaining test would be the next step")
        print("(not run automatically here; flag and build if this triggers).")
    else:
        print("\nStage 2b found no signal, consistent with every earlier CLARE")
        print("test. Nothing to chain -- richer ECG+EDA features did not unlock")
        print("EEG predictability. Skipping 2c.")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
