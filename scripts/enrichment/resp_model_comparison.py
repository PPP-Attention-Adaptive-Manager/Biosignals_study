"""
resp_model_comparison.py
=========================
Trains RF, XGBoost, CatBoost, and ARIMAX on Cog Lab V2 features
(13 SWELL-style counts + HR + RMSSD + SCL) targeting resp_bpm
direction and magnitude. Picks the best model, then produces a
ready-to-use enriched SWELL-KW dataset.

Run from: ~/biosignals_data/
Output:   ~/biosignals_data/outputs/resp_model_results.csv
          ~/biosignals_data/outputs/swell_kw_enriched.csv
          ~/biosignals_data/models/  (saved best model)
"""
from __future__ import annotations
import sys, os as _os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "features"))
import os, warnings, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────
CACHE_DIR   = os.path.expanduser("~/biosignals_data/data/cache/proxy_cache_swellstyle")
COG_LAB_DIR = os.path.expanduser("~/biosignals_data/data/cog_lab")
SWELL_FILE  = os.path.expanduser(
    "~/biosignals_data/data/swell_kw/Behavioral-features - per minute.xlsx")
SAVE_DIR    = os.path.expanduser("~/biosignals_data/models")
OUT_CSV     = os.path.expanduser("~/biosignals_data/outputs/resp_model_results.csv")
SWELL_OUT   = os.path.expanduser("~/biosignals_data/outputs/swell_kw_enriched.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

EXCLUDE    = {"S2", "S17"}
ALWAYS_NAN = {"SnRightClicked","SnDoubleClicked","SnDragged"}
REMAINING_TARGETS = [
    "acc_movement","acc_jerk","eda_tonic_slope","eda_phasic_count","resp_bpm",
    "eeg_theta_alpha","eeg_engagement","eeg_alpha_asym",
    "fnirs_hbo_slope_L","fnirs_hbo_slope_R"
]
RESP_IDX = REMAINING_TARGETS.index("resp_bpm")

HCI_COLS_SWELL = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio"
]

# ── 1. LOAD COG LAB CACHE ─────────────────────────────────────────
print("Loading Cog Lab cache...")
subjects   = sorted(d for d in os.listdir(COG_LAB_DIR)
                    if d.startswith("S") and d[1:].isdigit() and d not in EXCLUDE)
z0         = np.load(os.path.join(CACHE_DIR,f"{subjects[0]}.npz"), allow_pickle=True)
swell_names= list(z0["swell_names"])
keep_idx   = [i for i,n in enumerate(swell_names) if n not in ALWAYS_NAN]
kept_names = [swell_names[i] for i in keep_idx]

data = {}   # sid → (X_v2, starts, resp_raw)
for sid in subjects:
    z    = np.load(os.path.join(CACHE_DIR,f"{sid}.npz"), allow_pickle=True)
    order= np.argsort(z["starts"])
    Xc   = z["X_counts"][order][:,keep_idx].astype(float)  # (N,13) counts
    Xb   = z["X_input_biosig"][order].astype(float)         # (N,3)  HR/RMSSD/SCL
    Y    = z["Y_remaining"][order].astype(float)
    X_v2 = np.concatenate([Xc,Xb], axis=1)                 # (N,16) V2
    resp = Y[:,RESP_IDX]
    data[sid] = (X_v2, z["starts"][order], resp)

n_total = sum(len(v[2]) for v in data.values())
n_resp  = sum(np.isfinite(v[2]).sum() for v in data.values())
print(f"  {len(data)} subjects | {n_total} windows | "
      f"{n_resp} with valid resp_bpm ({100*n_resp/n_total:.1f}%)\n")

# ── 2. LABEL FUNCTIONS ────────────────────────────────────────────
def direction_labels(resp):
    delta = np.diff(resp, prepend=np.nan)
    lab   = np.where(np.isfinite(delta), (delta > 0).astype(float), np.nan)
    return lab

def magnitude_labels(resp):
    delta = np.diff(resp, prepend=np.nan)
    ok    = np.isfinite(delta)
    lab   = np.full(len(resp), np.nan)
    if ok.sum() < 5: return lab
    sd        = np.nanstd(delta[ok])
    lab[ok]   = 1
    lab[ok & (delta >  0.5*sd)] = 2
    lab[ok & (delta < -0.5*sd)] = 0
    return lab

# ── 3. MODEL FACTORY ─────────────────────────────────────────────
def get_model(name):
    if name == "RF":
        return RandomForestClassifier(
            300, min_samples_leaf=5, max_depth=10,
            class_weight="balanced", random_state=0, n_jobs=-1)
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=0,
            n_jobs=-1, verbosity=0)
    elif name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            random_seed=0, verbose=0)

# ── 4. STATIC LOSO (RF, XGBoost, CatBoost) ───────────────────────
def loso_static(data, subs, label_fn, model_name):
    accs, f1s = [], []
    for held in subs:
        others = [s for s in subs if s != held]
        Xtr   = np.vstack([data[s][0] for s in others])
        Rtr   = np.concatenate([data[s][2] for s in others])
        Xte, _, Rte = data[held]

        ytr = label_fn(Rtr)
        yte = label_fn(Rte)

        tr_ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
        te_ok = np.isfinite(Xte).all(1) & np.isfinite(yte)
        if tr_ok.sum() < 20 or te_ok.sum() < 5:
            continue

        mu = Xtr[tr_ok].mean(0); sd = Xtr[tr_ok].std(0) + 1e-9
        m  = get_model(model_name)
        m.fit((Xtr[tr_ok]-mu)/sd, ytr[tr_ok].astype(int))
        pred = m.predict((Xte[te_ok]-mu)/sd)
        yt   = yte[te_ok].astype(int)
        accs.append(accuracy_score(yt, pred))
        f1s.append(f1_score(yt, pred, average="macro", zero_division=0))
    return np.array(accs), np.array(f1s)

results = {}

# check which optional packages are installed
available_models = ["RF"]
try:
    import xgboost; available_models.append("XGBoost")
    print("XGBoost available")
except ImportError:
    print("XGBoost not installed — skipping (pip install xgboost)")
try:
    import catboost; available_models.append("CatBoost")
    print("CatBoost available")
except ImportError:
    print("CatBoost not installed — skipping (pip install catboost)")

print()
print("Running LOSO for static models (direction + magnitude)...")
for mname in available_models:
    print(f"  {mname}...", end=" ", flush=True)
    d_acc, d_f1 = loso_static(data, subjects, direction_labels, mname)
    m_acc, m_f1 = loso_static(data, subjects, magnitude_labels, mname)
    results[mname] = {
        "dir_acc": d_acc.mean(), "dir_f1": d_f1.mean(),
        "mag_acc": m_acc.mean(), "mag_f1": m_f1.mean(),
        "note": "LOSO N=16"
    }
    print(f"dir={d_acc.mean():.3f}  mag={m_acc.mean():.3f}")

# ── 5. ARIMAX (time-series, per-user temporal split) ─────────────
print("\n  ARIMAX (per-user 80/20 temporal split)...", flush=True)
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    arimax_dir_accs, arimax_mag_accs = [], []

    for sid in subjects:
        X_v2, starts, resp = data[sid]
        ok = np.isfinite(resp) & np.isfinite(X_v2).all(1)
        if ok.sum() < 80:
            continue

        resp_ok = resp[ok]
        X_ok    = X_v2[ok]
        mu = X_ok.mean(0); sd = X_ok.std(0) + 1e-9
        X_z = (X_ok - mu) / sd

        split = int(len(resp_ok) * 0.8)
        if split < 40 or (len(resp_ok) - split) < 15:
            continue

        y_tr, X_tr = resp_ok[:split],  X_z[:split]
        y_te, X_te = resp_ok[split:],  X_z[split:]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit  = SARIMAX(y_tr, exog=X_tr, order=(2,0,1)).fit(disp=False)
                pred = fit.forecast(steps=len(y_te), exog=X_te)

            delta_te   = np.diff(y_te,   prepend=y_te[0])
            delta_pred = np.diff(pred,   prepend=pred[0])
            dir_true   = (delta_te   > 0).astype(int)
            dir_pred   = (delta_pred > 0).astype(int)
            arimax_dir_accs.append(accuracy_score(dir_true, dir_pred))

            sd_d = np.std(delta_te) + 1e-9
            mt   = np.where(delta_te   >  0.5*sd_d, 2, np.where(delta_te   < -0.5*sd_d, 0, 1))
            mp   = np.where(delta_pred >  0.5*sd_d, 2, np.where(delta_pred < -0.5*sd_d, 0, 1))
            arimax_mag_accs.append(accuracy_score(mt, mp))
        except Exception as e:
            print(f"    {sid}: ARIMAX error: {e}")

    results["ARIMAX"] = {
        "dir_acc": np.mean(arimax_dir_accs) if arimax_dir_accs else np.nan,
        "dir_f1":  np.nan,
        "mag_acc": np.mean(arimax_mag_accs) if arimax_mag_accs else np.nan,
        "mag_f1":  np.nan,
        "note":    "per-user 80/20 temporal — NOT comparable to LOSO"
    }
    print(f"    dir={results['ARIMAX']['dir_acc']:.3f}  "
          f"mag={results['ARIMAX']['mag_acc']:.3f}")
except ImportError:
    print("    statsmodels not installed (pip install statsmodels)")

# ── 6. RESULTS TABLE ─────────────────────────────────────────────
print()
print("="*80)
print("RESP_BPM MODEL COMPARISON")
print("="*80)
print(f"{'Model':10s} {'Dir acc':>9s} {'Dir F1':>8s} {'Mag acc':>9s} {'Mag F1':>8s}  Note")
print("-"*80)
# reference from gate test
print(f"{'[RF V2 gate]':10s} {'0.802':>9s} {'—':>8s} {'0.602':>9s} {'—':>8s}  "
      f"reference from gate_dynamics (already validated)")
print("-"*80)
for mname, r in results.items():
    f1d = f"{r['dir_f1']:.3f}" if not np.isnan(r.get('dir_f1',np.nan)) else "—"
    f1m = f"{r['mag_f1']:.3f}" if not np.isnan(r.get('mag_f1',np.nan)) else "—"
    print(f"{mname:10s} {r['dir_acc']:9.3f} {f1d:>8s} {r['mag_acc']:9.3f} "
          f"{f1m:>8s}  {r['note']}")
print(f"{'chance':10s} {'0.500':>9s} {'—':>8s} {'0.333':>9s} {'—':>8s}")

# save results table
df_res = pd.DataFrame(results).T.reset_index().rename(columns={"index":"model"})
df_res.to_csv(OUT_CSV, index=False)
print(f"\nResults saved to: {OUT_CSV}")

# ── 7. PICK BEST MODEL + RETRAIN ON ALL COG LAB DATA ─────────────
static_models = {k:v for k,v in results.items() if k != "ARIMAX"}
if static_models:
    best_name = max(static_models, key=lambda k: static_models[k]["dir_acc"])
    print(f"\nBest static model: {best_name} "
          f"(dir={static_models[best_name]['dir_acc']:.3f})")

    # pool all Cog Lab subjects
    X_all = np.vstack([data[s][0] for s in subjects])
    R_all = np.concatenate([data[s][2] for s in subjects])
    y_dir = direction_labels(R_all)
    y_mag = magnitude_labels(R_all)

    ok_d = np.isfinite(X_all).all(1) & np.isfinite(y_dir)
    ok_m = np.isfinite(X_all).all(1) & np.isfinite(y_mag)

    mu_final = X_all[ok_d].mean(0)
    sd_final = X_all[ok_d].std(0) + 1e-9

    m_dir = get_model(best_name)
    m_dir.fit((X_all[ok_d]-mu_final)/sd_final, y_dir[ok_d].astype(int))
    m_mag = get_model(best_name)
    m_mag.fit((X_all[ok_m]-mu_final)/sd_final, y_mag[ok_m].astype(int))

    joblib.dump(m_dir, os.path.join(SAVE_DIR, "resp_dir_model.pkl"))
    joblib.dump(m_mag, os.path.join(SAVE_DIR, "resp_mag_model.pkl"))
    np.save(os.path.join(SAVE_DIR, "resp_mu.npy"), mu_final)
    np.save(os.path.join(SAVE_DIR, "resp_sd.npy"), sd_final)
    print(f"Models saved to: {SAVE_DIR}/")

    # ── 8. FILL SWELL-KW WITH resp_rising + resp_magnitude ───────
    print("\nBuilding enriched SWELL-KW dataset...")
    df_sw = pd.read_excel(SWELL_FILE)
    df_sw = df_sw[~df_sw["Condition"].isin({"R"})].copy()
    for c in ["HR","RMSSD","SCL"]:
        df_sw[c] = df_sw[c].replace(999, np.nan)

    # V2 input for SWELL-KW: same 13 counts (drop always-NaN) + HR/RMSSD/SCL
    swell_keep = [n for n in swell_names if n not in ALWAYS_NAN]
    Xc_sw = np.nan_to_num(df_sw[swell_keep].to_numpy(float))
    Xb_sw = df_sw[["HR","RMSSD","SCL"]].to_numpy(float)
    X_sw_v2 = np.concatenate([Xc_sw, Xb_sw], axis=1)   # (N_sw, 16)

    # mark rows with missing heart/skin inputs as unreliable
    has_biosig = np.isfinite(Xb_sw).all(1)
    X_sw_z = (X_sw_v2 - mu_final) / sd_final

    # explicit array assignment — avoids np.where/nan shape issue
    # with CatBoost output + newer numpy/pandas versions
    resp_dir_arr = np.full(len(df_sw), np.nan)
    resp_dir_arr[has_biosig] = m_dir.predict(
        X_sw_z[has_biosig]).ravel().astype(float)
    df_sw["resp_rising"] = resp_dir_arr

    resp_mag_arr = np.full(len(df_sw), np.nan)
    resp_mag_arr[has_biosig] = m_mag.predict(
        X_sw_z[has_biosig]).ravel().astype(float)
    df_sw["resp_magnitude"] = resp_mag_arr

    df_sw["resp_imputed"] = True

    print(f"  resp_rising   filled: {df_sw['resp_rising'].notna().sum()} / {len(df_sw)}")
    print(f"  resp_magnitude filled: {df_sw['resp_magnitude'].notna().sum()} / {len(df_sw)}")

    # also add native direction labels for HR/RMSSD/SCL
    df_sw = df_sw.sort_values(["PP","Condition"]).reset_index(drop=True)
    for col in ["HR","RMSSD","SCL"]:
        df_sw[col+"_delta"]   = df_sw.groupby(["PP","Condition"])[col].diff()
        df_sw[col+"_rising"]  = (df_sw[col+"_delta"] > 0).astype(float)
        # per-subject magnitude
        df_sw["_sd"] = df_sw.groupby("PP")[col+"_delta"].transform("std")
        df_sw[col+"_magnitude"] = 1
        df_sw.loc[df_sw[col+"_delta"] >  0.5*df_sw["_sd"], col+"_magnitude"] = 2
        df_sw.loc[df_sw[col+"_delta"] < -0.5*df_sw["_sd"], col+"_magnitude"] = 0
        df_sw.drop(columns=["_sd"], inplace=True)
    df_sw.drop(columns=["HR_delta","RMSSD_delta","SCL_delta"], inplace=True)

    df_sw.to_csv(SWELL_OUT, index=False)
    print(f"\nSaved enriched SWELL-KW to: {SWELL_OUT}")
    print(f"Shape: {df_sw.shape}")
    print(f"\nNew columns added:")
    new_cols = [c for c in df_sw.columns if any(
        k in c for k in ["rising","magnitude","imputed"])]
    for c in new_cols:
        print(f"  {c}")

    # ── 9. SCHEMA OF THE ENRICHED DATASET ────────────────────────
    print("\n" + "="*60)
    print("ENRICHED SWELL-KW SCHEMA")
    print("="*60)
    print("NATIVE (from original dataset, ground truth):")
    print("  PP, Condition, [18 HCI count cols], HR, RMSSD, SCL")
    print()
    print("ADDED — direction labels (binary 0/1):")
    print("  HR_rising, RMSSD_rising, SCL_rising")
    print("  resp_rising  ← IMPUTED from Cog Lab model [flag: resp_imputed=True]")
    print()
    print("ADDED — magnitude labels (3-class: 0=fall, 1=flat, 2=rise):")
    print("  HR_magnitude, RMSSD_magnitude, SCL_magnitude")
    print("  resp_magnitude  ← IMPUTED from Cog Lab model")
    print()
    print("USE THIS FILE FOR:")
    print("  - CCA with 4 physiological targets (HR,RMSSD,SCL,resp)")
    print("  - Direction/magnitude classifiers on all 4 targets")
    print("  - Training the 4-target model for BEHACOM transfer")
    print()
    print("CAVEAT: resp columns are soft — trained on N=16 Cog Lab subjects,")
    print("  single passive task, transferred. Use only where conf is high")
    print("  (i.e. rows where HR/RMSSD/SCL were all non-NaN).")
else:
    print("\nNo static models ran successfully.")
    print("Install packages: pip install xgboost catboost statsmodels")

print("\nDone.")
