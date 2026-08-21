"""
diagnose_cca_permutation.py
==============================
Decisive check: does the properly-grouped CCA result (0.262
GroupShuffleSplit, 0.206 true LOSO) survive a permutation test, or is
it indistinguishable from the noise floor at this sample size?

WHY THIS IS NECESSARY
------------------------
Correlation coefficients computed on small test folds have large
sampling variance. With ~10 rows per GroupShuffleSplit test fold, the
standard error on a Pearson r is large enough that moderate-looking
values (0.2-0.3) can arise from pure noise. The SD on both grouped
results (0.263, 0.771) is itself evidence of this instability. Every
other number in this session that looked like a moderate win got
checked against a permutation baseline before being trusted -- the CCA
result has not been, until now.

METHOD
--------
1. Pool all out-of-fold predictions across GroupShuffleSplit folds into
   ONE combined set, then compute ONE overall correlation -- more
   stable than averaging many small noisy per-fold correlations, and
   the standard way to report LOSO/grouped-CV correlation honestly.
2. Permutation control: shuffle the PP-to-physiology-row mapping (break
   the true pairing between a participant's HCI pattern and their own
   physiology, while preserving the group structure), rerun the exact
   same grouped CV, repeat 200 times to build a null distribution.
3. Compare the REAL pooled correlation against this null distribution
   directly -- if real sits within the range permutation regularly
   produces, the result does not survive.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, glob, warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/biosignals_data")
_matches = glob.glob(os.path.join(BASE, "data", "swell_kw", "**",
                                   "Behavioral-features - per minute.xlsx"),
                     recursive=True)
if not _matches:
    raise FileNotFoundError("SWELL-KW file not found")
SWELL_FILE = _matches[0]
print(f"Using: {SWELL_FILE}\n")

HCI_COLS = [
    "SnMouseAct","SnLeftClicked","SnRightClicked","SnDoubleClicked",
    "SnWheel","SnDragged","SnMouseDistance","SnKeyStrokes","SnChars",
    "SnSpecialKeys","SnDirectionKeys","SnErrorKeys","SnShortcutKeys",
    "SnSpaces","SnAppChange","SnTabfocusChange","CharactersRatio","ErrorKeyRatio",
]
PHYSIO_COLS_ACTIVE = ["HR", "RMSSD"]
N_PERMUTATIONS = 200


def build_agg():
    df = pd.read_excel(SWELL_FILE)
    df = df[df["Condition"].isin(["N", "I", "T"])].copy()
    for c in ["HR", "RMSSD", "SCL"]:
        df[c] = df[c].replace(999, np.nan)
    agg = df.groupby(["PP","Condition"])[HCI_COLS + PHYSIO_COLS_ACTIVE].mean().reset_index()
    for col in HCI_COLS + PHYSIO_COLS_ACTIVE:
        agg[col+"_z"] = agg.groupby("PP")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))
    clean = agg[["PP"] + [c+"_z" for c in HCI_COLS] + [c+"_z" for c in PHYSIO_COLS_ACTIVE]].dropna()
    print(f"agg table: {len(clean)} rows, {clean.PP.nunique()} participants\n")
    return clean


def pooled_grouped_cv_r(Xz, Ycca, groups, splitter, grouped=True):
    """Pools ALL out-of-fold predictions, computes ONE correlation --
    more stable than averaging many small noisy per-fold correlations."""
    all_Xt, all_Yt = [], []
    split_iter = splitter.split(Xz, groups=groups) if grouped else splitter.split(Xz)
    for tr, te in split_iter:
        try:
            c = CCA(n_components=1, max_iter=2000).fit(Xz[tr], Ycca[tr])
            Xt, Yt = c.transform(Xz[te], Ycca[te])
            all_Xt.append(Xt[:,0]); all_Yt.append(Yt[:,0])
        except Exception:
            continue
    if not all_Xt:
        return np.nan
    Xt_pool = np.concatenate(all_Xt); Yt_pool = np.concatenate(all_Yt)
    if np.std(Xt_pool) < 1e-9 or np.std(Yt_pool) < 1e-9:
        return np.nan
    return float(np.corrcoef(Xt_pool, Yt_pool)[0,1])


def main():
    print("=" * 78)
    print("CCA PERMUTATION TEST — is 0.262/0.206 real or noise-floor?")
    print("=" * 78)
    print()

    clean = build_agg()
    Xz = clean[[c+"_z" for c in HCI_COLS]].to_numpy(float)
    Ycca = clean[[c+"_z" for c in PHYSIO_COLS_ACTIVE]].to_numpy(float)
    groups = clean["PP"].to_numpy()
    pp_unique = np.unique(groups)

    print("=== REAL (pooled, properly grouped) ===")
    gss = GroupShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
    real_gss = pooled_grouped_cv_r(Xz, Ycca, groups, gss)
    print(f"  Pooled GroupShuffleSplit r: {real_gss:.3f}")

    logo = LeaveOneGroupOut()
    real_loso = pooled_grouped_cv_r(Xz, Ycca, groups, logo)
    print(f"  Pooled true-LOSO r:         {real_loso:.3f}\n")

    print(f"=== PERMUTATION NULL DISTRIBUTION ({N_PERMUTATIONS} shuffles) ===")
    print("Shuffling which participant's physiology is paired with which")
    print("participant's HCI pattern (breaking the true pairing, keeping")
    print("group structure and marginal distributions intact)...\n")

    rng = np.random.default_rng(0)
    perm_rs_gss, perm_rs_loso = [], []

    for i in range(N_PERMUTATIONS):
        # shuffle physio rows AMONG participants: assign each participant's
        # HCI rows the PHYSIO values of a randomly chosen OTHER participant
        # (preserves within-participant HCI structure & the 3-condition
        # design, breaks the true HCI<->own-physiology link)
        shuffled_pp_map = dict(zip(pp_unique, rng.permutation(pp_unique)))
        # build shuffled Y by reassigning rows: for each row, pull physio
        # from the row(s) belonging to shuffled_pp_map[this row's PP]
        Y_shuffled = np.zeros_like(Ycca)
        for pp in pp_unique:
            src_pp = shuffled_pp_map[pp]
            dst_idx = np.where(groups == pp)[0]
            src_idx = np.where(groups == src_pp)[0]
            n = min(len(dst_idx), len(src_idx))
            Y_shuffled[dst_idx[:n]] = Ycca[src_idx[:n]]

        r_gss = pooled_grouped_cv_r(Xz, Y_shuffled, groups,
                                    GroupShuffleSplit(n_splits=10, test_size=0.2,
                                                      random_state=i))
        if np.isfinite(r_gss):
            perm_rs_gss.append(r_gss)

    perm_rs_gss = np.array(perm_rs_gss)
    print(f"Permutation null (GroupShuffleSplit): "
          f"mean={perm_rs_gss.mean():+.3f}  std={perm_rs_gss.std():.3f}")
    print(f"  5th/95th percentile: [{np.percentile(perm_rs_gss,5):+.3f}, "
          f"{np.percentile(perm_rs_gss,95):+.3f}]")

    p_value = float((np.abs(perm_rs_gss) >= abs(real_gss)).mean())

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"\n  Real pooled r (GroupShuffleSplit): {real_gss:.3f}")
    print(f"  Permutation null range (5-95%):    "
          f"[{np.percentile(perm_rs_gss,5):+.3f}, {np.percentile(perm_rs_gss,95):+.3f}]")
    print(f"  Empirical p-value: {p_value:.3f}  "
          f"(fraction of permuted |r| >= real |r|)")

    if p_value < 0.05 and real_gss > np.percentile(perm_rs_gss, 95):
        print(f"""
  [REAL, DIMINISHED] The properly-grouped result clears the permutation
  null (p={p_value:.3f}). There IS genuine signal here -- weaker than the
  0.581/0.549 originally claimed (those were leakage-inflated), but not
  zero. This becomes the new, honest Tier 1 number.
""")
    else:
        print(f"""
  [DOES NOT CLEAR PERMUTATION] The properly-grouped result does not
  reliably separate from what random pairing produces at this sample
  size (p={p_value:.3f}). Combined with HR_rising/RMSSD_rising/
  RMSSD_magnitude all failing today, this means NOTHING in the
  originally-claimed Tier 1 proxy has survived honest validation.
""")


if __name__ == "__main__":
    main()
