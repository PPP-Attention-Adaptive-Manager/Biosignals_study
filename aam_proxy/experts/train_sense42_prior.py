"""
train_sense42_prior.py
=========================
Builds the SENSE-42 expert -- structurally DIFFERENT from the Cog Lab
expert. This is NOT a trained regression head predicting a continuous
value from HCI. It's a CATEGORICAL PRIOR: given known task identity
(which AAM's router already has for a multi-task session), what does
cardiac-locked EEG amplitude look like for that task, relative to
other tasks in the same session?

WHY A LOOKUP, NOT A MODEL
------------------------------
The validated finding (sense42_hep_analysis.py, confound-safe-window
confirmed) is a TASK-PAIR CONTRAST, not an HCI->EEG mapping:
    file_mgr vs mail:    d=+0.44  p=0.018*  (confound-safe window)
    notes vs browser:    d=-0.52  p=0.006** (confound-safe window)
    notes vs mail:       d=-0.03  NOT significant -- correctly null

Every attempt to chain this INTO an HCI-predictable form failed the
redundancy check (HEP regression: adding HCI features on top of app-
identity made prediction WORSE, gap C-A = -0.046). Task identity itself
is the carrier of the signal -- not something to re-derive from HCI,
since AAM's router already has task identity directly. Building a
"model" here would just be re-deriving a label the system already has.

WHAT THIS PRODUCES
----------------------
A lookup table: task_type -> expected HEP amplitude (z-scored per
participant, mean + std across the validated SENSE-42 population),
usable as a PRIOR/BIAS the fusion model can consult when task identity
is known, NOT a per-window prediction from behavioral features.

ONLY the two contrasts that survived the confound-safe-window check are
included as directional priors. notes_vs_mail is included as an
explicit NULL entry (no expected difference) rather than omitted, so
downstream code has a documented answer for that pair too, instead of
silently returning nothing.

Input:  outputs/sense42_hep_amplitudes.csv (participant x task, already
        computed by sense42_hep_analysis.py)
Output: sense42_prior_artifacts/task_hep_lookup.json
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/biosignals_data")
HEP_CSV = os.path.join(BASE, "outputs", "sense42_hep_amplitudes.csv")
OUT_DIR = os.path.join(BASE, "aam_proxy", "experts", "sense42_prior_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

# Confirmed via sense42_hep_analysis.py's confound-safe-window recheck
# (R-R interval check ruled out the tachycardia-into-next-QRS confound).
VALIDATED_CONTRASTS = {
    ("file_mgr", "mail"):    {"d": 0.44, "p": 0.018, "direction": "file_mgr_higher"},
    ("notes", "browser"):    {"d": -0.52, "p": 0.006, "direction": "browser_higher"},
    ("notes", "mail"):       {"d": -0.03, "p": 0.88, "direction": "no_difference"},
}


def main():
    print("=" * 78)
    print("BUILDING SENSE-42 PRIOR — categorical task-contrast lookup")
    print("=" * 78)

    if not os.path.isfile(HEP_CSV):
        print(f"\n{HEP_CSV} not found. Run sense42_hep_analysis.py first "
              f"to produce the cached per-task HEP amplitudes.")
        return

    df = pd.read_csv(HEP_CSV)
    print(f"\nLoaded {len(df)} participant x task rows")

    # per-participant z-score, matching how the original contrast tests
    # normalized (removes individual amplitude baseline differences)
    df["hep_z"] = df.groupby("participant")["hep_amplitude_uv"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9) if x.std() > 1e-9 else 0.0)

    task_means = df.groupby("app")["hep_z"].agg(["mean", "std", "count"])
    print("\nPer-task mean HEP amplitude (z-scored per participant):")
    print(task_means.round(3).to_string())

    lookup = {
        "task_amplitude_z": {
            task: {"mean": float(row["mean"]), "std": float(row["std"]),
                  "n": int(row["count"])}
            for task, row in task_means.iterrows()
        },
        "validated_contrasts": {
            f"{a}_vs_{b}": v for (a, b), v in VALIDATED_CONTRASTS.items()
        },
        "usage_note": (
            "This is a PRIOR, not a per-window prediction. Given known "
            "task identity (from AAM's router, not inferred from HCI), "
            "look up task_amplitude_z[task]['mean'] as the EXPECTED "
            "cardiac-locked EEG amplitude bias for that task, relative "
            "to that person's own session-mean. Only file_mgr/mail and "
            "notes/browser contrasts are validated as significant "
            "task-pair DIFFERENCES (see validated_contrasts) -- the "
            "per-task means themselves are descriptive, not all "
            "individually significance-tested pairwise."
        ),
    }

    with open(os.path.join(OUT_DIR, "task_hep_lookup.json"), "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"\nSaved: {OUT_DIR}/task_hep_lookup.json")
    print("\nValidated contrasts included:")
    for (a, b), v in VALIDATED_CONTRASTS.items():
        sig = "significant" if v["p"] < 0.05 else "NOT significant (correct null)"
        print(f"  {a} vs {b}: d={v['d']:+.2f}  p={v['p']:.3f}  ({sig})")


if __name__ == "__main__":
    main()
