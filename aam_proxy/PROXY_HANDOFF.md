# AAM Biosignal Proxy — Handoff

**Status as of this document: complete rewrite.** Everything before this
rewrite (SWELL-KW HR/RMSSD/SCL/CCA as the "core proxy") is **dead** —
confirmed below chance or indistinguishable from permuted noise, checked
multiple independent ways. If you find `train_biosignal_proxy.py`,
`apply_biosignal_proxy.py`, or `proxy_artifacts/` in this repo, **do not
use them** — they're kept only as a record of what was tried and ruled out.

---

## What this actually is now

Not one proxy — a **router + three independent experts**, each trained
on the dataset it's structurally suited to, each activated only under
the session conditions it was validated for. No expert is used outside
its validated scope.

```
aam_proxy/
  session_router.py                   <- decides which expert fires
  experts/
    train_coglab_expert.py            <- HCI-only, ✅ VALIDATED, USE THIS
    coglab_expert_artifacts/          <- trained models
    train_sense42_prior.py            <- needs known task identity
    sense42_prior_artifacts/          <- categorical lookup table
    train_clare_expert.py             <- needs REAL EEG input
    clare_expert_artifacts/           <- trained models
```

---

## FOR BEHACOM SPECIFICALLY: only the Cog Lab expert applies

BEHACOM has **zero biosignals**. Walking through the router:

| Expert | Fires when | Applies to BEHACOM? |
|---|---|---|
| `coglab_expert` | single sustained task, HCI only | **YES — this is the one** |
| `sense42_prior` | multi-task, task identity known | No — needs task categories matching SENSE-42's specific set (mail/notes/file_mgr/browser/trash); BEHACOM's own taxonomy won't map onto these |
| `clare_expert` | real EEG calibration present | No — structurally requires EEG, BEHACOM has none |

**Don't expect to see the other two experts activate on BEHACOM data.**
They exist for AAM's own live sessions, where task identity and
(eventually) calibration EEG are available. For BEHACOM, this handoff
is really about one thing: the Cog Lab expert.

---

## The Cog Lab expert — what it actually is

**Targets** (only these three — everything else tested was excluded,
see below):

| Target | LOSO accuracy | Empirical chance | Over chance |
|---|---|---|---|
| `acc_jerk_direction` | 0.697 | 0.526 | **+0.171** |
| `acc_jerk_magnitude` | 0.659 | 0.481 | **+0.178** |
| `eeg_engagement_direction` | 0.612 | 0.526 | **+0.086** |

Validated with true per-participant `LeaveOneGroupOut` (N=16 Cog Lab
subjects, S2/S17 excluded — S2 has no HCI folder, S17 is a **confirmed
byte-identical duplicate of S1's data**, verified via `diff`), plus a
permutation control on every number above.

**Input**: HCI count *deltas* (window-to-window change), the same
13-column schema SWELL-KW/Cog Lab share:

```
SnKeyStrokes, SnChars, SnSpecialKeys, SnDirectionKeys, SnErrorKeys,
SnShortcutKeys, SnSpaces, CharactersRatio, ErrorKeyRatio,
SnLeftClicked, SnWheel, SnMouseDistance, SnMouseAct
```

**Not** the richer keyboard-LSTM/mouse-stats encoder pipeline — that
was built and tested today, and **lost head-to-head against this flat
baseline on every single target** (`head_to_head_test.py`). The
adapters/encoders are real, reusable infrastructure, but as tested they
didn't beat the simpler input. Use the flat counts.

---

## Mapping BEHACOM's raw columns onto this 13-column schema

Reuse the mapping guidance from the old `apply_biosignal_proxy.py`
docstring — that part is still valid, it's the schema itself, not tied
to whichever model consumes it. You need **13 of the original 18**
columns (drop `SnRightClicked`, `SnDoubleClicked`, `SnDragged`,
`SnAppChange`, `SnTabfocusChange` — Cog Lab's expert never used these,
since Cog Lab sessions are single-window and never generate those event
types).

**Validation checklist before trusting any output** (same discipline as
everything in this project):
1. Every row should get a real (non-NaN) value for all 13 columns —
   gaps mean the BEHACOM mapping missed something.
2. Compute deltas (window `i` minus window `i-1`, **within the same
   user**, never across users) — the expert was trained on deltas, not
   absolute counts. Feeding it absolute values will silently produce
   garbage, not an error.
3. z-score the deltas using the expert's own saved
   `metadata.json["input_normalization"]` mean/std — don't re-normalize
   from scratch on BEHACOM's own distribution, that would break the
   correspondence to what the model was actually trained on.

---

## How to run it

```python
from aam_proxy.session_router import (
    SessionContext, route, apply_expert, load_coglab_expert
)

ctx = SessionContext(
    task_node_dwell_times={"the_one_app": total_seconds},  # single task
    n_switches=0,   # or low — BEHACOM sessions in one app
)
decision = route(ctx)
assert decision.expert == "coglab_expert"

expert = load_coglab_expert()
# hci_delta_vector: 13-dim, z-scored using expert["metadata"]["input_normalization"]
result = apply_expert(decision, hci_delta_vector=your_zscored_13dim_vector)
# result: {"acc_jerk_direction": 0/1, "acc_jerk_magnitude": 0/1/2,
#          "eeg_engagement_direction": 0/1}
```

---

## What's explicitly EXCLUDED, and why (so you don't rediscover these the hard way)

| Target | Result | Why excluded |
|---|---|---|
| `HR_rising` (SWELL-KW) | 0.790 vs true chance 0.820 | **Below chance** |
| `RMSSD_rising` (SWELL-KW) | 0.775 vs true chance 0.801 | **Below chance** |
| `RMSSD_magnitude` (SWELL-KW) | 0.329 vs chance 0.333 | Exact chance |
| `SCL_rising` (SWELL-KW) | ~0.49-0.52 vs chance 0.50 | Exact chance |
| CCA (SWELL-KW, properly grouped) | r=0.225 | Does not clear permutation null [-0.296, +0.378] |
| `resp_bpm` (Cog Lab, direction+mag) | Both flagged | Confirmed false positive — real chance was 0.847, not the assumed 0.50 |
| HCI→interruption condition (SWELL-KW, direct) | 3-class and binary both tested | Both match permutation — confirmed null |
| `eeg_theta_alpha`, `eeg_alpha_asym`, `fnirs_hbo_slope_L/R` (Cog Lab) | All negligible or below chance | Never cleared the bar |

Every one of these was checked with true LOSO + empirical chance +
permutation control, not assumed. If you're tempted to re-add any of
them because a number "looks good" — check it against its own real
chance and a permutation baseline first. Several of these looked
genuinely strong before that check and turned out to be artifacts
(class imbalance, validation leakage, or target-construction bugs).

---

## Honest scope statement

This is a **narrow, validated signal** — three targets, one dataset,
N=16. It is not a comprehensive cognitive-state readout, and it should
not be marketed or documented as one. Treat it as: "in a single-task,
HCI-only session, we have real, if modest, evidence that mouse-motion
jerk and one EEG-engagement-direction marker correlate with typing/
mouse behavior." That is the honest, defensible claim — use it as such.
