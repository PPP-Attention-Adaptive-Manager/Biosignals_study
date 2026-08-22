# AAM Biosignal Proxy — Handoff

**Status as of this document: complete rewrite.** Everything before this
rewrite (SWELL-KW HR/RMSSD/SCL/CCA as the "core proxy") is **dead** —
confirmed below chance or indistinguishable from permuted noise, checked
multiple independent ways. If you find `train_biosignal_proxy.py`,
`apply_biosignal_proxy.py`, or `proxy_artifacts/` in this repo, **do not
use them** — they're kept only as a record of what was tried and ruled out.

---

## Full Architecture Recap

```
                          ┌─────────────────────┐
                          │   session_router.py   │
                          │  (decides which expert │
                          │   fires, if any)        │
                          └──────────┬───────────┘
                                    │
        ┌───────────────┬──────────┼──────────┬────────────────┐
        │                │                    │                │
   single_task      multi_task        EEG calibration   interruption
        │                │              available            present
        ▼                ▼                    ▼                ▼
  coglab_expert    sense42_prior       clare_expert      no_expert_available
   (HCI-only,      (categorical         (needs REAL         (confirmed null,
    REAL, ready)    lookup, needs        EEG input,           see below)
                    known task ID)       REAL, ready)
```

Plus a **personalization layer** (warm-start for returning users) — designed
into `SessionContext`/`PersonalizationStatus` but permanently `is_active=False`
until multi-session-per-user AAM data exists to validate a leave-one-session-out
model against. Currently every user is treated as cold-start, the only
scenario any expert has actually been validated for.

### Expert 1 — `coglab_expert` (the one that matters for BEHACOM)

| | |
|---|---|
| **Fires when** | single sustained task, minimal window switching |
| **Input** | 13-column HCI count deltas (window-to-window change) — `SnKeyStrokes, SnChars, SnSpecialKeys, SnDirectionKeys, SnErrorKeys, SnShortcutKeys, SnSpaces, CharactersRatio, ErrorKeyRatio, SnLeftClicked, SnWheel, SnMouseDistance, SnMouseAct` |
| **Output** | `acc_jerk_direction` (2-class), `acc_jerk_magnitude` (3-class), `eeg_engagement_direction` (2-class) |
| **Validated on** | Cog Lab, N=16 (S1,S3-S16,S18 — S2 excluded: no HCI folder; S17 excluded: confirmed byte-identical duplicate of S1, verified via `diff`) |
| **Method** | True per-participant `LeaveOneGroupOut`, empirical chance, permutation control |
| **Results** | `acc_jerk_direction`: acc=0.697, chance=0.526, **over=+0.171**. `acc_jerk_magnitude`: acc=0.659, chance=0.481, **over=+0.178**. `eeg_engagement_direction`: acc=0.612, chance=0.526, **over=+0.086** |
| **Note** | A richer input (real keyboard-LSTM + mouse-stats encoder embeddings, 86-dim) was built and head-to-head tested against this flat baseline — **the baseline won on every target**. Use the flat counts, not the encoder pipeline. |

### Expert 2 — `sense42_prior` (categorical lookup, not a trained model)

| | |
|---|---|
| **Fires when** | multi-task session, task identity known |
| **Input** | task label only (`file_mgr`, `mail`, `notes`, `browser`, `trash`) — **never inferred from HCI** |
| **Output** | expected cardiac-locked EEG amplitude bias (z-scored) for that task |
| **Validated on** | SENSE-42, N=34-40, confound-safe rewindowed (R-R interval check ruled out tachycardia-into-next-QRS artifact) |
| **Results** | `file_mgr vs mail`: d=+0.44, p=0.018*. `notes vs browser`: d=-0.52, p=0.006**. `notes vs mail`: d=-0.03, p=0.88 — correctly returns "no difference," not omitted |
| **Note** | Chaining this into an HCI-predictable form was tested and failed the redundancy check (adding HCI features on top of app-identity made prediction *worse*, gap=-0.046) — task identity itself carries the signal, not something to re-derive from behavior. **Not applicable to BEHACOM** — its task taxonomy won't map onto SENSE-42's specific 5 categories. |

### Expert 3 — `clare_expert` (needs real EEG, structurally can't run on HCI alone)

| | |
|---|---|
| **Fires when** | real EEG calibration reading present (`engagement_index` computed from actual EEG) |
| **Input** | `engagement_index` (real EEG) + `hr_mean`, `hrv_rmssd` (ECG) + `eda_tonic_mean`, `eda_tonic_slope` (EDA) |
| **Output** | `log_frontal_alpha`, `frontal_theta_alpha_ratio` — **`frontal_theta` deliberately excluded** (R²=0.887-0.890 in testing, mostly circular since `engagement_index = beta/(alpha+theta)` shares theta in its own denominator) |
| **Validated on** | CLARE, N=19, true LOSO |
| **Results** | `log_frontal_alpha`: R²=0.129 (up from 0.057 engagement-alone — real improvement from adding ECG/EDA). `frontal_theta_alpha_ratio`: R²=0.349 |
| **Note** | Both confirmed clean against a temporal-site (TP9/TP10) control — no target there crossed significance. **Not applicable to BEHACOM** — no EEG exists in that dataset at all. |

### The interruption branch — confirmed null, not a gap

Four independent tests, all converging on the same answer:
1. HCI→cardio/skin (SWELL-KW): `HR_rising` 0.790 vs true chance 0.820, `RMSSD_rising` 0.775 vs 0.801 — **both below chance**
2. HCI→CCA properly grouped: r=0.225, doesn't clear permutation *at this sample size* — see reframed note below
3. HCI→resp_bpm chain: confirmed no improvement over HCI-alone baseline
4. HCI→interruption condition, direct, no biosignal step (3-class and binary): both match permutation

`session_router.py` correctly returns `no_expert_available` for interruption-present sessions.

### On the CCA specifically — reframed, not simply "null"

A power analysis (triggered by checking a literature claim that r=0.15–0.30 is
typical for this kind of relationship) found our SWELL-KW sample (N=19-25) had
only **~27% power to detect r=0.30** and **~9-11% power to detect r=0.15** — the
field-typical range. The honest statement is **not** "confirmed no relationship,"
it's: *"statistically indistinguishable from permuted noise at this sample size;
underpowered to detect a field-typical small effect; would need N≥85-150 for a
conclusive test."* Still not deployed — a real-but-unconfirmed small effect and
an actually-null effect look identical at this N, and CCA's own instability
(SD=0.263 on a mean of 0.225) is a separate reliability problem regardless of the
population-level answer. Flagged here as an open question for future work with
adequate sample size, not a closed dead end.

---



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
| CCA (SWELL-KW, properly grouped) | r=0.225, N=19-25, ~9-27% power | **Reframed**: underpowered, not confirmed null — see architecture section above |
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
