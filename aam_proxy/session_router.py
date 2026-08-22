"""
session_router.py
====================
Decides which expert(s) fire for a given AAM session, based on session-
level characteristics detectable from the switching graph, task
metadata, and whether an EEG calibration reading exists.

CURRENT STATE -- ONLY ONE EXPERT IS ACTUALLY TRAINED
---------------------------------------------------------
    Cog Lab expert:  READY  (train_coglab_expert.py, validated)
    SENSE-42 prior:  NOT YET BUILT (categorical HEP contrast, needs a
                      different mechanism than a trained regression head
                      -- see PROXY_ARCHITECTURE.md)
    CLARE expert:    NOT YET BUILT (requires real EEG calibration input,
                      not HCI-only -- different activation condition)
    SWELL-KW:        NO VALIDATED SIGNAL. Interruption-detected sessions
                      currently route to NOTHING. This is an open gap,
                      not a routing bug -- do not build a stub expert
                      here until a real result exists to back it.

This router is deliberately conservative: it returns "no_expert_available"
for any condition that doesn't have a trained, validated expert behind
it, rather than silently falling back to something unvalidated.

TWO SEPARATE VALIDATION QUESTIONS -- COLD-START VS WARM-START
-------------------------------------------------------------------
Every dataset trained on so far -- SWELL-KW, Cog Lab, SENSE-42 -- has
EXACTLY ONE SESSION PER PARTICIPANT. That structural fact means the
LOSO validation used throughout this project (leave-one-USER-out, that
user's ENTIRE data held out of training) can only ever answer one
question: "how does this perform on someone the model has never seen
at all" -- the cold-start case, a brand new user's first-ever session.

It cannot answer a different, equally real question: "how does this
perform on a RETURNING user, once AAM has accumulated several of their
own prior sessions" -- warm-start. That would need a different
validation scheme (leave-one-SESSION-out, where a user's OTHER sessions
stay in training) and NONE of the datasets used so far have more than
one session per person to test this with. This isn't a validation
choice that was skipped -- it's currently untestable with the data on
hand.

PersonalizationLayer below is a STUB for this second question. It does
NOT perform any real personalization yet -- there is no multi-session-
per-user data to train or validate it against. It exists so the
warm-start use case is part of the documented architecture from day
one, rather than something bolted on later once real longitudinal AAM
data exists. Activation condition: user_session_history >= 
MIN_SESSIONS_FOR_PERSONALIZATION. Until then, every user is treated as
cold-start, which is the only scenario currently validated for any
expert in this system.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


N_SWITCH_THRESHOLD = 3   # max switches away from majority node to still
                         # count as "single sustained task"
MIN_DWELL_S = 30.0       # minimum time on a task node to count as real
                         # engagement with it, not a quick glance

# Placeholder only -- no data exists yet to justify a specific number.
# Once multi-session-per-user AAM data accumulates, this threshold
# itself needs to be chosen empirically (e.g. via a leave-one-session-
# out sweep: at what n does warm-start meaningfully beat cold-start),
# not picked in advance.
MIN_SESSIONS_FOR_PERSONALIZATION = 5


@dataclass
class SessionContext:
    """What the router needs to know about a session to decide routing.
    Populate this from AAM's actual switching graph + interruption log +
    EEG calibration status before calling route()."""
    task_node_dwell_times: dict           # {task_id: seconds_spent}
    n_switches: int
    interruption_events: int = 0
    eeg_calibration_engagement_index: Optional[float] = None
    session_duration_s: float = 0.0
    # -- warm-start fields, see module docstring --
    user_id: Optional[str] = None
    user_session_history: int = 0         # count of THIS user's prior
                                          # sessions AAM has stored


@dataclass
class RoutingDecision:
    expert: str                            # which expert fires
    reason: str                            # why, for logging/debugging
    confidence_basis: str                  # what evidence backs this expert
    available_targets: list = field(default_factory=list)
    personalization: "PersonalizationStatus" = None


@dataclass
class PersonalizationStatus:
    """
    Whether warm-start personalization COULD apply to this session, and
    why it currently never actually changes the expert's output.

    is_eligible: True once user_session_history clears the (placeholder)
    threshold -- meaning AAM HAS enough of this user's own history that
    a warm-start model, once built and validated, would have something
    to work with.

    is_active: ALWAYS False right now. No dataset used to train or
    validate any expert in this system has more than one session per
    participant, so there is no leave-one-session-out validated
    personalization model to activate -- only the cold-start (LOSO)
    experts exist. Flipping this to True before that validation exists
    would mean shipping an unvalidated adjustment on top of already-
    validated predictions, which is exactly the mistake this whole
    project spent a full session learning to avoid.
    """
    is_eligible: bool
    is_active: bool
    reason: str


def personalization_status(ctx: SessionContext) -> PersonalizationStatus:
    eligible = ctx.user_session_history >= MIN_SESSIONS_FOR_PERSONALIZATION
    if eligible:
        reason = (f"User has {ctx.user_session_history} prior sessions "
                 f"(>= threshold {MIN_SESSIONS_FOR_PERSONALIZATION}) -- "
                 f"eligible for warm-start ONCE a leave-one-session-out "
                 f"validated model exists. Not active: no such model has "
                 f"been built or validated yet (no multi-session-per-user "
                 f"dataset exists to build it from).")
    else:
        reason = (f"User has {ctx.user_session_history} prior sessions, "
                 f"below the placeholder threshold "
                 f"({MIN_SESSIONS_FOR_PERSONALIZATION}). Treated as "
                 f"cold-start -- the only validated scenario currently.")
    return PersonalizationStatus(is_eligible=eligible, is_active=False,
                                 reason=reason)


def classify_task_structure(ctx: SessionContext) -> str:
    """Returns 'single_task', 'multi_task', or 'ambiguous'."""
    real_tasks = {t: d for t, d in ctx.task_node_dwell_times.items()
                 if d >= MIN_DWELL_S}
    if len(real_tasks) <= 1 and ctx.n_switches <= N_SWITCH_THRESHOLD:
        return "single_task"
    if len(real_tasks) >= 2:
        return "multi_task"
    return "ambiguous"


CLARE_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experts", "clare_expert_artifacts")
SENSE42_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experts", "sense42_prior_artifacts")
COGLAB_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experts", "coglab_expert_artifacts")


def _artifacts_exist(path, required_files):
    return os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, f)) for f in required_files)


def load_clare_expert():
    """Loads the real trained models -- raises clearly if artifacts are
    missing rather than silently returning something unusable."""
    import json, joblib
    meta_path = os.path.join(CLARE_ARTIFACTS_DIR, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    models = {}
    for tgt in meta["targets"]:
        models[tgt] = joblib.load(os.path.join(CLARE_ARTIFACTS_DIR, f"{tgt}_model.pkl"))
    return {"models": models, "metadata": meta}


def apply_clare_expert(expert, engagement_index, hr_mean, hrv_rmssd,
                       eda_tonic_mean, eda_tonic_slope):
    """Runs the real CLARE expert. Requires a genuine EEG-derived
    engagement_index -- this is never inferable from HCI alone, by
    design (see train_clare_expert.py docstring)."""
    import numpy as np
    meta = expert["metadata"]
    mu, sd = np.array(meta["train_mu"]), np.array(meta["train_sd"])
    x = np.array([engagement_index, hr_mean, hrv_rmssd,
                  eda_tonic_mean, eda_tonic_slope])
    xz = ((x - mu) / sd).reshape(1, -1)
    return {tgt: float(m.predict(xz)[0]) for tgt, m in expert["models"].items()}


def load_sense42_prior():
    import json
    with open(os.path.join(SENSE42_ARTIFACTS_DIR, "task_hep_lookup.json")) as f:
        return json.load(f)


def apply_sense42_prior(prior, task_name):
    """Returns the expected HEP amplitude bias (z-scored) for a known
    task, or None if that task isn't in the validated lookup -- never
    fabricates a number for an unseen task."""
    entry = prior["task_amplitude_z"].get(task_name)
    if entry is None:
        return None
    return entry


def load_coglab_expert():
    import json, joblib
    meta_path = os.path.join(COGLAB_ARTIFACTS_DIR, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    models = {}
    for tgt in meta["targets"]:
        models[tgt] = joblib.load(os.path.join(COGLAB_ARTIFACTS_DIR, f"{tgt}.pkl"))
    return {"models": models, "metadata": meta}


def apply_coglab_expert(expert, hci_delta_vector):
    """hci_delta_vector must already be z-scored using the expert's own
    saved mean/std (metadata['input_normalization']) -- caller's
    responsibility, since the raw delta computation depends on the
    session's own HCI schema, not something this function can assume."""
    x = hci_delta_vector.reshape(1, -1)
    out = {}
    for tgt, m in expert["models"].items():
        pred = m.predict(x)[0]
        out[tgt] = int(pred) if hasattr(m, "classes_") else float(pred)
    return out


def apply_expert(decision: "RoutingDecision", **kwargs):
    """
    Single dispatch entrypoint: given a RoutingDecision from route(),
    loads and runs the correct expert with the right inputs. Raises
    clearly rather than guessing if the required kwargs are missing for
    whichever expert fired.
    """
    if decision.expert == "coglab_expert":
        if "hci_delta_vector" not in kwargs:
            raise ValueError("coglab_expert requires hci_delta_vector")
        expert = load_coglab_expert()
        return apply_coglab_expert(expert, kwargs["hci_delta_vector"])

    if decision.expert == "clare_expert":
        needed = ["engagement_index","hr_mean","hrv_rmssd",
                 "eda_tonic_mean","eda_tonic_slope"]
        missing = [k for k in needed if k not in kwargs]
        if missing:
            raise ValueError(f"clare_expert requires {missing}")
        expert = load_clare_expert()
        return apply_clare_expert(expert, **{k: kwargs[k] for k in needed})

    if decision.expert == "sense42_prior":
        if "task_name" not in kwargs:
            raise ValueError("sense42_prior requires task_name")
        prior = load_sense42_prior()
        return apply_sense42_prior(prior, kwargs["task_name"])

    raise ValueError(f"No apply_* implementation for expert={decision.expert!r} "
                     f"-- check decision.reason for why nothing fired.")


def route(ctx: SessionContext) -> RoutingDecision:
    task_structure = classify_task_structure(ctx)
    pstatus = personalization_status(ctx)

    # ---- interruption-present sessions: no validated expert exists ----
    if ctx.interruption_events > 0:
        return RoutingDecision(
            expert="no_expert_available",
            reason=f"{ctx.interruption_events} interruption event(s) "
                  f"detected, but no dataset produced a validated "
                  f"HCI-only signal in this context (SWELL-KW's own "
                  f"interruption-condition data was tested directly and "
                  f"came back below chance).",
            confidence_basis="none -- open problem, not yet solvable",
            personalization=pstatus)

    # ---- EEG calibration available: CLARE expert, NOW REAL ----
    if ctx.eeg_calibration_engagement_index is not None:
        ready = _artifacts_exist(CLARE_ARTIFACTS_DIR,
                                 ["metadata.json", "log_frontal_alpha_model.pkl",
                                  "frontal_theta_alpha_ratio_model.pkl"])
        return RoutingDecision(
            expert="clare_expert" if ready else "clare_expert_ARTIFACTS_MISSING",
            reason="Real EEG calibration reading present -- CLARE expert "
                  "trigger condition. Predicts frontal_alpha and "
                  "theta_alpha_ratio from engagement_index + real ECG/EDA. "
                  "frontal_theta deliberately excluded (circular with its "
                  "own predictor, see train_clare_expert.py).",
            confidence_basis="true LOSO validated: log_frontal_alpha R2=0.129, "
                            "frontal_theta_alpha_ratio R2=0.349, both clean "
                            "against the temporal-site control",
            available_targets=["log_frontal_alpha", "frontal_theta_alpha_ratio"]
                              if ready else [],
            personalization=pstatus)

    # ---- single sustained task: Cog Lab expert, READY ----
    if task_structure == "single_task":
        return RoutingDecision(
            expert="coglab_expert",
            reason="Single sustained task, minimal window switching -- "
                  "matches Cog Lab's native session structure.",
            confidence_basis="true LOSO + empirical chance validated: "
                            "acc_jerk direction over=+0.171, magnitude "
                            "over=+0.178, eeg_engagement direction over=+0.086",
            available_targets=["acc_jerk_direction", "acc_jerk_magnitude",
                              "eeg_engagement_direction"],
            personalization=pstatus)

    # ---- multi-task: SENSE-42 prior, NOW REAL ----
    if task_structure == "multi_task":
        ready = _artifacts_exist(SENSE42_ARTIFACTS_DIR, ["task_hep_lookup.json"])
        return RoutingDecision(
            expert="sense42_prior" if ready else "sense42_prior_ARTIFACTS_MISSING",
            reason="Multiple distinct task nodes with real dwell time -- "
                  "matches SENSE-42's structure. Returns an EXPECTED "
                  "cardiac-locked EEG amplitude bias for the CURRENT known "
                  "task (categorical lookup, not a per-window prediction "
                  "from behavior -- see train_sense42_prior.py).",
            confidence_basis="confound-safe-window-checked task contrasts: "
                            "file_mgr vs mail d=+0.44 p=0.018*, notes vs "
                            "browser d=-0.52 p=0.006** -- notes vs mail "
                            "correctly returns no_difference (p=0.88)",
            available_targets=["task_hep_amplitude_z"] if ready else [],
            personalization=pstatus)

    return RoutingDecision(
        expert="no_expert_available",
        reason=f"Task structure ambiguous (task_structure={task_structure}) "
              f"-- doesn't clearly match any validated expert's native "
              f"session shape.",
        confidence_basis="none",
        personalization=pstatus)


if __name__ == "__main__":
    # quick self-test with a few illustrative session shapes
    examples = [
        ("single sustained task, new user", SessionContext(
            task_node_dwell_times={"notes_app": 1800}, n_switches=2,
            user_id="u1", user_session_history=0)),
        ("single sustained task, returning user (8 prior sessions)", SessionContext(
            task_node_dwell_times={"notes_app": 1800}, n_switches=2,
            user_id="u2", user_session_history=8)),
        ("multi-task session", SessionContext(
            task_node_dwell_times={"mail": 600, "browser": 500, "notes": 700},
            n_switches=14)),
        ("interruption present", SessionContext(
            task_node_dwell_times={"notes_app": 1800}, n_switches=2,
            interruption_events=1)),
        ("EEG calibration available", SessionContext(
            task_node_dwell_times={"notes_app": 1800}, n_switches=2,
            eeg_calibration_engagement_index=0.42)),
    ]
    for label, ctx in examples:
        d = route(ctx)
        print(f"\n[{label}]")
        print(f"  -> expert: {d.expert}")
        print(f"  -> reason: {d.reason}")
        if d.available_targets:
            print(f"  -> targets: {d.available_targets}")
        print(f"  -> personalization: eligible={d.personalization.is_eligible} "
              f"active={d.personalization.is_active}")
        print(f"     {d.personalization.reason}")


############################################################
# FILE: /home/hefouzinho/biosignals_data/scripts/baselines/run_baseline.py
