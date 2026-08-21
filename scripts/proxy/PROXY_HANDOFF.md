# Biosignal Proxy — Handoff for BEHACOM Application

## What this is

A frozen model that predicts physiological state (heart rate direction,
heart-rate-variability direction/magnitude) from mouse and keyboard
behavior alone. Trained on SWELL-KW (25 participants, 3 experimentally
induced cognitive-load conditions, simultaneous HCI + ECG + skin
conductance). Validated: CCA cross-validated correlation r=0.581 between
behavior and physiology, direction-prediction accuracy 77–85% depending
on target (chance = 50%).

**Purpose:** apply it to BEHACOM to generate per-window pseudo-physiological
labels, either as (a) auxiliary training targets for a separate cognitive-
state model, or (b) a validity check on whether BEHACOM's behavioral
patterns look physiologically plausible.

You don't need to understand the training methodology to run this — the
two scripts below handle everything. This document is here so you can
map BEHACOM's columns correctly and sanity-check the output.

---

## What you need to do

### Step 1 — Map BEHACOM's columns onto the required schema

The proxy expects exactly these 18 columns, one row per time window
(ideally per-minute, matching how it was trained):

| Column | Meaning |
|---|---|
| `SnMouseAct` | fraction of the window with mouse activity (0–1) |
| `SnLeftClicked` | left-click count |
| `SnRightClicked` | right-click count |
| `SnDoubleClicked` | double-click count |
| `SnWheel` | scroll-wheel event count |
| `SnDragged` | drag event count |
| `SnMouseDistance` | total mouse movement distance (pixels) |
| `SnKeyStrokes` | total keystroke count |
| `SnChars` | printable character keystrokes |
| `SnSpecialKeys` | non-printable key presses (shift, ctrl, etc.) |
| `SnDirectionKeys` | arrow key presses |
| `SnErrorKeys` | backspace/delete count |
| `SnShortcutKeys` | keyboard shortcut count (ctrl+combinations) |
| `SnSpaces` | space bar presses |
| `SnAppChange` | number of application switches in the window |
| `SnTabfocusChange` | number of window/tab focus changes |
| `CharactersRatio` | `SnChars / SnKeyStrokes` |
| `ErrorKeyRatio` | `SnErrorKeys / SnKeyStrokes` |

**If BEHACOM doesn't have an exact match for a column**, use the closest
available quantity or set it to 0 — but note which columns you approximated
when you report results back, since the CCA weights specific columns
differently (SnKeyStrokes carries the most weight, +0.83 loading on the
main component; if that one is missing or wrong, results are unreliable).

BEHACOM's raw data is typically organized around mouse movement events,
click events, keyboard events, and app-switching logs at fine granularity
— you'll need to aggregate into fixed windows (1-minute recommended, to
match SWELL-KW's native resolution) before this mapping. If your BEHACOM
export uses different native column names, match by concept, not by name.

Also add one column identifying the user/participant: default name
`user_id`. If your file uses a different name, edit the `USER_COL`
variable at the top of `apply_biosignal_proxy.py`.

### Step 2 — Run the apply script

```bash
pip install numpy pandas scikit-learn joblib
python apply_biosignal_proxy.py --input your_mapped_behacom.csv --output proxy_output.csv
```

This prints the proxy's own training performance first (so you can see
what accuracy level to expect), then applies it and runs a validation
checklist automatically.

### Step 3 — Check the validation output

The script checks this for you and prints results, but here's what a
healthy run looks like:

- **Coverage** should be ~100% on all output columns. If not, some of
  the 18 input columns have missing values — go back to Step 1.
- **CCA load score** should be roughly zero-centered per user (it's
  z-scored per user before projection), with per-user std somewhere in
  the 0.2–0.5 range. Reference from the original BEHACOM run: std range
  0.232–0.432 across 12 users.
- **hr_rising_prob** should vary meaningfully across users, not sit at
  a flat ~0.5 for everyone. If it's flat, check that `SnMouseAct` and
  `SnKeyStrokes` aren't constant/zero in your mapped data.
- **Silent overload pattern** (informational, not a bug check): the
  original BEHACOM run found 2 of 12 users with low mouse activity but
  high predicted HR-rising probability — behaviorally quiet but
  physiologically predicted-aroused. If this recurs in your run, it's a
  plausible finding worth flagging back, not an error.

---

## What NOT to use, and why

**`scl_rising_prob_FLAGGED`** — deliberately named to discourage casual
use. The skin-conductance model performs at chance (50.3% direction
accuracy) on its own source data. Worse, when tested as a fourth CCA
target alongside heart rate and HRV, it actively *hurt* the projection
(cross-validated r dropped from 0.581 to 0.497). It's included in the
output only for completeness/future revisiting at longer aggregation
windows — skin conductance moves on a 1–3 minute timescale and per-minute
windows may simply be too fast for it. Do not wire it into any training
loop as an active target without re-validating first.

**Don't treat `cca_load_score` or the direction probabilities as ground
truth.** They're model predictions with real but limited accuracy (77–85%
range depending on target). Appropriate as a soft auxiliary training
signal (lower loss weight), not appropriate as a hard label you'd trust
at face value for any single window.

---

## Background, if useful

The proxy exists because AAM's real training labels (NASA-TLX self-report)
are session-level and far too sparse for window-level supervision. This
proxy provides denser, physiologically-grounded pseudo-labels to
supplement that.

The key methodological finding behind why this specific design works:
the CCA was trained on **condition-level aggregates** (participant ×
condition mean, from SWELL-KW's 3 deliberately induced load conditions),
not on raw per-minute windows. That collapse-to-condition-mean step is
what exposes the induced-load signal — training directly on per-minute
data without deliberate condition contrast has consistently failed in
every other dataset tested (a separate, much larger investigation using
a 42-participant naturalistic dataset found no usable HCI→physiology
signal at any window size, precisely because that dataset had no
deliberately induced load variation). BEHACOM is naturalistic too, which
is exactly why this is being used as a **frozen, pre-trained** proxy
applied to BEHACOM rather than something re-trained on BEHACOM directly
— BEHACOM's own naturalistic variation likely isn't strong enough to
train a proxy from scratch, but a proxy trained elsewhere on genuine
condition contrast can still be *applied* to it.

Questions or odd results — send back the validation checklist output and
a few rows of your mapped input CSV, and it'll be easy to diagnose from
there.
