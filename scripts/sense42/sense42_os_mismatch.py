"""
sense42_os_mismatch.py
=======================
Tests whether interface-layout mismatch produces a measurable motor cost,
using SENSE-42's randomised OS-style manipulation.

THE DESIGN (which we previously missed entirely)
------------------------------------------------
In the experiment source, `style_randomizer` sits inside the outer trials
loop and re-randomises the whole interface every outer iteration:

    class Stylizer:
        STYLES = ["windows", "mac"]
        def reset(self, style=None):
            self.current_style = np.random.choice(self.STYLES)

For P002 that yielded 31 randomised blocks (19 windows / 14 mac).
We had concluded "SENSE-42 has no experimental conditions". Wrong -- it
has a randomised, within-subject, ~31-repetition UI manipulation.

Crossing presented style with each participant's HABITUAL OS (from
Demographics/participant_enrollment.csv) gives:

                      presented: windows    presented: mac
    habitual windows       MATCH               MISMATCH
    habitual mac           MISMATCH            MATCH

WHY window_close IS THE RIGHT DEPENDENT VARIABLE
-------------------------------------------------
From the calibration screen text in the experiment:

    "With Windows style layout, title bar buttons are presented on the
     top-right corner of the window."

So the manipulation's most concrete consequence is WHERE THE CLOSE BUTTON
IS -- top-right (Windows) vs top-left (Mac). A Mac-habitual participant
facing a Windows layout must override an automatic motor program and
search the opposite corner.

That is extraneous cognitive load in Sweller's sense: load imposed by
interface design rather than by the task itself. It should show up as
slower closes, longer mouse paths, and more hesitation.

This analysis needs NO EEG and NO clock alignment. It is pure behaviour
against a randomised factor, so it is immune to the ~104 s misalignment
bug that invalidated every earlier SENSE-42 result.

DEPENDENT MEASURES (per close event)
-------------------------------------
    close_time_s      total time from window_close onset to the click
    path_length       summed mouse displacement (normalised units)
    path_efficiency   straight-line distance / path_length
                      1.0 = perfectly direct, lower = wandering
    n_direction_rev   direction reversals in x (hesitation / re-targeting)
    time_to_first_move latency before the mouse starts moving
    initial_x_sign    sign of the first horizontal displacement
                      THE KEY MEASURE -- did they initially move toward
                      the wrong corner? A habitual-Mac user under a
                      Windows layout who first moves LEFT has revealed a
                      prepotent motor program being overridden.

STATISTICS
----------
Per-participant paired comparison (each participant contributes both
MATCH and MISMATCH blocks), then a paired test across participants.
This is a within-subject design, so between-participant differences in
baseline speed cancel out entirely.

Dual-OS users (Windows;MacOS etc, n=9) are EXCLUDED -- they have no
single habitual layout, so "mismatch" is undefined for them. That leaves
33 participants with an exclusive OS preference.

Run from: ~/biosignals_data/
Output:   outputs/sense42_os_mismatch_events.csv
          outputs/sense42_os_mismatch_results.json
"""
from __future__ import annotations
import os, glob, json, ast, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE    = os.path.expanduser("~/biosignals_data")
SENSE   = os.path.join(BASE, "data", "sense_42")
CSV_DIR = os.path.join(SENSE, "Behavioural", "CSV")
ENROLL  = os.path.join(SENSE, "Demographics", "participant_enrollment.csv")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_EV   = os.path.join(OUT_DIR, "sense42_os_mismatch_events.csv")
OUT_JSON = os.path.join(OUT_DIR, "sense42_os_mismatch_results.json")

# task-scope prefixes that carry their own window_close mouse arrays
SCOPES = ["mail", "file_manager_dragging_task", "file_manager_opening_task",
          "trash_bin", "notes", "browser"]

OS_COL = "What operating system(s) do you use most frequently? (select ALL that apply):"
PID_COL = "Participant ID"

MEASURES = ["close_time_s", "path_length", "path_efficiency",
            "n_direction_rev", "time_to_first_move", "moved_wrong_way"]


# ── habitual OS ───────────────────────────────────────────────────────

def load_habitual_os() -> dict:
    """
    participant id -> 'windows' | 'mac'.
    Dual users are dropped: mismatch is undefined without a single
    habitual layout. ChromeOS/Linux are ignored as secondary systems
    since neither implies a title-bar convention on its own.
    """
    e = pd.read_csv(ENROLL)
    out = {}
    for _, r in e.iterrows():
        raw = str(r.get(OS_COL, ""))
        has_win = "Windows" in raw
        has_mac = "MacOS" in raw
        if has_win and has_mac:
            continue                       # dual user -> exclude
        if has_win:
            out[int(r[PID_COL])] = "windows"
        elif has_mac:
            out[int(r[PID_COL])] = "mac"
    return out


# ── per-event motor features ──────────────────────────────────────────

def parse_list(v):
    try:
        out = ast.literal_eval(str(v))
        return out if isinstance(out, list) else None
    except Exception:
        return None


def close_event_features(xs, ys, lbs, ts):
    """
    Motor features for one close event.
    Trajectory is truncated at the first click, since everything after it
    is post-decision movement and would dilute the measure.
    """
    n = min(len(xs), len(ys), len(lbs), len(ts))
    if n < 5:
        return None
    xs, ys = np.asarray(xs[:n], float), np.asarray(ys[:n], float)
    lbs, ts = np.asarray(lbs[:n]), np.asarray(ts[:n], float)

    click = np.where(lbs == 1)[0]
    end = click[0] if len(click) else n - 1
    if end < 3:
        return None
    xs, ys, ts = xs[:end + 1], ys[:end + 1], ts[:end + 1]

    dx, dy = np.diff(xs), np.diff(ys)
    step = np.sqrt(dx ** 2 + dy ** 2)
    path = float(step.sum())
    straight = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))

    moving = np.where(step > 0.001)[0]
    t_first = float(ts[moving[0]]) if len(moving) else np.nan

    # direction reversals in x, ignoring sub-threshold jitter
    sig = np.sign(dx[np.abs(dx) > 0.001])
    n_rev = int((np.diff(sig) != 0).sum()) if len(sig) > 1 else 0

    # first meaningful horizontal displacement -- prepotent-response probe
    big = np.where(np.abs(dx) > 0.005)[0]
    init_sign = float(np.sign(dx[big[0]])) if len(big) else 0.0

    return {
        "close_time_s":       float(ts[-1] - ts[0]),
        "path_length":        path,
        "path_efficiency":    float(straight / path) if path > 1e-9 else np.nan,
        "n_direction_rev":    n_rev,
        "time_to_first_move": t_first,
        "initial_x_sign":     init_sign,
        "start_x":            float(xs[0]),
        "end_x":              float(xs[-1]),
        "n_samples":          int(len(xs)),
    }


# ── per participant ───────────────────────────────────────────────────

def process_participant(csv_path, pid, habitual):
    df = pd.read_csv(csv_path, low_memory=False)
    cols = df.columns

    if "style_randomizer.started" not in cols or "window_close.started" not in cols:
        return []

    # style blocks: (onset, style), forward-filled to label later events
    sr = df[["style_randomizer.started", "operating_system_style"]].dropna()
    blocks = [(float(t), str(s)) for t, s in
              zip(sr["style_randomizer.started"], sr["operating_system_style"])]
    blocks.sort()
    if not blocks:
        return []

    def style_at(t):
        cur = None
        for onset, s in blocks:
            if onset <= t:
                cur = s
            else:
                break
        return cur

    rows = []
    for scope in SCOPES:
        xc = f"{scope}.window_close_mouse.x"
        if xc not in cols:
            continue
        yc  = f"{scope}.window_close_mouse.y"
        lc  = f"{scope}.window_close_mouse.leftButton"
        tc  = f"{scope}.window_close_mouse.time"
        if not all(c in cols for c in (yc, lc, tc)):
            continue

        sub = df[df[xc].notna()]
        for _, r in sub.iterrows():
            xs, ys = parse_list(r[xc]), parse_list(r[yc])
            lbs, ts = parse_list(r[lc]), parse_list(r[tc])
            if not all(v is not None for v in (xs, ys, lbs, ts)):
                continue
            f = close_event_features(xs, ys, lbs, ts)
            if f is None:
                continue

            onset = pd.to_numeric(r.get("window_close.started"), errors="coerce")
            if not np.isfinite(onset):
                onset = pd.to_numeric(r.get("window_close_mouse.started"),
                                      errors="coerce")
            if not np.isfinite(onset):
                continue
            style = style_at(float(onset))
            if style not in ("windows", "mac"):
                continue

            hab = habitual[pid]
            f.update({
                "participant": f"P{pid:03d}", "pid": pid, "scope": scope,
                "onset_s": float(onset), "presented_style": style,
                "habitual_os": hab,
                "condition": "match" if style == hab else "mismatch",
            })
            # Windows close is top-RIGHT, Mac top-LEFT. Moving away from
            # the correct side on the first displacement = prepotent
            # response toward the habitual corner.
            correct = +1.0 if style == "windows" else -1.0
            f["moved_wrong_way"] = float(
                f["initial_x_sign"] != 0 and f["initial_x_sign"] != correct)
            rows.append(f)
    return rows


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 76)
    print("SENSE-42 : OS LAYOUT MATCH vs MISMATCH")
    print("=" * 76)
    print("Randomised within-subject UI manipulation. No EEG, no clock")
    print("alignment -- immune to the ~104 s misalignment bug.\n")

    habitual = load_habitual_os()
    n_win = sum(1 for v in habitual.values() if v == "windows")
    n_mac = sum(1 for v in habitual.values() if v == "mac")
    print(f"Exclusive-OS participants: {len(habitual)}  "
          f"({n_win} windows, {n_mac} mac)")
    print("Dual-OS users excluded (mismatch undefined).\n")

    all_rows = []
    for f in sorted(glob.glob(os.path.join(CSV_DIR, "*.csv"))):
        pid = int(os.path.basename(f).split("_")[0])
        if pid not in habitual:
            continue
        try:
            r = process_participant(f, pid, habitual)
        except Exception as e:
            print(f"  P{pid:03d} ERROR: {e}")
            continue
        if r:
            m = sum(1 for x in r if x["condition"] == "match")
            print(f"  P{pid:03d} [{habitual[pid]:7s}] {len(r):3d} closes "
                  f"({m} match / {len(r) - m} mismatch)")
            all_rows.extend(r)

    if not all_rows:
        print("\nNo events extracted.")
        return

    ev = pd.DataFrame(all_rows)
    ev.to_csv(OUT_EV, index=False)
    print(f"\nTotal close events: {len(ev)} "
          f"from {ev.participant.nunique()} participants")
    print(f"  match    {(ev.condition == 'match').sum()}")
    print(f"  mismatch {(ev.condition == 'mismatch').sum()}")

    # ── paired within-subject test ───────────────────────────────────
    print("\n" + "=" * 76)
    print("PAIRED TEST (each participant contributes both conditions)")
    print("=" * 76)
    print(f"\n{'measure':20s} {'match':>9s} {'mismatch':>9s} {'diff':>9s} "
          f"{'t':>7s} {'p':>8s} {'d':>7s}  n")
    print("-" * 76)

    results = {}
    for meas in MEASURES:
        if meas not in ev.columns:
            continue
        pm, pmm = [], []
        for pid, g in ev.groupby("participant"):
            a = g.loc[g.condition == "match", meas].dropna()
            b = g.loc[g.condition == "mismatch", meas].dropna()
            if len(a) >= 3 and len(b) >= 3:
                pm.append(a.mean()); pmm.append(b.mean())
        if len(pm) < 8:
            print(f"{meas:20s}  too few paired participants ({len(pm)})")
            continue
        pm, pmm = np.array(pm), np.array(pmm)
        d = pmm - pm
        t, p = stats.ttest_rel(pmm, pm)
        # Cohen's d for paired samples
        dz = float(d.mean() / (d.std(ddof=1) + 1e-12))
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{meas:20s} {pm.mean():9.4f} {pmm.mean():9.4f} "
              f"{d.mean():+9.4f} {t:7.2f} {p:8.4f} {dz:+7.2f}  {len(pm)}{star}")
        results[meas] = {"match": float(pm.mean()), "mismatch": float(pmm.mean()),
                         "diff": float(d.mean()), "t": float(t), "p": float(p),
                         "cohens_dz": dz, "n_participants": int(len(pm))}

    # ── the prepotent-response probe ─────────────────────────────────
    print("\n" + "=" * 76)
    print("PREPOTENT RESPONSE: first mouse movement toward the WRONG corner")
    print("=" * 76)
    print("Windows close = top-RIGHT, Mac close = top-LEFT.")
    print("If habit drives the first movement, mismatch blocks should show")
    print("more initial movements toward the habitual (now wrong) side.\n")
    if "moved_wrong_way" in ev.columns:
        ct = ev.groupby("condition")["moved_wrong_way"].agg(["mean", "count"])
        print(ct.to_string())
        a = ev.loc[ev.condition == "match", "moved_wrong_way"].dropna()
        b = ev.loc[ev.condition == "mismatch", "moved_wrong_way"].dropna()
        if len(a) > 30 and len(b) > 30:
            chi = stats.chi2_contingency([[a.sum(), len(a) - a.sum()],
                                          [b.sum(), len(b) - b.sum()]])
            print(f"\nchi2 = {chi[0]:.3f}   p = {chi[1]:.4f}")
            results["wrong_way_chi2"] = {"chi2": float(chi[0]), "p": float(chi[1]),
                                         "match_rate": float(a.mean()),
                                         "mismatch_rate": float(b.mean())}

    # ── per task scope ───────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("BY TASK SCOPE (close_time_s)")
    print("=" * 76)
    print(f"{'scope':32s} {'match':>9s} {'mismatch':>9s} {'diff':>9s}  n")
    print("-" * 70)
    for scope, g in ev.groupby("scope"):
        a = g.loc[g.condition == "match", "close_time_s"].dropna()
        b = g.loc[g.condition == "mismatch", "close_time_s"].dropna()
        if len(a) > 10 and len(b) > 10:
            print(f"{scope:32s} {a.mean():9.4f} {b.mean():9.4f} "
                  f"{b.mean() - a.mean():+9.4f}  {len(a) + len(b)}")

    # ── interpretation ───────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("INTERPRETATION")
    print("=" * 76)
    sig = [m for m, r in results.items()
           if isinstance(r, dict) and r.get("p", 1) < 0.05]
    if sig:
        print(f"\nSignificant at p<0.05: {sig}")
        print("\nA reliable mismatch cost means the manipulation induces")
        print("measurable extraneous cognitive load. That gives a within-")
        print("subject, randomised load contrast -- structurally the same")
        print("thing that made SWELL-KW work (CV r=0.58), which we had")
        print("concluded SENSE-42 lacked.")
        print("\nNext step: use match/mismatch as the label and test whether")
        print("HCI features alone can classify it. That is a well-posed")
        print("proxy question with a randomised ground truth.")
    else:
        print("\nNo measure reached p<0.05.")
        print("\nPossible reasons, in order of plausibility:")
        print("  1. Close buttons may be large and easy targets, so the")
        print("     motor cost is real but too small to detect.")
        print("  2. Participants adapt within a block -- the cost may be")
        print("     confined to the FIRST close after each style switch.")
        print("     Re-run restricted to first-close-per-block.")
        print("  3. The layout difference may not be salient enough to")
        print("     engage a prepotent response at all.")
        print("\nWorth checking 2 before concluding anything -- adaptation")
        print("within ~5 closes per block would mask a genuine effect.")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {OUT_EV}")
    print(f"       {OUT_JSON}")


if __name__ == "__main__":
    main()
