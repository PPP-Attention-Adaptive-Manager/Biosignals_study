"""
coglab_mouse_adapter.py
==========================
Layer 1 adapter: Cog Lab's raw mouse CSV -> the pre-embedder's exact
expected MouseEvent format:
    MouseEvent(timestamp, x, y, dx, dy, speed, event_type, button)

COORDINATE CHOICE -- screen_x/screen_y, NOT page_x/page_y
----------------------------------------------------------------
Record Tool captures mouse position at the OS level (libinput), across
native apps -- it has no concept of "page-relative" DOM coordinates.
Cog Lab's page_x/page_y are DOM/webpage-relative (meaningful only
because Cog Lab's collector is a browser extension); screen_x/screen_y
are global screen-absolute, the correct structural match.

dx, dy, speed -- COMPUTED, not present in Cog Lab's export
----------------------------------------------------------------
Record Tool's `speed` is documented as "pre-computed by libinput" --
Cog Lab has no equivalent. Computed here as simple consecutive-sample
differences: dx = x[i]-x[i-1], dy likewise, speed = sqrt(dx^2+dy^2)/dt.
This will NOT be identical to whatever smoothing libinput applies
internally, but it's a legitimate, direct approximation -- not a
structural gap like the keyboard's hold-duration issue, which had no
computable substitute at all.

EVENT TYPE MAPPING -- VERIFIED, NOT ASSUMED
----------------------------------------------
Only "Mouse Down" and "Mouse Move" were visible in the original 2-line
sample. This adapter does NOT assume the rest of the mapping -- run
diagnose_mouse_event_types() first (see bottom of this file) to confirm
what actually appears in the full files before trusting MAPPED_TYPES
below. Any type NOT in the map is passed through with a WARNING logged,
never silently dropped or silently misclassified.

BUTTON MAPPING
----------------
Cog Lab's `button` column follows standard JS MouseEvent.button coding
(0=left, 1=middle, 2=right) for click events; for non-click events
(Mouse Move) button is set to None, matching the encoder's own
Optional[str] contract.

Run: adapt_coglab_mouse(csv_path) -> list[MouseEvent-shaped dicts]
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from collections import Counter

# Confirmed only for "Mouse Down"/"Mouse Move" from the original sample.
# Anything else encountered gets flagged, not guessed.
MAPPED_TYPES = {
    "Mouse Down":  "mouse_press",
    "Mouse Up":    "mouse_release",
    "Mouse Move":  "mouse_move",
    "Mouse Wheel": "scroll",
    "Mouse Scroll": "scroll",
}

# CONFIRMED via diagnose_mouse_event_semantics.py (Part 1, run on the
# full 16-subject population): 70-91% of "Left Click" events land
# within 500ms of an existing "Mouse Down" for the SAME subject,
# consistently across every single subject checked. This is a
# synthesized "completed click" event Cog Lab's logger fires ON TOP OF
# the raw press/release pair (browsers do this: mousedown -> mouseup ->
# click, three distinct DOM events for one physical click). Mapping
# these to a NEW mouse_press would double-count presses and directly
# corrupt the encoder's click_rate feature (E1 in MouseWindowStats).
# "Right Click" is the same pattern for the secondary button.
# "Page Scroll" is dropped for the analogous reason -- redundant with
# "Mouse Wheel", which already captures the physical scroll input; Page
# Scroll is the resulting DOM scroll-position event, which can fire at
# a different rate (e.g. during momentum/smooth-scroll) and would
# distort scroll_velocity/scroll_reversal_count if treated as a
# separate physical action.
DROPPED_TYPES = {"Left Click", "Right Click", "Page Scroll"}

# CONFIRMED via diagnose_mouse_event_semantics.py (Part 2): Mouse Move
# sampling rate is TIGHT and consistent across every subject (median
# dt = 16.67-16.83ms, ~60Hz, standard vsync-throttled browser mousemove
# -- NOT the cause of the wild speed variance seen in the batch run).
# The actual cause is real large single-sample coordinate jumps (S5's
# p95=485,921 px/s implies a ~7,000px jump in one ~14ms sample --
# consistent with a multi-monitor setup where the cursor crossing
# screens produces a discontinuous screen_x jump, not fast physical
# movement). This is a CLIPPING decision, not a root-cause fix -- the
# underlying mechanism (multi-monitor jumps vs a logging glitch) was
# not further isolated. 5000 px/s is generous for even a fast real
# flick on a single monitor; anything above it is treated as a jump
# artifact and capped, not deleted (the event itself is kept, only its
# speed value is bounded).
SPEED_CLIP_PX_S = 5000.0

BUTTON_MAP = {0: "left", 1: "middle", 2: "right"}


def diagnose_mouse_event_types(csv_path: str) -> dict:
    """Run this FIRST on real data -- reports every distinct `type`
    value actually present, and flags any not in MAPPED_TYPES."""
    df = pd.read_csv(csv_path)
    counts = Counter(df["type"])
    unmapped = [t for t in counts if t not in MAPPED_TYPES]
    return {"counts": dict(counts), "unmapped_types": unmapped}


def adapt_coglab_mouse(csv_path: str) -> list[dict]:
    """
    Returns a list of dicts matching MouseEvent's fields, sorted by
    timestamp. DROPPED_TYPES rows are filtered out BEFORE computing
    dx/dy/dt/speed, so the remaining event stream stays continuous and
    diffs are only ever computed between real physical mouse actions --
    filtering after diff computation would corrupt the delta for
    whichever real event happened to follow a dropped one.

    Any event `type` not in MAPPED_TYPES and not in DROPPED_TYPES is
    kept with its RAW type string (prefixed "UNMAPPED:") -- still
    surfaced, never silently dropped or silently misclassified.
    """
    df = pd.read_csv(csv_path)
    df = df[~df["type"].isin(DROPPED_TYPES)].copy()
    df = df.sort_values("time").reset_index(drop=True)

    ts = df["time"].to_numpy(float)
    x  = df["screen_x"].to_numpy(float)
    y  = df["screen_y"].to_numpy(float)

    dt = np.diff(ts, prepend=ts[0])
    dt[0] = np.nan
    dt = np.where(dt < 1e-6, np.nan, dt)   # guard div-by-zero / dup timestamps

    dx = np.diff(x, prepend=x[0]); dx[0] = 0.0
    dy = np.diff(y, prepend=y[0]); dy[0] = 0.0

    speed = np.sqrt(dx**2 + dy**2) / dt
    speed = np.nan_to_num(speed, nan=0.0)
    speed = np.clip(speed, 0.0, SPEED_CLIP_PX_S)   # see SPEED_CLIP_PX_S docstring

    events = []
    unmapped_seen = set()
    for i in range(len(df)):
        raw_type = df["type"].iloc[i]
        mapped = MAPPED_TYPES.get(raw_type)
        if mapped is None:
            unmapped_seen.add(raw_type)
            mapped = f"UNMAPPED:{raw_type}"

        btn_raw = df["button"].iloc[i] if "button" in df.columns else None
        button = (BUTTON_MAP.get(int(btn_raw))
                 if mapped == "mouse_press" and pd.notna(btn_raw) else None)

        events.append({
            "timestamp":  float(ts[i]),
            "x": float(x[i]), "y": float(y[i]),
            "dx": float(dx[i]), "dy": float(dy[i]),
            "speed": float(speed[i]),
            "event_type": mapped,
            "button": button,
        })

    if unmapped_seen:
        print(f"  WARNING: unmapped event types encountered: {unmapped_seen} "
              f"-- these were kept as 'UNMAPPED:<type>', review before "
              f"feeding to the encoder")

    return events


def adapt_coglab_mouse_report(csv_path: str) -> dict:
    events = adapt_coglab_mouse(csv_path)
    type_counts = Counter(e["event_type"] for e in events)
    speeds = [e["speed"] for e in events if e["speed"] > 0]
    n_clipped = sum(1 for s in speeds if s >= SPEED_CLIP_PX_S)
    return {
        "n_events": len(events),
        "type_counts": dict(type_counts),
        "unmapped_types": [t for t in type_counts if t.startswith("UNMAPPED:")],
        "speed_mean_px_s": float(np.mean(speeds)) if speeds else None,
        "speed_p95_px_s": float(np.percentile(speeds, 95)) if speeds else None,
        "pct_speed_clipped": 100 * n_clipped / len(speeds) if speeds else 0.0,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python coglab_mouse_adapter.py <path_to_D3_Sxx_mouse.csv>")
        sys.exit(1)

    print("=== Event type diagnostic (run first, before trusting the mapping) ===")
    diag = diagnose_mouse_event_types(path)
    for t, n in sorted(diag["counts"].items(), key=lambda kv: -kv[1]):
        flag = "  <- UNMAPPED, check MAPPED_TYPES" if t in diag["unmapped_types"] else ""
        print(f"  {t!r:20s} {n:8d}{flag}")

    print("\n=== Adapter report ===")
    report = adapt_coglab_mouse_report(path)
    print(f"n_events: {report['n_events']}")
    print(f"type breakdown: {report['type_counts']}")
    if report["speed_mean_px_s"] is not None:
        print(f"speed: mean={report['speed_mean_px_s']:.1f} px/s  "
              f"p95={report['speed_p95_px_s']:.1f} px/s  "
              f"({report['pct_speed_clipped']:.1f}% of samples hit the "
              f"{SPEED_CLIP_PX_S:.0f} px/s clip ceiling)")
