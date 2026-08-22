"""
coglab_keyboard_adapter.py
=============================
Layer 1 adapter: Cog Lab's raw keyboard CSV -> the pre-embedder's exact
expected input format (list of {code, hold, ikl} dicts, consumed by
normalize_sequence() in pre_embedders/keyboard/preprocess.py).

STRUCTURAL GAP -- HOLD DURATION IS UNAVAILABLE, NOT JUST MISSING
----------------------------------------------------------------------
Confirmed directly: `cut -d',' -f2 D3_S10_keyboard.csv | sort -u` shows
ONLY "Keyboard Keydown" -- no keyup/keyrelease event exists anywhere in
Cog Lab's export. hold = release.timestamp - press.timestamp cannot be
computed; there is no release timestamp to use, ever, for this dataset.

MECHANICAL CONSEQUENCE (verified directly against the encoder's own
z_score() implementation): setting hold to a constant (0) for every
event means normalize_sequence()'s zero-variance guard
(`if std < 1e-6: return np.zeros_like(x)`) fires automatically. The
encoder receives [0.0, ikl_norm, code_norm] for every keystroke -- a
well-defined degenerate input, not NaN, not a crash. It runs one input
channel blind for this dataset only. This is a real compromise, stated
plainly, not hidden inside the adapter.

KEY CODE MAPPING
-------------------
Cog Lab's key_code column is a raw JS KeyboardEvent.keyCode integer.
The encoder's own SPECIAL_KEY_MAP (space=32, enter=13, backspace=8,
shift=16, ctrl=17, alt=18, tab=9, ...) is built from the SAME standard
JS keyCode space -- confirmed by direct comparison, not assumed. So
Cog Lab's key_code maps DIRECTLY (mod 256, matching encode_key()'s own
convention) without needing a name-based lookup at all.

IKL
-----
Computed directly from consecutive "Keyboard Keydown" timestamps within
the same session -- this is exactly what Record Tool's own interval_ms
represents, just computed here instead of arriving pre-computed.

Run: adapt_coglab_keyboard(csv_path) -> list[dict] ready for
pre_embedders.keyboard.preprocess.normalize_sequence()
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def adapt_coglab_keyboard(csv_path: str) -> list[dict]:
    """
    Returns a list of {code, hold, ikl} dicts, sorted by timestamp,
    matching the exact format normalize_sequence() expects.

    hold is ALWAYS 0.0 -- confirmed structural gap, not a bug. See
    module docstring. This is passed through explicitly rather than
    silently, so any downstream caller can see exactly what happened.
    """
    df = pd.read_csv(csv_path)

    # confirmed: only "Keyboard Keydown" exists in this export
    df = df[df["type"] == "Keyboard Keydown"].copy()
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < 2:
        return []

    # IKL: consecutive keydown timestamp deltas, in ms (matches Record
    # Tool's interval_ms convention: time since the PREVIOUS keydown)
    ts = df["time"].to_numpy(float)
    ikl_ms = np.diff(ts, prepend=ts[0]) * 1000.0
    ikl_ms[0] = 0.0   # first keystroke has no predecessor

    codes = df["key_code"].to_numpy(int)
    codes = np.mod(codes, 256)   # match encode_key()'s own convention

    events = []
    for i in range(len(df)):
        events.append({
            "code": int(codes[i]),
            "hold": 0.0,          # STRUCTURAL GAP -- see docstring.
                                  # Will z-score to a flat 0.0 channel
                                  # via the encoder's own zero-variance
                                  # guard, not a fabricated value.
            "ikl":  float(ikl_ms[i]),
        })
    return events


def adapt_coglab_keyboard_report(csv_path: str) -> dict:
    """Diagnostic wrapper -- reports what was actually extracted, so
    the gap is visible in logs rather than silently baked into output."""
    events = adapt_coglab_keyboard(csv_path)
    return {
        "n_events": len(events),
        "hold_channel_status": "ALWAYS 0.0 -- Cog Lab has no keyup events, "
                               "hold duration is structurally unavailable "
                               "for this dataset (confirmed via direct "
                               "check, not inferred)",
        "code_range": (min(e["code"] for e in events),
                      max(e["code"] for e in events)) if events else None,
        "ikl_mean_ms": float(np.mean([e["ikl"] for e in events[1:]]))
                      if len(events) > 1 else None,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python coglab_keyboard_adapter.py <path_to_D3_Sxx_keyboard.csv>")
        sys.exit(1)
    report = adapt_coglab_keyboard_report(path)
    print(f"Adapted {report['n_events']} keystroke events")
    print(f"  {report['hold_channel_status']}")
    print(f"  code range: {report['code_range']}")
    print(f"  mean IKL: {report['ikl_mean_ms']:.1f} ms")


def ikl_distribution(csv_path: str) -> dict:
    """
    Diagnostic: is the high mean IKL driven by rare long gaps (reading/
    thinking pauses between typing bursts -- expected for a learning
    task) or is it uniformly elevated (suggesting our diff-based
    computation doesn't match Record Tool's actual convention)?
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["type"] == "Keyboard Keydown"].sort_values("time")
    ts = df["time"].to_numpy(float)
    if len(ts) < 3:
        return {}
    ikl_ms = np.diff(ts) * 1000.0

    pct = np.percentile(ikl_ms, [10, 25, 50, 75, 90, 95, 99])
    return {
        "n": len(ikl_ms),
        "p10": pct[0], "p25": pct[1], "median": pct[2],
        "p75": pct[3], "p90": pct[4], "p95": pct[5], "p99": pct[6],
        "mean": float(ikl_ms.mean()),
        "pct_under_500ms": float((ikl_ms < 500).mean()),
        "pct_over_5000ms": float((ikl_ms > 5000).mean()),
    }
