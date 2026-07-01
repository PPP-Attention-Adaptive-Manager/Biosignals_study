"""
swell_style_counts.py
======================
Computes the 16 SWELL-KW-equivalent count features from Cog Lab raw
mouse/keyboard events. 2 of SWELL-KW's 18 columns (SnAppChange,
SnTabfocusChange) are structurally absent from Cog Lab — this dataset
has a single passive task with no app-switching at all (confirmed in
earlier analysis: "SWELL-KW has 3 deliberately varied stress conditions
and includes app-switching which Cog Lab had zero of"). Those two are
returned as np.nan, never as 0 — 0 would falsely claim "zero switches
observed" when the truth is "this signal doesn't exist for this task."

TIMESTAMP CONTRACT — read before wiring this in
-------------------------------------------------
This function does NOT do any windowing or timestamp alignment itself.
It expects mouse_window / key_window to already be pre-sliced to the
exact window your existing biosignal extractor used — the same
windowing loop already validated when proxy_cache/*.npz was built.

Call this from INSIDE that existing loop, where mouse_window/key_window
are already in scope. Do not build a second independent windowing
system — if X and Y come from two different windowing passes, they
will silently misalign even if both look correct in isolation.

Cog Lab keyboard CSV is keydown-only (confirmed across all 17 subjects,
no keyup events exist) — dwell/hold time is unrecoverable, but none of
the 18 SWELL columns need it, so this is a non-issue here.
"""

import numpy as np
import pandas as pd

SWELL_COL_ORDER = [
    "SnKeyStrokes", "SnChars", "SnSpecialKeys", "SnDirectionKeys",
    "SnErrorKeys", "SnShortcutKeys", "SnSpaces",
    "CharactersRatio", "ErrorKeyRatio",
    "SnLeftClicked", "SnRightClicked", "SnDoubleClicked",
    "SnWheel", "SnDragged", "SnMouseDistance", "SnMouseAct",
    "SnAppChange", "SnTabfocusChange",
]

SPECIAL_KEYS = {8, 9, 13, 27, 32, 45, 46}   # backspace, tab, enter, esc, space, ins, del
DIRECTION_KEYS = {37, 38, 39, 40}            # arrow keys


def compute_swell_counts(mouse_window: pd.DataFrame, key_window: pd.DataFrame) -> dict:
    """
    mouse_window: rows already filtered to one window. Expected columns
                  ['x', 'y', 'button', 'timestamp'] — rename below if
                  your mouse.csv headers differ.
    key_window:   rows already filtered to one window. Expected columns
                  ['keycode', 'timestamp'] — rename below if different.

    Returns a dict with the 16 computable SWELL-style features.
    SnAppChange / SnTabfocusChange are always np.nan for Cog Lab.
    """
    # ---- keyboard ----
    total_keys = len(key_window)
    if total_keys > 0:
        codes = key_window["keycode"].to_numpy()
        backspace = int((codes == 8).sum())
        special   = int(np.isin(codes, list(SPECIAL_KEYS)).sum())
        direction = int(np.isin(codes, list(DIRECTION_KEYS)).sum())
        shortcut  = int(((codes < 32) & ~np.isin(codes, list(SPECIAL_KEYS))).sum())
        spaces    = int((codes == 32).sum())
        printable = int(((codes > 32) & (codes < 127) &
                         ~np.isin(codes, list(SPECIAL_KEYS))).sum())
    else:
        backspace = special = direction = shortcut = spaces = printable = 0

    chars_ratio = printable / max(total_keys, 1)
    error_ratio = backspace / max(total_keys, 1)

    # ---- mouse ----
    if len(mouse_window) > 0 and "button" in mouse_window.columns:
        btn = mouse_window["button"]
        left_clicks   = int((btn == "left").sum())
        right_clicks  = int((btn == "right").sum())
        double_clicks = int((btn == "double").sum())
        scrolls       = int((btn == "scroll").sum())
        drags         = int((btn == "drag").sum())
    else:
        left_clicks = right_clicks = double_clicks = scrolls = drags = 0

    if len(mouse_window) > 1:
        xy = mouse_window[["x", "y"]].to_numpy(dtype=float)
        deltas = np.diff(xy, axis=0)
        distance = float(np.sqrt((deltas ** 2).sum(axis=1)).sum())
        t = mouse_window["timestamp"].to_numpy(dtype=float)
        gaps = np.diff(t)
        active_seconds = float((gaps < 2.0).sum())
        window_seconds = max(t[-1] - t[0], 1.0)
        mouse_act = active_seconds / window_seconds
    else:
        distance = 0.0
        mouse_act = 0.0

    return {
        "SnKeyStrokes": total_keys,
        "SnChars": printable,
        "SnSpecialKeys": special,
        "SnDirectionKeys": direction,
        "SnErrorKeys": backspace,
        "SnShortcutKeys": shortcut,
        "SnSpaces": spaces,
        "CharactersRatio": chars_ratio,
        "ErrorKeyRatio": error_ratio,
        "SnLeftClicked": left_clicks,
        "SnRightClicked": right_clicks,
        "SnDoubleClicked": double_clicks,
        "SnWheel": scrolls,
        "SnDragged": drags,
        "SnMouseDistance": distance,
        "SnMouseAct": mouse_act,
        "SnAppChange": np.nan,        # structurally absent in Cog Lab
        "SnTabfocusChange": np.nan,   # structurally absent in Cog Lab
    }


def counts_to_array(counts: dict) -> np.ndarray:
    """Convert dict to ordered array matching SWELL_COL_ORDER."""
    return np.array([counts[c] for c in SWELL_COL_ORDER], dtype=np.float32)
