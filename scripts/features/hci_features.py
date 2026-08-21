"""
biosignal_proxy/hci_features.py
================================
Shared HCI feature extractor for the biosignal proxy.

Design rules (locked):
  * Every feature is recomputed from PRIMITIVES — time, position (x,y), key code,
    event_type — so definitions are byte-identical across Cog Lab D3 and
    AAM-collected data. Precomputed columns that exist in only one source
    (e.g. the AAM tool's libinput `speed`) are IGNORED and recomputed.
  * Keyboard is KEYDOWN-ONLY compatible: Cog Lab logs no key-release, so there
    are NO dwell/hold features. Only flight-time (IKI) / rate / rhythm / pauses.
  * OS autorepeat is collapsed: consecutive same-key events less than
    AUTOREPEAT_MS apart are merged into the initiating press. This removes the
    ~45ms OS repeat bursts while leaving genuine human re-presses (>=150ms) intact.

Canonical key code is consistent across sources:
  Cog Lab gives JS keyCodes directly (a=65, space=32, backspace=8).
  AAM gives key NAMES which we map to the same codes, so 'a'->65 == Cog Lab 65.

Output: per-window feature vector  [mouse_features | keyboard_features].
Window: WINDOW_S / STRIDE_S (default 30 / 15), fully parameterized.

--------------------------------------------------------------------------------
NEW (Phase 0 — SWELL-KW gate test): extract_session() now ALSO returns a second
per-window feature block — SWELL-KW-equivalent counts (SWELL_COUNT_NAMES) —
computed inside the SAME windowing loop as the original kinematic features.
This guarantees the two feature sets share identical window boundaries by
construction, with zero extra alignment work.

CONFIRMED — click-type event vocabulary (verified against real data):
  df['type'].unique() == ['Mouse Down','Mouse Move','Mouse Up','Left Click',
                           'Mouse Wheel','Page Scroll']
Right-click, double-click, and drag are NOT logged event types in Cog Lab —
they are not "zero", they are structurally unmeasurable, so
SnRightClicked/SnDoubleClicked/SnDragged are always np.nan here, matching
the SnAppChange/SnTabfocusChange treatment.

OPEN QUESTION — possible double-count between 'Mouse Down' and 'Left Click':
load_mouse()'s existing is_click logic (unchanged, pre-existing) treats
both 'Mouse Down' and 'Left Click' as a click event:
    is_click = etype in ("Mouse Down", "Left Click")
If a single physical click ever produces BOTH a 'Mouse Down' row AND a
separate 'Left Click' row (e.g. one logged by a low-level hook, one by a
higher-level synthesized event), SnLeftClicked here would count it twice.
Verify before trusting SnLeftClicked or ms_click_rate:
    df = pd.read_csv("D3_S1_mouse.csv")
    # check whether Mouse Down and Left Click rows share timestamps:
    print(df[df["type"].isin(["Mouse Down","Left Click"])]
          .sort_values("time").head(20))
If they're always paired at the same timestamp, divide left-click counts
by 2, or pick only one label, before using SnLeftClicked downstream.
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from __future__ import annotations   # FIX — same Python 3.8 compatibility
                                       # issue as build_phase0_dataset.py:
                                       # tuple[...] generic syntax below
                                       # needs this on Python < 3.9.

import json
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config
WINDOW_S        = 30.0
STRIDE_S        = 15.0
AUTOREPEAT_MS   = 90.0     # same-key gap below this == OS autorepeat -> collapse
IDLE_SPEED_PX_S = 5.0      # below this speed == idle (matches MouseWindowStats)
SPATIAL_BINS    = 20

# JS keyCode sets for "printable"
_PRINTABLE_CODES = set([32]) | set(range(48, 58)) | set(range(65, 91)) \
                 | set(range(96, 106)) | set(range(186, 223))
_BACKSPACE = 8

# AAM key-name -> JS keyCode (aligns with Cog Lab key_code)
_SPECIAL = {
    "space": 32, "enter": 13, "return": 13, "backspace": 8, "shift": 16,
    "ctrl": 17, "control": 17, "alt": 18, "tab": 9, "escape": 27, "esc": 27,
    "delete": 46, "caps_lock": 20, "capslock": 20,
    "up": 38, "down": 40, "left": 37, "right": 39,
}

def _key_to_code(key) -> int:
    """Map a key (AAM name string OR Cog Lab numeric) to a canonical JS keyCode."""
    if isinstance(key, (int, np.integer)):
        return int(key)
    s = str(key).strip().lower()
    if s.isdigit():
        return int(s)
    if s in _SPECIAL:
        return _SPECIAL[s]
    if len(s) == 1:
        return ord(s.upper())   # 'a' -> 65, matches Cog Lab
    return 0


# ============================================================================ loaders
def load_keyboard(path: str, source: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (times, codes) for keydown events, autorepeat-collapsed, time-sorted.

    source = 'coglab' : columns time, key_code   (keydown-only)
    source = 'aam'    : columns timestamp, event_type, key  (use key_press only)
    """
    df = pd.read_csv(path)
    if source == "coglab":
        t = df["time"].to_numpy(float)
        codes = df["key_code"].to_numpy()
    elif source == "aam":
        df = df[df["event_type"] == "key_press"]
        t = df["timestamp"].to_numpy(float)
        codes = df["key"].to_numpy()
    else:
        raise ValueError(source)

    order = np.argsort(t)
    t, codes = t[order], codes[order]
    codes = np.array([_key_to_code(c) for c in codes], dtype=int)

    # ---- autorepeat collapse: drop same-code events < AUTOREPEAT_MS after prev row
    keep = np.ones(len(t), dtype=bool)
    for i in range(1, len(t)):
        if codes[i] == codes[i - 1] and (t[i] - t[i - 1]) * 1000.0 < AUTOREPEAT_MS:
            keep[i] = False
    return t[keep], codes[keep]


def load_mouse(path: str, source: str) -> dict:
    """
    Return canonical mouse arrays with RECOMPUTED kinematics (source speed ignored).
    Keys: t, x, y, dx, dy, speed, is_click, etype  (all length N, time-sorted).

    NEW: `etype` (the raw event-type string, e.g. "Mouse Down", "Left Click")
    is now preserved and sorted alongside the rest, so window-level click-type
    breakdown (left/right/double/scroll/drag) is possible downstream. It was
    previously discarded right after computing is_click.
    """
    df = pd.read_csv(path)
    if source == "coglab":
        t = df["time"].to_numpy(float)
        x = df["screen_x"].to_numpy(float)
        y = df["screen_y"].to_numpy(float)
        etype = df["type"].astype(str).to_numpy()
        is_click = np.array([s in ("Mouse Down", "Left Click") for s in etype])
    elif source == "aam":
        t = df["timestamp"].to_numpy(float)
        x = df["x"].to_numpy(float)
        y = df["y"].to_numpy(float)
        etype = df["event_type"].astype(str).to_numpy()
        is_click = etype == "mouse_press"
    else:
        raise ValueError(source)

    order = np.argsort(t)
    t, x, y, is_click, etype = t[order], x[order], y[order], is_click[order], etype[order]  # CHANGED — etype added

    n  = len(t)
    dx = np.zeros(n); dy = np.zeros(n); speed = np.zeros(n)
    if n > 1:
        dt = np.diff(t); dt = np.where(dt < 1e-6, 1e-6, dt)
        dx[1:] = np.diff(x)
        dy[1:] = np.diff(y)
        speed[1:] = np.sqrt(dx[1:] ** 2 + dy[1:] ** 2) / dt
    return dict(t=t, x=x, y=y, dx=dx, dy=dy, speed=speed, is_click=is_click, etype=etype)  # CHANGED — etype added


# ============================================================ keyboard window features
KB_FEATURES = [
    "kb_rate", "kb_iki_mean", "kb_iki_std", "kb_iki_cv",
    "kb_burstiness", "kb_pause_count", "kb_pause_ratio",
    "kb_backspace_ratio", "kb_printable_ratio",
]

def keyboard_window_features(t: np.ndarray, codes: np.ndarray, win_dur: float) -> np.ndarray:
    f = np.zeros(len(KB_FEATURES), dtype=np.float32)
    if len(t) < 2 or win_dur <= 0:
        return f
    iki = np.diff(t) * 1000.0                              # ms, keydown->keydown (flight)
    mu, sd = float(iki.mean()), float(iki.std())
    f[0] = len(t) / win_dur                                # keystrokes / sec
    f[1] = mu
    f[2] = sd
    f[3] = sd / mu if mu > 1e-9 else 0.0                   # CV
    f[4] = (sd - mu) / (sd + mu) if (sd + mu) > 1e-9 else 0.0   # Goh-Barabasi burstiness
    pauses = iki > 1000.0                                  # >1s gap == pause
    f[5] = float(pauses.sum())
    f[6] = float(iki[pauses].sum() / (iki.sum() + 1e-9))   # fraction of time in pauses
    f[7] = float((codes == _BACKSPACE).mean())             # editing/error proxy
    f[8] = float(np.isin(codes, list(_PRINTABLE_CODES)).mean())
    return f


# =============================================================== mouse window features
MOUSE_FEATURES = [
    "ms_speed_mean", "ms_speed_std", "ms_speed_max",
    "ms_path_efficiency", "ms_direction_reversals", "ms_submovements",
    "ms_curvature_mean", "ms_angle_delta_mean",
    "ms_idle_ratio", "ms_spatial_entropy", "ms_click_rate", "ms_convex_extent",
]

def mouse_window_features(m: dict, idx: np.ndarray, win_dur: float) -> np.ndarray:
    f = np.zeros(len(MOUSE_FEATURES), dtype=np.float32)
    if idx.sum() < 10 or win_dur <= 0:                     # MIN_EVENTS_FOR_WINDOW = 10
        return f
    t   = m["t"][idx]; x = m["x"][idx]; y = m["y"][idx]
    dx  = m["dx"][idx]; dy = m["dy"][idx]; speed = m["speed"][idx]
    is_click = m["is_click"][idx]

    f[0] = float(speed.mean()); f[1] = float(speed.std()); f[2] = float(speed.max())

    seg  = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    path = float(seg.sum())
    disp = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    f[3] = np.clip(disp / path, 0, 1) if path > 1e-9 else 1.0

    vx = dx; vy = dy
    f[4] = int((np.diff(np.sign(vx)) != 0).sum() + (np.diff(np.sign(vy)) != 0).sum())
    if len(speed) > 2:
        f[5] = int(((speed[1:-1] < speed[:-2]) & (speed[1:-1] < speed[2:])).sum())

    theta = np.arctan2(dy, dx)
    ang   = np.abs(np.diff(np.unwrap(theta)))
    f[7]  = float(ang.mean()) if len(ang) else 0.0
    # curvature ~ angle change per distance
    f[6]  = float((ang / (seg[:len(ang)] + 1e-6)).mean()) if len(ang) else 0.0

    f[8]  = float((speed < IDLE_SPEED_PX_S).mean())        # idle ratio
    H, _, _ = np.histogram2d(x, y, bins=SPATIAL_BINS)
    H = H[H > 0]
    f[9]  = float(-(H / H.sum() * np.log(H / H.sum())).sum()) if H.size else 0.0
    f[10] = float(is_click.sum() / win_dur)                # clicks / sec
    f[11] = float((x.max() - x.min()) * (y.max() - y.min()))   # bounding-box extent
    return f


# ===================================================== NEW: SWELL-KW-style window counts
SWELL_COUNT_NAMES = [
    "SnKeyStrokes", "SnChars", "SnSpecialKeys", "SnDirectionKeys",
    "SnErrorKeys", "SnShortcutKeys", "SnSpaces",
    "CharactersRatio", "ErrorKeyRatio",
    "SnLeftClicked", "SnRightClicked", "SnDoubleClicked",
    "SnWheel", "SnDragged", "SnMouseDistance", "SnMouseAct",
]
# NOTE: SWELL-KW has 18 columns total. SnAppChange / SnTabfocusChange are
# DELIBERATELY excluded here, not just zeroed — Cog Lab's single passive
# task has no app-switching signal at all (confirmed earlier: "SWELL-KW...
# includes app-switching which Cog Lab had zero of"). If you need to pad
# to exactly 18 columns to match SWELL-KW's column order elsewhere, append
# two np.nan columns for those, never 0.

_SPECIAL_KEYS_SWELL = {8, 9, 13, 27, 32, 45, 46}
_DIRECTION_KEYS_SWELL = {37, 38, 39, 40}

def swell_count_features(
    mt: np.ndarray, mx: np.ndarray, my: np.ndarray, metype: np.ndarray,
    kt_w: np.ndarray, kc_w: np.ndarray, win_dur: float
) -> np.ndarray:
    """
    Computes the 16 SWELL-KW-equivalent count features for one window.
    Called from inside extract_session()'s existing loop — mt/mx/my/metype
    and kt_w/kc_w are already sliced to this window by the caller.
    """
    f = np.zeros(len(SWELL_COUNT_NAMES), dtype=np.float32)
    if win_dur <= 0:
        return f

    # ---- keyboard ----
    n_keys = len(kt_w)
    if n_keys > 0:
        backspace = int((kc_w == 8).sum())
        special   = int(np.isin(kc_w, list(_SPECIAL_KEYS_SWELL)).sum())
        direction = int(np.isin(kc_w, list(_DIRECTION_KEYS_SWELL)).sum())
        shortcut  = int(((kc_w < 32) & ~np.isin(kc_w, list(_SPECIAL_KEYS_SWELL))).sum())
        spaces    = int((kc_w == 32).sum())
        printable = int(((kc_w > 32) & (kc_w < 127) &
                         ~np.isin(kc_w, list(_SPECIAL_KEYS_SWELL))).sum())
    else:
        backspace = special = direction = shortcut = spaces = printable = 0
    chars_ratio = printable / max(n_keys, 1)
    error_ratio = backspace / max(n_keys, 1)

    # ---- mouse: click-type breakdown from raw event-type strings ----
    # CONFIRMED against actual Cog Lab data (df['type'].unique() ==
    # ['Mouse Down','Mouse Move','Mouse Up','Left Click','Mouse Wheel',
    # 'Page Scroll']). Right-click, double-click, and drag DO NOT EXIST
    # as event types in this dataset — they are not "zero", they are
    # structurally unmeasurable, so they are np.nan, matching the
    # SnAppChange/SnTabfocusChange treatment elsewhere in this file.
    if len(metype) > 0:
        left  = int(np.sum(np.isin(metype, ["Mouse Down", "Left Click"])))
        wheel = int(np.sum(np.isin(metype, ["Mouse Wheel", "Page Scroll"])))
    else:
        left = wheel = 0
    right  = np.nan   # not a logged event type in Cog Lab
    double = np.nan   # not a logged event type in Cog Lab
    drag   = np.nan   # not a logged event type in Cog Lab

    if len(mt) > 1:
        dxy = np.sqrt(np.diff(mx) ** 2 + np.diff(my) ** 2)
        distance = float(dxy.sum())
        gaps = np.diff(mt)
        active_seconds = float((gaps < 2.0).sum())
        mouse_act = active_seconds / win_dur
    else:
        distance = 0.0
        mouse_act = 0.0

    f[:] = [n_keys, printable, special, direction, backspace, shortcut, spaces,
            chars_ratio, error_ratio,
            left, right, double, wheel, drag, distance, mouse_act]
    return f


# ===================================================================== session windows
FEATURE_NAMES = MOUSE_FEATURES + KB_FEATURES

def extract_session(
    mouse_path: str,
    kb_path: str,
    source: str,
    t_start: float | None = None,
    t_end: float | None = None,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide aligned windows over one session; return (X, swell_counts, starts).
      X            : (n_windows, len(FEATURE_NAMES))       float32 — original kinematic features
      swell_counts : (n_windows, len(SWELL_COUNT_NAMES))   float32 — NEW SWELL-KW-equivalent counts
      starts       : (n_windows,) window start timestamps

    t_start / t_end: crop to the common biosignal window (PB 'Task' span). If None,
    uses the overlap of the mouse+keyboard streams.

    CHANGED: now returns 3 values instead of 2. Any caller doing
    `X, starts = extract_session(...)` needs updating to
    `X, swell_counts, starts = extract_session(...)`.
    """
    m = load_mouse(mouse_path, source)
    kt, kc = load_keyboard(kb_path, source)

    lo = max(m["t"].min(), kt.min()) if t_start is None else t_start
    hi = min(m["t"].max(), kt.max()) if t_end   is None else t_end

    rows, swell_rows, starts = [], [], []
    ws = lo
    while ws + window_s <= hi:
        we = ws + window_s
        m_idx = (m["t"] >= ws) & (m["t"] < we)
        k_sel = (kt >= ws) & (kt < we)
        feats = np.concatenate([
            mouse_window_features(m, m_idx, window_s),
            keyboard_window_features(kt[k_sel], kc[k_sel], window_s),
        ])
        swell_feats = swell_count_features(                          # NEW
            m["t"][m_idx], m["x"][m_idx], m["y"][m_idx], m["etype"][m_idx],
            kt[k_sel], kc[k_sel], window_s,
        )
        rows.append(feats); swell_rows.append(swell_feats); starts.append(ws)
        ws += stride_s

    X = np.array(rows, dtype=np.float32) if rows else np.empty((0, len(FEATURE_NAMES)), np.float32)
    SC = (np.array(swell_rows, dtype=np.float32) if swell_rows
          else np.empty((0, len(SWELL_COUNT_NAMES)), np.float32))
    return X, SC, np.array(starts)


def per_user_zscore(X: np.ndarray) -> np.ndarray:
    """Per-subject z-score across windows (locked AAM normalization principle)."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    return (X - mu) / sd
