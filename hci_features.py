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
"""

from __future__ import annotations
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
    Keys: t, x, y, dx, dy, speed, is_click  (all length N, time-sorted).
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
    t, x, y, is_click = t[order], x[order], y[order], is_click[order]

    n  = len(t)
    dx = np.zeros(n); dy = np.zeros(n); speed = np.zeros(n)
    if n > 1:
        dt = np.diff(t); dt = np.where(dt < 1e-6, 1e-6, dt)
        dx[1:] = np.diff(x)
        dy[1:] = np.diff(y)
        speed[1:] = np.sqrt(dx[1:] ** 2 + dy[1:] ** 2) / dt
    return dict(t=t, x=x, y=y, dx=dx, dy=dy, speed=speed, is_click=is_click)


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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide aligned windows over one session; return (X, starts).
      X      : (n_windows, len(FEATURE_NAMES))  float32
      starts : (n_windows,) window start timestamps

    t_start / t_end: crop to the common biosignal window (PB 'Task' span). If None,
    uses the overlap of the mouse+keyboard streams.
    """
    m = load_mouse(mouse_path, source)
    kt, kc = load_keyboard(kb_path, source)

    lo = max(m["t"].min(), kt.min()) if t_start is None else t_start
    hi = min(m["t"].max(), kt.max()) if t_end   is None else t_end

    rows, starts = [], []
    ws = lo
    while ws + window_s <= hi:
        we = ws + window_s
        m_idx = (m["t"] >= ws) & (m["t"] < we)
        k_sel = (kt >= ws) & (kt < we)
        feats = np.concatenate([
            mouse_window_features(m, m_idx, window_s),
            keyboard_window_features(kt[k_sel], kc[k_sel], window_s),
        ])
        rows.append(feats); starts.append(ws)
        ws += stride_s

    X = np.array(rows, dtype=np.float32) if rows else np.empty((0, len(FEATURE_NAMES)), np.float32)
    return X, np.array(starts)


def per_user_zscore(X: np.ndarray) -> np.ndarray:
    """Per-subject z-score across windows (locked AAM normalization principle)."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    return (X - mu) / sd