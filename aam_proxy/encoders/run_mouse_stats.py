"""
run_mouse_stats.py
=====================
Computes the REAL 22-feature MouseWindowStats vector from Cog Lab
mouse data, using extract_window_stats() and its dependencies copied
VERBATIM from pre_embedders/mouse/mouse_encoder.py -- not
reimplemented, to guarantee an exact match with Record Tool's own
computation rather than a close approximation.

Deliberately BYPASSES the untrained neural components (MouseTCNEncoder,
PreClickSubsequenceEncoder, stats_mlp, fusion_proj) -- confirmed via
direct inspection that no saved checkpoint exists for the mouse
encoder anywhere (unlike keyboard's kb_encoder_lstm.pt), so those
layers would produce noise, not signal, if run as-is. The 22
handcrafted stats are pure computation, no learned weights involved --
fully usable today.

Windowing: SlidingWindowExtractor's own convention (5s windows, 1s
stride by default) -- adjustable via WINDOW_SEC/STRIDE_SEC below.

Run from: ~/biosignals_data/
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from scipy.stats import entropy as scipy_entropy
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapters"))
from coglab_mouse_adapter import adapt_coglab_mouse

BASE = os.path.expanduser("~/biosignals_data")
COG_LAB_DIR = os.path.join(BASE, "data", "cog_lab")
EXCLUDE = {"S2", "S17"}

WINDOW_SEC = 30.0   # matches the 30s window used everywhere else in
                    # this project's Cog Lab work, for consistency
                    # with acc_jerk/eeg_engagement's own windowing
STRIDE_SEC = 15.0   # 50% overlap

IDLE_SPEED_THRESHOLD = 5.0
SPATIAL_GRID_CELLS = 20
SAMPEN_M = 2
SAMPEN_R_FACTOR = 0.2
MIN_EVENTS_FOR_WINDOW = 10
PRE_CLICK_WINDOW_MS = 200.0


@dataclass
class MouseEvent:
    timestamp: float; x: float; y: float; dx: float; dy: float
    speed: float; event_type: str; button: Optional[str]


@dataclass
class MouseWindowStats:
    speed_mean: float = 0.0; speed_std: float = 0.0; speed_max: float = 0.0
    jerk_mean: float = 0.0; curvature_mean: float = 0.0
    angle_delta_mean: float = 0.0; path_efficiency: float = 0.0
    direction_reversals: int = 0; sub_movement_count: int = 0
    convex_hull_area: float = 0.0; spatial_entropy: float = 0.0
    sample_entropy: float = 0.0; fractal_dimension: float = 1.0
    autocorr_lag1: float = 0.0; idle_ratio: float = 0.0
    mean_idle_duration: float = 0.0; click_rate: float = 0.0
    ici_cv: float = 0.0; click_hesitation: float = 0.0
    post_click_pause: float = 0.0; scroll_reversal_count: int = 0
    scroll_velocity: float = 0.0

    def to_array(self):
        return np.array([
            self.speed_mean, self.speed_std, self.speed_max,
            self.jerk_mean, self.curvature_mean, self.angle_delta_mean,
            self.path_efficiency, float(self.direction_reversals),
            float(self.sub_movement_count), self.convex_hull_area,
            self.spatial_entropy, self.sample_entropy, self.fractal_dimension,
            self.autocorr_lag1, self.idle_ratio, self.mean_idle_duration,
            self.click_rate, self.ici_cv, self.click_hesitation,
            self.post_click_pause, float(self.scroll_reversal_count),
            self.scroll_velocity,
        ], dtype=np.float32)

    @staticmethod
    def feature_names():
        return ["speed_mean","speed_std","speed_max","jerk_mean",
               "curvature_mean","angle_delta_mean","path_efficiency",
               "direction_reversals","sub_movement_count","convex_hull_area",
               "spatial_entropy","sample_entropy","fractal_dimension",
               "autocorr_lag1","idle_ratio","mean_idle_duration",
               "click_rate","ici_cv","click_hesitation","post_click_pause",
               "scroll_reversal_count","scroll_velocity"]


# ── verbatim from pre_embedders/mouse/mouse_encoder.py ─────────────────

def compute_per_event_derivatives(events):
    n = len(events)
    ts = np.array([e.timestamp for e in events], dtype=np.float64)
    dx = np.array([e.dx for e in events], dtype=np.float64)
    dy = np.array([e.dy for e in events], dtype=np.float64)
    speed = np.array([e.speed for e in events], dtype=np.float64)
    dt = np.diff(ts)
    dt = np.where(dt < 1e-9, 1e-9, dt)
    accel = np.zeros(n); accel[1:] = np.diff(speed) / dt
    jerk = np.zeros(n); jerk[2:] = np.diff(accel[1:]) / dt[1:]
    theta = np.arctan2(dy, dx)
    angle_delta = np.zeros(n); angle_delta[1:] = np.abs(np.diff(np.unwrap(theta)))
    ang_vel = np.zeros(n); ang_vel[1:] = angle_delta[1:] / dt
    vx = np.zeros(n); vy = np.zeros(n)
    vx[1:] = dx[1:] / dt; vy[1:] = dy[1:] / dt
    ax = np.zeros(n); ay = np.zeros(n)
    ax[1:] = np.diff(vx) / dt; ay[1:] = np.diff(vy) / dt
    cross = np.abs(vx * ay - vy * ax)
    speed_cubed = np.where(speed**3 < 1e-9, 1e-9, speed**3)
    curvature = cross / speed_cubed
    is_idle = (speed < IDLE_SPEED_THRESHOLD).astype(np.float32)
    return {"ts": ts, "speed": speed, "accel": accel, "jerk": jerk,
           "curvature": curvature, "angle_delta": angle_delta,
           "ang_vel": ang_vel, "is_idle": is_idle, "vx": vx, "vy": vy, "dt": dt}


def _sample_entropy(series, m, r, chunk=500):
    n = len(series)
    if n < m + 2 or r <= 0:
        return 0.0
    def _count(length):
        L = n - length + 1
        if L < 2:
            return 0
        idx = np.arange(length)[None, :] + np.arange(L)[:, None]
        W = series[idx]
        total = 0
        for start in range(0, L, chunk):
            end = min(start + chunk, L)
            block = W[start:end]
            diffs = np.abs(block[:, None, :] - W[None, :, :]).max(axis=2)
            for local_i, global_i in enumerate(range(start, end)):
                total += int(np.sum(diffs[local_i, global_i + 1:] < r))
        return total
    B = _count(m); A = _count(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


def _higuchi_fd(series, k_max=8):
    N = len(series)
    if N < k_max * 2:
        return 1.0
    L = []; x = np.arange(1, k_max + 1)
    for k in x:
        Lk = 0.0
        for m in range(1, k + 1):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2:
                continue
            Lm = np.sum(np.abs(np.diff(series[idxs])))
            Lm *= (N - 1) / (len(idxs) * k)
            Lk += Lm
        L.append(Lk / k)
    L = np.array(L)
    valid = L > 0
    if valid.sum() < 2:
        return 1.0
    slope, _ = np.polyfit(np.log(1.0 / x[valid]), np.log(L[valid]), 1)
    return float(np.clip(slope, 1.0, 2.0))


def extract_window_stats(events):
    if len(events) < MIN_EVENTS_FOR_WINDOW:
        return None
    window_duration = events[-1].timestamp - events[0].timestamp
    if window_duration < 1e-9:
        return None
    d = compute_per_event_derivatives(events)
    ts, speed, jerk, curv = d["ts"], d["speed"], d["jerk"], d["curvature"]
    ang_d, is_idle, vx, vy = d["angle_delta"], d["is_idle"], d["vx"], d["vy"]
    xs = np.array([e.x for e in events]); ys = np.array([e.y for e in events])

    f = MouseWindowStats()
    f.speed_mean = float(np.mean(speed)); f.speed_std = float(np.std(speed))
    f.speed_max = float(np.max(speed)); f.jerk_mean = float(np.mean(np.abs(jerk)))
    f.curvature_mean = float(np.mean(curv)); f.angle_delta_mean = float(np.mean(ang_d))

    diffs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    path_length = float(np.sum(diffs))
    displacement = float(np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2))
    f.path_efficiency = (float(np.clip(displacement/path_length, 0.0, 1.0))
                         if path_length > 1e-9 else 1.0)
    f.direction_reversals = (int(np.sum(np.diff(np.sign(vx)) != 0)) +
                             int(np.sum(np.diff(np.sign(vy)) != 0)))
    f.sub_movement_count = int(np.sum(
        (speed[1:-1] < speed[:-2]) & (speed[1:-1] < speed[2:])))

    pts = np.stack([xs, ys], axis=1)
    try:
        hull = ConvexHull(pts)
        f.convex_hull_area = float(hull.volume)
    except Exception:
        f.convex_hull_area = float((xs.max()-xs.min()) * (ys.max()-ys.min()))

    H, _, _ = np.histogram2d(xs, ys, bins=SPATIAL_GRID_CELLS)
    H_flat = H.flatten(); H_flat = H_flat[H_flat > 0]
    f.spatial_entropy = float(scipy_entropy(H_flat / H_flat.sum()))
    f.sample_entropy = _sample_entropy(speed, SAMPEN_M,
                                       SAMPEN_R_FACTOR * (np.std(speed)+1e-9))
    f.fractal_dimension = _higuchi_fd(speed)

    speed_c = speed - speed.mean(); var = np.var(speed)
    if len(speed) > 1 and var > 1e-12:
        f.autocorr_lag1 = float(np.mean(speed_c[:-1]*speed_c[1:]) / var)

    idle_durations = []; in_idle = False; idle_start = 0.0
    for i, flag in enumerate(is_idle):
        if flag and not in_idle:
            in_idle = True; idle_start = ts[i]
        elif not flag and in_idle:
            in_idle = False; idle_durations.append(ts[i] - idle_start)
    if in_idle:
        idle_durations.append(ts[-1] - idle_start)
    f.idle_ratio = float(sum(idle_durations) / window_duration)
    f.mean_idle_duration = float(np.mean(idle_durations)) if idle_durations else 0.0

    click_events = [(i, e) for i, e in enumerate(events)
                    if e.event_type == "mouse_press" and e.button == "left"]
    f.click_rate = len(click_events) / window_duration
    if len(click_events) >= 2:
        click_ts = np.array([ts[i] for i, _ in click_events])
        icis = np.diff(click_ts); mu = float(np.mean(icis))
        f.ici_cv = float(np.std(icis) / mu) if mu > 1e-9 else 0.0

    hesitations = []
    for idx, _ in click_events:
        for j in range(idx-1, max(idx-50, -1), -1):
            if is_idle[j] == 1.0:
                hesitations.append(ts[idx]-ts[j+1] if j+1 < idx else 0.0)
                break
    f.click_hesitation = float(np.mean(hesitations)) if hesitations else 0.0

    post_pauses = []
    for idx, _ in click_events:
        for j in range(idx+1, min(idx+100, len(events))):
            if is_idle[j] == 0.0:
                post_pauses.append(ts[j]-ts[idx]); break
    f.post_click_pause = float(np.mean(post_pauses)) if post_pauses else 0.0

    scroll_evts = [(e.timestamp, e.dy) for e in events if e.event_type == "scroll"]
    if len(scroll_evts) >= 2:
        dy_signs = np.sign([dy for _, dy in scroll_evts])
        f.scroll_reversal_count = int(np.sum(np.diff(dy_signs) != 0))
        ts_sc = np.array([t for t, _ in scroll_evts])
        dy_sc = np.array([dy for _, dy in scroll_evts])
        dur_sc = ts_sc[-1] - ts_sc[0] + 1e-9
        f.scroll_velocity = float(np.sum(np.abs(dy_sc)) / dur_sc)

    return f

# ── end verbatim block ──────────────────────────────────────────────────


def window_by_time(events_dicts, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC):
    """Returns (windows, window_starts) -- BOTH same length as the raw
    sliding-window sequence, BEFORE any extract_window_stats() filtering.
    Callers must zip these together and only keep the start time for
    windows that survive filtering -- never reconstruct start times from
    the OUTPUT index after filtering, since that silently breaks the
    moment any window gets dropped (the original bug here)."""
    events = [MouseEvent(**e) for e in events_dicts]
    if not events:
        return [], []
    t0, tN = events[0].timestamp, events[-1].timestamp
    windows, starts = [], []
    w0 = t0
    while w0 + window_sec <= tN:
        w1 = w0 + window_sec
        chunk = [e for e in events if w0 <= e.timestamp < w1]
        windows.append(chunk)
        starts.append(w0)
        w0 += stride_sec
    return windows, starts


def encode_subject_stats(sid):
    path = os.path.join(COG_LAB_DIR, sid, "HCI", f"D3_{sid}_mouse.csv")
    if not os.path.isfile(path):
        return None
    events_dicts = adapt_coglab_mouse(path)
    windows, window_starts = window_by_time(events_dicts)   # both same length as input windows

    stats_arr = []
    valid_starts = []
    for w, w_start in zip(windows, window_starts):
        stats = extract_window_stats(w)
        if stats is not None:
            stats_arr.append(stats.to_array())
            valid_starts.append(w_start)   # ONLY keep the start time for
                                           # windows that actually survived
                                           # -- this is the fix: never assume
                                           # a dense index correspondence

    if not stats_arr:
        return None, None
    return np.stack(stats_arr), np.array(valid_starts)   # (n_windows, 22), (n_windows,)


def main():
    print("=" * 78)
    print("MOUSE STATS — 22 REAL handcrafted features, no untrained network")
    print("=" * 78)

    all_dirs = sorted(d for d in os.listdir(COG_LAB_DIR)
                      if d.startswith("S") and d[1:].isdigit())
    subjects = [d for d in all_dirs if d not in EXCLUDE]

    out_dir = os.path.join(BASE, "aam_proxy", "encoders", "mouse_stats")
    os.makedirs(out_dir, exist_ok=True)

    for sid in subjects:
        result = encode_subject_stats(sid)
        if result is None or result[0] is None:
            print(f"  {sid}: no data / no valid windows")
            continue
        arr, starts = result
        np.save(os.path.join(out_dir, f"{sid}_mouse_stats.npy"), arr)
        np.save(os.path.join(out_dir, f"{sid}_mouse_starts.npy"), starts)
        print(f"  {sid}: {arr.shape[0]} windows -> 22-dim stats "
              f"(click_rate mean={arr[:,16].mean():.3f}, "
              f"speed_mean mean={arr[:,0].mean():.1f})")

    print(f"\nSaved to {out_dir}/")
    print(f"Feature order: {MouseWindowStats.feature_names()}")


if __name__ == "__main__":
    main()
