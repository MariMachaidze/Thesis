#!/usr/bin/env python3
"""
sphero_navigate.py — Smooth path-following navigation for Sphero BOLT using pure pursuit.

Flow:
  1. Load paper calibration, open RealSense, connect Sphero
  2. Aim calibration: rotate shell so glowing tail faces you, press ENTER
  3. Click multiple waypoints on the display window
     - Smooth spline path automatically interpolates between waypoints
     - Visual preview shows the path that will be followed
  4. Press ENTER to start navigation
  5. Sphero continuously follows the smooth interpolated path using pure pursuit
     - Lookahead point guides heading
     - Speed scales smoothly based on distance to end
     - Cross-track error monitored for closed-loop correction
  6. Arrives when path is complete

Features:
  • Smooth cubic spline interpolation between waypoints
  • Pure pursuit path following (10–30 Hz control loop)
  • Distance-scaled speed throughout entire path
  • Cross-track error monitoring for research metrics
  • Visual path preview before navigation
  • Responsive heading correction every frame
  • 0.08 s update interval for tight closed-loop control

Usage:
  python3 sphero_navigate.py --sphero SB-FD03
  python3 sphero_navigate.py --sphero SB-FD03 --heading_offset 2.0
"""

import argparse
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from scipy.interpolate import PchipInterpolator
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI

# ── Constants ─────────────────────────────────────────────────────────────────
SPEED               = 50
UPDATE_INTERVAL     = 0.12   # s between heading updates (tuned for Bluetooth reliability)
MIN_SPEED           = 25     # floor speed (keeps robot moving even at path end)
HDG_CHANGE_THRESH   = 3      # degrees: only send roll() if heading changed by this much
SPEED_CHANGE_THRESH = 2      # speed units: only send roll() if speed changed by this much

# Curvature-aware control (predictive)
SHARP_TURN_THRESHOLD    = 25   # degrees: turn sharper than this reduces speed
CURVATURE_LOOKAHEAD_PTS = 15   # path points ahead to estimate upcoming curvature

# Adaptive lookahead
LOOKAHEAD_MIN_CM = 4.0   # minimum lookahead (sharp turns)
LOOKAHEAD_MAX_CM = 10.0  # maximum lookahead (straight segments)

# Stuck detection (path-progress-based, immune to distance noise)
STUCK_CYCLES       = 20   # cycles window (~2.4s at 0.12s interval)
STUCK_MIN_PROGRESS = 2    # minimum path_idx points that must advance in that window

# Waypoint navigation
SEARCH_WINDOW    = 40    # local search window (10% of 400-point path)
ARRIVAL_DIST_CM  = 4.0   # robot must be within this distance of final waypoint to arrive
FINAL_APPROACH_CM = 8.0  # decelerate smoothly within this distance of goal
PAPER_X_CM      = 85.0
PAPER_Y_CM      = 60.0
RECT_W          = 850
RECT_H          = 600
WIDTH, HEIGHT, FPS = 1280, 720, 30

# Hough — hardcoded
P2         = 21
MIN_R      = 25
MAX_R      = 40
BLUR_K     = 6
CLAHE_CLIP = 0.6
HOUGH_DP   = 1.2
HOUGH_DIST = 40
HOUGH_P1   = 80

# One Euro Filter
OEF_MIN_CUTOFF = 0.6
OEF_BETA       = 0.03

TRAIL_LEN = 80
MAX_LOST  = 10
MAX_JUMP  = 120   # px

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))

STATE_AIM     = 'aim'
STATE_WAIT_B  = 'wait_b'
STATE_NAV     = 'nav'
STATE_ARRIVED = 'arrived'


# ── One Euro Filter ───────────────────────────────────────────────────────────
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = None
        self.dx_prev    = 0.0
        self.t_prev     = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev = x; self.t_prev = t; return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        a_d  = self._alpha(self.d_cutoff, dt)
        dx   = (x - self.x_prev) / dt
        dx_h = a_d * dx + (1 - a_d) * self.dx_prev
        cut  = self.min_cutoff + self.beta * abs(dx_h)
        a    = self._alpha(cut, dt)
        x_h  = a * x + (1 - a) * self.x_prev
        self.x_prev  = x_h
        self.dx_prev = dx_h
        self.t_prev  = t
        return x_h

    def reset(self):
        self.x_prev = None; self.dx_prev = 0.0; self.t_prev = None


# ── Tracker ───────────────────────────────────────────────────────────────────
class SpheroTracker:
    def __init__(self):
        self._lock    = threading.Lock()
        self.last_px  = None
        self._uv      = None
        self._fuv     = None
        self._history = deque(maxlen=12)
        self._trail   = deque(maxlen=TRAIL_LEN)
        self.lost     = 0
        self._oef_u   = OneEuroFilter(OEF_MIN_CUTOFF, OEF_BETA)
        self._oef_v   = OneEuroFilter(OEF_MIN_CUTOFF, OEF_BETA)

    @property
    def filtered_uv(self):
        with self._lock:
            return self._fuv if self._fuv is not None else self._uv

    @property
    def trail(self):
        with self._lock:
            return list(self._trail)

    def update(self, candidates, t):
        det = _pick_best(candidates, self.last_px)
        with self._lock:
            if det:
                cx, cy, r    = det
                self.last_px = (cx, cy)
                self.lost    = 0
                u = cx / RECT_W
                v = cy / RECT_H
                self._uv = (u, v)
                self._history.append((u, v))
                fu = self._oef_u(u, t)
                fv = self._oef_v(v, t)
                self._fuv = (fu, fv)
                self._trail.append((cx, cy))
            else:
                self.lost += 1
                if self.lost > MAX_LOST:
                    self.last_px = None
        return det

    def stable_uv(self, n=6, radius_cm=2.0):
        with self._lock:
            hist = list(self._history)
        if len(hist) < n:
            return None
        recent = hist[-n:]
        cu, cv = float(np.mean([u for u, v in recent])), float(np.mean([v for u, v in recent]))
        for u, v in recent:
            if math.hypot((u - cu) * PAPER_X_CM, (v - cv) * PAPER_Y_CM) > radius_cm:
                return None
        return (cu, cv)

    def clear_history(self):
        with self._lock:
            self._history.clear()
            self._trail.clear()
            self._oef_u.reset()
            self._oef_v.reset()
            self._fuv = None


def _pick_best(candidates, last_px):
    if not candidates:
        return None
    if last_px is None:
        return candidates[0]
    lx, ly = last_px
    best, bd = None, float('inf')
    for cx, cy, r in candidates:
        d = math.hypot(cx - lx, cy - ly)
        if d < bd and d < MAX_JUMP:
            bd, best = d, (cx, cy, r)
    return best if best is not None else candidates[0]


# ── Path Interpolation ────────────────────────────────────────────────────────
def interpolate_smooth_path(waypoints, num_points=400):
    """
    Interpolate waypoints to a smooth path using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).
    PCHIP preserves monotonicity and avoids overshoot, making it safer than CubicSpline for sharp-corner paths.

    Args:
        waypoints: list of (u, v) tuples
        num_points: resolution of interpolated path

    Returns:
        list of (u, v) points along the smooth path
    """
    if len(waypoints) < 2:
        return waypoints
    if len(waypoints) == 2:
        # Linear interpolation for 2 points
        u_vals = np.linspace(waypoints[0][0], waypoints[1][0], num_points)
        v_vals = np.linspace(waypoints[0][1], waypoints[1][1], num_points)
        return list(zip(u_vals, v_vals))

    # PCHIP interpolation: shape-preserving, no overshoot
    waypoints = np.array(waypoints)
    t = np.linspace(0, 1, len(waypoints))
    pchip_u = PchipInterpolator(t, waypoints[:, 0])
    pchip_v = PchipInterpolator(t, waypoints[:, 1])

    t_smooth = np.linspace(0, 1, num_points)
    u_smooth = pchip_u(t_smooth)
    v_smooth = pchip_v(t_smooth)

    # Clamp to valid UV range [0, 1]
    u_smooth = np.clip(u_smooth, 0, 1)
    v_smooth = np.clip(v_smooth, 0, 1)

    return list(zip(u_smooth, v_smooth))


def find_closest_path_point(pos, path, search_start_idx=0):
    """
    Find closest point on path to current position.
    Limits search to a local window to prevent snap-ahead on looping paths.
    Returns (closest_point, closest_idx, distance_cm).
    """
    if not path:
        return None, 0, float('inf')

    min_dist = float('inf')
    min_idx = search_start_idx

    # Search forward from last known position, but limit to local window
    search_end = min(search_start_idx + SEARCH_WINDOW, len(path))
    for i in range(search_start_idx, search_end):
        px, pv = path[i]
        dist = math.hypot((px - pos[0]) * PAPER_X_CM, (pv - pos[1]) * PAPER_Y_CM)
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    return path[min_idx], min_idx, min_dist


def get_lookahead_point(pos, path, current_idx, lookahead_cm):
    """
    Get point on path that is lookahead_cm ahead of robot's current position.
    Uses pure pursuit logic.
    """
    if not path or current_idx >= len(path):
        return path[-1] if path else pos

    for i in range(current_idx, len(path)):
        px, pv = path[i]
        dist = math.hypot((px - pos[0]) * PAPER_X_CM, (pv - pos[1]) * PAPER_Y_CM)
        if dist >= lookahead_cm:
            return (px, pv)

    return path[-1]


def estimate_upcoming_curvature(path, path_idx):
    """
    Estimate curvature CURVATURE_LOOKAHEAD_PTS ahead along the path.
    Returns heading change in degrees — larger means sharper upcoming turn.
    """
    n = len(path)
    pts = CURVATURE_LOOKAHEAD_PTS
    if path_idx + pts >= n:
        return 0.0

    p0 = path[path_idx]
    p1 = path[path_idx + pts // 2]
    p2 = path[path_idx + pts]

    du1 = (p1[0] - p0[0]) * PAPER_X_CM
    dv1 = (p1[1] - p0[1]) * PAPER_Y_CM
    du2 = (p2[0] - p1[0]) * PAPER_X_CM
    dv2 = (p2[1] - p1[1]) * PAPER_Y_CM

    if math.hypot(du1, dv1) < 1e-6 or math.hypot(du2, dv2) < 1e-6:
        return 0.0

    h1 = math.degrees(math.atan2(du1, dv1)) % 360
    h2 = math.degrees(math.atan2(du2, dv2)) % 360

    diff = abs(h1 - h2)
    if diff > 180:
        diff = 360 - diff
    return diff


def compute_adaptive_lookahead(speed, curvature_deg):
    """
    Compute adaptive lookahead distance based on current speed and upcoming curvature.
    - Faster speed → larger lookahead (smoother straights)
    - Sharper turn ahead → smaller lookahead (tighter corner tracking)
    """
    speed_factor = speed / SPEED
    lookahead = LOOKAHEAD_MIN_CM + speed_factor * (LOOKAHEAD_MAX_CM - LOOKAHEAD_MIN_CM)

    if curvature_deg > SHARP_TURN_THRESHOLD:
        reduction_ratio = (curvature_deg - SHARP_TURN_THRESHOLD) / 90.0
        lookahead -= reduction_ratio * (lookahead - LOOKAHEAD_MIN_CM) * 0.6

    return max(LOOKAHEAD_MIN_CM, min(LOOKAHEAD_MAX_CM, lookahead))


# ── Calibration / camera ──────────────────────────────────────────────────────
def load_calibration(path='~/.ros/paper_calibration.yaml'):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def build_homography(calib):
    p1 = np.array(calib['plane_center'])
    ax = np.array(calib['paper_x_axis'])
    ay = np.array(calib['paper_y_axis'])
    Lx = float(calib['paper_x_m'])
    Ly = float(calib['paper_y_m'])
    fx = float(calib['intrinsics']['fx'])
    fy = float(calib['intrinsics']['fy'])
    cx = float(calib['intrinsics']['cx'])
    cy = float(calib['intrinsics']['cy'])
    corners = [p1, p1+ax*Lx, p1+ax*Lx+ay*Ly, p1+ay*Ly]
    src = np.float32([[fx*X/Z + cx, fy*Y/Z + cy] for X, Y, Z in corners])
    dst = np.float32([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]])
    return cv2.getPerspectiveTransform(src, dst)


def start_camera():
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe.start(cfg)
    return pipe


def get_frame(pipe):
    frames = pipe.wait_for_frames(timeout_ms=5000)
    cf = frames.get_color_frame()
    if not cf:
        return None
    return cv2.flip(np.asanyarray(cf.get_data()), 1)


# ── Detection ─────────────────────────────────────────────────────────────────
def detect_sphero(rect):
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    enh  = _clahe.apply(gray)
    k    = max(3, BLUR_K | 1)
    blr  = cv2.GaussianBlur(enh, (k, k), 0)
    cirs = cv2.HoughCircles(
        blr, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_DIST,
        param1=HOUGH_P1, param2=max(1, P2),
        minRadius=max(1, MIN_R), maxRadius=max(2, MAX_R),
    )
    if cirs is None:
        return []
    return [(int(cx), int(cy), int(r)) for cx, cy, r in np.round(cirs[0]).astype(int)]


# ── Coordinate helpers ────────────────────────────────────────────────────────
def disp_pt(cx, cy):
    return (cx, RECT_H - 1 - cy)


def click_to_uv(dx, dy):
    return (dx / RECT_W, (RECT_H - 1 - dy) / RECT_H)


def uv_to_disp(u, v):
    return disp_pt(int(u * RECT_W), int(v * RECT_H))


# ── Navigator ─────────────────────────────────────────────────────────────────
class Navigator:
    def __init__(self, droid, tracker, heading_offset):
        self.droid              = droid
        self.tracker            = tracker
        self.heading_offset     = heading_offset
        self.path               = []
        self.path_idx           = 0
        self.active             = False
        self.arrived            = False
        self.last_heading       = None
        self.last_speed         = None
        self.last_dist_cm       = None
        self.last_progress      = 0.0
        self.last_cmd_time      = 0.0
        self.start_time         = 0.0
        self.trajectory         = []  # actual path followed during navigation
        self._thread            = None

        # Stuck detection
        self.idx_history = deque(maxlen=STUCK_CYCLES)
        self.stuck_counter = 0
        self.in_stuck_recovery = False
        self.recovery_start_time = 0.0

        # Metrics tracking
        self.cte_history = []
        self.max_cte = 0.0
        self.waypoints = []

    def start(self, smooth_path, waypoints=None):
        """Start pure pursuit navigation along a smooth interpolated path."""
        self.path        = list(smooth_path)
        self.path_idx    = 0
        self.active      = True
        self.arrived     = False
        self.start_time  = time.time()
        self.trajectory  = []
        self.waypoints   = waypoints or []

        # Reset tracking variables
        self.idx_history.clear()
        self.stuck_counter = 0
        self.in_stuck_recovery = False
        self.cte_history = []
        self.max_cte = 0.0

        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False

    def _log_metrics(self):
        """Log trajectory metrics to CSV file."""
        total_time = time.time() - self.start_time
        total_distance = sum(
            math.hypot((self.trajectory[i][0] - self.trajectory[i-1][0]) * PAPER_X_CM,
                      (self.trajectory[i][1] - self.trajectory[i-1][1]) * PAPER_Y_CM)
            for i in range(1, len(self.trajectory))
        )
        avg_cte = sum(self.cte_history) / len(self.cte_history) if self.cte_history else 0.0
        overshoot = self.last_dist_cm  # distance remaining at arrival (negative if overshot)

        # CSV filename with timestamp
        import time as time_module
        ts = int(time_module.time())
        filename = f"thesis_ws/trajectory_metrics_{ts}.csv"

        with open(filename, 'w') as f:
            f.write("metric,value\n")
            f.write(f"total_time_sec,{total_time:.3f}\n")
            f.write(f"total_distance_cm,{total_distance:.1f}\n")
            f.write(f"max_cross_track_error_cm,{self.max_cte:.1f}\n")
            f.write(f"avg_cross_track_error_cm,{avg_cte:.1f}\n")
            f.write(f"final_distance_to_target_cm,{overshoot:.1f}\n")
            f.write(f"path_length_points,{len(self.path)}\n")
            f.write(f"waypoint_count,{len(self.waypoints)}\n")

        print(f"[NAV] metrics logged to {filename}")

    def _run(self):
        """Main navigation loop using pure pursuit path following."""
        _was_lost = False
        while self.active:
            pos = self.tracker.filtered_uv
            if pos is None:
                if not _was_lost:
                    print("[NAV] detection lost — stopping Sphero")
                    _was_lost = True
                self.droid.set_speed(0)
                time.sleep(0.05)
                continue

            if _was_lost:
                print(f"[NAV] detection regained at u={pos[0]:.3f} v={pos[1]:.3f}")
                _was_lost = False

            # Find closest point on path (to update tracking)
            _, new_idx, cross_track_error = find_closest_path_point(
                pos, self.path, self.path_idx
            )
            self.path_idx = max(self.path_idx, new_idx)

            # Predictive curvature: estimate upcoming turn before computing lookahead/speed
            upcoming_curvature = estimate_upcoming_curvature(self.path, self.path_idx)

            # Adaptive lookahead: smaller on sharp turns, larger on straights
            lookahead_cm = compute_adaptive_lookahead(self.last_speed or SPEED, upcoming_curvature)
            target = get_lookahead_point(pos, self.path, self.path_idx, lookahead_cm)

            # Compute error
            du = target[0] - pos[0]
            dv = target[1] - pos[1]
            dist_to_end = math.hypot(
                (self.path[-1][0] - pos[0]) * PAPER_X_CM,
                (self.path[-1][1] - pos[1]) * PAPER_Y_CM
            )
            self.last_dist_cm = dist_to_end

            # Check if reached end of path (both path_idx near end and physically close to final waypoint)
            if self.path_idx >= len(self.path) - 3 and dist_to_end < ARRIVAL_DIST_CM:
                self.droid.set_speed(0)
                self.arrived = True
                print(f"[NAV] arrived at destination (path_idx={self.path_idx}, dist_to_end={dist_to_end:.1f}cm)")
                break

            # Progress metric (0 to 1)
            self.last_progress = min(1.0, self.path_idx / float(len(self.path)))

            # Metrics tracking: accumulate CTE
            self.cte_history.append(cross_track_error)
            self.max_cte = max(self.max_cte, cross_track_error)

            # Stuck detection: track path_idx advancement (immune to distance noise)
            self.idx_history.append(self.path_idx)
            if len(self.idx_history) == STUCK_CYCLES:
                idx_advance = self.path_idx - self.idx_history[0]
                if idx_advance < STUCK_MIN_PROGRESS:
                    self.stuck_counter += 1
                    if self.stuck_counter >= 3 and not self.in_stuck_recovery:
                        print(f"[NAV] STUCK DETECTED: path_idx advanced only {idx_advance} points in {STUCK_CYCLES} cycles")
                        self.in_stuck_recovery = True
                        self.recovery_start_time = time.time()
                else:
                    self.stuck_counter = 0

            # Compute heading toward lookahead point
            paper_angle = math.degrees(math.atan2(du, dv)) % 360
            heading = int((paper_angle + self.heading_offset) % 360)

            # Stuck recovery: briefly stop then boost to full speed
            if self.in_stuck_recovery:
                recovery_elapsed = time.time() - self.recovery_start_time
                if recovery_elapsed < 0.3:
                    speed = 0
                    print(f"[NAV] recovery: stopping")
                else:
                    self.in_stuck_recovery = False
                    speed = SPEED
            else:
                near_path_end = self.path_idx >= len(self.path) * 0.9
                if near_path_end:
                    # Softer final approach: smooth ramp over last FINAL_APPROACH_CM
                    approach_factor = min(dist_to_end / FINAL_APPROACH_CM, 1.0)
                    speed = max(MIN_SPEED, int(SPEED * approach_factor))
                else:
                    speed = SPEED

                # Apply predictive curvature reduction in all phases
                if upcoming_curvature > SHARP_TURN_THRESHOLD:
                    turn_factor = 1.0 - (upcoming_curvature - SHARP_TURN_THRESHOLD) / 90.0
                    turn_factor = max(0.3, turn_factor)
                    speed = max(MIN_SPEED, int(speed * turn_factor))

            # Save previous values BEFORE overwriting (for change detection)
            prev_heading = self.last_heading
            prev_speed = self.last_speed

            self.last_heading = heading
            self.last_speed = speed

            # Track actual trajectory
            self.trajectory.append(pos)

            print(f"[NAV] progress={self.last_progress:.1%}  "
                  f"pos=({pos[0]:.3f},{pos[1]:.3f})  "
                  f"lookahead=({target[0]:.3f},{target[1]:.3f})  "
                  f"dist_to_end={dist_to_end:.1f}cm  hdg={heading}°  speed={speed}  "
                  f"la={lookahead_cm:.1f}cm  curv={upcoming_curvature:.0f}°  cte={cross_track_error:.1f}cm")

            # Only send command if heading or speed changed significantly, or if too long since last command
            t_now = time.time()
            should_send = (
                self.last_cmd_time == 0.0  # First command
                or (prev_heading is not None and abs(heading - prev_heading) >= HDG_CHANGE_THRESH)
                or (prev_speed is not None and abs(speed - prev_speed) >= SPEED_CHANGE_THRESH)
                or (t_now - self.last_cmd_time) > UPDATE_INTERVAL * 1.5  # Failsafe: re-send if stale
            )

            if should_send:
                try:
                    self.droid.roll(heading, speed, UPDATE_INTERVAL)
                    self.last_cmd_time = t_now
                except TimeoutError:
                    print(f"[NAV] Bluetooth timeout at hdg={heading}° speed={speed}")
                except Exception as e:
                    print(f"[NAV] Bluetooth error: {type(e).__name__}: {e}")
            else:
                time.sleep(0.01)  # Brief sleep to avoid busy-waiting

        self.droid.set_speed(0)

        # Log trajectory metrics
        if self.arrived:
            self._log_metrics()


# ── Display helpers ───────────────────────────────────────────────────────────
def draw_grid(img, step_px=100, color=(55, 55, 55)):
    for rx in range(step_px, RECT_W, step_px):
        cv2.line(img, (rx, 0), (rx, RECT_H), color, 1)
        cv2.putText(img, f"{rx // 10}cm", (rx + 2, RECT_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    for ry in range(step_px, RECT_H, step_px):
        dy = RECT_H - 1 - ry   # rectified y → display y (vertical flip)
        cv2.line(img, (0, dy), (RECT_W, dy), color, 1)
        cv2.putText(img, f"{ry // 10}cm", (2, dy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)


def draw_overlay(img, lines):
    h = 14 + 26 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (RECT_W, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 22 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_target(img, uv):
    dp = uv_to_disp(*uv)
    cv2.drawMarker(img, dp, (0, 60, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(img, dp, 18, (0, 60, 255), 1)
    cv2.putText(img, "B", (dp[0] + 14, dp[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 255), 2, cv2.LINE_AA)


def draw_start(img, uv):
    dp = uv_to_disp(*uv)
    cv2.drawMarker(img, dp, (0, 220, 0), cv2.MARKER_CROSS, 18, 1)
    cv2.putText(img, "A", (dp[0] + 8, dp[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA)


def draw_waypoint_marker(img, uv):
    """Draw waypoint marker (cross + circle) without label."""
    dp = uv_to_disp(*uv)
    cv2.drawMarker(img, dp, (0, 60, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(img, dp, 18, (0, 60, 255), 1)


def draw_smooth_path(img, path, color=(100, 150, 255), thickness=1):
    """Draw smooth interpolated path as a line."""
    if len(path) < 2:
        return
    for i in range(1, len(path)):
        p1 = uv_to_disp(*path[i-1])
        p2 = uv_to_disp(*path[i])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_trajectory(img, trajectory, color=(0, 255, 100), thickness=2):
    """Draw actual trajectory followed during navigation (thicker, brighter)."""
    if len(trajectory) < 2:
        return
    for i in range(1, len(trajectory)):
        p1 = uv_to_disp(*trajectory[i-1])
        p2 = uv_to_disp(*trajectory[i])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sphero',         type=str,   default='')
    ap.add_argument('--heading_offset', type=float, default=0.0,
                    help='Offset from sphero_calib_test (default 0)')
    args = ap.parse_args()

    print("Loading calibration…")
    calib  = load_calibration()
    H_rect = build_homography(calib)
    print(f"  Paper: {calib['paper_x_m']*100:.0f} × {calib['paper_y_m']*100:.0f} cm")

    print("Starting RealSense…")
    pipe = start_camera()

    print(f"Connecting to Sphero{(' ' + args.sphero) if args.sphero else ''}…")
    toy = scanner.find_toy(toy_name=args.sphero) if args.sphero else scanner.find_toy()
    print(f"Connected: {toy.name}")

    WIN = "Sphero Navigate  [Q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, RECT_W, RECT_H)

    tracker        = SpheroTracker()
    state          = STATE_AIM
    aim_phase      = 0          # 0 = show instructions, 1 = stabilisation off
    waypoints      = [[]]       # list of waypoint lists; [0] holds current plan
    smooth_path    = [[]]       # interpolated smooth path; [0] holds current path
    start_uv       = [None]
    nav            = [None]
    _last_det_log  = 0.0        # throttle detection prints to 1 Hz

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state in (STATE_WAIT_B, STATE_NAV):
            wp = click_to_uv(x, y)
            waypoints[0].append(wp)
            print(f"  Waypoint {len(waypoints[0])}: u={wp[0]:.3f}  v={wp[1]:.3f}")
            # If already navigating, let current plan finish; next click restarts
            if state == STATE_WAIT_B and len(waypoints[0]) > 0:
                # Auto-start when first waypoint is set in wait state
                pass

    cv2.setMouseCallback(WIN, mouse_cb)

    with SpheroEduAPI(toy) as droid:
        while True:
            # ── Capture & detect ─────────────────────────────────────────────
            raw = get_frame(pipe)
            if raw is None:
                continue
            rect  = cv2.warpPerspective(raw, H_rect, (RECT_W, RECT_H))
            t_now = time.time()
            cands = detect_sphero(rect)
            det   = tracker.update(cands, t_now)
            fuv   = tracker.filtered_uv

            if t_now - _last_det_log >= 1.0:
                _last_det_log = t_now
                if fuv is not None:
                    print(f"[DET] {len(cands)} candidate(s)  "
                          f"fuv=({fuv[0]:.3f},{fuv[1]:.3f})  lost={tracker.lost}")
                else:
                    print(f"[DET] no detection  lost={tracker.lost}")

            # ── Build display (vertical flip: far edge at top) ────────────────
            disp = cv2.flip(rect, 0)
            draw_grid(disp)

            # trail
            trail = tracker.trail
            for i in range(1, len(trail)):
                a   = i / len(trail)
                col = (int(50*a), int(200*a), int(255*(1-a) + 80*a))
                cv2.line(disp, disp_pt(*trail[i-1]), disp_pt(*trail[i]), col, max(1, int(3*a)))

            # all Hough candidates (faint)
            for cx, cy, r in cands:
                cv2.circle(disp, disp_pt(cx, cy), r, (80, 80, 80), 1)

            # filtered position
            if fuv is not None:
                dp     = uv_to_disp(*fuv)
                stable = tracker.stable_uv() is not None
                col    = (0, 255, 0) if stable else (0, 200, 255)
                cv2.circle(disp, dp, 7, col, -1)
                cv2.putText(disp, f"u={fuv[0]:.3f}  v={fuv[1]:.3f}",
                            (dp[0] + 10, dp[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

            # ── State overlays ────────────────────────────────────────────────
            if state == STATE_AIM:
                if aim_phase == 0:
                    draw_overlay(disp, [
                        "AIM CALIBRATION",
                        "Press ENTER to begin.",
                    ])
                else:
                    draw_overlay(disp, [
                        "Rotate Sphero shell: glowing TAIL toward YOU (near edge).",
                        "Press ENTER when aimed.",
                    ])

            elif state == STATE_WAIT_B:
                stable_pos = tracker.stable_uv()
                if stable_pos is None:
                    draw_overlay(disp, ["Waiting for stable detection…"])
                elif len(waypoints[0]) == 0:
                    draw_overlay(disp, ["Sphero detected.  Click waypoints on the paper (ENTER to start)."])
                else:
                    # Generate and show smooth path preview
                    smooth_path[0] = interpolate_smooth_path(waypoints[0], num_points=400)
                    draw_smooth_path(disp, smooth_path[0], color=(100, 150, 255), thickness=1)
                    # Show all waypoints set so far
                    for i, wp in enumerate(waypoints[0]):
                        draw_waypoint_marker(disp, wp)
                        dp = uv_to_disp(*wp)
                        cv2.putText(disp, f"W{i+1}", (dp[0] - 18, dp[1] - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 1, cv2.LINE_AA)
                    draw_overlay(disp, [f"{len(waypoints[0])} waypoint(s) set.  Click more or press ENTER to start."])

            elif state == STATE_NAV:
                if start_uv[0]:
                    draw_start(disp, start_uv[0])
                # Draw smooth path
                if smooth_path[0]:
                    draw_smooth_path(disp, smooth_path[0], color=(100, 150, 255), thickness=2)
                # Show waypoints
                for i, wp in enumerate(waypoints[0]):
                    draw_waypoint_marker(disp, wp)
                    dp = uv_to_disp(*wp)
                    cv2.putText(disp, f"W{i+1}", (dp[0] - 18, dp[1] - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 1, cv2.LINE_AA)
                n = nav[0]
                if n:
                    d = n.last_dist_cm
                    h = n.last_heading
                    p = n.last_progress
                    col = (0, 255, 255)
                    info = (f"NAVIGATING  progress={p:.0%}  dist_to_end={d:.1f}cm  hdg={h}°  |  R=restart"
                            if d is not None else "NAVIGATING…")
                    cv2.putText(disp, info, (8, RECT_H - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            elif state == STATE_ARRIVED:
                if start_uv[0]:
                    draw_start(disp, start_uv[0])
                # Draw planned path (light green)
                if smooth_path[0]:
                    draw_smooth_path(disp, smooth_path[0], color=(100, 200, 100), thickness=1)
                # Draw actual trajectory (bright green, thicker)
                if nav[0] and nav[0].trajectory:
                    draw_trajectory(disp, nav[0].trajectory, color=(0, 255, 0), thickness=2)
                for i, wp in enumerate(waypoints[0]):
                    draw_waypoint_marker(disp, wp)
                    dp = uv_to_disp(*wp)
                    cv2.putText(disp, f"W{i+1}", (dp[0] - 18, dp[1] - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 1, cv2.LINE_AA)
                draw_overlay(disp, [
                    "ARRIVED at all waypoints!",
                    "Planned path (light green)  |  Actual trajectory (bright green)",
                    "Press R for new route  |  Q to quit",
                ])

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF

            # ── Global keys ───────────────────────────────────────────────────
            if key == ord('q'):
                if nav[0]:
                    nav[0].stop()
                break

            # ── State transitions ─────────────────────────────────────────────
            if state == STATE_AIM:
                if key == 13:   # Enter
                    if aim_phase == 0:
                        droid.set_stabilization(False)
                        droid.set_back_led(255)
                        aim_phase = 1
                        print("  Stabilisation off — rotate Sphero, then press ENTER.")
                    else:
                        droid.reset_aim()
                        droid.set_stabilization(True)
                        droid.set_back_led(0)
                        aim_phase = 0
                        state = STATE_WAIT_B
                        print("[STATE] aim → wait_b  (heading=0 locked to Sphero forward direction)")

            elif state == STATE_WAIT_B:
                if key == 13 and len(waypoints[0]) > 0:  # ENTER with waypoints set
                    fuv_now = tracker.filtered_uv
                    if fuv_now is not None:
                        start_uv[0] = fuv_now
                        tracker.clear_history()
                        # Prepend current position to waypoints so path goes: START → WP1 → WP2 → ... → WPN
                        path_waypoints = [fuv_now] + waypoints[0]
                        smooth_path[0] = interpolate_smooth_path(path_waypoints, num_points=400)
                        n = Navigator(droid, tracker, args.heading_offset)
                        n.start(smooth_path[0], waypoints=waypoints[0])
                        nav[0] = n
                        state  = STATE_NAV
                        au, av = start_uv[0]
                        print(f"[STATE] wait_b → nav  "
                              f"A=({au * PAPER_X_CM:.1f},{av * PAPER_Y_CM:.1f})cm  "
                              f"waypoints={len(waypoints[0])}  smooth_path_points={len(smooth_path[0])}  "
                              f"heading_offset={args.heading_offset}°")

            elif state == STATE_NAV:
                # Check if navigation is complete
                if nav[0] and nav[0].arrived:
                    state = STATE_ARRIVED
                    print("[STATE] nav → arrived  (all waypoints reached)")
                elif key == ord('r'):
                    if nav[0]:
                        nav[0].stop()
                    waypoints[0] = []
                    smooth_path[0] = []
                    start_uv[0]  = None
                    nav[0]       = None
                    tracker.clear_history()
                    state = STATE_WAIT_B
                    print("[STATE] nav → wait_b  (restart route)")

            elif state == STATE_ARRIVED:
                if key == ord('r'):
                    if nav[0]:
                        nav[0].stop()
                    waypoints[0] = []
                    smooth_path[0] = []
                    start_uv[0]  = None
                    nav[0]       = None
                    tracker.clear_history()
                    state = STATE_WAIT_B
                    print("[STATE] arrived → wait_b  (ready for new route)")

    pipe.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
