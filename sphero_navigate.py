#!/usr/bin/env python3
"""
sphero_navigate.py — Navigate Sphero BOLT from A to B using camera feedback.

Flow:
  1. Load paper calibration, open RealSense, connect Sphero
  2. Aim calibration: rotate shell so glowing tail faces you, press ENTER
  3. Click target B on the display window
  4. Sphero navigates at speed 50, updating heading every 0.5 s
  5. Stops within 3 cm of B

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
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI

# ── Constants ─────────────────────────────────────────────────────────────────
SPEED           = 50
UPDATE_INTERVAL = 0.3   # s between heading updates
HOVER_STOP_CM   = 4.0   # cm — stop rolling and enter hover within this radius
HOVER_RESUME_CM = 9.0   # cm — exit hover and navigate again if drift exceeds this
SLOW_DIST_CM    = 20.0  # cm — start reducing speed within this radius
MIN_SPEED       = 20    # floor speed when very close
COAST_FACTOR    = 0.28  # empirical: coast_cm ≈ COAST_FACTOR × speed
NEAR_DIST_CM    = 20.0  # cm — switch to shorter update interval
UPDATE_NEAR     = 0.1   # s — interval when within NEAR_DIST_CM
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
        self.droid          = droid
        self.tracker        = tracker
        self.heading_offset = heading_offset
        self.target         = None
        self.active         = False
        self.last_heading   = None
        self.last_dist_cm   = None
        self._hovering      = False
        self._thread        = None

    @property
    def hovering(self):
        return self._hovering

    def start(self, target_uv):
        self.target  = target_uv
        self.active  = True
        self.arrived = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False

    def _run(self):
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
                print(f"[NAV] detection regained at u={pos[0]:.3f} v={pos[1]:.3f} — resuming")
                _was_lost = False

            du = self.target[0] - pos[0]
            dv = self.target[1] - pos[1]
            dist_cm = math.hypot(du * PAPER_X_CM, dv * PAPER_Y_CM)
            self.last_dist_cm = dist_cm
            speed = max(MIN_SPEED, int(SPEED * min(dist_cm / SLOW_DIST_CM, 1.0)))

            if self._hovering:
                if dist_cm > HOVER_RESUME_CM:
                    self._hovering = False
                    print(f"[NAV] HOVER → NAVIGATE  dist={dist_cm:.1f}cm > {HOVER_RESUME_CM}cm")
                else:
                    time.sleep(0.1)
                    continue
            elif dist_cm < max(HOVER_STOP_CM, speed * COAST_FACTOR):
                self.droid.set_speed(0)
                self._hovering = True
                print(f"[NAV] NAVIGATE → HOVER"
                      f"  pos=({pos[0]*PAPER_X_CM:.1f},{pos[1]*PAPER_Y_CM:.1f})cm"
                      f"  target=({self.target[0]*PAPER_X_CM:.1f},{self.target[1]*PAPER_Y_CM:.1f})cm"
                      f"  dist={dist_cm:.1f}cm")
                time.sleep(0.1)
                continue

            paper_angle = math.degrees(math.atan2(du, dv)) % 360
            heading  = int((paper_angle + self.heading_offset) % 360)
            interval = UPDATE_NEAR if dist_cm < NEAR_DIST_CM else UPDATE_INTERVAL
            self.last_heading = heading
            print(f"[NAV] pos=({pos[0]:.3f},{pos[1]:.3f})  target=({self.target[0]:.3f},{self.target[1]:.3f})"
                  f"  du={du:+.3f} dv={dv:+.3f}  dist={dist_cm:.1f}cm"
                  f"  paper_angle={paper_angle:.1f}°  heading={heading}°  speed={speed}  dt={interval}s")
            self.droid.roll(heading, speed, interval)

        self.droid.set_speed(0)


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
    target_uv      = [None]
    start_uv       = [None]
    nav            = [None]
    _last_det_log  = 0.0        # throttle detection prints to 1 Hz

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state in (STATE_WAIT_B, STATE_NAV):
            target_uv[0] = click_to_uv(x, y)
            if nav[0] is not None:
                nav[0].target   = target_uv[0]
                nav[0]._hovering = False   # force re-navigation to new target
            print(f"  Target B: u={target_uv[0][0]:.3f}  v={target_uv[0][1]:.3f}")

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
                if target_uv[0] is None:
                    stable_pos = tracker.stable_uv()
                    if stable_pos is None:
                        draw_overlay(disp, ["Waiting for stable detection…"])
                    else:
                        draw_overlay(disp, ["Sphero detected.  Click TARGET B on the paper."])
                else:
                    draw_target(disp, target_uv[0])

            elif state == STATE_NAV:
                if start_uv[0]:
                    draw_start(disp, start_uv[0])
                if target_uv[0]:
                    draw_target(disp, target_uv[0])
                n = nav[0]
                if n:
                    d = n.last_dist_cm
                    h = n.last_heading
                    if n.hovering:
                        col  = (0, 255, 0)
                        info = (f"HOVERING  dist={d:.1f}cm  |  R=new target"
                                if d is not None else "HOVERING  |  R=new target")
                    else:
                        col  = (0, 255, 255)
                        if fuv and target_uv[0]:
                            cv2.line(disp, uv_to_disp(*fuv), uv_to_disp(*target_uv[0]),
                                     (255, 255, 0), 1, cv2.LINE_AA)
                        info = (f"NAVIGATING  dist={d:.1f}cm  hdg={h}deg  |  R=new target"
                                if d is not None else "NAVIGATING…")
                    cv2.putText(disp, info, (8, RECT_H - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            elif state == STATE_ARRIVED:
                if start_uv[0]:
                    draw_start(disp, start_uv[0])
                if target_uv[0]:
                    draw_target(disp, target_uv[0])
                draw_overlay(disp, [
                    "ARRIVED at B!",
                    "Press R to navigate again  |  Q to quit",
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
                if target_uv[0] is not None:
                    fuv_now = tracker.filtered_uv
                    if fuv_now is not None:
                        start_uv[0] = fuv_now
                        tracker.clear_history()
                        n = Navigator(droid, tracker, args.heading_offset)
                        n.start(target_uv[0])
                        nav[0] = n
                        state  = STATE_NAV
                        au, av = start_uv[0]
                        bu, bv = target_uv[0]
                        print(f"[STATE] wait_b → nav"
                              f"  A=({au * PAPER_X_CM:.1f},{av * PAPER_Y_CM:.1f})cm"
                              f"  B=({bu * PAPER_X_CM:.1f},{bv * PAPER_Y_CM:.1f})cm"
                              f"  heading_offset={args.heading_offset}°")

            elif state == STATE_NAV:
                if key == ord('r'):
                    if nav[0]:
                        nav[0].stop()
                    target_uv[0] = None
                    start_uv[0]  = None
                    nav[0]       = None
                    tracker.clear_history()
                    state = STATE_WAIT_B
                    print("[STATE] nav → wait_b  (new target)")

            elif state == STATE_ARRIVED:
                if key == ord('r'):
                    if nav[0]:
                        nav[0].stop()
                    target_uv[0] = None
                    start_uv[0]  = None
                    nav[0]       = None
                    tracker.clear_history()
                    state = STATE_WAIT_B
                    print("[STATE] arrived → wait_b  (ready for new target)")

    pipe.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
