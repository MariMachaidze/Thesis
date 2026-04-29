#!/usr/bin/env python3
"""
sphero_calib_test.py — Measure Sphero heading offset and speed from camera.

Steps:
  1. Loads paper calibration from ~/.ros/paper_calibration.yaml
  2. Opens RealSense + shows rectified paper view (1 px = 1 mm)
  3. Tune Hough trackbars until the Sphero is reliably detected
  4. Press ENTER to start the roll (heading=0, configurable speed/duration)
  5. Camera measures actual displacement → prints heading_offset and cm/s

Keys (in the display window):
  ENTER — start roll (only when Sphero is detected)
  Q     — quit

Usage:
  python3 sphero_calib_test.py
  python3 sphero_calib_test.py --speed 50 --duration 2.0 --sphero SB-FD03
"""

import argparse
import math
import os
import threading
import time
import yaml
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# ── Constants ─────────────────────────────────────────────────────────────────
WIDTH, HEIGHT, FPS  = 1280, 720, 30
RECT_W, RECT_H      = 850, 600   # rectified paper view: 1 px = 1 mm
PAPER_X_CM          = 85.0
PAPER_Y_CM          = 60.0

# Sphero BOLT on rectified view (1 px = 1 mm):
#   physical diameter ≈ 75 mm  →  radius ≈ 37-38 px
# Use a slightly wider range to handle partial occlusion / perspective distortion.
DEF_P2    = 25   # accumulator threshold — lower = more sensitive
DEF_MIN_R = 28   # px
DEF_MAX_R = 48   # px
DEF_BLUR  = 5    # Gaussian blur kernel (must be odd; enforced below)
DEF_CLAHE = 20   # CLAHE clip limit × 10  (trackbar stores int; divide by 10)

HOUGH_DP      = 1.2
HOUGH_MINDIST = 40   # px — ignore circles closer than this
HOUGH_P1      = 80   # Canny upper threshold inside HoughCircles

BLUE   = Color(r=0,   g=0,   b=255)
YELLOW = Color(r=255, g=200, b=0  )
OFF    = Color(r=0,   g=0,   b=0  )

STATE_WAITING  = 'waiting'
STATE_ROLLING  = 'rolling'
STATE_SETTLING = 'settling'
STATE_DONE     = 'done'


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


# ── Calibration ───────────────────────────────────────────────────────────────

def load_calibration(path='~/.ros/paper_calibration.yaml'):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Calibration not found: {path}\n"
            "Run the full ROS system first and click the 4 paper corners.")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError("Calibration file is empty.")
    return data


def build_homography(calib):
    p1 = np.array(calib['plane_center'])
    ax = np.array(calib['paper_x_axis'])
    ay = np.array(calib['paper_y_axis'])
    Lx = float(calib['paper_x_m'])
    Ly = float(calib['paper_y_m'])
    fx = float(calib['intrinsics']['fx'])
    fy = float(calib['intrinsics']['fy'])
    cx_i = float(calib['intrinsics']['cx'])
    cy_i = float(calib['intrinsics']['cy'])

    corners = [p1, p1+ax*Lx, p1+ax*Lx+ay*Ly, p1+ay*Ly]
    src = np.float32([[fx*X/Z + cx_i, fy*Y/Z + cy_i] for X, Y, Z in corners])
    dst = np.float32([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]])
    return cv2.getPerspectiveTransform(src, dst)


# ── Camera ────────────────────────────────────────────────────────────────────

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
    # Mirror to match d435i_driver_node output
    return cv2.flip(np.asanyarray(cf.get_data()), 1)


# ── Detection ─────────────────────────────────────────────────────────────────

def make_clahe(clip):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))


def detect_sphero(rect, p2, min_r, max_r, blur_k, clahe):
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    enh  = clahe.apply(gray)
    k    = max(3, blur_k | 1)   # ensure odd
    blr  = cv2.GaussianBlur(enh, (k, k), 0)
    cirs = cv2.HoughCircles(
        blr, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MINDIST,
        param1=HOUGH_P1, param2=max(1, p2),
        minRadius=max(1, min_r), maxRadius=max(2, max_r))
    if cirs is None:
        return [], blr
    return [(int(cx), int(cy), int(r)) for cx, cy, r in np.round(cirs[0]).astype(int)], blr


def pick_best(candidates, last_px, max_jump=80):
    if not candidates:
        return None
    if last_px is None:
        return candidates[0]
    lx, ly = last_px
    best, bd = None, float('inf')
    for cx, cy, r in candidates:
        d = math.hypot(cx-lx, cy-ly)
        if d < bd and d < max_jump:
            bd, best = d, (cx, cy, r)
    return best if best is not None else candidates[0]


# ── Tracker ───────────────────────────────────────────────────────────────────

class SpheroTracker:
    def __init__(self):
        self._lock    = threading.Lock()
        self.last_px  = None
        self._uv      = None
        self._fuv     = None   # OEF-filtered position
        self._history = deque(maxlen=12)
        self.lost     = 0
        self._oef_u   = OneEuroFilter(min_cutoff=1.0, beta=0.05)
        self._oef_v   = OneEuroFilter(min_cutoff=1.0, beta=0.05)

    def set_oef_params(self, min_cutoff, beta):
        self._oef_u.min_cutoff = min_cutoff
        self._oef_u.beta       = beta
        self._oef_v.min_cutoff = min_cutoff
        self._oef_v.beta       = beta

    @property
    def uv(self):
        with self._lock:
            return self._uv

    @property
    def filtered_uv(self):
        with self._lock:
            return self._fuv if self._fuv is not None else self._uv

    def update(self, candidates, t):
        det = pick_best(candidates, self.last_px)
        with self._lock:
            if det:
                cx, cy, r    = det
                self.last_px = (cx, cy)
                self.lost    = 0
                u = cx / RECT_W
                v = cy / RECT_H
                self._history.append((u, v))
                self._uv  = (u, v)
                fu = self._oef_u(u, t)
                fv = self._oef_v(v, t)
                self._fuv = (fu, fv)
            else:
                self.lost += 1
                if self.lost > 10:
                    self.last_px = None
        return det

    def stable_uv(self, n=5, radius_cm=2.5):
        with self._lock:
            hist = list(self._history)
        if len(hist) < n:
            return None
        recent = hist[-n:]
        cu, cv = np.mean(recent, axis=0)
        for u, v in recent:
            d = math.sqrt(((u-cu)*PAPER_X_CM)**2 + ((v-cv)*PAPER_Y_CM)**2)
            if d > radius_cm:
                return None
        return (float(cu), float(cv))

    def clear_history(self):
        with self._lock:
            self._history.clear()
            self._oef_u.reset()
            self._oef_v.reset()
            self._fuv = None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed',    type=int,   default=50,   help='Roll speed 0-255')
    ap.add_argument('--duration', type=float, default=2.0,  help='Roll duration (s)')
    ap.add_argument('--heading',      type=int,   default=0,    help='Roll heading 0-359 deg')
    ap.add_argument('--pre_rotate_s', type=float, default=0.8,  help='Seconds to rotate in place before rolling')
    ap.add_argument('--pause',        type=float, default=0.6,  help='Settle pause after roll (s)')
    ap.add_argument('--use_compass',  action='store_true',       help='Calibrate heading with magnetometer at startup')
    ap.add_argument('--compass_north',type=int,   default=0,    help='Compass heading that means "toward far edge of paper"')
    ap.add_argument('--sphero',       type=str,   default='',   help='Sphero name e.g. SB-FD03')
    ap.add_argument('--p2',       type=int,   default=DEF_P2)
    ap.add_argument('--min_r',    type=int,   default=DEF_MIN_R)
    ap.add_argument('--max_r',    type=int,   default=DEF_MAX_R)
    ap.add_argument('--blur',     type=int,   default=DEF_BLUR)
    args = ap.parse_args()

    print("Loading calibration…")
    calib  = load_calibration()
    H_rect = build_homography(calib)
    print(f"  Paper: {calib['paper_x_m']*100:.0f} × {calib['paper_y_m']*100:.0f} cm")

    print("Starting RealSense…")
    pipe = start_camera()

    print(f"Scanning for Sphero{(' ' + args.sphero) if args.sphero else ''}…")
    toy = scanner.find_toy(toy_name=args.sphero) if args.sphero else scanner.find_toy()
    print(f"Connected: {toy.name}")
    print("\nAdjust Hough trackbars until Sphero is detected, then press ENTER.\n")

    WIN_MAIN = "Sphero Calib  [ENTER=roll | Q=quit]"
    WIN_DBG  = "Hough input (CLAHE + blur)"
    WIN_TB   = "Hough Controls"

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, RECT_W, RECT_H)
    cv2.namedWindow(WIN_DBG, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_DBG, RECT_W // 2, RECT_H // 2)
    cv2.namedWindow(WIN_TB, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("heading (deg)",     WIN_TB, args.heading,            359, lambda _: None)
    cv2.createTrackbar("speed  (0-255)",    WIN_TB, args.speed,              255, lambda _: None)
    cv2.createTrackbar("dur x10 (0.1-5s)", WIN_TB, int(args.duration * 10), 50,  lambda _: None)
    cv2.createTrackbar("p2  (lower=more)",  WIN_TB, args.p2,                 80,  lambda _: None)
    cv2.createTrackbar("min radius (px)",   WIN_TB, args.min_r,              100, lambda _: None)
    cv2.createTrackbar("max radius (px)",   WIN_TB, args.max_r,              200, lambda _: None)
    cv2.createTrackbar("blur kernel (px)",  WIN_TB, args.blur,               21,  lambda _: None)
    cv2.createTrackbar("CLAHE clip x10",    WIN_TB, DEF_CLAHE,               60,  lambda _: None)
    cv2.createTrackbar("OEF cutoff x10",    WIN_TB, 10,                      50,  lambda _: None)
    cv2.createTrackbar("OEF beta   x100",   WIN_TB, 5,                       200, lambda _: None)

    tracker  = SpheroTracker()
    trail    = deque(maxlen=400)
    state    = STATE_WAITING
    start_uv = None
    end_uv   = None
    results  = None
    clahe    = make_clahe(DEF_CLAHE / 10.0)
    last_clahe_val = DEF_CLAHE

    with SpheroEduAPI(toy) as droid:
        droid.set_main_led(BLUE)

        if args.use_compass:
            print("\n" + "=" * 54)
            print("  STEP 1 — Magnetometer calibration")
            print("  The Sphero will now spin in a figure-8 to calibrate its compass.")
            print("  Place it on a flat surface and do NOT touch it.")
            input("  Press ENTER when ready > ")
            droid.calibrate_compass()
            print("  Magnetometer calibrated.")
            print()
            print("  STEP 2 — Set heading=0")
            print("  The back LED (tail light) will turn BRIGHT BLUE.")
            print("  Rotate the Sphero shell so the blue light points AWAY from the far edge.")
            print("  The front (no light) should face the far edge of the paper.")
            droid.set_back_led(255)      # bright blue tail light — shows where the back is
            droid.set_stabilization(False)
            input("  Press ENTER when aimed > ")
            compass_deg = droid.get_compass_direction()
            droid.reset_aim()
            droid.set_stabilization(True)
            droid.set_back_led(0)        # turn off tail light
            print(f"  Compass reading at far-edge direction: {compass_deg} deg")
            print(f"  Heading=0 locked to far edge of paper.")
            print(f"  Save this for next session:  --compass_north {compass_deg}")
            print("=" * 54 + "\n")
        elif args.compass_north != 0:
            print("\n" + "=" * 54)
            print(f"  RESTORING AIM from saved compass heading {args.compass_north}°")
            droid.set_compass_direction(args.compass_north)
            print("  Done. Heading=0 = toward far edge of paper.")
            print("=" * 54 + "\n")

        def do_roll(heading, speed, duration):
            nonlocal state, start_uv, end_uv, results

            # Grab stable start position — prefer filtered UV for accuracy
            su = tracker.stable_uv(n=5, radius_cm=2.5)
            if su is None:
                su = tracker.filtered_uv
            start_uv = su
            print(f"  Start: ({start_uv[0]:.3f}, {start_uv[1]:.3f})")
            print(f"  Pre-rotating to heading={heading}° for {args.pre_rotate_s:.1f}s…")

            state = STATE_ROLLING
            droid.set_main_led(YELLOW)
            droid.roll(heading, 0, args.pre_rotate_s)   # rotate in place, no movement

            print(f"  Rolling heading={heading}  speed={speed}  duration={duration:.1f}s…")
            droid.roll(heading, speed, duration)
            droid.set_speed(0)

            state = STATE_SETTLING
            tracker.clear_history()
            time.sleep(args.pause)
            time.sleep(0.25)   # a few extra frames to fill tracker history

            eu = tracker.stable_uv(n=4, radius_cm=3.0)
            if eu is None:
                eu = tracker.filtered_uv
            end_uv = eu

            droid.set_main_led(BLUE)

            if start_uv is None or end_uv is None:
                print("  ERROR: could not read Sphero position before or after roll.")
                state = STATE_WAITING
                return

            du = end_uv[0] - start_uv[0]
            dv = end_uv[1] - start_uv[1]
            dist_cm       = math.sqrt((du*PAPER_X_CM)**2 + (dv*PAPER_Y_CM)**2)
            cm_per_sec    = dist_cm / duration if duration > 0 else 0.0

            # paper_direction compatible with sphero_pointing_node's atan2(du,-dv).
            # calib_test UV is both horizontally and vertically mirrored vs detection_node UV
            # (d435i_driver flips once; calib homography orientation ends up opposite).
            # atan2(-du, dv) on calib UV = atan2(du_det, -dv_det) on detection UV. ✓
            # Convention in this formula: 0° = far edge, 180° = near edge.
            paper_direction = math.degrees(math.atan2(-du, dv)) % 360

            # heading_offset: correction to use in sphero_pointing_node / sphero_move_test.
            # Derived from:  sphero_heading = heading_offset - paper_angle
            # → heading_offset = heading_used + paper_direction
            # With compass calibration this should be ≈ 0 (no correction needed).
            heading_offset = (heading + paper_direction) % 360

            results = dict(du=du, dv=dv, dist_cm=dist_cm,
                           heading_used=heading, paper_direction=paper_direction,
                           heading_offset=heading_offset,
                           cm_per_sec=cm_per_sec, speed=speed, duration=duration)

            print()
            print("=" * 54)
            print(f"  start            = ({start_uv[0]:.3f}, {start_uv[1]:.3f})")
            print(f"  end              = ({end_uv[0]:.3f}, {end_uv[1]:.3f})")
            print(f"  heading used     = {heading} deg")
            print(f"  du={du:+.3f}  dv={dv:+.3f}  dist={dist_cm:.1f} cm")
            print(f"  paper direction  = {paper_direction:.1f} deg  (UV-space angle of movement)")
            print(f"  heading_offset   = {heading_offset:.1f} deg  (use in sphero_pointing_node)")
            print(f"  measured speed   = {cm_per_sec:.1f} cm/s  (speed={speed})")
            print("=" * 54)
            print()

            state = STATE_DONE

        while True:
            # Read trackbars
            tb_heading  = cv2.getTrackbarPos("heading (deg)",     WIN_TB)
            tb_speed    = cv2.getTrackbarPos("speed  (0-255)",    WIN_TB)
            tb_dur_x10  = max(1, cv2.getTrackbarPos("dur x10 (0.1-5s)", WIN_TB))
            tb_duration = tb_dur_x10 / 10.0
            p2          = cv2.getTrackbarPos("p2  (lower=more)",  WIN_TB)
            min_r       = cv2.getTrackbarPos("min radius (px)",   WIN_TB)
            max_r       = cv2.getTrackbarPos("max radius (px)",   WIN_TB)
            blur        = cv2.getTrackbarPos("blur kernel (px)",  WIN_TB)
            clahe_val    = cv2.getTrackbarPos("CLAHE clip x10",    WIN_TB)
            oef_cut_val  = max(1, cv2.getTrackbarPos("OEF cutoff x10",  WIN_TB))
            oef_beta_val = max(1, cv2.getTrackbarPos("OEF beta   x100", WIN_TB))
            if clahe_val != last_clahe_val:
                clahe = make_clahe(max(0.1, clahe_val / 10.0))
                last_clahe_val = clahe_val
            tracker.set_oef_params(oef_cut_val / 10.0, oef_beta_val / 100.0)

            frame = get_frame(pipe)
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            t_now          = time.time()
            rect           = cv2.warpPerspective(frame, H_rect, (RECT_W, RECT_H))
            cands, blr_img = detect_sphero(rect, p2, min_r, max_r, blur, clahe)
            det            = tracker.update(cands, t_now)
            fuv            = tracker.filtered_uv

            if det and fuv is not None:
                trail.append((int(fuv[0] * RECT_W), int(fuv[1] * RECT_H)))

            # ── Vertical-flip display (far side of paper at top) ───────────────
            disp = cv2.flip(rect, 0)

            def d(x, y):
                return (x, RECT_H - 1 - y)

            # Grid (1 cm = 10 px in rectified view)
            for x in range(0, RECT_W+1, 100):
                cv2.line(disp, (x, 0), (x, RECT_H), (45, 45, 45), 1)
            for y in range(0, RECT_H+1, 100):
                cv2.line(disp, (0, y), (RECT_W, y), (45, 45, 45), 1)

            # Trail
            pts = list(trail)
            for i in range(1, len(pts)):
                a   = i / len(pts)
                col = (int(50*a), int(200*a), int(255*(1-a) + 80*a))
                cv2.line(disp, d(*pts[i-1]), d(*pts[i]), col, max(1, int(3*a)))

            # All candidates (faint)
            for cx, cy, r in cands:
                cv2.circle(disp, d(cx, cy), r, (60, 60, 60), 1)

            # Confirmed detection
            stable = tracker.stable_uv()
            if det:
                cx, cy, r = det
                dx, dy = d(cx, cy)
                # Raw Hough ring — faint grey so jitter is still visible
                cv2.circle(disp, (dx, dy), max(r, 6), (100, 100, 100), 1)
                # Filtered position — solid green dot
                if fuv is not None:
                    fx = int(fuv[0] * RECT_W)
                    fy = int((1.0 - fuv[1]) * RECT_H)
                    dot_col = (0, 255, 0) if stable else (0, 200, 255)
                    cv2.circle(disp, (fx, fy), 7, dot_col, -1)
                    label = f"u={fuv[0]:.3f}  v={fuv[1]:.3f}  r={r}px"
                    if stable:
                        label += "  [STABLE]"
                    cv2.putText(disp, label, (fx+10, fy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, dot_col, 1)
            else:
                cv2.putText(disp, "Searching…  tune Hough Controls", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

            # Start marker
            if start_uv is not None:
                sx = int(start_uv[0] * RECT_W)
                sy = int((1.0 - start_uv[1]) * RECT_H)
                cv2.drawMarker(disp, (sx, sy), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(disp, "START", (sx+10, sy-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            # End marker
            if end_uv is not None:
                ex = int(end_uv[0] * RECT_W)
                ey = int((1.0 - end_uv[1]) * RECT_H)
                cv2.drawMarker(disp, (ex, ey), (0, 100, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(disp, "END", (ex+10, ey-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 100, 255), 1)

            # Arrow start → end
            if start_uv is not None and end_uv is not None:
                sx = int(start_uv[0] * RECT_W)
                sy = int((1.0 - start_uv[1]) * RECT_H)
                ex = int(end_uv[0] * RECT_W)
                ey = int((1.0 - end_uv[1]) * RECT_H)
                cv2.arrowedLine(disp, (sx, sy), (ex, ey), (255, 255, 0), 2, tipLength=0.15)

            # State banner / status bar
            if state == STATE_WAITING:
                prompt = "ENTER to roll" if (det or tracker.uv) else "No detection — tune Hough then ENTER"
                cv2.putText(disp, prompt, (10, RECT_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                params = (f"heading={tb_heading}deg  speed={tb_speed}  dur={tb_duration:.1f}s  "
                          f"p2={p2}  r={min_r}-{max_r}  blur={max(3,blur|1)}  CLAHE={clahe_val/10:.1f}")
                cv2.putText(disp, params, (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)
            elif state == STATE_ROLLING:
                cv2.putText(disp, "ROLLING…", (RECT_W//2 - 75, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 3)
                if results:
                    cv2.putText(disp,
                                f"h={results['heading_used']}deg  spd={results['speed']}  dur={results['duration']:.1f}s",
                                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)
            elif state == STATE_SETTLING:
                cv2.putText(disp, "Settling…", (RECT_W//2 - 65, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 200), 2)
            elif state == STATE_DONE and results:
                res = results
                line = (f"h={res['heading_used']}deg  spd={res['speed']}  dur={res['duration']:.1f}s  "
                        f"dist={res['dist_cm']:.1f}cm  paper_dir={res['paper_direction']:.1f}deg  "
                        f"offset={res['heading_offset']:.1f}deg  {res['cm_per_sec']:.1f}cm/s")
                cv2.putText(disp, line, (10, RECT_H - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 1)
                cv2.putText(disp, "ENTER = roll again   Q = quit",
                            (10, RECT_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

            cv2.imshow(WIN_MAIN, disp)

            # Debug window: what Hough sees (CLAHE + blurred grayscale)
            dbg = cv2.cvtColor(blr_img, cv2.COLOR_GRAY2BGR)
            for cx, cy, r in cands:
                cv2.circle(dbg, (cx, cy), max(r, 4), (0, 255, 255), 1)
                cv2.circle(dbg, (cx, cy), 3, (0, 0, 255), -1)
            if det:
                cx, cy, r = det
                cv2.circle(dbg, (cx, cy), max(r, 6), (0, 255, 0), 2)
            cv2.imshow(WIN_DBG, dbg)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == 13 and state in (STATE_WAITING, STATE_DONE):   # Enter
                if tracker.uv is None:
                    print("No Sphero detected yet — cannot roll.")
                else:
                    trail.clear()
                    start_uv = end_uv = results = None
                    state = STATE_WAITING
                    print("\n" + "=" * 54)
                    print(f"  ROLL  heading={tb_heading}deg  speed={tb_speed}  dur={tb_duration:.1f}s")
                    threading.Thread(
                        target=do_roll,
                        args=(tb_heading, tb_speed, tb_duration),
                        daemon=True
                    ).start()

        droid.set_speed(0)
        droid.set_main_led(OFF)

    cv2.destroyAllWindows()
    pipe.stop()
    print("Done.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nInterrupted.")
