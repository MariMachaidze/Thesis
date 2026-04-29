#!/usr/bin/env python3
"""
sphero_move_test.py — Standalone Sphero movement tester.
No ROS required.

Uses RealSense camera + Hough detection on rectified paper view.
Loads paper calibration from ~/.ros/paper_calibration.yaml.

Controls (in the display window):
  Click        → move Sphero to clicked position
  1-5          → preset positions (TL, TR, Center, BL, BR)
  C            → redo heading calibration
  Q            → quit

Usage:
  python3 sphero_move_test.py
  python3 sphero_move_test.py --speed 80 --step 1.0 --sphero SB-FD03
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
WIDTH, HEIGHT, FPS = 1280, 720, 30
RECT_W, RECT_H     = 850, 600   # rectified paper: 1 px = 1 mm

BLUE   = Color(r=0,   g=0,   b=255)
YELLOW = Color(r=255, g=200, b=0  )
ORANGE = Color(r=255, g=100, b=0  )
GREEN  = Color(r=0,   g=200, b=0  )
OFF    = Color(r=0,   g=0,   b=0  )

HOUGH_DP, HOUGH_MINDIST, HOUGH_P1      = 1.2, 40, 80
DEF_P2, DEF_MINR, DEF_MAXR, DEF_BLUR  = 30, 18, 42, 6
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

PAPER_X_CM, PAPER_Y_CM = 85.0, 60.0

PRESETS = {
    ord('1'): (0.15, 0.15),
    ord('2'): (0.85, 0.15),
    ord('3'): (0.50, 0.50),
    ord('4'): (0.15, 0.85),
    ord('5'): (0.85, 0.85),
}
PRESET_LABELS = {
    ord('1'): 'TL', ord('2'): 'TR', ord('3'): 'C',
    ord('4'): 'BL', ord('5'): 'BR',
}


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
    cx = float(calib['intrinsics']['cx'])
    cy = float(calib['intrinsics']['cy'])

    corners = [p1, p1+ax*Lx, p1+ax*Lx+ay*Ly, p1+ay*Ly]
    src = np.float32([[fx*X/Z+cx, fy*Y/Z+cy] for X, Y, Z in corners])
    dst = np.float32([[0,0],[RECT_W,0],[RECT_W,RECT_H],[0,RECT_H]])
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
    # Mirror to match d435i_driver_node which publishes cv2.flip(frame, 1)
    return cv2.flip(np.asanyarray(cf.get_data()), 1)

# ── Sphero detection ──────────────────────────────────────────────────────────

def detect_sphero(rect, p2, min_r, max_r, blur_k):
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    enh  = _clahe.apply(gray)
    k    = max(3, blur_k | 1)
    blr  = cv2.GaussianBlur(enh, (k, k), 0)
    cirs = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT,
                            dp=HOUGH_DP, minDist=HOUGH_MINDIST,
                            param1=HOUGH_P1, param2=max(1, p2),
                            minRadius=max(1, min_r), maxRadius=max(2, max_r))
    if cirs is None:
        return []
    return [(int(cx), int(cy), int(r)) for cx, cy, r in np.round(cirs[0]).astype(int)]


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


class SpheroTracker:
    """Thread-safe Sphero position tracker."""

    def __init__(self):
        self._lock   = threading.Lock()
        self.last_px = None
        self._uv     = None
        self._history = deque(maxlen=8)
        self.lost    = 0

    @property
    def uv(self):
        with self._lock:
            return self._uv

    def update(self, candidates):
        det = pick_best(candidates, self.last_px)
        with self._lock:
            if det:
                cx, cy, r    = det
                self.last_px = (cx, cy)
                self.lost    = 0
                uv = (cx / RECT_W, cy / RECT_H)
                self._history.append(uv)
                self._uv = uv
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


# ── Mover ─────────────────────────────────────────────────────────────────────

class SpheroMover:
    """
    Heading calibration + closed-loop movement.

    Movement runs in a background thread so the display loop never freezes.
    Step duration is scaled by distance using cm_per_sec measured at calibration,
    so the Sphero does not overshoot on the first step.
    """

    def __init__(self, droid, tracker):
        self.droid          = droid
        self.tracker        = tracker
        self.heading_offset = 0.0
        self.cm_per_sec     = 0.0     # measured at calibration; 0 = unknown
        self.current_uv     = np.array([0.5, 0.5])
        self.is_moving      = False   # True while calibrating or moving

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _wait_stable(self, timeout=12.0):
        """Block until tracker reports a stable position, or timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.tracker.stable_uv()
            if s:
                return s
            time.sleep(0.1)
        return None

    def _wait_settled(self, pause):
        """
        After a roll, clear old history and sleep long enough for the
        main display loop to grab fresh post-roll frames into the tracker.
        """
        self.tracker.clear_history()
        time.sleep(pause)          # Sphero coasts to stop
        time.sleep(0.2)            # a few extra frames for main loop to fill history

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, args):
        """
        Roll heading=0, measure actual direction + speed from camera.
        Designed to run in a background thread while the main loop stays live.
        Sets heading_offset and cm_per_sec.
        """
        print("\n" + "="*52)
        print("  HEADING CALIBRATION — rolling heading=0")

        start = self._wait_stable(timeout=12.0)
        if start is None:
            print("  Camera cannot find Sphero — skipped (heading_offset unchanged)")
            print("="*52 + "\n")
            self.is_moving = False
            return

        print(f"  Start confirmed: ({start[0]:.3f}, {start[1]:.3f})")
        print(f"  Rolling for {args.step:.1f}s at speed {args.speed}…")

        self.droid.set_main_led(ORANGE)
        self.droid.roll(0, args.speed, args.step)   # blocks in bg thread — fine
        self.droid.set_speed(0)
        self._wait_settled(args.pause)

        # Use stable_uv for end — single tracker.uv can be a false positive at edges
        end = self.tracker.stable_uv(n=4, radius_cm=3.0)
        if end is None:
            end = self.tracker.uv   # fallback if not enough samples yet

        self.droid.set_main_led(BLUE)

        if end is None:
            print("  Lost Sphero after roll — skipped")
            print("="*52 + "\n")
            self.is_moving = False
            return

        du = end[0] - start[0]
        dv = end[1] - start[1]
        dist_cm = math.sqrt((du*PAPER_X_CM)**2 + (dv*PAPER_Y_CM)**2)

        if dist_cm < 1.5:
            print(f"  Barely moved ({dist_cm:.1f} cm) — try --speed {args.speed+20} or --step {args.step+0.5:.1f}")
            print("="*52 + "\n")
            self.is_moving = False
            return

        self.heading_offset = math.degrees(math.atan2(du, -dv)) % 360
        self.cm_per_sec     = dist_cm / args.step
        self.current_uv     = np.array(end)
        print(f"  start=({start[0]:.3f},{start[1]:.3f})  end=({end[0]:.3f},{end[1]:.3f})")

        print(f"  Moved {dist_cm:.1f} cm → heading_offset={self.heading_offset:.1f}°  "
              f"measured speed={self.cm_per_sec:.1f} cm/s")
        print("="*52 + "\n")
        self.is_moving = False

    # ── Movement ──────────────────────────────────────────────────────────────

    def move_to_bg(self, target_uv, args):
        """
        Single-shot movement. Runs in a background thread.

        Computes exact roll duration from measured cm_per_sec, rolls once,
        then reads camera to report where the Sphero actually landed.
        """
        target = np.array(target_uv)

        # Read current position
        uv = self.tracker.uv
        if uv:
            self.current_uv = np.array(uv)

        du = target[0] - self.current_uv[0]
        dv = target[1] - self.current_uv[1]
        dist_cm = math.sqrt((du*PAPER_X_CM)**2 + (dv*PAPER_Y_CM)**2)

        if dist_cm < args.thresh:
            print(f"  Already within threshold ({dist_cm:.1f} cm) — no move")
            self.is_moving = False
            return

        angle_paper = math.degrees(math.atan2(du, -dv)) % 360
        heading     = int(round(self.heading_offset - angle_paper)) % 360

        if self.cm_per_sec > 0:
            duration = dist_cm / self.cm_per_sec
        else:
            duration = args.step   # fallback if calibration didn't measure speed

        print(f"\n→ ({self.current_uv[0]:.3f},{self.current_uv[1]:.3f}) "
              f"→ target ({target[0]:.3f},{target[1]:.3f})  "
              f"dist={dist_cm:.1f}cm  h={heading}°  dur={duration:.2f}s")

        self.droid.set_main_led(YELLOW)
        self.droid.roll(heading, args.speed, duration)
        self.droid.set_speed(0)

        self._wait_settled(args.pause)
        self.droid.set_main_led(BLUE)

        uv = self.tracker.uv
        if uv:
            self.current_uv = np.array(uv)
            err_mm = math.sqrt(((uv[0]-target[0])*PAPER_X_CM*10)**2 +
                               ((uv[1]-target[1])*PAPER_Y_CM*10)**2)
            print(f"  landed ({uv[0]:.3f},{uv[1]:.3f})  err={err_mm:.0f} mm")
        else:
            print("  No camera reading after roll")

        self.is_moving = False


# ── Mouse callback ────────────────────────────────────────────────────────────

_click_uv = [None]

def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Display is flipped vertically — un-flip v to get paper (u,v)
        _click_uv[0] = (x / RECT_W, 1.0 - y / RECT_H)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed',  type=int,   default=80,   help='Roll speed 0-255')
    ap.add_argument('--step',   type=float, default=1.0,  help='Max step duration (s)')
    ap.add_argument('--pause',  type=float, default=0.5,  help='Settle pause after each step (s)')
    ap.add_argument('--thresh', type=float, default=3.0,  help='Arrival threshold (cm)')
    ap.add_argument('--sphero', type=str,   default='',   help='Sphero name e.g. SB-FD03')
    ap.add_argument('--p2',     type=int,   default=DEF_P2)
    ap.add_argument('--min_r',  type=int,   default=DEF_MINR)
    ap.add_argument('--max_r',  type=int,   default=DEF_MAXR)
    ap.add_argument('--blur',   type=int,   default=DEF_BLUR)
    args = ap.parse_args()

    print("Loading calibration…")
    calib  = load_calibration()
    H_rect = build_homography(calib)
    print(f"  Paper: {calib['paper_x_m']*100:.0f} × {calib['paper_y_m']*100:.0f} cm")

    print("Starting RealSense…")
    pipe = start_camera()

    print(f"Scanning for Sphero{(' ' + args.sphero) if args.sphero else ''}…")
    toy = scanner.find_toy(toy_name=args.sphero) if args.sphero else scanner.find_toy()
    print(f"Connected: {toy.name}\n")

    WIN    = "Sphero Test  [click=target | 1-5=presets | C=calibrate | Q=quit]"
    WIN_TB = "Hough Controls"

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, RECT_W, RECT_H)
    cv2.setMouseCallback(WIN, _on_mouse)
    cv2.namedWindow(WIN_TB, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("p2 (sensitivity)", WIN_TB, args.p2,    80,  lambda _: None)
    cv2.createTrackbar("min radius (px)",  WIN_TB, args.min_r, 100, lambda _: None)
    cv2.createTrackbar("max radius (px)",  WIN_TB, args.max_r, 200, lambda _: None)
    cv2.createTrackbar("blur kernel",      WIN_TB, args.blur,  15,  lambda _: None)

    tracker   = SpheroTracker()
    trail     = deque(maxlen=60)
    target_uv = None

    with SpheroEduAPI(toy) as droid:
        droid.set_main_led(BLUE)
        mover = SpheroMover(droid, tracker)

        # Initial calibration — runs in background so display stays live
        mover.is_moving = True
        cal_thread = threading.Thread(target=mover.calibrate, args=(args,), daemon=True)
        cal_thread.start()

        print("Controls:")
        print("  Click on window → set target")
        print("  1-5             → presets (TL, TR, Center, BL, BR)")
        print("  C               → redo heading calibration")
        print("  Q               → quit\n")

        # ── Main display loop ─────────────────────────────────────────────────
        while True:
            # Read trackbars
            args.p2    = cv2.getTrackbarPos("p2 (sensitivity)", WIN_TB)
            args.min_r = cv2.getTrackbarPos("min radius (px)",  WIN_TB)
            args.max_r = cv2.getTrackbarPos("max radius (px)",  WIN_TB)
            args.blur  = cv2.getTrackbarPos("blur kernel",      WIN_TB)

            # Grab frame and detect — always running
            frame = get_frame(pipe)
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            rect  = cv2.warpPerspective(frame, H_rect, (RECT_W, RECT_H))
            cands = detect_sphero(rect, args.p2, args.min_r, args.max_r, args.blur)
            det   = tracker.update(cands)

            if det:
                trail.append((det[0], det[1]))

            # ── Draw — vertical flip so far side appears at top ──────────────────
            # d(x,y): rectified → display coords after vertical flip.
            # duv(u,v): normalised paper coords → display coords.
            disp = cv2.flip(rect, 0)

            def d(x, y):
                return (x, RECT_H - 1 - y)

            def duv(u, v):
                return (int(u * RECT_W), int((1.0 - v) * RECT_H))

            # Grid (symmetric — looks the same after 180° rotation)
            for x in range(0, RECT_W+1, 100):
                cv2.line(disp, (x, 0), (x, RECT_H), (50, 50, 50), 1)
            for y in range(0, RECT_H+1, 100):
                cv2.line(disp, (0, y), (RECT_W, y), (50, 50, 50), 1)

            # Trail
            pts = list(trail)
            for i in range(1, len(pts)):
                a   = i / len(pts)
                col = (int(50*a), int(200*a), int(255*(1-a)+80*a))
                cv2.line(disp, d(*pts[i-1]), d(*pts[i]), col, max(1, int(3*a)))

            # All Hough candidates (faint)
            for cx, cy, r in cands:
                cv2.circle(disp, d(cx, cy), r, (60, 60, 60), 1)

            # Confirmed Sphero
            if det:
                cx, cy, r = det
                dx, dy = d(cx, cy)
                cv2.circle(disp, (dx, dy), max(r, 4), (0, 255, 255), 2)
                cv2.circle(disp, (dx, dy), 4, (0, 0, 255), -1)
                cv2.putText(disp, f"u={cx/RECT_W:.3f}  v={cy/RECT_H:.3f}",
                            (dx+r+4, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            else:
                cv2.putText(disp, "Searching…", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            # Current target crosshair
            if target_uv:
                tx, ty = duv(*target_uv)
                cv2.line(disp, (tx-20, ty), (tx+20, ty), (0, 255, 0), 2)
                cv2.line(disp, (tx, ty-20), (tx, ty+20), (0, 255, 0), 2)
                cv2.circle(disp, (tx, ty), 10, (0, 255, 0), 2)

            # Preset labels
            for key, (pu, pv) in PRESETS.items():
                px, py = duv(pu, pv)
                cv2.putText(disp, PRESET_LABELS[key], (px-8, py+6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            # Status bar
            status = (f"speed={args.speed}  step={args.step:.1f}s  "
                      f"offset={mover.heading_offset:.1f}°  "
                      f"cm/s={mover.cm_per_sec:.1f}")
            cv2.putText(disp, status, (6, RECT_H-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Moving banner
            if mover.is_moving:
                cv2.putText(disp, "MOVING", (RECT_W//2-55, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

            cv2.imshow(WIN, disp)

            # ── Input ─────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c') and not mover.is_moving:
                mover.is_moving = True
                threading.Thread(target=mover.calibrate, args=(args,), daemon=True).start()

            elif key in PRESETS and not mover.is_moving:
                target_uv = PRESETS[key]
                print(f"Preset {PRESET_LABELS[key]}: ({target_uv[0]:.2f},{target_uv[1]:.2f})")
                mover.is_moving = True
                threading.Thread(target=mover.move_to_bg, args=(target_uv, args), daemon=True).start()

            elif _click_uv[0] is not None and not mover.is_moving:
                target_uv = _click_uv[0]
                _click_uv[0] = None
                print(f"Click: ({target_uv[0]:.3f},{target_uv[1]:.3f})")
                mover.is_moving = True
                threading.Thread(target=mover.move_to_bg, args=(target_uv, args), daemon=True).start()

        # ── Cleanup ───────────────────────────────────────────────────────────
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
