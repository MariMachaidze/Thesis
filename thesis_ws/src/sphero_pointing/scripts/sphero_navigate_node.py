#!/usr/bin/env python3
"""
sphero_navigate_node.py
Navigates Sphero BOLT toward the pointing target with hover behaviour.

Subscribed:
  /d435i/rgb/image_raw      sensor_msgs/Image
  /camera/calibration       camera_calibration/CalibrationData  (latched)
  /sphero/detection         geometry_msgs/Point  (x=u, y=v, z=radius_px)
  /pointing/target          pointing_localization/PointingTarget

Parameters:
  ~sphero_name     str    e.g. "SB-FD03"
  ~heading_offset  float  degrees (default 0.0)
  ~fx, ~fy, ~cx, ~cy       camera intrinsics (same defaults as detection node)

Keys (display window): Q = quit
"""

import math
import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI

from camera_calibration.msg import CalibrationData
from pointing_localization.msg import PointingTarget

# ── Constants ─────────────────────────────────────────────────────────────────
SPEED             = 50
UPDATE_INTERVAL   = 0.15  # roll command duration (s) — short so heading updates quickly
ARRIVAL_RADIUS_CM = 3.0   # cm — pass-through radius; robot switches to next waypoint here
STUCK_TIMEOUT_S   = 2.5   # s  — skip waypoint if pos hasn't moved for this long
STUCK_MOVE_CM     = 1.5   # cm — minimum displacement to reset stuck timer
QUEUE_DWELL     = 3     # 10 Hz samples (300 ms) finger must stay in same square to enqueue
PAPER_X_CM      = 85.0
PAPER_Y_CM      = 60.0
BORDER_CM       = 5.0
SQUARE_CM       = 10.0
RECT_W          = 850
RECT_H          = 600


def _build_grid():
    centers = []
    x = BORDER_CM + SQUARE_CM / 2.0
    while x <= PAPER_X_CM - BORDER_CM - SQUARE_CM / 2.0 + 1e-9:
        y = BORDER_CM + SQUARE_CM / 2.0
        while y <= PAPER_Y_CM - BORDER_CM - SQUARE_CM / 2.0 + 1e-9:
            centers.append((round(x, 1), round(y, 1)))
            y += SQUARE_CM
        x += SQUARE_CM
    return centers


GRID = _build_grid()


def _nearest_square_uv(u, v):
    """Map raw (u,v) to the center of the nearest grid square, returned as (u,v)."""
    px, py = u * PAPER_X_CM, v * PAPER_Y_CM
    best, best_d = GRID[0], float('inf')
    for cx, cy in GRID:
        d = math.hypot(px - cx, py - cy)
        if d < best_d:
            best_d, best = d, (cx, cy)
    return (best[0] / PAPER_X_CM, best[1] / PAPER_Y_CM)

STATE_AIM      = 'aim'
STATE_TRACKING = 'tracking'


# ── Navigator ─────────────────────────────────────────────────────────────────
class Navigator:
    """Runs in a daemon thread — reads shared UV, drives Sphero."""

    def __init__(self, droid, get_sphero_uv, get_target_uv, heading_offset, on_arrive=None):
        self.droid          = droid
        self.get_sphero_uv  = get_sphero_uv
        self.get_target_uv  = get_target_uv
        self.heading_offset = heading_offset
        self.active         = False
        self.last_heading   = None
        self.last_dist_cm   = None
        self._thread        = None
        self._on_arrive     = on_arrive

    def start(self):
        self.active  = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False

    def _run(self):
        _was_lost    = False
        _stuck_pos   = None
        _stuck_since = None
        _in_border   = False

        BORDER_U = BORDER_CM / PAPER_X_CM
        BORDER_V = BORDER_CM / PAPER_Y_CM

        while self.active and not rospy.is_shutdown():
            pos    = self.get_sphero_uv()
            target = self.get_target_uv()

            if pos is None:
                if not _was_lost:
                    rospy.loginfo_throttle(2, "[NAV] waiting for Sphero detection…")
                    _was_lost = True
                self.droid.set_speed(0)
                time.sleep(0.1)
                continue

            if _was_lost:
                rospy.loginfo("[NAV] detection regained — resuming")
                _was_lost = False

            if target is None:
                self.droid.set_speed(0)
                time.sleep(0.1)
                continue

            # Border escape: reverse if robot drifted into exclusion zone
            in_border_now = (pos[0] < BORDER_U or pos[0] > 1.0 - BORDER_U or
                             pos[1] < BORDER_V or pos[1] > 1.0 - BORDER_V)
            if in_border_now:
                if not _in_border:
                    rospy.logwarn("[NAV] Border zone (%.1f,%.1f)cm — reversing",
                                  pos[0]*PAPER_X_CM, pos[1]*PAPER_Y_CM)
                    _in_border   = True
                    _stuck_pos   = None
                    _stuck_since = None
                reverse = (self.last_heading + 180) % 360 if self.last_heading is not None else 0
                self.droid.roll(reverse, SPEED, UPDATE_INTERVAL)
                continue

            if _in_border:
                rospy.loginfo("[NAV] Escaped border — resuming navigation")
                _in_border = False

            du = target[0] - pos[0]
            dv = target[1] - pos[1]
            dist_cm = math.hypot(du * PAPER_X_CM, dv * PAPER_Y_CM)

            # Waypoint pass-through: robot sweeps through target, no stopping
            if dist_cm < ARRIVAL_RADIUS_CM:
                rospy.loginfo("[NAV] WAYPOINT  dist=%.1fcm  target=(%.1f,%.1f)cm",
                              dist_cm, target[0]*PAPER_X_CM, target[1]*PAPER_Y_CM)
                if self._on_arrive:
                    self._on_arrive()
                _stuck_pos   = None
                _stuck_since = None
                time.sleep(0.05)
                continue

            # Stuck detection: skip waypoint if not making progress
            if _stuck_pos is None:
                _stuck_pos   = pos
                _stuck_since = time.time()
            else:
                moved = math.hypot((pos[0]-_stuck_pos[0])*PAPER_X_CM,
                                   (pos[1]-_stuck_pos[1])*PAPER_Y_CM)
                if moved > STUCK_MOVE_CM:
                    _stuck_pos   = pos
                    _stuck_since = time.time()
                elif time.time() - _stuck_since > STUCK_TIMEOUT_S:
                    rospy.loginfo("[NAV] STUCK → skip  pos=(%.1f,%.1f)cm  target=(%.1f,%.1f)cm  dist=%.1fcm",
                                  pos[0]*PAPER_X_CM, pos[1]*PAPER_Y_CM,
                                  target[0]*PAPER_X_CM, target[1]*PAPER_Y_CM, dist_cm)
                    if self._on_arrive:
                        self._on_arrive()
                    _stuck_pos   = None
                    _stuck_since = None
                    time.sleep(0.1)
                    continue

            paper_angle = math.degrees(math.atan2(du, dv)) % 360
            heading     = int(round(self.heading_offset + paper_angle)) % 360
            self.last_heading = heading
            self.droid.roll(heading, SPEED, UPDATE_INTERVAL)

        self.droid.set_speed(0)


# ── Display helpers ───────────────────────────────────────────────────────────
def disp_pt(cx, cy):
    return (cx, RECT_H - 1 - cy)

def uv_to_disp(u, v):
    return disp_pt(int(u * RECT_W), int(v * RECT_H))

def draw_grid(img, step_px=100, color=(110, 110, 110)):
    for rx in range(step_px, RECT_W, step_px):
        cv2.line(img, (rx, 0), (rx, RECT_H), color, 1)
        cv2.putText(img, f"{rx//10}cm", (rx+2, RECT_H-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    for ry in range(step_px, RECT_H, step_px):
        dy = RECT_H - 1 - ry
        cv2.line(img, (0, dy), (RECT_W, dy), color, 1)
        cv2.putText(img, f"{ry//10}cm", (2, dy-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

def draw_overlay(img, lines):
    h = 14 + 26 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (RECT_W, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 22 + 26*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def draw_sphero(img, uv, radius_px=0):
    dp = uv_to_disp(*uv)
    r  = max(int(radius_px), 8)
    cv2.circle(img, dp, r, (0, 255, 0), 2)
    cv2.circle(img, dp, 4, (0, 255, 0), -1)

def draw_target(img, uv):
    dp = uv_to_disp(*uv)
    cv2.drawMarker(img, dp, (0, 60, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(img, dp, 18, (0, 60, 255), 1)


# ── Node ──────────────────────────────────────────────────────────────────────
class SpheroNavigateNode:

    def __init__(self):
        rospy.init_node('sphero_navigate', anonymous=False)

        self.heading_offset = rospy.get_param('~heading_offset', 0.0)
        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 642.351)
        self.cy = rospy.get_param('~cy', 349.990)

        self.bridge = CvBridge()
        self.lock   = threading.Lock()

        self._frame           = None
        self._H_rect          = None
        self._sphero_uv       = None
        self._sphero_r        = 0
        self._sphero_last_det = 0.0   # time.time() of last valid detection message

        # Live pointing (updated every message)
        self._current_pointing_uv = None   # raw border-clamped (u,v)
        self._current_pointing_sq = None   # nearest grid square — used only for change detection

        # Navigation queue: robot drives to each raw (u,v) in order
        self._target_queue     = deque()   # pending raw positions
        self._last_queued_sq   = None      # last grid square enqueued (dedup)
        self._target_uv        = None      # currently active navigation target (raw)
        self._target_accept_time = None

        # Dwell tracking: finger must stay in same square for QUEUE_DWELL samples
        self._dwell_sq         = None      # square currently being dwelt in
        self._dwell_count      = 0         # consecutive samples in _dwell_sq

        rospy.Timer(rospy.Duration(1.0/60.0), self._queue_sample_cb)  # 60 Hz

        self._travel_pub = rospy.Publisher('/sphero/travel_time_s', Float64, queue_size=10)

        rospy.Subscriber('/d435i/rgb/image_raw',  Image,
                         self._image_cb,    queue_size=1)
        rospy.Subscriber('/camera/calibration',   CalibrationData,
                         self._calib_cb,    queue_size=1)
        rospy.Subscriber('/sphero/detection',     Point,
                         self._detection_cb, queue_size=1)
        rospy.Subscriber('/pointing/target',      PointingTarget,
                         self._target_cb,   queue_size=1)

        rospy.loginfo("SpheroNavigateNode ready — connecting to Sphero…")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr_throttle(5, "imgmsg_to_cv2: %s", e)
            return
        with self.lock:
            self._frame = frame

    def _calib_cb(self, msg):
        if not msg.is_calibrated:
            return
        p1 = np.array([msg.plane_center.x,  msg.plane_center.y,  msg.plane_center.z])
        ax = np.array([msg.paper_x_axis.x,   msg.paper_x_axis.y,  msg.paper_x_axis.z])
        ay = np.array([msg.paper_y_axis.x,   msg.paper_y_axis.y,  msg.paper_y_axis.z])
        Lx, Ly = float(msg.paper_x_m), float(msg.paper_y_m)
        corners = [p1, p1+ax*Lx, p1+ay*Ly, p1+ax*Lx+ay*Ly]
        src = np.float32([(self.fx*X/Z+self.cx, self.fy*Y/Z+self.cy) for X,Y,Z in corners])
        dst = np.float32([[0,0],[RECT_W,0],[0,RECT_H],[RECT_W,RECT_H]])
        H   = cv2.getPerspectiveTransform(src, dst)
        with self.lock:
            self._H_rect = H
        rospy.loginfo("Calibration received — homography ready.")

    def _detection_cb(self, msg):
        with self.lock:
            self._sphero_uv       = (float(msg.x), float(msg.y))
            self._sphero_r        = int(msg.z)
            self._sphero_last_det = time.time()

    def _target_cb(self, msg):
        if not msg.is_valid:
            return
        border_u = BORDER_CM / PAPER_X_CM
        border_v = BORDER_CM / PAPER_Y_CM
        raw_u = float(np.clip(msg.u_normalized, border_u, 1.0 - border_u))
        raw_v = float(np.clip(msg.v_normalized, border_v, 1.0 - border_v))
        sq    = _nearest_square_uv(raw_u, raw_v)   # for change detection only
        with self.lock:
            self._current_pointing_uv = (raw_u, raw_v)
            self._current_pointing_sq = sq

    def _queue_sample_cb(self, event):
        """10 Hz — enqueue if finger has dwelt in a new grid square for QUEUE_DWELL samples."""
        with self.lock:
            if self._current_pointing_uv is None:
                return
            sq     = self._current_pointing_sq
            raw_uv = self._current_pointing_uv

            # Dwell counter: reset on square change
            if sq != self._dwell_sq:
                self._dwell_sq    = sq
                self._dwell_count = 1
            else:
                self._dwell_count += 1

            # Only enqueue exactly when dwell threshold is reached
            if self._dwell_count != QUEUE_DWELL:
                return
            if sq == self._last_queued_sq:
                return   # same square as last queued — don't enqueue again

            self._target_queue.append(raw_uv)
            self._last_queued_sq = sq
            # If robot is idle, activate immediately
            if self._target_uv is None:
                self._target_uv        = self._target_queue.popleft()
                self._target_accept_time = time.time()
            q_len = len(self._target_queue)
        rospy.loginfo("[QUEUE] enqueued (%.1f,%.1f)cm  sq=(%.0f,%.0f)cm  pending=%d",
                      raw_uv[0] * PAPER_X_CM, raw_uv[1] * PAPER_Y_CM,
                      sq[0] * PAPER_X_CM, sq[1] * PAPER_Y_CM, q_len)

    # ── Shared-state accessors for Navigator ──────────────────────────────────

    def _get_sphero_uv(self):
        with self.lock:
            if self._sphero_uv is None:
                return None
            if time.time() - self._sphero_last_det > 1.0:
                return None   # stale — Sphero out of camera frame
            return self._sphero_uv

    def _get_target_uv(self):
        with self.lock:
            return self._target_uv

    def _on_arrive(self):
        with self.lock:
            t0 = self._target_accept_time
            if self._target_queue:
                self._target_uv        = self._target_queue.popleft()
                self._target_accept_time = time.time()
                rospy.loginfo("[QUEUE] next target (%.1f,%.1f)cm  remaining=%d",
                              self._target_uv[0] * PAPER_X_CM,
                              self._target_uv[1] * PAPER_Y_CM,
                              len(self._target_queue))
            else:
                self._target_uv        = None
                self._target_accept_time = None
                rospy.loginfo("[QUEUE] empty — robot idle")
        if t0 is not None:
            elapsed = time.time() - t0
            self._travel_pub.publish(Float64(data=elapsed))
            rospy.loginfo("[TIMER] travel %.2fs", elapsed)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def spin(self, droid):
        WIN = "Sphero Navigate  [Q=quit]"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, RECT_W, RECT_H)

        nav       = None
        state     = STATE_AIM
        aim_phase = 0   # 0=instructions, 1=stabilisation off

        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            with self.lock:
                frame  = self._frame.copy() if self._frame is not None else None
                H_rect = self._H_rect
                suv    = self._sphero_uv
                sr     = self._sphero_r
                tuv    = self._target_uv
                puv    = self._current_pointing_uv
                q_len  = len(self._target_queue)

            # ── Build display ───────────────────────────────────────────────
            if frame is not None and H_rect is not None:
                rect = cv2.warpPerspective(frame, H_rect, (RECT_W, RECT_H))
                disp = cv2.flip(rect, 0)
            else:
                disp = np.zeros((RECT_H, RECT_W, 3), dtype=np.uint8)
                if frame is None:
                    cv2.putText(disp, "Waiting for camera…", (20, RECT_H//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 2)
                else:
                    cv2.putText(disp, "Waiting for calibration…", (20, RECT_H//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 2)

            draw_grid(disp)

            if suv is not None:
                draw_sphero(disp, suv, sr)
            if state == STATE_TRACKING:
                # Live pointing crosshair (where finger is now)
                if puv is not None:
                    dp_p = uv_to_disp(*puv)
                    cv2.drawMarker(disp, dp_p, (0, 220, 255), cv2.MARKER_CROSS, 16, 1)
                # Active navigation target
                if tuv is not None:
                    dp_t = uv_to_disp(*tuv)
                    cv2.drawMarker(disp, dp_t, (0, 60, 255), cv2.MARKER_CROSS, 24, 2)
                    cv2.circle(disp, dp_t, 18, (0, 60, 255), 1)
                    if suv is not None:
                        cv2.line(disp, uv_to_disp(*suv), dp_t, (0, 200, 255), 1)

            # ── State overlays ───────────────────────────────────────────────
            if state == STATE_AIM:
                if aim_phase == 0:
                    draw_overlay(disp, ["AIM CALIBRATION", "Press ENTER to begin."])
                else:
                    draw_overlay(disp, [
                        "Rotate Sphero shell: glowing TAIL toward YOU.",
                        "Press ENTER when aimed.",
                    ])

            elif state == STATE_TRACKING:
                if tuv is None and puv is None:
                    draw_overlay(disp, ["Waiting for pointing target…"])
                elif nav is not None:
                    d = nav.last_dist_cm
                    h = nav.last_heading
                    if tuv is None:
                        col  = (180, 180, 180)
                        info = f"IDLE  queue={q_len} — point to a new location"
                    else:
                        col  = (0, 200, 255)
                        info = (f"NAVIGATING  dist={d:.1f}cm  hdg={h}deg  queue={q_len}"
                                if d is not None else f"NAVIGATING…  queue={q_len}")
                    cv2.putText(disp, info, (8, RECT_H-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                if nav:
                    nav.stop()
                break

            # ── State transitions ────────────────────────────────────────────
            if state == STATE_AIM and key == 13:
                if aim_phase == 0:
                    droid.set_stabilization(False)
                    droid.set_back_led(255)
                    aim_phase = 1
                    rospy.loginfo("Stabilisation off — rotate Sphero, then press ENTER.")
                else:
                    droid.reset_aim()
                    droid.set_stabilization(True)
                    droid.set_back_led(0)
                    aim_phase = 0
                    state = STATE_TRACKING
                    with self.lock:
                        self._target_queue.clear()
                        self._target_uv           = None
                        self._target_accept_time  = None
                        self._last_queued_sq      = None
                        self._current_pointing_uv = None
                        self._current_pointing_sq = None
                        self._dwell_sq            = None
                        self._dwell_count         = 0
                    rospy.loginfo("Aim reset. Queue cleared. Starting navigation.")
                    nav = Navigator(droid, self._get_sphero_uv, self._get_target_uv,
                                    self.heading_offset, on_arrive=self._on_arrive)
                    nav.start()

            rate.sleep()

        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    node = SpheroNavigateNode()

    sphero_name = rospy.get_param('~sphero_name', '')
    rospy.loginfo("Scanning for Sphero%s…", (' ' + sphero_name) if sphero_name else '')
    toy = (scanner.find_toy(toy_name=sphero_name) if sphero_name
           else scanner.find_toy())
    rospy.loginfo("Connected: %s", toy.name)

    # ROS spin in background so OpenCV runs on main thread
    spin_thread = threading.Thread(target=rospy.spin, daemon=True)
    spin_thread.start()

    with SpheroEduAPI(toy) as droid:
        node.spin(droid)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
