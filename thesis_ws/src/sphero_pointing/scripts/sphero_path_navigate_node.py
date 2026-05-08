#!/usr/bin/env python3
"""
sphero_path_navigate_node.py

Navigates Sphero BOLT along a PCHIP-interpolated path using pure pursuit.
As your finger moves, waypoints are committed every MIN_WAYPOINT_DIST_CM and
the path grows in real time — trace a route with your finger, the robot follows.

Flow:
  1. AIM state: calibrate Sphero heading (ENTER × 2)
  2. TRACKING state: move your finger over the paper.
     Every 1 cm of finger movement commits a new waypoint and extends the path.
     The robot follows the growing path using pure pursuit.
  3. ARRIVED: press R to trace a new route, Q to quit.

Subscribed:
  /d435i/rgb/image_raw    sensor_msgs/Image
  /camera/calibration     camera_calibration/CalibrationData (latched)
  /sphero/detection       geometry_msgs/Point  (x=u, y=v, z=radius_px)
  /pointing/target        pointing_localization/PointingTarget

Parameters:
  ~sphero_name      str    e.g. "SB-FD03"
  ~heading_offset   float  degrees (default 0.0)
  ~fx, ~fy, ~cx, ~cy       camera intrinsics
"""

import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from scipy.interpolate import PchipInterpolator
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from std_msgs.msg import Float64

from camera_calibration.msg import CalibrationData
from pointing_localization.msg import PointingTarget

# ── Navigation constants (identical to sphero_navigate.py) ───────────────────
SPEED               = 50
UPDATE_INTERVAL     = 0.12
MIN_SPEED           = 25
HDG_CHANGE_THRESH   = 3
SPEED_CHANGE_THRESH = 2

SHARP_TURN_THRESHOLD    = 25
CURVATURE_LOOKAHEAD_PTS = 15

LOOKAHEAD_MIN_CM = 4.0
LOOKAHEAD_MAX_CM = 10.0

STUCK_CYCLES       = 20
STUCK_MIN_PROGRESS = 2

SEARCH_WINDOW     = 40
ARRIVAL_DIST_CM   = 4.0
FINAL_APPROACH_CM = 8.0

PAPER_X_CM = 85.0
PAPER_Y_CM = 60.0
RECT_W     = 850
RECT_H     = 600

# Border clamping — keep robot away from paper edges
BORDER_CM = 5.0

# Waypoint distance threshold — finger must move this far from the last
# committed waypoint before a new one is added to the path
MIN_WAYPOINT_DIST_CM = 2.0

STATE_AIM      = 'aim'
STATE_TRACKING = 'tracking'


# ── Path functions (identical to sphero_navigate.py) ─────────────────────────
def interpolate_smooth_path(waypoints, num_points=400):
    if len(waypoints) < 2:
        return waypoints
    if len(waypoints) == 2:
        u_vals = np.linspace(waypoints[0][0], waypoints[1][0], num_points)
        v_vals = np.linspace(waypoints[0][1], waypoints[1][1], num_points)
        return list(zip(u_vals, v_vals))
    pts = np.array(waypoints)
    t   = np.linspace(0, 1, len(pts))
    pu  = PchipInterpolator(t, pts[:, 0])
    pv  = PchipInterpolator(t, pts[:, 1])
    ts  = np.linspace(0, 1, num_points)
    return list(zip(np.clip(pu(ts), 0, 1), np.clip(pv(ts), 0, 1)))


def find_closest_path_point(pos, path, search_start_idx=0):
    if not path:
        return None, 0, float('inf')
    min_dist = float('inf')
    min_idx  = search_start_idx
    search_end = min(search_start_idx + SEARCH_WINDOW, len(path))
    for i in range(search_start_idx, search_end):
        px, pv = path[i]
        dist = math.hypot((px - pos[0]) * PAPER_X_CM, (pv - pos[1]) * PAPER_Y_CM)
        if dist < min_dist:
            min_dist = dist
            min_idx  = i
    return path[min_idx], min_idx, min_dist


def get_lookahead_point(pos, path, current_idx, lookahead_cm):
    if not path or current_idx >= len(path):
        return path[-1] if path else pos
    for i in range(current_idx, len(path)):
        px, pv = path[i]
        dist = math.hypot((px - pos[0]) * PAPER_X_CM, (pv - pos[1]) * PAPER_Y_CM)
        if dist >= lookahead_cm:
            return (px, pv)
    return path[-1]


def estimate_upcoming_curvature(path, path_idx):
    n   = len(path)
    pts = CURVATURE_LOOKAHEAD_PTS
    if path_idx + pts >= n:
        return 0.0
    p0, p1, p2 = path[path_idx], path[path_idx + pts // 2], path[path_idx + pts]
    du1, dv1 = (p1[0]-p0[0])*PAPER_X_CM, (p1[1]-p0[1])*PAPER_Y_CM
    du2, dv2 = (p2[0]-p1[0])*PAPER_X_CM, (p2[1]-p1[1])*PAPER_Y_CM
    if math.hypot(du1, dv1) < 1e-6 or math.hypot(du2, dv2) < 1e-6:
        return 0.0
    h1 = math.degrees(math.atan2(du1, dv1)) % 360
    h2 = math.degrees(math.atan2(du2, dv2)) % 360
    diff = abs(h1 - h2)
    return 360 - diff if diff > 180 else diff


def compute_adaptive_lookahead(speed, curvature_deg):
    speed_factor = speed / SPEED
    lookahead    = LOOKAHEAD_MIN_CM + speed_factor * (LOOKAHEAD_MAX_CM - LOOKAHEAD_MIN_CM)
    if curvature_deg > SHARP_TURN_THRESHOLD:
        ratio = (curvature_deg - SHARP_TURN_THRESHOLD) / 90.0
        lookahead -= ratio * (lookahead - LOOKAHEAD_MIN_CM) * 0.6
    return max(LOOKAHEAD_MIN_CM, min(LOOKAHEAD_MAX_CM, lookahead))


# ── Navigator (adapted from sphero_navigate.py — uses get_pos callable) ──────
class Navigator:

    def __init__(self, droid, get_pos, heading_offset):
        self.droid          = droid
        self.get_pos        = get_pos
        self.heading_offset = heading_offset
        self.path           = []
        self.path_idx       = 0
        self.active         = False
        self.arrived        = False
        self.last_heading   = None
        self.last_speed     = None
        self.last_dist_cm   = None
        self.last_progress  = 0.0
        self.last_cmd_time  = 0.0
        self.start_time     = 0.0
        self.trajectory     = []
        self._thread        = None

        self.idx_history         = deque(maxlen=STUCK_CYCLES)
        self.stuck_counter       = 0
        self.in_stuck_recovery   = False
        self.recovery_start_time = 0.0

        self.cte_history = []
        self.max_cte     = 0.0
        self.waypoints   = []

    def start(self, smooth_path, waypoints=None):
        self.path       = list(smooth_path)
        self.path_idx   = 0
        self.active     = True
        self.arrived    = False
        self.start_time = time.time()
        self.trajectory = []
        self.waypoints  = waypoints or []
        self.idx_history.clear()
        self.stuck_counter       = 0
        self.in_stuck_recovery   = False
        self.cte_history         = []
        self.max_cte             = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False

    def extend_path(self, additional_points):
        """Append new segment to path without resetting path_idx (GIL-safe)."""
        self.path = self.path + list(additional_points)

    def _log_metrics(self):
        total_time = time.time() - self.start_time
        total_dist = sum(
            math.hypot((self.trajectory[i][0] - self.trajectory[i-1][0]) * PAPER_X_CM,
                       (self.trajectory[i][1] - self.trajectory[i-1][1]) * PAPER_Y_CM)
            for i in range(1, len(self.trajectory))
        ) if len(self.trajectory) > 1 else 0.0
        avg_cte = sum(self.cte_history) / len(self.cte_history) if self.cte_history else 0.0

        ts    = int(time.time())
        fname = os.path.expanduser(f"~/Desktop/thesis/thesis_ws/trajectory_metrics_{ts}.csv")
        with open(fname, 'w') as f:
            f.write("metric,value\n")
            f.write(f"total_time_sec,{total_time:.3f}\n")
            f.write(f"total_distance_cm,{total_dist:.1f}\n")
            f.write(f"max_cross_track_error_cm,{self.max_cte:.1f}\n")
            f.write(f"avg_cross_track_error_cm,{avg_cte:.1f}\n")
            f.write(f"final_distance_to_target_cm,{self.last_dist_cm:.1f}\n")
            f.write(f"path_length_points,{len(self.path)}\n")
            f.write(f"waypoint_count,{len(self.waypoints)}\n")
        rospy.loginfo("[NAV] metrics saved to %s", fname)

    def _run(self):
        _was_lost = False
        while self.active:
            pos = self.get_pos()
            if pos is None:
                if not _was_lost:
                    rospy.logwarn("[NAV] detection lost — stopping Sphero")
                    _was_lost = True
                self.droid.set_speed(0)
                time.sleep(0.05)
                continue

            if _was_lost:
                rospy.loginfo("[NAV] detection regained at (%.3f,%.3f)", pos[0], pos[1])
                _was_lost = False

            path_snap = self.path   # GIL-safe snapshot; may grow between cycles
            if not path_snap:
                time.sleep(0.05)
                continue

            _, new_idx, cross_track_error = find_closest_path_point(pos, path_snap, self.path_idx)
            self.path_idx = max(self.path_idx, new_idx)

            upcoming_curvature = estimate_upcoming_curvature(path_snap, self.path_idx)
            lookahead_cm       = compute_adaptive_lookahead(self.last_speed or SPEED, upcoming_curvature)
            target             = get_lookahead_point(pos, path_snap, self.path_idx, lookahead_cm)

            du = target[0] - pos[0]
            dv = target[1] - pos[1]
            dist_to_end = math.hypot(
                (path_snap[-1][0] - pos[0]) * PAPER_X_CM,
                (path_snap[-1][1] - pos[1]) * PAPER_Y_CM,
            )
            self.last_dist_cm = dist_to_end

            if self.path_idx >= len(path_snap) - 3 and dist_to_end < ARRIVAL_DIST_CM:
                self.droid.set_speed(0)
                self.arrived = True
                rospy.loginfo("[NAV] arrived (path_idx=%d  dist=%.1fcm)", self.path_idx, dist_to_end)
                break

            self.last_progress = min(1.0, self.path_idx / float(len(path_snap)))

            self.cte_history.append(cross_track_error)
            self.max_cte = max(self.max_cte, cross_track_error)

            self.idx_history.append(self.path_idx)
            if len(self.idx_history) == STUCK_CYCLES:
                advance = self.path_idx - self.idx_history[0]
                if advance < STUCK_MIN_PROGRESS:
                    self.stuck_counter += 1
                    if self.stuck_counter >= 3 and not self.in_stuck_recovery:
                        rospy.logwarn("[NAV] STUCK: advanced only %d pts in %d cycles",
                                      advance, STUCK_CYCLES)
                        self.in_stuck_recovery   = True
                        self.recovery_start_time = time.time()
                else:
                    self.stuck_counter = 0

            paper_angle = math.degrees(math.atan2(du, dv)) % 360
            heading     = int((paper_angle + self.heading_offset) % 360)

            if self.in_stuck_recovery:
                elapsed = time.time() - self.recovery_start_time
                if elapsed < 0.3:
                    speed = 0
                else:
                    self.in_stuck_recovery = False
                    speed = SPEED
            else:
                near_end = self.path_idx >= len(path_snap) * 0.9
                if near_end:
                    approach_factor = min(dist_to_end / FINAL_APPROACH_CM, 1.0)
                    speed = max(MIN_SPEED, int(SPEED * approach_factor))
                else:
                    speed = SPEED
                if upcoming_curvature > SHARP_TURN_THRESHOLD:
                    turn_factor = 1.0 - (upcoming_curvature - SHARP_TURN_THRESHOLD) / 90.0
                    speed = max(MIN_SPEED, int(speed * max(0.3, turn_factor)))

            prev_heading = self.last_heading
            prev_speed   = self.last_speed
            self.last_heading = heading
            self.last_speed   = speed
            self.trajectory.append(pos)

            t_now = time.time()
            should_send = (
                self.last_cmd_time == 0.0
                or (prev_heading is not None and abs(heading - prev_heading) >= HDG_CHANGE_THRESH)
                or (prev_speed   is not None and abs(speed - prev_speed)     >= SPEED_CHANGE_THRESH)
                or (t_now - self.last_cmd_time) > UPDATE_INTERVAL * 1.5
            )
            if should_send:
                try:
                    self.droid.roll(heading, speed, UPDATE_INTERVAL)
                    self.last_cmd_time = t_now
                except TimeoutError:
                    rospy.logwarn("[NAV] Bluetooth timeout hdg=%d spd=%d", heading, speed)
                except Exception as e:
                    rospy.logwarn("[NAV] Bluetooth error: %s: %s", type(e).__name__, e)
            else:
                time.sleep(0.01)

        self.droid.set_speed(0)
        if self.arrived:
            self._log_metrics()


# ── Display helpers (identical to sphero_navigate.py) ────────────────────────
def disp_pt(cx, cy):
    return (cx, RECT_H - 1 - cy)


def uv_to_disp(u, v):
    return disp_pt(int(u * RECT_W), int(v * RECT_H))


def draw_grid(img, step_px=100, color=(55, 55, 55)):
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


def draw_smooth_path(img, path, color=(100, 150, 255), thickness=1):
    for i in range(1, len(path)):
        cv2.line(img, uv_to_disp(*path[i-1]), uv_to_disp(*path[i]), color, thickness, cv2.LINE_AA)


def draw_trajectory(img, traj, color=(0, 255, 100), thickness=2):
    for i in range(1, len(traj)):
        cv2.line(img, uv_to_disp(*traj[i-1]), uv_to_disp(*traj[i]), color, thickness, cv2.LINE_AA)


def draw_sphero(img, uv, radius_px=8):
    dp = uv_to_disp(*uv)
    r  = max(int(radius_px), 8)
    cv2.circle(img, dp, r, (0, 255, 0), 2)
    cv2.circle(img, dp, 4, (0, 255, 0), -1)


# ── ROS Node ──────────────────────────────────────────────────────────────────
class SpheroPathNavigateNode:

    def __init__(self):
        rospy.init_node('sphero_path_navigate', anonymous=False)

        self.heading_offset = rospy.get_param('~heading_offset', 0.0)
        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 642.351)
        self.cy = rospy.get_param('~cy', 349.990)

        self.bridge = CvBridge()
        self.lock   = threading.Lock()

        # Camera / display
        self._frame  = None
        self._H_rect = None

        # Sphero position (from /sphero/detection)
        self._sphero_uv       = None
        self._sphero_r        = 0
        self._sphero_last_det = 0.0

        # Pointing target (from /pointing/target)
        self._pointing_uv = None

        # Waypoint accumulation
        self._waypoints  = []    # all committed waypoints (for display + metrics)
        self._last_wp_uv = None  # UV of last committed waypoint (distance gate)

        # Navigation state
        self._state       = STATE_AIM
        self._smooth_path = []
        self._start_uv    = None
        self._nav         = None
        self._droid       = None   # set in spin()

        self._travel_pub = rospy.Publisher('/sphero/travel_time_s', Float64, queue_size=10)

        rospy.Subscriber('/d435i/rgb/image_raw',  Image,           self._image_cb,     queue_size=1)
        rospy.Subscriber('/camera/calibration',   CalibrationData, self._calib_cb,     queue_size=1)
        rospy.Subscriber('/sphero/detection',     Point,           self._detection_cb, queue_size=1)
        rospy.Subscriber('/pointing/target',      PointingTarget,  self._target_cb,    queue_size=1)

        rospy.loginfo("SpheroPathNavigateNode ready — enter AIM state.")

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
        p1 = np.array([msg.plane_center.x, msg.plane_center.y, msg.plane_center.z])
        ax = np.array([msg.paper_x_axis.x,  msg.paper_x_axis.y,  msg.paper_x_axis.z])
        ay = np.array([msg.paper_y_axis.x,  msg.paper_y_axis.y,  msg.paper_y_axis.z])
        Lx, Ly = float(msg.paper_x_m), float(msg.paper_y_m)
        corners = [p1, p1+ax*Lx, p1+ax*Lx+ay*Ly, p1+ay*Ly]
        src = np.float32([[self.fx*X/Z + self.cx, self.fy*Y/Z + self.cy] for X, Y, Z in corners])
        dst = np.float32([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]])
        H = cv2.getPerspectiveTransform(src, dst)
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
            with self.lock:
                self._pointing_uv = None
            return
        border_u = BORDER_CM / PAPER_X_CM
        border_v = BORDER_CM / PAPER_Y_CM
        u = float(np.clip(msg.u_normalized, border_u, 1.0 - border_u))
        v = float(np.clip(msg.v_normalized, border_v, 1.0 - border_v))
        with self.lock:
            self._pointing_uv = (u, v)
            state = self._state
        if state == STATE_TRACKING:
            self._maybe_add_waypoint((u, v))

    # ── Waypoint accumulation ─────────────────────────────────────────────────

    def _maybe_add_waypoint(self, wp):
        """Commit wp as a new waypoint if finger moved >= MIN_WAYPOINT_DIST_CM."""
        with self.lock:
            last_wp   = self._last_wp_uv
            nav       = self._nav
            smooth    = list(self._smooth_path)
            sphero_uv = self._sphero_uv

        # Distance gate — only add if finger moved far enough
        if last_wp is not None:
            d = math.hypot((wp[0]-last_wp[0])*PAPER_X_CM,
                           (wp[1]-last_wp[1])*PAPER_Y_CM)
            if d < MIN_WAYPOINT_DIST_CM:
                return

        with self.lock:
            self._last_wp_uv = wp
            self._waypoints.append(wp)

        rospy.logdebug("[WP] +1 waypoint (%.1f,%.1f)cm  total=%d",
                       wp[0]*PAPER_X_CM, wp[1]*PAPER_Y_CM, len(self._waypoints))

        if nav is None:
            if sphero_uv is not None:
                self._start_navigation(sphero_uv, wp)
            else:
                rospy.logwarn_throttle(2, "[NAV] pointing received but Sphero not detected yet")

        elif nav.active:
            # Extend path from its current end to the new waypoint
            last_pt = smooth[-1] if smooth else wp
            dist_cm = math.hypot((wp[0]-last_pt[0])*PAPER_X_CM,
                                 (wp[1]-last_pt[1])*PAPER_Y_CM)
            if dist_cm < 0.3:
                return
            n_pts   = max(10, int(dist_cm * 5))   # ~5 pts/cm
            segment = interpolate_smooth_path([last_pt, wp], num_points=n_pts)
            tail    = list(segment[1:])            # drop duplicate of last_pt
            with self.lock:
                self._smooth_path = self._smooth_path + tail
            nav.extend_path(tail)

        elif nav.arrived:
            # Finger moved after arrival — start a fresh route from current Sphero pos
            if sphero_uv is not None:
                self._reset_route()
                self._start_navigation(sphero_uv, wp)

    def _start_navigation(self, start_uv, wp):
        smooth = interpolate_smooth_path([start_uv, wp], num_points=400)
        nav    = Navigator(self._droid, self._get_pos, self.heading_offset)
        with self.lock:
            self._start_uv    = start_uv
            self._smooth_path = smooth
            self._nav         = nav
        nav.start(smooth, waypoints=list(self._waypoints))
        rospy.loginfo("[NAV] started: start=(%.1f,%.1f)cm  first_wp=(%.1f,%.1f)cm  pts=%d",
                      start_uv[0]*PAPER_X_CM, start_uv[1]*PAPER_Y_CM,
                      wp[0]*PAPER_X_CM, wp[1]*PAPER_Y_CM, len(smooth))

    def _reset_route(self):
        with self.lock:
            if self._nav:
                self._nav.stop()
            self._nav         = None
            self._smooth_path = []
            self._start_uv    = None
            self._waypoints   = []
            self._last_wp_uv  = None
        rospy.loginfo("[STATE] route reset — waiting for pointing target")

    # ── Position accessor for Navigator thread ────────────────────────────────

    def _get_pos(self):
        with self.lock:
            if self._sphero_uv is None:
                return None
            if time.time() - self._sphero_last_det > 1.0:
                return None
            return self._sphero_uv

    # ── Main loop ─────────────────────────────────────────────────────────────

    def spin(self, droid):
        self._droid = droid

        WIN = "Sphero Path Navigate  [Q=quit]"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, RECT_W, RECT_H)

        state     = STATE_AIM
        aim_phase = 0
        rate      = rospy.Rate(30)

        while not rospy.is_shutdown():
            # ── Snapshot shared state ─────────────────────────────────────────
            with self.lock:
                frame       = self._frame.copy() if self._frame is not None else None
                H_rect      = self._H_rect
                sphero_uv   = self._sphero_uv
                sphero_r    = self._sphero_r
                pointing_uv = self._pointing_uv
                smooth_path = list(self._smooth_path)
                start_uv    = self._start_uv
                nav         = self._nav
                n_wps       = len(self._waypoints)

            # ── Build display ─────────────────────────────────────────────────
            if frame is not None and H_rect is not None:
                rect = cv2.warpPerspective(frame, H_rect, (RECT_W, RECT_H))
                disp = cv2.flip(rect, 0)
            else:
                disp = np.zeros((RECT_H, RECT_W, 3), dtype=np.uint8)
                cv2.putText(disp, "Waiting for camera / calibration…",
                            (20, RECT_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (80, 80, 80), 2)

            draw_grid(disp)

            if sphero_uv is not None:
                draw_sphero(disp, sphero_uv, sphero_r)

            # ── State-specific display ────────────────────────────────────────
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

            elif state == STATE_TRACKING:
                arrived = nav is not None and nav.arrived

                # Growing planned path
                if smooth_path:
                    path_col = (100, 200, 100) if arrived else (100, 150, 255)
                    draw_smooth_path(disp, smooth_path, color=path_col, thickness=2)

                # Actual trajectory overlay on arrival
                if arrived and nav.trajectory:
                    draw_trajectory(disp, nav.trajectory, color=(0, 255, 0), thickness=2)

                # Start marker
                if start_uv is not None:
                    dp = uv_to_disp(*start_uv)
                    cv2.drawMarker(disp, dp, (0, 220, 0), cv2.MARKER_CROSS, 18, 1)
                    cv2.putText(disp, "A", (dp[0]+8, dp[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA)

                # Live finger crosshair
                if pointing_uv is not None:
                    cv2.drawMarker(disp, uv_to_disp(*pointing_uv),
                                   (0, 220, 255), cv2.MARKER_CROSS, 20, 1)

                # Status bar
                if arrived:
                    draw_overlay(disp, [
                        "ARRIVED!  Move finger to trace a new route.",
                        "Planned (blue)  |  Actual (green)  |  R = full reset  |  Q = quit",
                    ])
                elif nav is None:
                    draw_overlay(disp, [
                        "Move your finger over the paper to trace a route.",
                    ])
                else:
                    d = nav.last_dist_cm
                    h = nav.last_heading
                    p = nav.last_progress
                    info = (f"NAVIGATING  {p:.0%}  dist={d:.1f}cm  hdg={h}°  "
                            f"waypoints={n_wps}  |  R = reset"
                            if d is not None else "NAVIGATING…  |  R = reset")
                    cv2.putText(disp, info, (8, RECT_H-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF

            # ── Global keys ───────────────────────────────────────────────────
            if key == ord('q'):
                if nav:
                    nav.stop()
                break

            # ── State transitions ─────────────────────────────────────────────
            if state == STATE_AIM:
                if key == 13:   # ENTER
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
                            self._state = STATE_TRACKING
                        rospy.loginfo("[STATE] aim → tracking  (heading=0 locked)")

            elif state == STATE_TRACKING:
                if key == ord('r'):
                    self._reset_route()

            rate.sleep()

        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    node = SpheroPathNavigateNode()

    sphero_name = rospy.get_param('~sphero_name', '')
    rospy.loginfo("Scanning for Sphero%s…", (' ' + sphero_name) if sphero_name else '')
    toy = (scanner.find_toy(toy_name=sphero_name) if sphero_name
           else scanner.find_toy())
    rospy.loginfo("Connected: %s", toy.name)

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
