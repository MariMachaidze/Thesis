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

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI

from camera_calibration.msg import CalibrationData
from pointing_localization.msg import PointingTarget

# ── Constants ─────────────────────────────────────────────────────────────────
SPEED           = 50
UPDATE_INTERVAL = 0.3
UPDATE_NEAR     = 0.1
NEAR_DIST_CM    = 20.0
SLOW_DIST_CM    = 20.0
MIN_SPEED       = 25
HOVER_STOP_CM   = 3.0
HOVER_RESUME_CM = 9.0
COAST_FACTOR    = 0.28
STUCK_TIMEOUT_S = 2.5   # s  — force hover if pos hasn't moved for this long
STUCK_MOVE_CM   = 1.5   # cm — minimum displacement to reset stuck timer
PAPER_X_CM      = 85.0
PAPER_Y_CM      = 60.0
RECT_W          = 850
RECT_H          = 600

STATE_AIM      = 'aim'
STATE_TRACKING = 'tracking'


# ── Navigator ─────────────────────────────────────────────────────────────────
class Navigator:
    """Runs in a daemon thread — reads shared UV, drives Sphero."""

    def __init__(self, droid, get_sphero_uv, get_target_uv, heading_offset):
        self.droid          = droid
        self.get_sphero_uv  = get_sphero_uv
        self.get_target_uv  = get_target_uv
        self.heading_offset = heading_offset
        self.active         = False
        self._hovering      = False
        self.last_heading   = None
        self.last_dist_cm   = None
        self._thread        = None

    @property
    def hovering(self):
        return self._hovering

    def start(self):
        self.active    = True
        self._hovering = False
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False

    def _run(self):
        _was_lost   = False
        _stuck_pos  = None
        _stuck_since = None
        while self.active and not rospy.is_shutdown():
            pos    = self.get_sphero_uv()
            target = self.get_target_uv()

            if pos is None or target is None:
                if not _was_lost:
                    rospy.loginfo_throttle(2, "[NAV] waiting — pos=%s target=%s",
                                           pos is not None, target is not None)
                    _was_lost = True
                self.droid.set_speed(0)
                time.sleep(0.1)
                continue

            if _was_lost:
                rospy.loginfo("[NAV] detection/target regained — resuming")
                _was_lost = False

            du = target[0] - pos[0]
            dv = target[1] - pos[1]
            dist_cm = math.hypot(du * PAPER_X_CM, dv * PAPER_Y_CM)
            self.last_dist_cm = dist_cm
            speed = max(MIN_SPEED, int(SPEED * min(dist_cm / SLOW_DIST_CM, 1.0)))

            if self._hovering:
                if dist_cm > HOVER_RESUME_CM:
                    self._hovering = False
                    _stuck_pos   = None
                    _stuck_since = None
                    rospy.loginfo("[NAV] HOVER → NAVIGATE  dist=%.1fcm", dist_cm)
                else:
                    time.sleep(0.1)
                    continue
            elif dist_cm < max(HOVER_STOP_CM, speed * COAST_FACTOR):
                self.droid.set_speed(0)
                self._hovering = True
                rospy.loginfo("[NAV] NAVIGATE → HOVER  pos=(%.1f,%.1f)cm  target=(%.1f,%.1f)cm  dist=%.1fcm",
                              pos[0]*PAPER_X_CM, pos[1]*PAPER_Y_CM,
                              target[0]*PAPER_X_CM, target[1]*PAPER_Y_CM, dist_cm)
                time.sleep(0.1)
                continue

            # Stuck detection: if position hasn't changed, force hover
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
                    self.droid.set_speed(0)
                    self._hovering = True
                    _stuck_pos   = None
                    _stuck_since = None
                    rospy.loginfo("[NAV] STUCK → HOVER  pos=(%.1f,%.1f)cm  target=(%.1f,%.1f)cm  dist=%.1fcm",
                                  pos[0]*PAPER_X_CM, pos[1]*PAPER_Y_CM,
                                  target[0]*PAPER_X_CM, target[1]*PAPER_Y_CM, dist_cm)
                    time.sleep(0.1)
                    continue

            # heading: atan2(du, dv) matches our aim convention
            # (tail toward user → heading=0 = toward far edge = large v = positive dv)
            paper_angle = math.degrees(math.atan2(du, dv)) % 360
            heading     = int(round(self.heading_offset + paper_angle)) % 360
            interval    = UPDATE_NEAR if dist_cm < NEAR_DIST_CM else UPDATE_INTERVAL
            self.last_heading = heading
            rospy.loginfo_throttle(0.3,
                "[NAV] pos=(%.3f,%.3f) target=(%.3f,%.3f) du=%+.3f dv=%+.3f "
                "dist=%.1fcm angle=%.1f° hdg=%d° spd=%d dt=%.2fs",
                pos[0], pos[1], target[0], target[1], du, dv,
                dist_cm, paper_angle, heading, speed, interval)
            self.droid.roll(heading, speed, interval)

        self.droid.set_speed(0)


# ── Display helpers ───────────────────────────────────────────────────────────
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

        self._frame      = None
        self._H_rect     = None
        self._sphero_uv  = None
        self._sphero_r   = 0
        self._target_uv  = None   # last valid pointing target
        self._target_locked = False  # True while navigating; blocks new pointing

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
            self._sphero_uv = (float(msg.x), float(msg.y))
            self._sphero_r  = int(msg.z)

    def _target_cb(self, msg):
        if msg.is_valid:
            with self.lock:
                if self._target_locked:
                    return
                self._target_uv = (float(np.clip(msg.u_normalized, 0, 1)),
                                   float(np.clip(msg.v_normalized, 0, 1)))
                rospy.loginfo_throttle(1.0, "[NAV] New target accepted: (%.3f, %.3f)",
                                       *self._target_uv)

    # ── Shared-state accessors for Navigator ──────────────────────────────────

    def _get_sphero_uv(self):
        with self.lock:
            return self._sphero_uv

    def _get_target_uv(self):
        with self.lock:
            return self._target_uv

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
            # Update target lock: locked only when navigating AND a target already exists
            # (if no target yet, always accept so the first pointing isn't blocked)
            if nav is not None:
                with self.lock:
                    self._target_locked = (nav.active and not nav.hovering
                                           and self._target_uv is not None)
            else:
                with self.lock:
                    self._target_locked = False

            with self.lock:
                frame   = self._frame.copy() if self._frame is not None else None
                H_rect  = self._H_rect
                suv     = self._sphero_uv
                sr      = self._sphero_r
                tuv     = self._target_uv
                locked  = self._target_locked

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
            if tuv is not None and state == STATE_TRACKING:
                target_col = (80, 80, 80) if locked else (0, 60, 255)
                dp_t = uv_to_disp(*tuv)
                cv2.drawMarker(disp, dp_t, target_col, cv2.MARKER_CROSS, 24, 2)
                cv2.circle(disp, dp_t, 18, target_col, 1)
                if suv is not None:
                    line_col = (80, 80, 80) if locked else (255, 255, 0)
                    cv2.line(disp, uv_to_disp(*suv), dp_t, line_col, 1)

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
                if tuv is None:
                    draw_overlay(disp, ["Waiting for pointing target…"])
                elif nav is not None:
                    d = nav.last_dist_cm
                    h = nav.last_heading
                    if nav.hovering:
                        col  = (0, 255, 0)
                        info = f"ARRIVED  dist={d:.1f}cm — point to new target" if d is not None else "ARRIVED"
                    else:
                        col  = (0, 200, 255)
                        info = (f"NAVIGATING  dist={d:.1f}cm  hdg={h}deg  [pointing locked]"
                                if d is not None else "NAVIGATING… [pointing locked]")
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
                    rospy.loginfo("Aim reset. heading=0 locked. Starting navigation.")
                    nav = Navigator(droid, self._get_sphero_uv, self._get_target_uv,
                                    self.heading_offset)
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
