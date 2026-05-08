#!/usr/bin/env python3
"""
sphero_path_detection_node.py

Detects Sphero BOLT using CLAHE + Hough circles with the SpheroTracker
from sphero_navigate.py (12-sample history, time-based One Euro Filter,
stable_uv check).

Subscribed:
  /d435i/rgb/image_raw    sensor_msgs/Image
  /camera/calibration     camera_calibration/CalibrationData (latched)

Published:
  /sphero/detection       geometry_msgs/Point  (x=u_filtered, y=v_filtered, z=radius_px)
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

from camera_calibration.msg import CalibrationData

# ── Detection parameters (identical to sphero_navigate.py) ───────────────────
P2         = 21
MIN_R      = 25
MAX_R      = 40
BLUR_K     = 6
CLAHE_CLIP = 0.6
HOUGH_DP   = 1.2
HOUGH_DIST = 40
HOUGH_P1   = 80

OEF_MIN_CUTOFF = 0.6
OEF_BETA       = 0.03

TRAIL_LEN  = 80
MAX_LOST   = 10
MAX_JUMP   = 120   # px — max inter-frame jump in rectified space

RECT_W     = 850
RECT_H     = 600
PAPER_X_CM = 85.0
PAPER_Y_CM = 60.0

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))


# ── One Euro Filter (time-based, identical to sphero_navigate.py) ─────────────
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


# ── SpheroTracker (identical to sphero_navigate.py) ──────────────────────────
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
        cu  = float(np.mean([u for u, v in recent]))
        cv_ = float(np.mean([v for u, v in recent]))
        for u, v in recent:
            if math.hypot((u - cu) * PAPER_X_CM, (v - cv_) * PAPER_Y_CM) > radius_cm:
                return None
        return (cu, cv_)

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


# ── ROS Node ──────────────────────────────────────────────────────────────────
class SpheroPathDetectionNode:

    def __init__(self):
        rospy.init_node('sphero_path_detection', anonymous=False)

        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 642.351)
        self.cy = rospy.get_param('~cy', 349.990)

        self.bridge  = CvBridge()
        self.lock    = threading.Lock()
        self.tracker = SpheroTracker()

        self._frame  = None
        self._H_rect = None

        self.det_pub = rospy.Publisher('/sphero/detection', Point, queue_size=1)

        rospy.Subscriber('/d435i/rgb/image_raw',  Image,           self._image_cb, queue_size=1)
        rospy.Subscriber('/camera/calibration',   CalibrationData, self._calib_cb, queue_size=1)

        rospy.loginfo("SpheroPathDetectionNode ready — waiting for calibration…")

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
        corners = [p1, p1 + ax*Lx, p1 + ax*Lx + ay*Ly, p1 + ay*Ly]
        src = np.float32([[self.fx*X/Z + self.cx, self.fy*Y/Z + self.cy] for X, Y, Z in corners])
        dst = np.float32([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]])
        H = cv2.getPerspectiveTransform(src, dst)
        with self.lock:
            self._H_rect = H
        rospy.loginfo("Calibration received — rectification homography ready.")

    def spin(self):
        rate = rospy.Rate(30)
        _last_log = 0.0

        while not rospy.is_shutdown():
            with self.lock:
                frame  = self._frame.copy() if self._frame is not None else None
                H_rect = self._H_rect

            if frame is None or H_rect is None:
                rate.sleep()
                continue

            rect  = cv2.warpPerspective(frame, H_rect, (RECT_W, RECT_H))
            t_now = time.time()
            cands = detect_sphero(rect)
            det   = self.tracker.update(cands, t_now)
            fuv   = self.tracker.filtered_uv

            if fuv is not None:
                r_px = float(det[2]) if det else 0.0
                self.det_pub.publish(Point(x=fuv[0], y=fuv[1], z=r_px))

            if t_now - _last_log >= 1.0:
                _last_log = t_now
                if fuv is not None:
                    rospy.loginfo("[DET] fuv=(%.3f,%.3f)  lost=%d  cands=%d",
                                  fuv[0], fuv[1], self.tracker.lost, len(cands))
                else:
                    rospy.loginfo("[DET] no detection  lost=%d  cands=%d",
                                  self.tracker.lost, len(cands))

            rate.sleep()


if __name__ == '__main__':
    try:
        node = SpheroPathDetectionNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Detection node stopped.")
