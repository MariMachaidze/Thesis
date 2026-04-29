#!/usr/bin/env python3
"""
sphero_detection_node.py
Detects Sphero BOLT on calibrated paper using Hough circles.
Applies One Euro Filter to u,v and publishes on /sphero/detection.

Subscribed:
  /d435i/rgb/image_raw      sensor_msgs/Image
  /camera/calibration       camera_calibration/CalibrationData  (latched)

Published:
  /sphero/detection         geometry_msgs/Point  (x=u, y=v, z=radius_px)

Keys: Q = quit
"""

import math
import threading
from collections import deque

import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from camera_calibration.msg import CalibrationData

# ── Hough parameters (tuned for Sphero BOLT) ─────────────────────────────────
HOUGH_DP      = 1.2
HOUGH_MINDIST = 40
HOUGH_PARAM1  = 80
DEF_PARAM2    = 30    # accumulator threshold — lower = more sensitive
DEF_MIN_R     = 18   # px
DEF_MAX_R     = 42   # px
DEF_BLUR      = 6    # Gaussian blur kernel (forced odd)

# ── Tracking ──────────────────────────────────────────────────────────────────
TRAIL_LEN = 50
MAX_LOST  = 8
MAX_JUMP  = 80    # px — in rectified space

# ── CLAHE ─────────────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ── Rectified view: 1 px/mm → paper 850 mm × 600 mm ─────────────────────────
RECT_W = 850
RECT_H = 600


def _nothing(_):
    pass


# ── One Euro Filter ───────────────────────────────────────────────────────────

class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.1, dcutoff=1.0):
        self._freq       = freq
        self._min_cutoff = min_cutoff
        self._beta       = beta
        self._dcutoff    = dcutoff
        self._x          = None
        self._dx         = 0.0

    def _alpha(self, cutoff):
        te  = 1.0 / self._freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self._x is None:
            self._x = x
            return x
        dx        = (x - self._x) * self._freq
        a_d       = self._alpha(self._dcutoff)
        self._dx  = a_d * dx + (1 - a_d) * self._dx
        cutoff    = self._min_cutoff + self._beta * abs(self._dx)
        a         = self._alpha(cutoff)
        self._x   = a * x + (1 - a) * self._x
        return self._x


# ── Node ──────────────────────────────────────────────────────────────────────

class SpheroDetectionNode:

    def __init__(self):
        rospy.init_node('sphero_detection_node', anonymous=False)

        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 642.351)
        self.cy = rospy.get_param('~cy', 349.990)

        self.bridge = CvBridge()
        self.lock   = threading.Lock()

        self.frame      = None
        self.H_rect     = None

        self.last_px    = None
        self.lost_count = 0
        self.trail      = deque(maxlen=TRAIL_LEN)

        self._filt_u = OneEuroFilter()
        self._filt_v = OneEuroFilter()

        self.det_pub = rospy.Publisher('/sphero/detection', Point, queue_size=1)

        rospy.Subscriber('/d435i/rgb/image_raw', Image,
                         self._image_cb, queue_size=1)
        rospy.Subscriber('/camera/calibration', CalibrationData,
                         self._calib_cb, queue_size=1)

        rospy.loginfo("SpheroDetectionNode ready — waiting for /camera/calibration …")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr_throttle(5, f"imgmsg_to_cv2: {e}")
            return
        with self.lock:
            self.frame = frame

    def _calib_cb(self, msg):
        if not msg.is_calibrated:
            return
        p1 = np.array([msg.plane_center.x, msg.plane_center.y, msg.plane_center.z])
        ax = np.array([msg.paper_x_axis.x, msg.paper_x_axis.y, msg.paper_x_axis.z])
        ay = np.array([msg.paper_y_axis.x, msg.paper_y_axis.y, msg.paper_y_axis.z])
        Lx, Ly = float(msg.paper_x_m), float(msg.paper_y_m)

        corners_3d = np.array([p1, p1 + ax*Lx, p1 + ay*Ly, p1 + ax*Lx + ay*Ly])
        corners_px = []
        for X, Y, Z in corners_3d:
            if Z <= 0:
                rospy.logwarn("Paper corner behind camera — bad calibration?")
                return
            corners_px.append((self.fx*X/Z + self.cx, self.fy*Y/Z + self.cy))
        corners_px = np.array(corners_px, dtype=np.float32)

        dst    = np.array([[0,0],[RECT_W,0],[0,RECT_H],[RECT_W,RECT_H]], dtype=np.float32)
        H_rect = cv2.getPerspectiveTransform(corners_px, dst)
        with self.lock:
            self.H_rect = H_rect
        rospy.loginfo("Calibration received — rectification homography ready.")

    # ── Detection helpers ──────────────────────────────────────────────────────

    def _detect(self, rectified, p2, min_r, max_r, blur_k):
        gray     = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        enhanced = _clahe.apply(gray)
        k        = max(3, blur_k | 1)
        blur     = cv2.GaussianBlur(enhanced, (k, k), 0)
        circles  = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP, minDist=HOUGH_MINDIST,
            param1=HOUGH_PARAM1, param2=max(1, p2),
            minRadius=max(1, min_r), maxRadius=max(2, max_r),
        )
        if circles is None:
            return []
        return [(int(cx_r), int(cy_r), int(r))
                for cx_r, cy_r, r in np.round(circles[0]).astype(int)]

    def _pick_best(self, candidates):
        if not candidates:
            return None
        if self.last_px is None:
            return candidates[0]
        lx, ly   = self.last_px
        best, bd = None, float('inf')
        for cx, cy, r in candidates:
            d = np.hypot(cx - lx, cy - ly)
            if d < bd and d < MAX_JUMP:
                bd, best = d, (cx, cy, r)
        return best if best is not None else candidates[0]

    def _draw_trail(self, img):
        pts = list(self.trail)
        n   = len(pts)
        for i in range(1, n):
            a   = i / n
            col = (int(50*a), int(200*a), int(255*(1-a) + 80*a))
            cv2.line(img, pts[i-1], pts[i], col, max(1, int(3*a)))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def spin(self):
        # WIN    = "Sphero Detection  [Q=quit]"
        # WIN_TB = "Hough Controls"

        # cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(WIN, RECT_W, RECT_H)
        # cv2.namedWindow(WIN_TB, cv2.WINDOW_NORMAL)

        # cv2.createTrackbar("Sensitivity (p2)\nlower=more", WIN_TB, DEF_PARAM2, 80,  _nothing)
        # cv2.createTrackbar("Min radius (px)",              WIN_TB, DEF_MIN_R,  100, _nothing)
        # cv2.createTrackbar("Max radius (px)",              WIN_TB, DEF_MAX_R,  200, _nothing)
        # cv2.createTrackbar("Blur kernel (px)",             WIN_TB, DEF_BLUR,   15,  _nothing)

        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            with self.lock:
                frame  = self.frame.copy() if self.frame is not None else None
                H_rect = self.H_rect

            if frame is None or H_rect is None:
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                rate.sleep()
                continue

            # p2     = cv2.getTrackbarPos("Sensitivity (p2)\nlower=more", WIN_TB)
            # min_r  = cv2.getTrackbarPos("Min radius (px)",              WIN_TB)
            # max_r  = cv2.getTrackbarPos("Max radius (px)",              WIN_TB)
            # blur_k = cv2.getTrackbarPos("Blur kernel (px)",             WIN_TB)

            rectified  = cv2.warpPerspective(frame, H_rect, (RECT_W, RECT_H))
            candidates = self._detect(rectified, DEF_PARAM2, DEF_MIN_R, DEF_MAX_R, DEF_BLUR)
            detected   = self._pick_best(candidates)

            u_out = v_out = None

            if detected is not None:
                cx, cy, r   = detected
                self.last_px    = (cx, cy)
                self.lost_count = 0
                self.trail.append((cx, cy))

                u_raw = float(np.clip(cx / RECT_W, 0.0, 1.0))
                v_raw = float(np.clip(cy / RECT_H, 0.0, 1.0))
                u_out = self._filt_u(u_raw)
                v_out = self._filt_v(v_raw)

                self.det_pub.publish(Point(x=u_out, y=v_out, z=float(r)))
                # rospy.loginfo_throttle(0.5, "Sphero u=%.3f v=%.3f r=%dpx", u_out, v_out, r)
            else:
                self.lost_count += 1
                if self.lost_count > MAX_LOST:
                    self.last_px = None
                    if self.trail:
                        self.trail.popleft()

            # ── Display (commented out — navigate node shows its own window) ──────
            # disp = cv2.flip(rectified, 0)
            # def f(cx, cy):
            #     return (cx, RECT_H - 1 - cy)
            # trail_pts = list(self.trail)
            # for i in range(1, len(trail_pts)):
            #     a   = i / len(trail_pts)
            #     col = (int(50*a), int(200*a), int(255*(1-a) + 80*a))
            #     cv2.line(disp, f(*trail_pts[i-1]), f(*trail_pts[i]), col, max(1, int(3*a)))
            # for cx_c, cy_c, r_c in candidates:
            #     cv2.circle(disp, f(cx_c, cy_c), r_c, (80, 80, 80), 1)
            # if detected is not None:
            #     cx, cy, r = detected
            #     dx, dy = f(cx, cy)
            #     cv2.circle(disp, (dx, dy), max(r, 4), (0, 255, 255), 2)
            #     cv2.circle(disp, (dx, dy), 4, (0, 0, 255), -1)
            #     cv2.putText(disp, f"u={u_out:.3f}  v={v_out:.3f}  r={r}px",
            #                 (dx + r + 4, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            # else:
            #     cv2.putText(disp, f"searching...  (lost {self.lost_count}f)",
            #                 (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)
            # cv2.imshow(WIN, disp)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            rate.sleep()


if __name__ == '__main__':
    try:
        node = SpheroDetectionNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        rospy.loginfo("Interrupted.")
