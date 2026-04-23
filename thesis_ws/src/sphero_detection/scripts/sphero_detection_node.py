#!/usr/bin/env python3
"""
sphero_detection_node.py
------------------------
Detects the Sphero Bolt on the calibrated paper playground using Hough circles.

Full pipeline
=============
  1. Wait for /camera/calibration (latched) — 4 paper corners in 3D camera frame.
  2. Project corners 3D → 2D image pixels using camera intrinsics (pinhole model).
  3. Compute a homography: image pixel → normalised paper (u,v) ∈ [0,1]².
  4. Each RGB frame: CLAHE + GaussianBlur + HoughCircles inside the paper bounding box.
  5. Pick best circle (closest to last known position, within MAX_JUMP px).
  6. Map detected pixel → paper (u,v) via homography.
  7. Print normalised coords; publish on /sphero/detection.

Three windows
=============
  "Sphero Detection"       full RGB frame
                           ├─ green quad  : paper outline projected from 3D
                           ├─ orange rect : bounding-box ROI used for Hough
                           ├─ coloured trail of past detections
                           └─ yellow circle + (u,v) label on ball
  "Hough ROI (CLAHE)"      grayscale CLAHE crop that Hough actually sees
  "Paper Map (top-down)"   bird's-eye rectangle; dot = ball in normalised (u,v)

Topics
======
  subscribed:
    /camera/calibration       camera_calibration/CalibrationData  (latched)
    /d435i/rgb/image_raw      sensor_msgs/Image
  published:
    /sphero/detection         geometry_msgs/Point  (x=u, y=v, z=radius_px)

Four windows
============
  "Hough Controls"          trackbars — tune detection live while the node runs

Keys: Q = quit
"""

import rospy
import numpy as np
import cv2
import threading
from collections import deque

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from camera_calibration.msg import CalibrationData


# ── Hough parameters ──────────────────────────────────────────────────────────
# At ~3 m, a 7.5 cm Sphero ≈ 10–15 px radius in a 1280×720 frame.
HOUGH_DP      = 1.2
HOUGH_MINDIST = 40     # px — minimum distance between two detected circle centres
HOUGH_PARAM1  = 80     # Canny upper threshold used internally by Hough
HOUGH_PARAM2  = 20     # accumulator threshold — lower = more sensitive / more noise
MIN_RADIUS    = 6      # px
MAX_RADIUS    = 40     # px
BLUR_KERNEL   = 5      # Gaussian blur kernel (forced odd)

# ── Tracking ──────────────────────────────────────────────────────────────────
TRAIL_LEN  = 60        # max past positions kept for the trail
MAX_LOST   = 8         # frames without detection before the trail starts fading
MAX_JUMP   = 80        # px — discard detections that jump too far between frames

# ── CLAHE contrast enhancer ───────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ── Rectified view output size ────────────────────────────────────────────────
# 1 pixel per mm → A1 paper (841 × 594 mm) = 841 × 594 px output.
# Scale down by RECT_SCALE if your monitor is small.
RECT_SCALE = 0.8
RECT_W = int(841 * RECT_SCALE)   # long edge  (paper x-axis)
RECT_H = int(594 * RECT_SCALE)   # short edge (paper y-axis)


def _nothing(_):
    pass


# ── Paper-map display size ────────────────────────────────────────────────────
MAP_W, MAP_H = 420, 297   # pixels — A3-ish aspect ratio (841:594 ≈ 1.41)
MAP_PAD      = 20          # inner padding so the dot is never clipped at the edge


# ─────────────────────────────────────────────────────────────────────────────

def project_3d_to_pixel(X, Y, Z, fx, fy, cx, cy):
    """
    Pinhole projection: 3D camera-frame point → image pixel (float).
    Returns None when Z ≤ 0 (point is behind the camera).
    """
    if Z <= 0:
        return None
    return fx * X / Z + cx, fy * Y / Z + cy


# ─────────────────────────────────────────────────────────────────────────────

class SpheroDetectionNode:

    def __init__(self):
        rospy.init_node('sphero_detection_node', anonymous=False)

        # Camera intrinsics — must match the values in camera_calibration_node
        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx_i = rospy.get_param('~cx', 637.649)
        self.cy_i = rospy.get_param('~cy', 349.990)

        self.bridge = CvBridge()
        self.lock   = threading.Lock()

        # Calibration-derived geometry (filled in _compute_geometry)
        self.corners_px = None   # (4,2) float32: TL TR BL BR in image pixels
        self.roi        = None   # (x1,y1,x2,y2) axis-aligned bounding box of corners
        self.homography = None   # 3×3: image pixel → normalised paper (u,v)
        self.H_rect     = None   # 3×3: image pixel → rectified top-down image pixel

        # Tracking state
        self.last_px    = None   # (cx, cy) of last confirmed detection in image
        self.lost_count = 0
        self.trail_px   = deque(maxlen=TRAIL_LEN)   # pixel-space trail
        self.trail_uv   = deque(maxlen=TRAIL_LEN)   # paper-UV trail (for map window)

        self.frame = None   # latest BGR frame from camera

        # Publisher
        self.det_pub = rospy.Publisher('/sphero/detection', Point, queue_size=1)

        # Subscribers
        rospy.Subscriber('/camera/calibration', CalibrationData,
                         self._calib_cb, queue_size=1)
        rospy.Subscriber('/d435i/rgb/image_raw', Image,
                         self._image_cb, queue_size=1)

        rospy.loginfo("SpheroDetectionNode ready — waiting for /camera/calibration …")

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _calib_cb(self, msg):
        if not msg.is_calibrated:
            return
        with self.lock:
            self._compute_geometry(msg)

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr_throttle(5, f"imgmsg_to_cv2: {e}")
            return
        with self.lock:
            self.frame = frame

    # ── 3D → 2D geometry ──────────────────────────────────────────────────────

    def _compute_geometry(self, msg):
        """
        Project the four paper corners from 3D camera-frame coordinates to image
        pixels, then compute a planar homography that maps any image pixel inside
        the paper onto normalised paper coordinates (u,v) ∈ [0,1]².

        Corner convention from camera_calibration_node
        ─────────────────────────────────────────────
          plane_center            = TL  →  paper (0, 0)
          TL + paper_x_axis * Lx = TR  →  paper (1, 0)
          TL + paper_y_axis * Ly = BL  →  paper (0, 1)
          TL + x·Lx  +  y·Ly    = BR  →  paper (1, 1)
        """
        p1 = np.array([msg.plane_center.x, msg.plane_center.y, msg.plane_center.z])
        ax = np.array([msg.paper_x_axis.x, msg.paper_x_axis.y, msg.paper_x_axis.z])
        ay = np.array([msg.paper_y_axis.x, msg.paper_y_axis.y, msg.paper_y_axis.z])
        Lx = float(msg.paper_x_m)
        Ly = float(msg.paper_y_m)

        # Four corners in 3D camera frame
        corners_3d = np.array([
            p1,                          # TL
            p1 + ax * Lx,               # TR
            p1 + ay * Ly,               # BL
            p1 + ax * Lx + ay * Ly,     # BR
        ])

        # Project each corner to image pixels
        corners_px = []
        for X, Y, Z in corners_3d:
            uv = project_3d_to_pixel(X, Y, Z, self.fx, self.fy, self.cx_i, self.cy_i)
            if uv is None:
                rospy.logwarn("A paper corner projects behind the camera — check calibration.")
                return
            corners_px.append(uv)
        corners_px = np.array(corners_px, dtype=np.float32)   # shape (4, 2)

        # Axis-aligned bounding box → Hough search ROI
        x1 = max(0, int(np.floor(corners_px[:, 0].min())))
        y1 = max(0, int(np.floor(corners_px[:, 1].min())))
        x2 = int(np.ceil(corners_px[:, 0].max()))
        y2 = int(np.ceil(corners_px[:, 1].max()))

        # Homography: image pixels → normalised paper
        #   src order: TL TR BL BR  (same as corners_px)
        #   dst order: (0,0) (1,0) (0,1) (1,1)
        dst_norm = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
        H, _ = cv2.findHomography(corners_px, dst_norm)

        # Rectification homography: image pixels → flat top-down output image
        # src: TL TR BL BR in image space
        # dst: four corners of a RECT_W × RECT_H rectangle (paper lies flat)
        dst_rect = np.array(
            [[0, 0], [RECT_W, 0], [0, RECT_H], [RECT_W, RECT_H]],
            dtype=np.float32,
        )
        H_rect = cv2.getPerspectiveTransform(corners_px[[0, 1, 2, 3]], dst_rect)

        self.corners_px = corners_px
        self.roi        = (x1, y1, x2, y2)
        self.homography = H
        self.H_rect     = H_rect

        rospy.loginfo(
            "3D→2D paper projection complete:\n"
            "  TL %s  TR %s\n  BL %s  BR %s\n"
            "  Hough ROI: (%d,%d) → (%d,%d)",
            np.round(corners_px[0]).astype(int), np.round(corners_px[1]).astype(int),
            np.round(corners_px[2]).astype(int), np.round(corners_px[3]).astype(int),
            x1, y1, x2, y2,
        )

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_hough(self, rectified, p2, min_r, max_r, blur_k):
        """
        Run CLAHE + HoughCircles on the full rectified (top-down) image.
        The ball always appears the same size here regardless of position,
        so one fixed radius range works everywhere.
        Returns list of (cx, cy, r) in rectified-image pixel coords.
        """
        gray     = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        enhanced = _clahe.apply(gray)
        k        = max(3, blur_k | 1)               # ensure odd
        blur     = cv2.GaussianBlur(enhanced, (k, k), 0)

        circles = cv2.HoughCircles(
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
        """
        From candidate circles, select the one closest to the last known position
        (within MAX_JUMP px).  Falls back to the strongest (first) Hough result
        if none are within range.
        """
        if not candidates:
            return None
        if self.last_px is None:
            return candidates[0]
        lx, ly = self.last_px
        best, best_d = None, float('inf')
        for cx, cy, r in candidates:
            d = np.hypot(cx - lx, cy - ly)
            if d < best_d and d < MAX_JUMP:
                best_d, best = d, (cx, cy, r)
        return best if best is not None else candidates[0]

    def _pixel_to_paper(self, cx, cy):
        """Map a full-frame image pixel to normalised paper (u, v) via homography."""
        pt  = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.homography)
        return float(res[0, 0, 0]), float(res[0, 0, 1])

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_paper_quad(self, img, corners_px):
        """
        Draw the paper outline (projected from 3D) as a green quadrilateral.
        Corner order in corners_px: TL TR BL BR
        Drawing order for a proper rectangle: TL → TR → BR → BL → TL
        """
        order  = [0, 1, 3, 2]    # TL, TR, BR, BL
        labels = ['TL', 'TR', 'BR', 'BL']
        for i in range(4):
            a = tuple(corners_px[order[i]].astype(int))
            b = tuple(corners_px[order[(i + 1) % 4]].astype(int))
            cv2.line(img, a, b, (0, 220, 0), 2)
        for i, lbl in enumerate(labels):
            pt = tuple(corners_px[order[i]].astype(int))
            cv2.circle(img, pt, 7, (0, 255, 0), -1)
            cv2.putText(img, lbl, (pt[0] + 9, pt[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    def _draw_trail(self, img, trail):
        pts = list(trail)
        n   = len(pts)
        for i in range(1, n):
            a   = i / n
            col = (int(50 * a), int(200 * a), int(255 * (1 - a) + 80 * a))
            cv2.line(img, pts[i - 1], pts[i], col, max(1, int(3 * a)))

    def _make_paper_map(self, u_now, v_now):
        """
        Render a top-down bird's-eye view of the paper with the ball position.
        The map shows the full normalised [0,1]² paper space.
        Returns a BGR image of size (MAP_H+2*MAP_PAD) × (MAP_W+2*MAP_PAD).
        """
        h = MAP_H + 2 * MAP_PAD
        w = MAP_W + 2 * MAP_PAD
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw paper rectangle
        cv2.rectangle(canvas, (MAP_PAD, MAP_PAD),
                      (MAP_PAD + MAP_W, MAP_PAD + MAP_H), (60, 60, 60), -1)
        cv2.rectangle(canvas, (MAP_PAD, MAP_PAD),
                      (MAP_PAD + MAP_W, MAP_PAD + MAP_H), (0, 200, 0), 2)

        # Corner labels
        corners_uv = [(0, 0, 'TL'), (1, 0, 'TR'), (0, 1, 'BL'), (1, 1, 'BR')]
        for cu, cv_, lbl in corners_uv:
            px = MAP_PAD + int(cu * MAP_W)
            py = MAP_PAD + int(cv_ * MAP_H)
            cv2.circle(canvas, (px, py), 5, (0, 200, 0), -1)
            ox = 6 if cu == 0 else -30
            oy = -6 if cv_ == 0 else 16
            cv2.putText(canvas, lbl, (px + ox, py + oy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)

        # UV axis labels
        cv2.putText(canvas, 'u →', (MAP_PAD + MAP_W // 2 - 20, MAP_PAD - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'v', (4, MAP_PAD + MAP_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, '↓', (4, MAP_PAD + MAP_H // 2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Draw UV trail
        uv_pts = list(self.trail_uv)
        n = len(uv_pts)
        for i in range(1, n):
            a   = i / n
            col = (int(50 * a), int(200 * a), int(255 * (1 - a) + 80 * a))
            au, av = uv_pts[i - 1]
            bu, bv = uv_pts[i]
            pa = (MAP_PAD + int(np.clip(au, 0, 1) * MAP_W),
                  MAP_PAD + int(np.clip(av, 0, 1) * MAP_H))
            pb = (MAP_PAD + int(np.clip(bu, 0, 1) * MAP_W),
                  MAP_PAD + int(np.clip(bv, 0, 1) * MAP_H))
            cv2.line(canvas, pa, pb, col, max(1, int(3 * a)))

        # Draw current ball position
        if u_now is not None:
            bx = MAP_PAD + int(np.clip(u_now, 0, 1) * MAP_W)
            by = MAP_PAD + int(np.clip(v_now, 0, 1) * MAP_H)
            cv2.circle(canvas, (bx, by), 10, (0, 255, 255), 2)
            cv2.circle(canvas, (bx, by),  4, (0, 0, 255), -1)
            cv2.putText(canvas, f"u={u_now:.3f}  v={v_now:.3f}",
                        (MAP_PAD, MAP_PAD + MAP_H + MAP_PAD - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return canvas

    # ── Main loop ─────────────────────────────────────────────────────────────

    def spin(self):
        WIN_MAIN = "Sphero Detection  [Q=quit]"
        WIN_RECT = "Rectified View (top-down)"
        WIN_DBG  = "Hough ROI (CLAHE)"
        WIN_MAP  = "Paper Map (top-down)"
        WIN_TB   = "Hough Controls"

        cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_MAIN, 1280, 720)
        cv2.namedWindow(WIN_RECT, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_RECT, RECT_W, RECT_H)
        cv2.namedWindow(WIN_MAP, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_MAP, MAP_W + 2 * MAP_PAD, MAP_H + 2 * MAP_PAD + 30)
        cv2.namedWindow(WIN_TB, cv2.WINDOW_NORMAL)

        # Trackbars — live Hough tuning
        cv2.createTrackbar("Sensitivity (p2)\nlower=more", WIN_TB, HOUGH_PARAM2, 80,  _nothing)
        cv2.createTrackbar("Min radius (px)",              WIN_TB, MIN_RADIUS,   100, _nothing)
        cv2.createTrackbar("Max radius (px)",              WIN_TB, MAX_RADIUS,   200, _nothing)
        cv2.createTrackbar("Blur kernel (px)",             WIN_TB, BLUR_KERNEL,  15,  _nothing)

        rate = rospy.Rate(30)
        u_now = v_now = None

        while not rospy.is_shutdown():
            with self.lock:
                frame      = self.frame.copy()      if self.frame      is not None else None
                corners_px = self.corners_px.copy() if self.corners_px is not None else None
                roi        = self.roi
                H          = self.homography
                H_rect     = self.H_rect

            # ── Waiting for camera ─────────────────────────────────────────────
            if frame is None:
                rospy.loginfo_throttle(5, "Waiting for camera frames …")
                rate.sleep()
                continue

            disp = frame.copy()

            # ── Waiting for calibration ────────────────────────────────────────
            if corners_px is None:
                cv2.putText(disp, "Waiting for /camera/calibration …",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 255), 2, cv2.LINE_AA)
                cv2.imshow(WIN_MAIN, disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                rate.sleep()
                continue

            # ── Read live trackbar values ──────────────────────────────────────
            p2     = cv2.getTrackbarPos("Sensitivity (p2)\nlower=more", WIN_TB)
            min_r  = cv2.getTrackbarPos("Min radius (px)",              WIN_TB)
            max_r  = cv2.getTrackbarPos("Max radius (px)",              WIN_TB)
            blur_k = cv2.getTrackbarPos("Blur kernel (px)",             WIN_TB)

            # ── Step 1: warp camera frame → rectified top-down image ───────────
            rectified = cv2.warpPerspective(frame, H_rect, (RECT_W, RECT_H))

            # ── Step 2: Hough runs on the rectified image ──────────────────────
            # Ball is always the same apparent size here (no perspective distortion).
            candidates = self._detect_hough(rectified, p2, min_r, max_r, blur_k)
            detected   = self._pick_best(candidates)   # coords in rectified px space

            u_now = v_now = None
            cam_cx = cam_cy = None   # ball pos projected back to camera space

            if detected is not None:
                cx, cy, r = detected   # rectified pixel coords
                self.last_px    = (cx, cy)
                self.lost_count = 0
                self.trail_px.append((cx, cy))

                # Normalised paper position: trivial division
                u_now = float(np.clip(cx / RECT_W, 0.0, 1.0))
                v_now = float(np.clip(cy / RECT_H, 0.0, 1.0))
                self.trail_uv.append((u_now, v_now))

                # Project ball back to camera pixel space (for main-view overlay)
                H_inv = np.linalg.inv(H_rect)
                pt_c  = cv2.perspectiveTransform(
                    np.array([[[float(cx), float(cy)]]], dtype=np.float32), H_inv)
                cam_cx, cam_cy = int(pt_c[0, 0, 0]), int(pt_c[0, 0, 1])

                self.det_pub.publish(Point(x=u_now, y=v_now, z=float(r)))
                rospy.loginfo_throttle(
                    0.5,
                    "Sphero → rect=(%d,%d)  paper=(u=%.3f, v=%.3f)  r=%d px",
                    cx, cy, u_now, v_now, r,
                )
            else:
                self.lost_count += 1
                if self.lost_count > MAX_LOST:
                    self.last_px = None
                    if self.trail_px:
                        self.trail_px.popleft()
                    if self.trail_uv:
                        self.trail_uv.popleft()

            # ── Rectified view: draw trail + candidates + detection ────────────
            rect_disp = rectified.copy()
            self._draw_trail(rect_disp, self.trail_px)

            for cx_c, cy_c, r_c in candidates:
                cv2.circle(rect_disp, (cx_c, cy_c), r_c, (80, 80, 80), 1)

            if detected is not None:
                cx, cy, r = detected
                cv2.circle(rect_disp, (cx, cy), max(r, 4), (0, 255, 255), 2)
                cv2.circle(rect_disp, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(rect_disp, f"u={u_now:.3f}  v={v_now:.3f}  r={r}px",
                            (cx + r + 4, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(rect_disp, f"searching…  (lost {self.lost_count}f)",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2, cv2.LINE_AA)

            cv2.rectangle(rect_disp, (0, 0), (RECT_W - 1, RECT_H - 1), (0, 200, 0), 2)
            cv2.putText(rect_disp,
                        f"p2={p2}  r={min_r}-{max_r}px  blur={max(3,blur_k|1)}  |  Q=quit",
                        (6, RECT_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(WIN_RECT, rect_disp)

            # ── Main camera view: reference only — shows paper outline + ball ──
            self._draw_paper_quad(disp, corners_px)
            if cam_cx is not None:
                cv2.circle(disp, (cam_cx, cam_cy), max(r, 4), (0, 255, 255), 2)
                cv2.circle(disp, (cam_cx, cam_cy), 4, (0, 0, 255), -1)
                cv2.putText(disp, f"u={u_now:.3f}  v={v_now:.3f}",
                            (cam_cx + r + 6, cam_cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, "Camera reference  |  Hough runs on Rectified View",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(WIN_MAIN, disp)

            # ── Debug: CLAHE of the rectified image (what Hough actually sees) ─
            gray_rect  = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
            clahe_rect = _clahe.apply(gray_rect)
            dbg        = cv2.cvtColor(clahe_rect, cv2.COLOR_GRAY2BGR)
            if detected is not None:
                cx, cy, r = detected
                cv2.circle(dbg, (cx, cy), max(r, 4), (0, 255, 255), 2)
                cv2.circle(dbg, (cx, cy), 3, (0, 0, 255), -1)
            cv2.namedWindow(WIN_DBG, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_DBG, RECT_W, RECT_H)
            cv2.imshow(WIN_DBG, dbg)

            # ── Paper map: top-down normalised view ────────────────────────────
            paper_map = self._make_paper_map(u_now, v_now)
            cv2.imshow(WIN_MAP, paper_map)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node = SpheroDetectionNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        rospy.loginfo("Interrupted.")
