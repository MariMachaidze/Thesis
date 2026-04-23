#!/usr/bin/env python3
"""
Finger Analysis Node
Analyzes index finger straightness and computes 3D positions
"""

import rospy
import numpy as np
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Vector3
from cv_bridge import CvBridge
import cv2

from std_msgs.msg import Float32

from hand_detection.msg import Hand
from finger_analysis.msg import FingerAnalysis


class OneEuroFilter:
    """
    One Euro Filter — adaptive low-pass filter (Casiez et al. 2012).
    Smooth when slow, responsive when fast.
    cutoff_freq = min_cutoff + beta * |velocity|
    """
    def __init__(self, min_cutoff=0.3, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        # Estimate derivative
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        # Filter the signal
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class FingerAnalysisNode:
    # Index finger landmarks: MCP(5), PIP(6), DIP(7), TIP(8)
    FINGER_INDICES = [5, 6, 7, 8]

    def __init__(self):
        rospy.init_node('finger_analysis_node', anonymous=False)

        # Subscribers
        self.hand_sub = rospy.Subscriber('/hand/detection', Hand, self.hand_callback)
        self.depth_sub = rospy.Subscriber('/d435i/depth/image_raw', Image, self.depth_callback)

        # Publishers
        self.finger_pub = rospy.Publisher('/finger/analysis', FingerAnalysis, queue_size=1)
        self.raw_pub = rospy.Publisher('/diag/straightness_raw', Float32, queue_size=1)

        # Straightness hysteresis thresholds
        self.straight_on_threshold = rospy.get_param('~straight_on_threshold', 0.65)
        self.straight_off_threshold = rospy.get_param('~straight_off_threshold', 0.45)
        self.is_straight_state = False

        # # Keep a small buffer for median filtering (commented out — One Euro is active)
        # from collections import deque
        # self.raw_history = deque(maxlen=3)

        # # EMA smoothing for straightness score before hysteresis (commented out — One Euro is active)
        # self.straightness_ema_alpha = rospy.get_param('~straightness_ema_alpha', 0.3)
        self.straightness_ema = None  # kept so message field assignment doesn't crash

        # Last valid values — used when depth fails so we publish last known instead of 0
        self.last_valid_raw = None
        self.last_valid_oe  = None

        # Camera intrinsics
        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 637.649)
        self.cy = rospy.get_param('~cy', 349.990)
        self.depth_scale = 0.001

        # Patch size for median depth filtering (applied to all joints equally)
        self.depth_patch_size = rospy.get_param('~depth_patch_size', 3)

        # One Euro Filter on straightness score (optional, gated by ~use_one_euro)
        self.use_one_euro = rospy.get_param('~use_one_euro', False)
        oe_min_cutoff = rospy.get_param('~one_euro_min_cutoff', 1.0)
        oe_beta       = rospy.get_param('~one_euro_beta', 0.007)
        oe_d_cutoff   = rospy.get_param('~one_euro_d_cutoff', 1.0)
        self.straightness_oe_filter = OneEuroFilter(oe_min_cutoff, oe_beta, oe_d_cutoff)

        # One Euro Filters — separate sets for knuckle origin and direction
        min_cutoff = rospy.get_param('~filter_min_cutoff', 1.0)
        beta = rospy.get_param('~filter_beta', 0.007)
        d_cutoff = rospy.get_param('~filter_d_cutoff', 1.0)

        # Filter knuckle 3D position (ray origin) — 3 axes
        self.knuckle_filters = (
            OneEuroFilter(min_cutoff, beta, d_cutoff),
            OneEuroFilter(min_cutoff, beta, d_cutoff),
            OneEuroFilter(min_cutoff, beta, d_cutoff),
        )
        # Filter direction vector (knuckle→tip) — 3 axes
        self.direction_filters = (
            OneEuroFilter(min_cutoff, beta, d_cutoff),
            OneEuroFilter(min_cutoff, beta, d_cutoff),
            OneEuroFilter(min_cutoff, beta, d_cutoff),
        )

        self.bridge = CvBridge()
        self.depth_image = None

        oe_status = f"One Euro ON (min_cutoff={oe_min_cutoff}, beta={oe_beta})" if self.use_one_euro else "One Euro OFF (EMA active)"
        rospy.loginfo(f"Finger Analysis Node: Ready — straightness filter: {oe_status}")

    def depth_callback(self, msg):
        """Store depth frame"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        except Exception as e:
            rospy.logerr(f"Depth error: {e}")

    def get_3d_point(self, x, y):
        """Convert pixel to 3D using median depth patch for noise reduction."""
        if self.depth_image is None:
            return None

        h, w = self.depth_image.shape[:2]
        x = int(np.clip(x, 0, w-1))
        y = int(np.clip(y, 0, h-1))

        half = self.depth_patch_size // 2
        y_min = max(0, y - half)
        y_max = min(h, y + half + 1)
        x_min = max(0, x - half)
        x_max = min(w, x + half + 1)

        patch = self.depth_image[y_min:y_max, x_min:x_max]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None

        depth_value = int(np.median(valid))
        if depth_value == 0 or depth_value > 10000:
            return None

        z = depth_value * self.depth_scale
        X = (x - self.cx) * z / self.fx
        Y = (y - self.cy) * z / self.fy

        return np.array([X, Y, z])

    def fit_finger_direction(self, points_3d):
        """Fit a line through finger joints using SVD for robust direction.
        Uses all joints with valid depth so one bad point (e.g. thin tip)
        doesn't dominate the direction estimate."""
        valid = [p for p in points_3d if p is not None]
        if len(valid) < 2:
            return None

        pts = np.array(valid)
        centroid = pts.mean(axis=0)
        centered = pts - centroid

        # SVD — first right singular vector = best-fit line direction
        _, _, Vt = np.linalg.svd(centered)
        direction = Vt[0]

        # Ensure direction points from knuckle toward tip
        if points_3d[0] is not None and points_3d[-1] is not None:
            rough_dir = points_3d[-1] - points_3d[0]
        else:
            rough_dir = valid[-1] - valid[0]
        if np.dot(direction, rough_dir) < 0:
            direction = -direction

        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        return direction / norm

    def calculate_straightness(self, p5, p6, p7, p8):
        """Calculate straightness from the 4 index finger joints directly."""
        if any(p is None for p in [p5, p6, p7, p8]):
            return None

        vec1 = p6 - p5
        vec2 = p7 - p6
        vec3 = p8 - p7

        dots = []
        for v1, v2 in [(vec1, vec2), (vec2, vec3)]:
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-4 or n2 < 1e-4:
                return None
            cos_a = np.dot(v1, v2) / (n1 * n2)
            dots.append(float(np.clip(cos_a, 0.0, 1.0)))
        return float(np.mean(dots))

    def hand_callback(self, msg):
        """Process hand — back-project to 3D, filter origin and direction separately"""
        if not msg.detected or self.depth_image is None:
            return
        if len(msg.keypoints_2d) < 9:
            return

        h, w = self.depth_image.shape[:2]
        stamp_sec = msg.header.stamp.to_sec()

        # Back-project raw 2D landmarks to 3D (with depth patch median)
        raw_3d = {}
        finger_2d = {}
        for idx in self.FINGER_INDICES:
            pt = msg.keypoints_2d[idx]
            px = int(pt.x * w)
            py = int(pt.y * h)
            finger_2d[idx] = (pt.x, pt.y)
            raw_3d[idx] = self.get_3d_point(px, py)

        finger_msg = FingerAnalysis()
        finger_msg.header.stamp = msg.header.stamp
        finger_msg.header.frame_id = msg.header.frame_id

        # 2D coords (raw)
        knuckle_2d = Point()
        knuckle_2d.x, knuckle_2d.y = finger_2d[5]
        finger_msg.knuckle_2d = knuckle_2d

        tip_2d = Point()
        tip_2d.x, tip_2d.y = finger_2d[8]
        finger_msg.tip_2d = tip_2d

        # Knuckle depth is required for a usable ray origin
        if raw_3d[5] is None:
            return

        kf = self.knuckle_filters
        knuckle_filt = np.array([
            kf[0](raw_3d[5][0], stamp_sec),
            kf[1](raw_3d[5][1], stamp_sec),
            kf[2](raw_3d[5][2], stamp_sec),
        ])
        finger_msg.knuckle_3d.x = knuckle_filt[0]
        finger_msg.knuckle_3d.y = knuckle_filt[1]
        finger_msg.knuckle_3d.z = knuckle_filt[2]

        if raw_3d[8] is not None:
            finger_msg.tip_3d.x = raw_3d[8][0]
            finger_msg.tip_3d.y = raw_3d[8][1]
            finger_msg.tip_3d.z = raw_3d[8][2]

        # Fit direction through all 4 joints (robust to noisy tip depth)
        direction = self.fit_finger_direction(
            [raw_3d[5], raw_3d[6], raw_3d[7], raw_3d[8]])
        if direction is not None:
            # Filter each component of the unit direction vector
            df = self.direction_filters
            filt_dir = np.array([
                df[0](direction[0], stamp_sec),
                df[1](direction[1], stamp_sec),
                df[2](direction[2], stamp_sec),
            ])
            # Re-normalize after filtering
            filt_norm = np.linalg.norm(filt_dir)
            if filt_norm > 1e-6:
                filt_dir = filt_dir / filt_norm
            finger_msg.direction_3d.x = filt_dir[0]
            finger_msg.direction_3d.y = filt_dir[1]
            finger_msg.direction_3d.z = filt_dir[2]

        raw = self.calculate_straightness(raw_3d[5], raw_3d[6], raw_3d[7], raw_3d[8])

        # Publish raw score for diagnostics (before any smoothing)
        if raw is not None:
            self.raw_pub.publish(Float32(data=raw))

        # # EMA computation commented out — One Euro filter is the active smoother
        # if raw is not None:
        #     self.raw_history.append(raw)
        #     filtered = float(np.median(self.raw_history))
        #     if self.straightness_ema is None:
        #         self.straightness_ema = filtered
        #     else:
        #         a = self.straightness_ema_alpha
        #         self.straightness_ema = a * filtered + (1 - a) * self.straightness_ema

        # One Euro Filter on raw straightness (computed every frame regardless of gate,
        # so the filter state stays warm and the value is always available in the message)
        oe_val = self.straightness_oe_filter(raw, stamp_sec) if raw is not None else None

        # Update hold-last trackers
        if raw    is not None: self.last_valid_raw = raw
        if oe_val is not None: self.last_valid_oe  = oe_val

        # Hysteresis: activate at on_threshold, deactivate at off_threshold
        # Driven by One Euro filter
        if oe_val is not None:
            if self.is_straight_state:
                if oe_val < self.straight_off_threshold:
                    self.is_straight_state = False
            else:
                if oe_val > self.straight_on_threshold:
                    self.is_straight_state = True

        # Populate all three filter outputs — use last valid value when depth fails
        finger_msg.straightness_raw       = self.last_valid_raw if self.last_valid_raw is not None else 0.0
        finger_msg.straightness_ema       = self.straightness_ema if self.straightness_ema is not None else 0.0
        finger_msg.straightness_one_euro  = self.last_valid_oe  if self.last_valid_oe  is not None else 0.0

        # Active output: One Euro when enabled, EMA otherwise
        if self.use_one_euro and oe_val is not None:
            finger_msg.straightness_score = oe_val
        else:
            finger_msg.straightness_score = finger_msg.straightness_ema

        finger_msg.is_straight = self.is_straight_state
        finger_msg.confidence = msg.confidences[8] if len(msg.confidences) > 8 else 0.0

        # # Verbose comparison print commented out
        # raw_s = f"{raw:.3f}" if raw is not None else "None"
        # rospy.loginfo(
        #     f"Straightness  raw={raw_s}"
        #     f"  ema={finger_msg.straightness_ema:.3f}"
        #     f"  oe={finger_msg.straightness_one_euro:.3f}"
        #     f"  score={finger_msg.straightness_score:.3f}"
        #     f"  straight={self.is_straight_state}"
        # )

        self.finger_pub.publish(finger_msg)

if __name__ == '__main__':
    try:
        node = FingerAnalysisNode()
        rospy.spin()
    except KeyboardInterrupt:
        pass
