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

        # Publisher
        self.finger_pub = rospy.Publisher('/finger/analysis', FingerAnalysis, queue_size=1)

        # Straightness hysteresis thresholds
        self.straight_on_threshold = rospy.get_param('~straight_on_threshold', 0.75)
        self.straight_off_threshold = rospy.get_param('~straight_off_threshold', 0.55)
        self.is_straight_state = False

        # EMA smoothing for straightness score before hysteresis
        self.straightness_ema_alpha = rospy.get_param('~straightness_ema_alpha', 0.3)
        self.straightness_ema = None

        # Camera intrinsics
        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 637.649)
        self.cy = rospy.get_param('~cy', 349.990)
        self.depth_scale = 0.001

        # Depth patch sizes for median filtering
        self.depth_patch_size = rospy.get_param('~depth_patch_size', 5)
        self.tip_patch_size = rospy.get_param('~tip_patch_size', 9)

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

        rospy.loginfo("Finger Analysis Node: Ready (depth patch median + 3D One Euro Filter)")

    def depth_callback(self, msg):
        """Store depth frame"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        except Exception as e:
            rospy.logerr(f"Depth error: {e}")

    def get_3d_point(self, x, y, landmark_idx=None):
        """Convert pixel to 3D using median depth patch for noise reduction.
        Uses larger patch for fingertip (landmark 8) which is thin."""
        if self.depth_image is None:
            return None

        h, w = self.depth_image.shape[:2]
        x = int(np.clip(x, 0, w-1))
        y = int(np.clip(y, 0, h-1))

        # Larger patch for thin fingertip
        patch_size = self.tip_patch_size if landmark_idx == 8 else self.depth_patch_size
        half = patch_size // 2
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
            return 0.0

        vec1 = p6 - p5
        vec2 = p7 - p6
        vec3 = p8 - p7

        dots = []
        for v1, v2 in [(vec1, vec2), (vec2, vec3)]:
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_a = np.dot(v1, v2) / (n1 * n2)
                dots.append(float(np.clip(cos_a, -1.0, 1.0)))

        if not dots:
            return 0.0

        # 1.0 = perfectly straight, 0.0 = bent 90 degrees, -1.0 = fully folded
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
            raw_3d[idx] = self.get_3d_point(px, py, landmark_idx=idx)

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

        # Filter knuckle 3D position (ray origin)
        if raw_3d[5] is not None:
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

        # Straightness: raw score → EMA smoothing → hysteresis
        raw_straightness = self.calculate_straightness(
            raw_3d[5], raw_3d[6], raw_3d[7], raw_3d[8])
        if self.straightness_ema is None:
            self.straightness_ema = raw_straightness
        else:
            a = self.straightness_ema_alpha
            self.straightness_ema = a * raw_straightness + (1 - a) * self.straightness_ema

        # Hysteresis: activate at on_threshold, deactivate at off_threshold
        if self.is_straight_state:
            if self.straightness_ema < self.straight_off_threshold:
                self.is_straight_state = False
        else:
            if self.straightness_ema > self.straight_on_threshold:
                self.is_straight_state = True

        finger_msg.straightness_score = self.straightness_ema
        finger_msg.is_straight = self.is_straight_state
        finger_msg.confidence = msg.confidences[8] if len(msg.confidences) > 8 else 0.0

        rospy.loginfo(f"Straightness  raw={raw_straightness:.3f}  ema={self.straightness_ema:.3f}  straight={self.is_straight_state}")

        self.finger_pub.publish(finger_msg)

if __name__ == '__main__':
    try:
        node = FingerAnalysisNode()
        rospy.spin()
    except KeyboardInterrupt:
        pass
