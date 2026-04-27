#!/usr/bin/env python3
"""
Finger Analysis Node
Analyzes index finger straightness and computes 3D positions
"""

import rospy
import numpy as np
import time
from collections import deque
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

        # Subscribers — queue_size=1 + buff_size so we never process stale depth frames
        self.hand_sub  = rospy.Subscriber('/hand/detection',         Hand,  self.hand_callback,  queue_size=1)
        self.depth_sub = rospy.Subscriber('/d435i/depth/image_raw',  Image, self.depth_callback, queue_size=1, buff_size=2**24)

        # Publishers
        self.finger_pub = rospy.Publisher('/finger/analysis', FingerAnalysis, queue_size=1)
        self.raw_pub = rospy.Publisher('/diag/straightness_raw', Float32, queue_size=1)

        # Straightness hysteresis thresholds
        self.straight_on_threshold = rospy.get_param('~straight_on_threshold', 0.65)
        self.straight_off_threshold = rospy.get_param('~straight_off_threshold', 0.45)
        self.is_straight_state = False

        # Last valid values — used when depth fails so we publish last known instead of 0
        self.last_valid_raw = None
        self.last_valid_oe  = None

        # Camera intrinsics
        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 642.351)
        self.cy = rospy.get_param('~cy', 349.990)
        self.depth_scale = 0.001

        # Patch size for median depth filtering (applied to all joints equally)
        self.depth_patch_size = rospy.get_param('~depth_patch_size', 3)

        # One Euro Filter on straightness score
        oe_min_cutoff = rospy.get_param('~one_euro_min_cutoff', 1.0)
        oe_beta       = rospy.get_param('~one_euro_beta', 0.007)
        oe_d_cutoff   = rospy.get_param('~one_euro_d_cutoff', 1.0)
        self.straightness_oe_filter = OneEuroFilter(oe_min_cutoff, oe_beta, oe_d_cutoff)

        # One Euro Filters — one set per finger joint (12 filters total for 4 joints × 3 axes).
        # Direction is derived from the *filtered* joints so origin and direction stay
        # temporally consistent (no separate direction filter on the unit vector).
        min_cutoff = rospy.get_param('~filter_min_cutoff', 1.0)
        beta = rospy.get_param('~filter_beta', 0.007)
        d_cutoff = rospy.get_param('~filter_d_cutoff', 1.0)
        self.joint_filters = {
            idx: tuple(OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3))
            for idx in self.FINGER_INDICES
        }

        # Depth buffer: match depth frame to the hand message's source stamp so we
        # don't pair a hand detected at frame N with depth from frame N+1/N+2.
        self.depth_buffer_size = int(rospy.get_param('~depth_buffer_size', 10))
        self.depth_buffer = deque(maxlen=self.depth_buffer_size)

        # Reset filters after N consecutive hand-undetected frames, so a re-acquired
        # hand isn't smeared toward the stale pre-gap position.
        self.hand_loss_reset_frames = int(rospy.get_param('~hand_loss_reset_frames', 5))
        self.frames_since_detected = 0

        self.bridge = CvBridge()

        rospy.loginfo(f"Finger Analysis Node: Ready — One Euro (min_cutoff={oe_min_cutoff}, beta={oe_beta})")

    def depth_callback(self, msg):
        """Buffer depth frame so hand_callback can pair it by timestamp."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            self.depth_buffer.append((msg.header.stamp.to_sec(), img))
        except Exception as e:
            rospy.logerr(f"Depth error: {e}")

    def _get_depth_for_stamp(self, target_stamp):
        """Return (depth_image, matched_ts) whose stamp is closest to target_stamp."""
        if not self.depth_buffer:
            return None, None
        target = target_stamp.to_sec()
        best_img, best_ts, best_diff = None, None, float('inf')
        for ts, img in self.depth_buffer:
            diff = abs(ts - target)
            if diff < best_diff:
                best_diff, best_img, best_ts = diff, img, ts
        return best_img, best_ts

    def _reset_filters(self):
        """Clear filter state after the hand has been lost for a while."""
        for filters in self.joint_filters.values():
            for f in filters:
                f.reset()
        self.straightness_oe_filter.reset()
        self.last_valid_raw = None
        self.last_valid_oe  = None
        self.is_straight_state = False

    def get_3d_point(self, x, y, depth_image):
        """Convert pixel to 3D using median depth patch for noise reduction."""
        if depth_image is None:
            return None

        h, w = depth_image.shape[:2]
        x = int(np.clip(x, 0, w-1))
        y = int(np.clip(y, 0, h-1))

        half = self.depth_patch_size // 2
        y_min = max(0, y - half)
        y_max = min(h, y + half + 1)
        x_min = max(0, x - half)
        x_max = min(w, x + half + 1)

        patch = depth_image[y_min:y_max, x_min:x_max]
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
        """Process hand — pair with stamp-matched depth, filter joints, derive direction"""
        # Hand-loss handling — reset filter state after a sustained gap
        if not msg.detected:
            self.frames_since_detected += 1
            if self.frames_since_detected == self.hand_loss_reset_frames:
                rospy.loginfo("Hand lost — resetting filters")
                self._reset_filters()
            return
        self.frames_since_detected = 0

        depth_image, matched_ts = self._get_depth_for_stamp(msg.header.stamp)
        if depth_image is None:
            rospy.logwarn_throttle(1.0, "[FA] No depth frame in buffer — skipping frame")
            return
        if len(msg.keypoints_2d) < 9:
            return

        h, w = depth_image.shape[:2]
        stamp_sec = msg.header.stamp.to_sec()
        depth_age_ms = abs(matched_ts - stamp_sec) * 1000.0
        rospy.loginfo_throttle(0.5, f"[FA] depth_age={depth_age_ms:.0f}ms  buf={len(self.depth_buffer)}")

        # Back-project each finger joint using the depth frame matched to this hand message
        raw_3d = {}
        finger_2d = {}
        for idx in self.FINGER_INDICES:
            pt = msg.keypoints_2d[idx]
            px = int(pt.x * w)
            py = int(pt.y * h)
            finger_2d[idx] = (pt.x, pt.y)
            raw_3d[idx] = self.get_3d_point(px, py, depth_image)

        valid_joints = [idx for idx in self.FINGER_INDICES if raw_3d[idx] is not None]
        rospy.loginfo_throttle(0.5, f"[FA] valid_3d={valid_joints}  ({len(valid_joints)}/4 joints have depth)")

        # Filter each joint's 3D position independently — direction is derived
        # from the filtered joints (no separate per-axis direction filter, which
        # would break the unit-vector assumption on S²).
        filt_3d = {}
        for idx in self.FINGER_INDICES:
            if raw_3d[idx] is None:
                filt_3d[idx] = None
                continue
            f = self.joint_filters[idx]
            filt_3d[idx] = np.array([
                f[0](raw_3d[idx][0], stamp_sec),
                f[1](raw_3d[idx][1], stamp_sec),
                f[2](raw_3d[idx][2], stamp_sec),
            ])

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

        # Ray origin = filtered MCP (joint 5). Required.
        if filt_3d[5] is None:
            return
        finger_msg.knuckle_3d.x = filt_3d[5][0]
        finger_msg.knuckle_3d.y = filt_3d[5][1]
        finger_msg.knuckle_3d.z = filt_3d[5][2]

        if filt_3d[8] is not None:
            finger_msg.tip_3d.x = filt_3d[8][0]
            finger_msg.tip_3d.y = filt_3d[8][1]
            finger_msg.tip_3d.z = filt_3d[8][2]

        # Direction from the filtered joints — no further smoothing on the unit vector
        direction = self.fit_finger_direction(
            [filt_3d[5], filt_3d[6], filt_3d[7], filt_3d[8]])
        if direction is not None:
            finger_msg.direction_3d.x = direction[0]
            finger_msg.direction_3d.y = direction[1]
            finger_msg.direction_3d.z = direction[2]
            rospy.loginfo_throttle(0.5, f"[FA] dir={direction.round(3).tolist()}  knuckle_z={filt_3d[5][2]:.3f}m")
        else:
            rospy.logwarn_throttle(0.5, "[FA] direction=None — too few valid depth joints for SVD")

        # Straightness computed on RAW joints so the filter gates on real noise
        raw = self.calculate_straightness(raw_3d[5], raw_3d[6], raw_3d[7], raw_3d[8])

        # Publish raw score for diagnostics (before smoothing)
        if raw is not None:
            self.raw_pub.publish(Float32(data=raw))

        # One Euro Filter — always active
        oe_val = self.straightness_oe_filter(raw, stamp_sec) if raw is not None else None

        if raw    is not None: self.last_valid_raw = raw
        if oe_val is not None: self.last_valid_oe  = oe_val

        # Hysteresis: activate at on_threshold, deactivate at off_threshold
        if oe_val is not None:
            if self.is_straight_state:
                if oe_val < self.straight_off_threshold:
                    self.is_straight_state = False
            else:
                if oe_val > self.straight_on_threshold:
                    self.is_straight_state = True

        finger_msg.straightness_raw      = self.last_valid_raw if self.last_valid_raw is not None else 0.0
        finger_msg.straightness_one_euro = self.last_valid_oe  if self.last_valid_oe  is not None else 0.0
        finger_msg.straightness_score    = finger_msg.straightness_one_euro

        finger_msg.is_straight = self.is_straight_state
        finger_msg.confidence = msg.confidences[8] if len(msg.confidences) > 8 else 0.0

        self.finger_pub.publish(finger_msg)

if __name__ == '__main__':
    try:
        node = FingerAnalysisNode()
        rospy.spin()
    except KeyboardInterrupt:
        pass
