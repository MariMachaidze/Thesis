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
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
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


class EMAFilter:
    """Exponential moving average filter."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.x_prev = None

    def __call__(self, x, t=None):
        if self.x_prev is None:
            self.x_prev = x
            return x
        x_hat = self.alpha * x + (1 - self.alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        self.x_prev = None


class FingerAnalysisNode:
    NUM_LANDMARKS = 21

    def __init__(self):
        rospy.init_node('finger_analysis_node', anonymous=False)

        # Subscribers
        self.hand_sub = rospy.Subscriber('/hand/detection', Hand, self.hand_callback)
        self.depth_sub = rospy.Subscriber('/d435i/depth/image_raw', Image, self.depth_callback)

        # Publisher
        self.finger_pub = rospy.Publisher('/finger/analysis', FingerAnalysis, queue_size=1)

        # Parameters
        self.straightness_threshold = rospy.get_param('~straightness_threshold', 0.7)

        # Camera intrinsics
        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 637.649)
        self.cy = rospy.get_param('~cy', 349.990)
        self.depth_scale = 0.001

        # Landmark smoothing filter
        self.filter_type = rospy.get_param('~filter_type', 'one_euro')  # "one_euro" or "ema"
        self.filter_min_cutoff = rospy.get_param('~filter_min_cutoff', 1.0)
        self.filter_beta = rospy.get_param('~filter_beta', 0.007)
        self.filter_d_cutoff = rospy.get_param('~filter_d_cutoff', 1.0)
        self.ema_alpha = rospy.get_param('~ema_alpha', 0.1)
        self._init_filters()

        self.bridge = CvBridge()
        self.depth_image = None

        rospy.loginfo(f"Finger Analysis Node: Ready (filter={self.filter_type})")
    
    def _init_filters(self):
        """Create per-landmark, per-axis smoothing filters."""
        self.filters_x = []
        self.filters_y = []
        for _ in range(self.NUM_LANDMARKS):
            if self.filter_type == 'one_euro':
                self.filters_x.append(OneEuroFilter(self.filter_min_cutoff, self.filter_beta, self.filter_d_cutoff))
                self.filters_y.append(OneEuroFilter(self.filter_min_cutoff, self.filter_beta, self.filter_d_cutoff))
            else:
                self.filters_x.append(EMAFilter(self.ema_alpha))
                self.filters_y.append(EMAFilter(self.ema_alpha))

    def _filter_landmarks(self, keypoints, stamp_sec):
        """Return a list of (x, y) tuples with smoothed normalized coords."""
        filtered = []
        for i, pt in enumerate(keypoints):
            if i < self.NUM_LANDMARKS:
                fx = self.filters_x[i](pt.x, stamp_sec)
                fy = self.filters_y[i](pt.y, stamp_sec)
            else:
                fx, fy = pt.x, pt.y
            filtered.append((fx, fy))
        return filtered

    def depth_callback(self, msg):
        """Store depth frame"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        except Exception as e:
            rospy.logerr(f"Depth error: {e}")
    
    def get_3d_point(self, x, y):
        """Convert pixel to 3D"""
        if self.depth_image is None:
            return None
        
        h, w = self.depth_image.shape[:2]
        x = int(np.clip(x, 0, w-1))
        y = int(np.clip(y, 0, h-1))
        
        depth_value = self.depth_image[y, x]
        if depth_value == 0 or depth_value > 10000:
            return None
        
        z = depth_value * self.depth_scale
        X = (x - self.cx) * z / self.fx
        Y = (y - self.cy) * z / self.fy
        
        return np.array([X, Y, z])
    
    def calculate_straightness(self, landmarks_3d):
        """Calculate straightness"""
        if len(landmarks_3d) < 9:
            return 0.0
        
        p5, p6, p7, p8 = landmarks_3d[5], landmarks_3d[6], landmarks_3d[7], landmarks_3d[8]
        
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
        """Process hand with filtered landmark coordinates"""
        if not msg.detected or self.depth_image is None:
            return

        h, w = self.depth_image.shape[:2]
        stamp_sec = msg.header.stamp.to_sec()

        # Smooth normalized landmark coordinates before 3D back-projection
        filtered = self._filter_landmarks(msg.keypoints_2d, stamp_sec)

        landmarks_3d = [self.get_3d_point(int(fx * w), int(fy * h))
                        for fx, fy in filtered]

        finger_msg = FingerAnalysis()
        finger_msg.header.stamp = msg.header.stamp
        finger_msg.header.frame_id = msg.header.frame_id

        # Publish the *filtered* 2D coords so downstream sees smoothed values
        knuckle_2d = Point()
        knuckle_2d.x, knuckle_2d.y = filtered[5]
        finger_msg.knuckle_2d = knuckle_2d

        tip_2d = Point()
        tip_2d.x, tip_2d.y = filtered[8]
        finger_msg.tip_2d = tip_2d

        if landmarks_3d[5] is not None:
            p = landmarks_3d[5]
            finger_msg.knuckle_3d.x, finger_msg.knuckle_3d.y, finger_msg.knuckle_3d.z = p

        if landmarks_3d[8] is not None:
            p = landmarks_3d[8]
            finger_msg.tip_3d.x, finger_msg.tip_3d.y, finger_msg.tip_3d.z = p

        if landmarks_3d[5] is not None and landmarks_3d[8] is not None:
            direction = landmarks_3d[8] - landmarks_3d[5]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
                finger_msg.direction_3d.x = direction[0]
                finger_msg.direction_3d.y = direction[1]
                finger_msg.direction_3d.z = direction[2]

        straightness = self.calculate_straightness(landmarks_3d)
        finger_msg.straightness_score = straightness
        finger_msg.is_straight = straightness > self.straightness_threshold
        finger_msg.confidence = msg.confidences[8] if len(msg.confidences) > 8 else 0.0

        self.finger_pub.publish(finger_msg)

if __name__ == '__main__':
    try:
        node = FingerAnalysisNode()
        rospy.spin()
    except KeyboardInterrupt:
        pass
