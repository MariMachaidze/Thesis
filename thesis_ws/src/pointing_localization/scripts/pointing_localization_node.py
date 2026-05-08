#!/usr/bin/env python3
"""
Pointing Localization Node
Calculates where the finger points on the paper plane
"""

import rospy
import numpy as np

from finger_analysis.msg import FingerAnalysis
from camera_calibration.msg import CalibrationData
from pointing_localization.msg import PointingTarget
from geometry_msgs.msg import Point


class OneEuroFilter:
    """
    One Euro Filter - adaptive low-pass filter.
    Smooth when slow, responsive when fast.

    Parameters:
        min_cutoff: minimum cutoff frequency (lower = more smoothing when still)
        beta: speed coefficient (higher = less lag when moving fast)
        d_cutoff: cutoff frequency for derivative estimation
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

    def __call__(self, x, t):
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


class PointingLocalizationNode:
    def __init__(self):
        rospy.init_node('pointing_localization_node', anonymous=False)

        # Subscribers
        self.finger_sub = rospy.Subscriber('/finger/analysis', FingerAnalysis, self.finger_callback)
        self.calib_sub = rospy.Subscriber('/camera/calibration', CalibrationData, self.calib_callback)

        # Publisher
        self.target_pub = rospy.Publisher('/pointing/target', PointingTarget, queue_size=1)

        # Calibration data
        self.calibration = None

        # U filter — responsive, higher beta is fine because U speed is moderate
        min_cutoff   = rospy.get_param('~filter_min_cutoff',   1.0)
        beta         = rospy.get_param('~filter_beta',         0.007)
        d_cutoff     = rospy.get_param('~filter_d_cutoff',     1.0)
        self.filter_u = OneEuroFilter(min_cutoff, beta, d_cutoff)

        # V filter — separate params: V is 1.4-1.8x faster than U and spends 2x
        # more time at the paper edges, so it needs more smoothing (lower beta).
        min_cutoff_v = rospy.get_param('~filter_min_cutoff_v', min_cutoff)
        beta_v       = rospy.get_param('~filter_beta_v',       beta)
        self.filter_v = OneEuroFilter(min_cutoff_v, beta_v, d_cutoff)

        # Diagnostic / bypass parameters
        self.diagnose_pointing = rospy.get_param('~diagnose_pointing', False)
        self.bypass_filters = rospy.get_param('~bypass_filters', False)
        if self.diagnose_pointing:
            self.diag_pointing_pub = rospy.Publisher('/diag/pointing_raw', Point, queue_size=1)
            rospy.loginfo("Pointing Localization Node: diagnose_pointing=True — diagnostic mode active")
        if self.bypass_filters:
            rospy.loginfo("Pointing Localization Node: bypass_filters=True — One Euro Filter disabled")

        self._was_valid = False
        self._not_straight_count = 0
        self._NOT_STRAIGHT_RESET = 5

        # Reset UV filters after a gap — if the intersection was out of bounds for
        # >UV_GAP_RESET_S seconds the filter state is stale; starting fresh from the
        # next raw value prevents the large lag spike on resume.
        self._UV_GAP_RESET_S = rospy.get_param('~uv_gap_reset_s', 0.5)
        self._last_valid_uv_t = None

        rospy.loginfo("Pointing Localization Node: Ready (One Euro Filter)")
    
    def calib_callback(self, msg):
        """Store calibration data and run sanity checks."""
        self.calibration = msg

        normal = np.array([msg.plane_normal.x, msg.plane_normal.y, msg.plane_normal.z])
        x_axis = np.array([msg.paper_x_axis.x, msg.paper_x_axis.y, msg.paper_x_axis.z])
        y_axis = np.array([msg.paper_y_axis.x, msg.paper_y_axis.y, msg.paper_y_axis.z])
        n_norm = float(np.linalg.norm(normal))
        x_norm = float(np.linalg.norm(x_axis))
        y_norm = float(np.linalg.norm(y_axis))
        ortho_xy = float(abs(np.dot(x_axis / max(x_norm, 1e-9),
                                    y_axis / max(y_norm, 1e-9))))
        rospy.loginfo(
            "[PL] Calibration: normal_norm=%.4f  x_norm=%.4f  y_norm=%.4f  "
            "x·y=%.4f  paper=%.3fx%.3fm",
            n_norm, x_norm, y_norm, ortho_xy, msg.paper_x_m, msg.paper_y_m)
        if abs(n_norm - 1.0) > 0.02:
            rospy.logwarn("[PL] Plane normal is not a unit vector: norm=%.4f — "
                          "recalibrate to fix systematic pointing drift", n_norm)
        if ortho_xy > 0.05:
            rospy.logwarn("[PL] Paper axes not orthogonal: |x·y|=%.4f — "
                          "corner clicks may have been misplaced", ortho_xy)
    
    def ray_plane_intersection(self, ray_origin, ray_direction, plane_normal, plane_point):
        """
        Calculate intersection of ray with plane
        
        Ray: P(t) = ray_origin + t * ray_direction
        Plane: normal · (X - plane_point) = 0
        """
        denom = np.dot(plane_normal, ray_direction)
        
        if abs(denom) < 1e-8:
            return None  # Ray parallel to plane
        
        t = np.dot(plane_normal, plane_point - ray_origin) / denom
        
        if t < 0:
            return None  # Intersection behind ray origin
        
        intersection = ray_origin + t * ray_direction
        return intersection
    
    def project_to_paper_coords(self, point_3d):
        """Project 3D point to paper 2D coordinates"""
        if self.calibration is None or not self.calibration.is_calibrated:
            return None, None
        
        # Paper origin and axes
        origin = np.array([
            self.calibration.plane_center.x,
            self.calibration.plane_center.y,
            self.calibration.plane_center.z
        ])
        
        x_axis = np.array([
            self.calibration.paper_x_axis.x,
            self.calibration.paper_x_axis.y,
            self.calibration.paper_x_axis.z
        ])
        
        y_axis = np.array([
            self.calibration.paper_y_axis.x,
            self.calibration.paper_y_axis.y,
            self.calibration.paper_y_axis.z
        ])
        
        # Vector from origin to point
        v = point_3d - origin
        
        # Project onto axes
        x_coord = np.dot(v, x_axis)
        y_coord = np.dot(v, y_axis)
        
        # Normalize: paper_x_m is the long edge (x_axis), paper_y_m the short edge (y_axis)
        x_norm = x_coord / self.calibration.paper_x_m
        y_norm = y_coord / self.calibration.paper_y_m

        # Convert to mm on paper
        x_mm = x_coord * 1000.0
        y_mm = y_coord * 1000.0
        
        return (x_norm, y_norm), (x_mm, y_mm)
    
    def finger_callback(self, msg):
        """Process finger analysis and publish pointing target"""
        rospy.loginfo_throttle(1.0,
            f"[PL] is_straight={msg.is_straight}  score={msg.straightness_score:.3f}  "
            f"knuckle=({msg.knuckle_3d.x:.3f},{msg.knuckle_3d.y:.3f},{msg.knuckle_3d.z:.3f})")

        if not msg.is_straight:
            self._not_straight_count += 1
            if self._not_straight_count >= self._NOT_STRAIGHT_RESET:
                # Sustained non-pointing — reset filters and clear display
                self.filter_u.reset()
                self.filter_v.reset()
                self._last_valid_uv_t = None
                if self._was_valid:
                    clear_msg = PointingTarget()
                    clear_msg.header.stamp = msg.header.stamp
                    clear_msg.is_valid = False
                    self.target_pub.publish(clear_msg)
                    self._was_valid = False
            return

        self._not_straight_count = 0  # back to straight — cancel any pending reset

        if self.calibration is None or not self.calibration.is_calibrated:
            rospy.logwarn_once("Calibration not yet received")
            return
        
        # Extract data
        # Ray origin: use the most distal joint finger_analysis could find (TIP→DIP→PIP).
        # finger_analysis populates tip_3d with the best available distal joint; z > 0
        # means a joint was found.  Using the distal joint instead of MCP (knuckle_3d)
        # reduces the ray-origin-to-plane distance by ~10-15 cm, which cuts the
        # angular amplification of direction errors and reduces the "near me" bias.
        if msg.tip_3d.z > 0.0:
            ray_origin = np.array([msg.tip_3d.x, msg.tip_3d.y, msg.tip_3d.z])
        else:
            ray_origin = np.array([msg.knuckle_3d.x, msg.knuckle_3d.y, msg.knuckle_3d.z])
        direction = np.array([msg.direction_3d.x, msg.direction_3d.y, msg.direction_3d.z])

        # Plane data
        plane_normal = np.array([
            self.calibration.plane_normal.x,
            self.calibration.plane_normal.y,
            self.calibration.plane_normal.z
        ])
        
        plane_point = np.array([
            self.calibration.plane_center.x,
            self.calibration.plane_center.y,
            self.calibration.plane_center.z
        ])
        
        # Calculate ray-plane intersection
        intersection_3d = self.ray_plane_intersection(ray_origin, direction, plane_normal, plane_point)
        
        # Create PointingTarget message
        target_msg = PointingTarget()
        target_msg.header.stamp = msg.header.stamp
        target_msg.header.frame_id = self.calibration.header.frame_id
        target_msg.straightness = msg.straightness_score
        target_msg.confidence = msg.confidence
        
        if intersection_3d is not None:
            # Project to paper coordinates
            coords_norm, coords_mm = self.project_to_paper_coords(intersection_3d)
            
            if coords_norm is not None:
                x_norm, y_norm = coords_norm
                x_mm, y_mm = coords_mm

                rospy.loginfo_throttle(0.5, f"[PL] raw_uv=({x_norm:.3f},{y_norm:.3f})")

                # Only update OEF and publish when the intersection is on the paper.
                # Out-of-bounds frames come from bad direction estimates during transitions
                # and would poison the OEF state — keeping the bounds check here ensures
                # the filter state stays clean regardless of transient noisy frames.
                if 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0:
                    # --- Diagnostic output (every valid frame) ---
                    if self.diagnose_pointing:
                        diag_pt = Point()
                        diag_pt.x = float(x_norm)
                        diag_pt.y = float(y_norm)
                        diag_pt.z = 0.0
                        self.diag_pointing_pub.publish(diag_pt)

                    # Apply One Euro Filter (or bypass)
                    if self.bypass_filters:
                        filtered_u = x_norm
                        filtered_v = y_norm
                    else:
                        t = msg.header.stamp.to_sec()
                        # Reset if the filter has a stale value from a long gap —
                        # this prevents the large lag spike seen when UV resumes
                        # after the intersection was briefly out of bounds.
                        if (self._last_valid_uv_t is not None and
                                t - self._last_valid_uv_t > self._UV_GAP_RESET_S):
                            self.filter_u.reset()
                            self.filter_v.reset()
                            rospy.loginfo_throttle(0.5,
                                f"[PL] UV gap {t - self._last_valid_uv_t:.2f}s — filter reset")
                        self._last_valid_uv_t = t
                        filtered_u = self.filter_u(x_norm, t)
                        filtered_v = self.filter_v(y_norm, t)

                    lag = np.hypot(filtered_u - x_norm, filtered_v - y_norm)
                    rospy.loginfo_throttle(0.5,
                        f"[PL] filt_uv=({filtered_u:.3f},{filtered_v:.3f})  OEF_lag={lag:.4f}")

                    target_msg.u_normalized = filtered_u
                    target_msg.v_normalized = filtered_v
                    target_msg.u_mm = filtered_u * self.calibration.paper_x_m * 1000.0
                    target_msg.v_mm = filtered_v * self.calibration.paper_y_m * 1000.0
                    target_msg.is_valid = True
                else:
                    target_msg.is_valid = False
            else:
                target_msg.is_valid = False
        else:
            target_msg.is_valid = False
        
        if target_msg.is_valid:
            self._was_valid = True
            self.target_pub.publish(target_msg)
        elif self._was_valid:
            self.target_pub.publish(target_msg)
            self._was_valid = False
    
    def spin(self):
        """Keep node alive"""
        rospy.spin()
    
    def cleanup(self):
        """Cleanup"""
        rospy.loginfo("Pointing Localization Node: Cleanup complete")

if __name__ == '__main__':
    try:
        node = PointingLocalizationNode()
        node.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        node.cleanup()
