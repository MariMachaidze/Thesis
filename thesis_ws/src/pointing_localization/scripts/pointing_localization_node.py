#!/usr/bin/env python3
"""
Pointing Localization Node
Calculates where the finger points on the paper plane
"""

import rospy
import numpy as np
import time

from finger_analysis.msg import FingerAnalysis
from camera_calibration.msg import CalibrationData
from pointing_localization.msg import PointingTarget


class OneEuroFilter:
    """
    One Euro Filter - adaptive low-pass filter.
    Smooth when slow, responsive when fast.

    Parameters:
        min_cutoff: minimum cutoff frequency (lower = more smoothing when still)
        beta: speed coefficient (higher = less lag when moving fast)
        d_cutoff: cutoff frequency for derivative estimation
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

    def filter(self, x, t=None):
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

        # One Euro Filter parameters
        min_cutoff = rospy.get_param('~filter_min_cutoff', 1.0)
        beta = rospy.get_param('~filter_beta', 0.007)
        d_cutoff = rospy.get_param('~filter_d_cutoff', 1.0)
        self.filter_u = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.filter_v = OneEuroFilter(min_cutoff, beta, d_cutoff)

        rospy.loginfo("Pointing Localization Node: Ready (One Euro Filter)")
    
    def calib_callback(self, msg):
        """Store calibration data"""
        self.calibration = msg
        rospy.loginfo("Calibration data received")
    
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
        
        # Normalize to paper dimensions
        x_norm = x_coord / (self.calibration.paper_width_mm / 1000.0)
        y_norm = y_coord / (self.calibration.paper_height_mm / 1000.0)
        
        # Convert to mm on paper
        x_mm = x_coord * 1000.0
        y_mm = y_coord * 1000.0
        
        return (x_norm, y_norm), (x_mm, y_mm)
    
    def finger_callback(self, msg):
        """Process finger analysis and publish pointing target"""
        if not msg.is_straight:
            return  # Only publish if finger is straight
        
        if self.calibration is None or not self.calibration.is_calibrated:
            rospy.logwarn_once("Calibration not yet received")
            return
        
        # Extract data
        knuckle_3d = np.array([msg.knuckle_3d.x, msg.knuckle_3d.y, msg.knuckle_3d.z])
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
        intersection_3d = self.ray_plane_intersection(knuckle_3d, direction, plane_normal, plane_point)
        
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
                
                # Check if within paper bounds
                if 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0:
                    # Apply One Euro Filter
                    t = msg.header.stamp.to_sec()
                    filtered_u = self.filter_u.filter(x_norm, t)
                    filtered_v = self.filter_v.filter(y_norm, t)

                    target_msg.u_normalized = filtered_u
                    target_msg.v_normalized = filtered_v
                    target_msg.u_mm = filtered_u * self.calibration.paper_width_mm
                    target_msg.v_mm = filtered_v * self.calibration.paper_height_mm
                    target_msg.is_valid = True
                else:
                    target_msg.is_valid = False
                    self.filter_u.reset()
                    self.filter_v.reset()
                    rospy.logdebug(f"Pointing target outside paper: ({x_norm:.2f}, {y_norm:.2f})")
            else:
                target_msg.is_valid = False
        else:
            target_msg.is_valid = False
        
        if target_msg.is_valid:
            self.target_pub.publish(target_msg)
    
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
