#!/usr/bin/env python3
"""
Visualization Node
Real-time display of hand tracking and pointing target
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from hand_detection.msg import Hand
from finger_analysis.msg import FingerAnalysis
from camera_calibration.msg import CalibrationData
from pointing_localization.msg import PointingTarget

class VisualizationNode:
    def __init__(self):
        rospy.init_node('visualization_node', anonymous=False)
        
        # Subscribers
        self.rgb_sub = rospy.Subscriber('/d435i/rgb/image_raw', Image, self.rgb_callback)
        self.hand_sub = rospy.Subscriber('/hand/detection', Hand, self.hand_callback)
        self.finger_sub = rospy.Subscriber('/finger/analysis', FingerAnalysis, self.finger_callback)
        self.calib_sub = rospy.Subscriber('/camera/calibration', CalibrationData, self.calib_callback)
        self.target_sub = rospy.Subscriber('/pointing/target', PointingTarget, self.target_callback)
        
        # Parameters
        self.show_landmarks = rospy.get_param('~show_landmarks', True)
        self.fx = rospy.get_param('~fx', 640.0)
        self.fy = rospy.get_param('~fy', 640.0)
        self.cx = rospy.get_param('~cx', 640.0)
        self.cy = rospy.get_param('~cy', 360.0)
        
        self.bridge = CvBridge()
        
        # Data storage
        self.rgb_frame = None
        self.hand_data = None
        self.finger_data = None
        self.calib_data = None
        self.target_data = None
        
        # Window setup
        cv2.namedWindow('Thesis System - Hand Pointing')
        cv2.namedWindow('Top-Down View')
        
        rospy.loginfo("Visualization Node: Ready")
    
    def rgb_callback(self, msg):
        """Store RGB frame"""
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting RGB: {e}")
    
    def hand_callback(self, msg):
        """Store hand data"""
        self.hand_data = msg
    
    def finger_callback(self, msg):
        """Store finger data"""
        self.finger_data = msg
    
    def calib_callback(self, msg):
        """Store calibration data"""
        self.calib_data = msg
    
    def target_callback(self, msg):
        """Store pointing target data"""
        self.target_data = msg
    
    def project_to_pixel(self, point_3d):
        """Project a 3D camera-space point to pixel coordinates"""
        X, Y, Z = point_3d
        if Z <= 0:
            return None
        u = int(self.fx * X / Z + self.cx)
        v = int(self.fy * Y / Z + self.cy)
        return u, v

    def ray_plane_intersection(self, ray_origin, ray_direction, plane_normal, plane_point):
        """Return the 3D point where the ray hits the plane, or None if parallel"""
        denom = np.dot(plane_normal, ray_direction)
        if abs(denom) < 1e-8:
            return None
        t = np.dot(plane_normal, plane_point - ray_origin) / denom
        if t < 0:
            return None
        return ray_origin + t * ray_direction

    def draw_hand(self, frame, hand_msg):
        """Draw hand landmarks on frame"""
        if not hand_msg.detected or len(hand_msg.keypoints_2d) == 0:
            return
        
        h, w = frame.shape[:2]
        
        # MediaPipe hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        
        # Draw connections
        for start, end in connections:
            if start < len(hand_msg.keypoints_2d) and end < len(hand_msg.keypoints_2d):
                pt1 = hand_msg.keypoints_2d[start]
                pt2 = hand_msg.keypoints_2d[end]
                
                x1 = int(pt1.x * w)
                y1 = int(pt1.y * h)
                x2 = int(pt2.x * w)
                y2 = int(pt2.y * h)
                
                cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
        
        # Draw landmarks
        colors = [(0, 255, 0)] * 21  # Green for all
        colors[8] = (0, 255, 255)  # Cyan for fingertip
        colors[5] = (255, 0, 0)  # Blue for knuckle
        
        for i, pt in enumerate(hand_msg.keypoints_2d):
            x = int(pt.x * w)
            y = int(pt.y * h)
            cv2.circle(frame, (x, y), 5, colors[i], -1)
    
    def draw_finger_info(self, frame, finger_msg, calib_msg=None):
        """Draw finger analysis info and 3D ray to paper plane"""
        h, w = frame.shape[:2]

        # Draw straightness info
        straightness_text = f"Straightness: {finger_msg.straightness_score:.2f}"
        color = (0, 255, 0) if finger_msg.is_straight else (0, 165, 255)
        cv2.putText(frame, straightness_text, (10, h-100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Knuckle and tip in pixels
        x1 = int(finger_msg.knuckle_2d.x * w)
        y1 = int(finger_msg.knuckle_2d.y * h)
        x2 = int(finger_msg.tip_2d.x * w)
        y2 = int(finger_msg.tip_2d.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 8, (0, 255, 255), -1)

        # Draw 3D ray to paper plane if calibrated
        if calib_msg and calib_msg.is_calibrated:
            knuckle_3d = np.array([finger_msg.knuckle_3d.x,
                                   finger_msg.knuckle_3d.y,
                                   finger_msg.knuckle_3d.z])
            direction = np.array([finger_msg.direction_3d.x,
                                  finger_msg.direction_3d.y,
                                  finger_msg.direction_3d.z])

            if np.linalg.norm(knuckle_3d) > 1e-6 and np.linalg.norm(direction) > 1e-6:
                plane_normal = np.array([calib_msg.plane_normal.x,
                                         calib_msg.plane_normal.y,
                                         calib_msg.plane_normal.z])
                plane_point = np.array([calib_msg.plane_center.x,
                                        calib_msg.plane_center.y,
                                        calib_msg.plane_center.z])

                intersection_3d = self.ray_plane_intersection(
                    knuckle_3d, direction, plane_normal, plane_point)

                if intersection_3d is not None:
                    px = self.project_to_pixel(intersection_3d)
                    if px is not None:
                        px_x, px_y = px
                        # Ray from fingertip to landing point
                        cv2.line(frame, (x2, y2), (px_x, px_y), (0, 200, 255), 2)
                        # Landing dot on the paper surface
                        cv2.circle(frame, (px_x, px_y), 12, (0, 0, 255), -1)
                        cv2.circle(frame, (px_x, px_y), 14, (255, 255, 255), 2)
    
    def draw_topdown(self, calib_msg, target_msg):
        """Draw top-down view of paper"""
        A1_WIDTH_CM = 59.4
        A1_HEIGHT_CM = 84.1
        DISPLAY_SCALE = 0.7
        
        width = int(A1_WIDTH_CM * DISPLAY_SCALE * 10)
        height = int(A1_HEIGHT_CM * DISPLAY_SCALE * 10)
        
        topdown = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Draw grid
        grid = int(DISPLAY_SCALE * 10)
        for i in range(0, width, grid):
            cv2.line(topdown, (i, 0), (i, height), (220, 220, 220), 1)
        for i in range(0, height, grid):
            cv2.line(topdown, (0, i), (width, i), (220, 220, 220), 1)
        
        # Draw border
        cv2.rectangle(topdown, (0, 0), (width-1, height-1), (0, 200, 255), 3)
        
        # Draw pointing target
        if target_msg and target_msg.is_valid:
            px = int(target_msg.u_normalized * width)
            py = int(target_msg.v_normalized * height)
            
            cv2.circle(topdown, (px, py), 20, (0, 0, 255), -1)
            cv2.circle(topdown, (px, py), 22, (255, 255, 255), 2)
            
            text = f"({target_msg.u_mm:.1f}, {target_msg.v_mm:.1f}) mm"
            cv2.putText(topdown, text, (max(10, px-80), min(height-10, py+40)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw title
        cv2.putText(topdown, "Top-Down View (A1 Paper)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return topdown
    
    def spin(self):
        """Main loop"""
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            if self.rgb_frame is not None:
                display = self.rgb_frame.copy()
                h, w = display.shape[:2]
                
                # Draw hand
                if self.hand_data and self.hand_data.detected:
                    self.draw_hand(display, self.hand_data)
                
                # Draw finger analysis
                if self.finger_data:
                    self.draw_finger_info(display, self.finger_data, self.calib_data)
                
                # Draw status
                if self.hand_data and self.hand_data.detected:
                    cv2.putText(display, "HAND DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "No hand detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw calibration status
                if self.calib_data and self.calib_data.is_calibrated:
                    cv2.putText(display, "Calibrated", (w-200, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw tilt info
                    tilt_deg = np.degrees(self.calib_data.tilt_angle_rad)
                    cv2.putText(display, f"Tilt: {tilt_deg:.1f}°", (10, h-50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                else:
                    cv2.putText(display, "Not calibrated", (w-200, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                cv2.imshow('Thesis System - Hand Pointing', display)
                
                # Draw top-down view
                if self.calib_data and self.calib_data.is_calibrated:
                    topdown = self.draw_topdown(self.calib_data, self.target_data)
                    cv2.imshow('Top-Down View', topdown)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            rate.sleep()
        
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """Cleanup"""
        cv2.destroyAllWindows()
        rospy.loginfo("Visualization Node: Cleanup complete")

if __name__ == '__main__':
    try:
        node = VisualizationNode()
        node.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        node.cleanup()
