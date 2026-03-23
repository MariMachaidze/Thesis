#!/usr/bin/env python3
"""
Debug script to see the ray calculation step by step
"""

import rospy
from pointing_localization.msg import PointingTarget
from finger_analysis.msg import FingerAnalysis
from camera_calibration.msg import CalibrationData
import numpy as np

class RayDebugger:
    def __init__(self):
        rospy.init_node('ray_debugger', anonymous=False)
        
        self.finger_sub = rospy.Subscriber('/finger/analysis', FingerAnalysis, self.finger_callback)
        self.calib_sub = rospy.Subscriber('/camera/calibration', CalibrationData, self.calib_callback)
        self.target_sub = rospy.Subscriber('/pointing/target', PointingTarget, self.target_callback)
        
        self.calibration = None
        self.finger = None
    
    def calib_callback(self, msg):
        self.calibration = msg
        print("\n" + "="*60)
        print("📐 CALIBRATION DATA RECEIVED")
        print("="*60)
        print(f"Plane Normal: ({msg.plane_normal.x:.4f}, {msg.plane_normal.y:.4f}, {msg.plane_normal.z:.4f})")
        print(f"Plane Center: ({msg.plane_center.x:.4f}, {msg.plane_center.y:.4f}, {msg.plane_center.z:.4f})")
        print(f"Tilt Angle: {np.degrees(msg.tilt_angle_rad):.2f}°")
        print(f"Distance to Paper: {msg.distance_to_paper_m:.3f}m")
        print(f"Calibrated: {msg.is_calibrated}")
    
    def finger_callback(self, msg):
        self.finger = msg
        print("\n" + "="*60)
        print("👆 FINGER ANALYSIS DATA")
        print("="*60)
        print(f"Knuckle 2D: ({msg.knuckle_2d.x:.4f}, {msg.knuckle_2d.y:.4f})")
        print(f"Tip 2D: ({msg.tip_2d.x:.4f}, {msg.tip_2d.y:.4f})")
        print(f"Knuckle 3D: ({msg.knuckle_3d.x:.4f}, {msg.knuckle_3d.y:.4f}, {msg.knuckle_3d.z:.4f})")
        print(f"Tip 3D: ({msg.tip_3d.x:.4f}, {msg.tip_3d.y:.4f}, {msg.tip_3d.z:.4f})")
        print(f"Direction: ({msg.direction_3d.x:.4f}, {msg.direction_3d.y:.4f}, {msg.direction_3d.z:.4f})")
        print(f"Straightness: {msg.straightness_score:.2f}")
        print(f"Is Straight: {msg.is_straight}")
    
    def target_callback(self, msg):
        print("\n" + "="*60)
        print("🎯 POINTING TARGET")
        print("="*60)
        print(f"Valid: {msg.is_valid}")
        print(f"Normalized Coords: ({msg.u_normalized:.4f}, {msg.v_normalized:.4f})")
        print(f"MM Coords: ({msg.u_mm:.2f}, {msg.v_mm:.2f})")
        print(f"Straightness: {msg.straightness:.2f}")
        print(f"Confidence: {msg.confidence:.4f}")
    
    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    node = RayDebugger()
    print("👀 Ray Debugger Started - watching all topics...\n")
    node.spin()
