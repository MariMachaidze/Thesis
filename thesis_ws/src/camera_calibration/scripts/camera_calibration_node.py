#!/usr/bin/env python3
"""
Camera Calibration Node
Detects white paper plane and stores calibration data
"""

import rospy
import numpy as np
import cv2
import os
import yaml
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Vector3
from cv_bridge import CvBridge
import sys

from camera_calibration.msg import CalibrationData

class CameraCalibrationNode:
    def __init__(self):
        rospy.init_node('camera_calibration_node', anonymous=False)
        
        # Subscribers
        self.rgb_sub = rospy.Subscriber('/d435i/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/d435i/depth/image_raw', Image, self.depth_callback)
        
        # Publisher
        self.calib_pub = rospy.Publisher('/camera/calibration', CalibrationData, queue_size=1, latch=True)
        
        # Parameters — paper dimensions in metres
        # paper_x_m: long edge, direction TL→TR (x_axis)  = 850 mm
        # paper_y_m: short edge, direction TL→BL (y_axis) = 600 mm
        self.paper_x_m = rospy.get_param('~paper_x_m', 0.850)
        self.paper_y_m = rospy.get_param('~paper_y_m', 0.600)

        # Camera intrinsics (D435i spec)
        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 642.351)
        self.cy = rospy.get_param('~cy', 349.990)
        self.depth_scale = 0.001
        
        self.bridge = CvBridge()
        self.rgb_frame = None
        self.depth_frame = None
        self.calibration_points = []
        self.calibration_depths_3d = []
        self.calibrated = False

        # Persistent calibration
        self.calibration_file = os.path.expanduser(
            rospy.get_param('~calibration_file', '~/.ros/paper_calibration.yaml')
        )
        self.recalibrate = rospy.get_param('~recalibrate', False)
        self.loaded_from_file = False

        # Try to load saved calibration; fall through to click UI if not available
        if not self.recalibrate and self._try_load_calibration():
            self.loaded_from_file = True
            rospy.loginfo(f"Loaded calibration from {self.calibration_file}")
            rospy.loginfo("Pass ~recalibrate:=true (or delete the file) to redo.")
        else:
            cv2.namedWindow('Camera Calibration')
            cv2.setMouseCallback('Camera Calibration', self.mouse_callback)
            rospy.loginfo("Camera Calibration Node: Ready. Click 4 corners of white paper.")
            rospy.loginfo("Corners: 1=Top-Left, 2=Top-Right, 3=Bottom-Left, 4=Bottom-Right")
    
    def _try_load_calibration(self):
        """Load a previously saved calibration and publish it. Returns True on success."""
        if not os.path.exists(self.calibration_file):
            return False
        try:
            with open(self.calibration_file, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                rospy.logwarn(f"Calibration file is empty: {self.calibration_file} — will re-click")
                return False

            # Warn if saved intrinsics diverge from current — calibration is baked
            # with the intrinsics used at capture time
            saved = data.get('intrinsics', {})
            def _diff(k, cur):
                return abs(saved.get(k, cur) - cur) > 0.01
            if any(_diff(k, v) for k, v in [('fx', self.fx), ('fy', self.fy),
                                            ('cx', self.cx), ('cy', self.cy)]):
                rospy.logwarn("Saved calibration intrinsics differ from current — consider ~recalibrate:=true")
                rospy.logwarn(f"  saved : fx={saved.get('fx')} fy={saved.get('fy')} cx={saved.get('cx')} cy={saved.get('cy')}")
                rospy.logwarn(f"  current: fx={self.fx} fy={self.fy} cx={self.cx} cy={self.cy}")

            msg = CalibrationData()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "d435i_depth_frame"
            msg.plane_normal.x, msg.plane_normal.y, msg.plane_normal.z = data['plane_normal']
            msg.plane_center.x, msg.plane_center.y, msg.plane_center.z = data['plane_center']
            msg.paper_x_axis.x, msg.paper_x_axis.y, msg.paper_x_axis.z = data['paper_x_axis']
            msg.paper_y_axis.x, msg.paper_y_axis.y, msg.paper_y_axis.z = data['paper_y_axis']
            msg.paper_x_m = float(data['paper_x_m'])
            msg.paper_y_m = float(data['paper_y_m'])
            msg.tilt_angle_rad = float(data['tilt_angle_rad'])
            msg.distance_to_paper_m = float(data['distance_to_paper_m'])
            msg.is_calibrated = True

            # Don't publish here — rospy hasn't registered the publisher with the
            # master yet so early publishes are silently dropped even on latched topics.
            # spin() will publish after a short delay once the node is running.
            self._loaded_calib_msg = msg
            self.calibrated = True
            return True
        except Exception as e:
            rospy.logwarn(f"Failed to load calibration from {self.calibration_file}: {e}")
            return False

    def _save_calibration(self, msg):
        """Persist calibration so next run skips the click UI."""
        def _f(v): return float(v)   # force Python float — yaml.safe_dump rejects numpy types
        data = {
            'plane_normal': [_f(msg.plane_normal.x), _f(msg.plane_normal.y), _f(msg.plane_normal.z)],
            'plane_center': [_f(msg.plane_center.x), _f(msg.plane_center.y), _f(msg.plane_center.z)],
            'paper_x_axis': [_f(msg.paper_x_axis.x), _f(msg.paper_x_axis.y), _f(msg.paper_x_axis.z)],
            'paper_y_axis': [_f(msg.paper_y_axis.x), _f(msg.paper_y_axis.y), _f(msg.paper_y_axis.z)],
            'paper_x_m': _f(msg.paper_x_m),
            'paper_y_m': _f(msg.paper_y_m),
            'tilt_angle_rad': _f(msg.tilt_angle_rad),
            'distance_to_paper_m': _f(msg.distance_to_paper_m),
            'intrinsics': {'fx': _f(self.fx), 'fy': _f(self.fy), 'cx': _f(self.cx), 'cy': _f(self.cy)},
        }
        try:
            os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
            with open(self.calibration_file, 'w') as f:
                yaml.safe_dump(data, f)
            rospy.loginfo(f"Saved calibration → {self.calibration_file}")
        except Exception as e:
            rospy.logwarn(f"Failed to save calibration: {e}")

    def get_depth_at(self, x, y, patch_size=5):
        """Get median depth around a pixel for robustness"""
        if self.depth_frame is None:
            return 0
        h, w = self.depth_frame.shape[:2]
        x = int(np.clip(x, 0, w-1))
        y = int(np.clip(y, 0, h-1))

        half = patch_size // 2
        y_min = max(0, y - half)
        y_max = min(h, y + half + 1)
        x_min = max(0, x - half)
        x_max = min(w, x + half + 1)

        patch = self.depth_frame[y_min:y_max, x_min:x_max]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return 0
        return int(np.median(valid))

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for calibration"""
        if event != cv2.EVENT_LBUTTONDOWN or self.calibrated or len(self.calibration_points) >= 4:
            return

        if self.depth_frame is None:
            rospy.logwarn("No depth frame yet - wait for camera to start")
            return

        depth_val = self.get_depth_at(x, y)
        if depth_val == 0 or depth_val > 10000:
            rospy.logwarn(f"Invalid depth at ({x}, {y}) - click again on the paper surface")
            return

        z = depth_val * self.depth_scale
        h, w = self.depth_frame.shape[:2]
        x_clip = int(np.clip(x, 0, w-1))
        y_clip = int(np.clip(y, 0, h-1))

        X = (x_clip - self.cx) * z / self.fx
        Y = (y_clip - self.cy) * z / self.fy

        self.calibration_points.append((x, y))
        self.calibration_depths_3d.append(np.array([X, Y, z]))

        corner_num = len(self.calibration_points)
        rospy.loginfo(f"Corner {corner_num}/4 marked at ({x}, {y}) - Depth: {z:.3f}m, 3D: ({X:.3f}, {Y:.3f}, {z:.3f})")

        if corner_num == 4:
            self.compute_calibration()
    
    def compute_calibration(self):
        """Compute camera plane from 4 calibration points"""
        if len(self.calibration_depths_3d) < 3:
            rospy.logwarn("Not enough valid depth points for calibration")
            return

        p1 = self.calibration_depths_3d[0]
        p2 = self.calibration_depths_3d[1]
        p3 = self.calibration_depths_3d[2]

        # Least-squares plane fit using all available clicked points via SVD.
        # Stack points, subtract centroid, decompose — the normal is the last
        # row of Vt (smallest singular value = direction of least variance).
        pts = np.array(self.calibration_depths_3d, dtype=float)
        centroid = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - centroid)
        normal = Vt[-1]

        # Ensure normal points toward the camera (positive Z component)
        if normal[2] < 0:
            normal = -normal

        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            rospy.logerr("Cannot compute plane — SVD produced zero normal")
            return
        normal = normal / norm

        self.paper_plane = (normal[0], normal[1], normal[2], -np.dot(normal, p1))
        
        # Calculate camera tilt
        cam_z = np.array([0.0, 0.0, 1.0])
        angle_rad = np.arccos(np.clip(np.dot(normal, cam_z), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Calculate average distance
        distances = [np.linalg.norm(p) for p in self.calibration_depths_3d if p is not None]
        avg_distance = np.mean(distances) if distances else 0.0
        
        # Create calibration message
        calib_msg = CalibrationData()
        calib_msg.header.stamp = rospy.Time.now()
        calib_msg.header.frame_id = "d435i_depth_frame"
        
        calib_msg.plane_normal.x = normal[0]
        calib_msg.plane_normal.y = normal[1]
        calib_msg.plane_normal.z = normal[2]
        
        # Origin must be p1 (top-left corner) so that x/y axes defined as
        # (p2-p1) and (p3-p1) give correct [0,1] normalized coords on the paper
        calib_msg.plane_center.x = p1[0]
        calib_msg.plane_center.y = p1[1]
        calib_msg.plane_center.z = p1[2]
        
        # Paper axes
        paper_x = (p2 - p1) / np.linalg.norm(p2 - p1)
        paper_y = (p3 - p1) / np.linalg.norm(p3 - p1)
        
        calib_msg.paper_x_axis.x = paper_x[0]
        calib_msg.paper_x_axis.y = paper_x[1]
        calib_msg.paper_x_axis.z = paper_x[2]
        
        calib_msg.paper_y_axis.x = paper_y[0]
        calib_msg.paper_y_axis.y = paper_y[1]
        calib_msg.paper_y_axis.z = paper_y[2]
        
        calib_msg.paper_x_m = self.paper_x_m   # long edge 0.841 m (x_axis direction)
        calib_msg.paper_y_m = self.paper_y_m   # short edge 0.594 m (y_axis direction)
        calib_msg.tilt_angle_rad = angle_rad
        calib_msg.distance_to_paper_m = avg_distance
        calib_msg.is_calibrated = True
        
        self.calib_pub.publish(calib_msg)

        self.calibrated = True
        rospy.loginfo("Calibration complete!")
        rospy.loginfo(f"  Plane normal: {normal}")
        rospy.loginfo(f"  Tilt angle: {angle_deg:.2f}°")
        rospy.loginfo(f"  Distance: {avg_distance:.3f}m")

        self._save_calibration(calib_msg)
    
    def rgb_callback(self, msg):
        """Store RGB frame"""
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting RGB: {e}")
    
    def depth_callback(self, msg):
        """Store depth frame"""
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        except Exception as e:
            rospy.logerr(f"Error converting depth: {e}")
    
    def spin(self):
        """Main loop"""
        # If calibration was loaded from file, publish it now that the node is
        # fully running and other nodes have had time to subscribe.
        if self.loaded_from_file:
            rospy.loginfo("Waiting 1 s for subscribers before publishing saved calibration...")
            rospy.sleep(1.0)
            self._loaded_calib_msg.header.stamp = rospy.Time.now()
            self.calib_pub.publish(self._loaded_calib_msg)
            rospy.loginfo("Saved calibration published.")
            rospy.loginfo("Pass ~recalibrate:=true (or delete the file) to redo.")
            rospy.spin()
            return

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.rgb_frame is not None:
                display = self.rgb_frame.copy()
                
                # Draw calibration points
                for i, (x, y) in enumerate(self.calibration_points):
                    cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
                    cv2.putText(display, str(i+1), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw lines between points
                if len(self.calibration_points) >= 2:
                    for i in range(len(self.calibration_points)-1):
                        cv2.line(display, self.calibration_points[i], self.calibration_points[i+1], (0, 255, 0), 2)
                
                # Close rectangle
                if len(self.calibration_points) == 4:
                    cv2.line(display, self.calibration_points[3], self.calibration_points[0], (0, 255, 0), 2)
                
                # Instructions
                if not self.calibrated:
                    instr = f"Click corner {len(self.calibration_points)+1}/4"
                    cv2.putText(display, instr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(display, "CALIBRATED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Camera Calibration', display)
            
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            
            rate.sleep()
        
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """Cleanup"""
        cv2.destroyAllWindows()
        rospy.loginfo("Camera Calibration Node: Cleanup complete")

if __name__ == '__main__':
    try:
        node = CameraCalibrationNode()
        node.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        node.cleanup()
