#!/usr/bin/env python3
"""
Visualization Node
Real-time display of hand tracking and pointing target.

  "Hand Detection" — full 1280×720 camera feed with hand skeleton + pointing ray
  "Paper View"     — perspective-corrected top-down paper view with hand + crosshair
  "Top-Down View"  — perspective-corrected top-down paper view with pointing target
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge

from hand_detection.msg import Hand
from finger_analysis.msg import FingerAnalysis
from camera_calibration.msg import CalibrationData
from pointing_localization.msg import PointingTarget


class VisualizationNode:
    # Output size of the warped paper view (pixels)
    # paper_x_m = 0.850 → WARP_W,  paper_y_m = 0.600 → WARP_H
    WARP_W = 850
    WARP_H = 600

    def __init__(self):
        rospy.init_node('visualization_node', anonymous=False)

        # Subscribers
        self.rgb_sub           = rospy.Subscriber('/d435i/rgb/image_raw',      Image,          self.rgb_callback,   queue_size=1, buff_size=2**24)
        self.hand_sub          = rospy.Subscriber('/hand/detection',            Hand,            self.hand_callback)
        self.finger_sub        = rospy.Subscriber('/finger/analysis',           FingerAnalysis,  self.finger_callback)
        self.calib_sub         = rospy.Subscriber('/camera/calibration',        CalibrationData, self.calib_callback)
        self.target_sub        = rospy.Subscriber('/pointing/target',           PointingTarget,  self.target_callback)
        self.raw_straight_sub  = rospy.Subscriber('/diag/straightness_raw',     Float32,         self._raw_straight_callback)
        self.diag_pointing_sub = rospy.Subscriber('/diag/pointing_raw',         Point,           self._diag_pointing_callback)

        # Parameters
        self.show_landmarks    = rospy.get_param('~show_landmarks',    True)
        self.diagnose_hand     = rospy.get_param('~diagnose_hand',     False)
        self.diagnose_pointing = rospy.get_param('~diagnose_pointing', False)
        self.fx = rospy.get_param('~fx', 640.0)
        self.fy = rospy.get_param('~fy', 640.0)
        self.cx = rospy.get_param('~cx', 640.0)
        self.cy = rospy.get_param('~cy', 360.0)

        self.bridge = CvBridge()

        # Data storage
        self.rgb_frame         = None
        self.hand_data         = None
        self.finger_data       = None
        self.calib_data        = None
        self.target_data       = None
        self.raw_straightness  = None
        self.diag_pointing_raw = None

        # Warp matrix cache
        self._warp_M           = None
        self._warp_M_inv       = None
        self._warp_calib_stamp = None

        # Diagnostic publisher
        self.diag_raw_pub = rospy.Publisher('/diag/straightness_raw', Float32, queue_size=1)

        cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Top-Down View',  cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Top-Down View', self.WARP_W, self.WARP_H)
        rospy.loginfo("Visualization Node: Ready")

    # ================================================================
    #  ROS callbacks
    # ================================================================
    def rgb_callback(self, msg):
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting RGB: {e}")

    def hand_callback(self, msg):   self.hand_data = msg
    def finger_callback(self, msg): self.finger_data = msg
    def calib_callback(self, msg):  self.calib_data = msg
    def target_callback(self, msg): self.target_data = msg

    def _raw_straight_callback(self, msg):
        self.raw_straightness = msg.data

    def _diag_pointing_callback(self, msg):
        self.diag_pointing_raw = msg

    # ================================================================
    #  Camera intrinsics helpers
    # ================================================================
    def project_to_pixel(self, point_3d):
        X, Y, Z = point_3d
        if Z <= 0:
            return None
        u = int(self.fx * X / Z + self.cx)
        v = int(self.fy * Y / Z + self.cy)
        return u, v

    # ================================================================
    #  Perspective warp
    # ================================================================
    def _get_warp_matrix(self, calib_msg):
        stamp = calib_msg.header.stamp.to_sec() if calib_msg.header.stamp.secs != 0 else None
        if self._warp_M is not None and stamp == self._warp_calib_stamp:
            return self._warp_M

        origin = np.array([calib_msg.plane_center.x,
                           calib_msg.plane_center.y,
                           calib_msg.plane_center.z])
        x_axis = np.array([calib_msg.paper_x_axis.x,
                           calib_msg.paper_x_axis.y,
                           calib_msg.paper_x_axis.z])
        y_axis = np.array([calib_msg.paper_y_axis.x,
                           calib_msg.paper_y_axis.y,
                           calib_msg.paper_y_axis.z])

        corners_3d = [
            origin,
            origin + calib_msg.paper_x_m * x_axis,
            origin + calib_msg.paper_x_m * x_axis + calib_msg.paper_y_m * y_axis,
            origin + calib_msg.paper_y_m * y_axis,
        ]

        src_pts = []
        for pt in corners_3d:
            px = self.project_to_pixel(pt)
            if px is None:
                return None
            src_pts.append(list(px))

        W, H = self.WARP_W, self.WARP_H
        src = np.float32(src_pts)
        dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

        self._warp_M           = cv2.getPerspectiveTransform(src, dst)
        self._warp_M_inv       = np.linalg.inv(self._warp_M)
        self._warp_calib_stamp = stamp
        return self._warp_M

    def _warp_frame(self, frame, calib_msg):
        if calib_msg is None or not calib_msg.is_calibrated:
            return None
        M = self._get_warp_matrix(calib_msg)
        if M is None:
            return None
        return cv2.warpPerspective(frame, M, (self.WARP_W, self.WARP_H))

    def _transform_pts(self, norm_pts, frame_shape, M):
        h, w = frame_shape[:2]
        raw = np.float32([[p[0] * w, p[1] * h] for p in norm_pts]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(raw, M)
        return [(int(p[0][0]), int(p[0][1])) for p in warped]

    # ================================================================
    #  Draw: Hand Detection (full camera frame)
    # ================================================================
    def draw_hand_detection_view(self, frame):
        out = frame  # caller already copied
        h, w = out.shape[:2]

        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
        ]

        if self.hand_data and self.hand_data.detected and len(self.hand_data.keypoints_2d) >= 21:
            kp  = self.hand_data.keypoints_2d
            pts = [(int(p.x * w), int(p.y * h)) for p in kp]

            for s, e in connections:
                cv2.line(out, pts[s], pts[e], (180, 180, 180), 2)

            for i, pt in enumerate(pts):
                color = (0, 255, 255) if i == 8 else (255, 80, 0) if i == 5 else (0, 200, 0)
                cv2.circle(out, pt, 7, color, -1)

            # Ray from fingertip to pointing target (projected back to camera pixels)
            if (self.finger_data and self.finger_data.is_straight
                    and self.target_data is not None and self.target_data.is_valid
                    and self._warp_M_inv is not None):
                src   = np.array([[[self.target_data.u_normalized * self.WARP_W,
                                    self.target_data.v_normalized * self.WARP_H]]], dtype=np.float32)
                dst   = cv2.perspectiveTransform(src, self._warp_M_inv)
                tx, ty = int(dst[0, 0, 0]), int(dst[0, 0, 1])
                tip = pts[8]
                cv2.line(out,   tip, (tx, ty), (0, 200, 255), 2)
                cv2.circle(out, (tx, ty), 12, (0, 0, 200), -1)
                cv2.circle(out, (tx, ty), 14, (255, 255, 255), 2)

        # Status label
        if self.hand_data and self.hand_data.detected:
            cv2.putText(out, "HAND DETECTED", (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(out, "No hand", (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 60, 220), 2, cv2.LINE_AA)

        if self.finger_data:
            s_text  = f"Straight: {self.finger_data.straightness_score:.2f}"
            s_color = (0, 255, 0) if self.finger_data.is_straight else (0, 140, 255)
            cv2.putText(out, s_text, (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_color, 2, cv2.LINE_AA)

        return out

    # ================================================================
    #  Draw: Paper View
    # ================================================================
    def draw_paper_view(self, warped, frame_shape, calib_msg):
        M = self._get_warp_matrix(calib_msg)
        W, H = self.WARP_W, self.WARP_H

        if self.hand_data and self.hand_data.detected and len(self.hand_data.keypoints_2d) >= 21:
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
            ]
            kp  = self.hand_data.keypoints_2d
            pts = self._transform_pts([(p.x, p.y) for p in kp], frame_shape, M)

            for s, e in connections:
                cv2.line(warped, pts[s], pts[e], (180, 180, 180), 1)

            for i, pt in enumerate(pts):
                color = (0, 255, 255) if i == 8 else (255, 80, 0) if i == 5 else (0, 200, 0)
                cv2.circle(warped, pt, 5, color, -1)

        if (self.finger_data and self.finger_data.is_straight
                and self.diag_pointing_raw is not None):
            tx = int(self.diag_pointing_raw.x * W)
            ty = int(self.diag_pointing_raw.y * H)
            if self.hand_data and len(self.hand_data.keypoints_2d) >= 9:
                tip   = self.hand_data.keypoints_2d[8]
                tip_w = self._transform_pts([(tip.x, tip.y)], frame_shape, M)[0]
                cv2.line(warped, tip_w, (tx, ty), (0, 200, 255), 2)
            cv2.circle(warped, (tx, ty), 10, (0, 0, 200), -1)
            cv2.circle(warped, (tx, ty), 12, (255, 255, 255), 2)

        if self.target_data and self.target_data.is_valid:
            tx  = int(self.target_data.u_normalized * W)
            ty  = int(self.target_data.v_normalized * H)
            arm = 18
            cv2.line(warped,   (tx - arm, ty), (tx + arm, ty), (0, 255, 0), 2)
            cv2.line(warped,   (tx, ty - arm), (tx, ty + arm), (0, 255, 0), 2)
            cv2.circle(warped, (tx, ty), 8, (0, 255, 0), 2)
            cv2.putText(warped, f"({self.target_data.u_normalized:.2f},{self.target_data.v_normalized:.2f})",
                        (tx + 14, ty - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        label = "HAND DETECTED" if (self.hand_data and self.hand_data.detected) else "No hand"
        color = (0, 255, 0)    if (self.hand_data and self.hand_data.detected) else (0, 60, 220)
        cv2.putText(warped, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        if self.finger_data:
            s_text  = f"Straight: {self.finger_data.straightness_score:.2f}"
            s_color = (0, 255, 0) if self.finger_data.is_straight else (0, 140, 255)
            cv2.putText(warped, s_text, (8, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 1, cv2.LINE_AA)

        cv2.putText(warped, "Paper View", (W - 110, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # ================================================================
    #  Draw: Top-Down View
    # ================================================================
    def draw_topdown_view(self, warped):
        W, H = self.WARP_W, self.WARP_H

        if self.target_data and self.target_data.is_valid:
            tx  = int(self.target_data.u_normalized * W)
            ty  = int(self.target_data.v_normalized * H)
            arm = 20
            cv2.line(warped,   (tx - arm, ty), (tx + arm, ty), (0, 255, 0), 2)
            cv2.line(warped,   (tx, ty - arm), (tx, ty + arm), (0, 255, 0), 2)
            cv2.circle(warped, (tx, ty), 10, (0, 255, 0), 2)
            cv2.putText(warped,
                        f"Target ({self.target_data.u_normalized:.2f},{self.target_data.v_normalized:.2f})",
                        (tx + 14, ty - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(warped, "Top-Down View", (W - 130, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ================================================================
    #  Main loop
    # ================================================================
    def spin(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.rgb_frame is None:
                rate.sleep()
                continue

            frame = self.rgb_frame.copy()

            # Hand Detection window: draw on a fresh copy, show immediately
            cv2.imshow('Hand Detection', self.draw_hand_detection_view(frame.copy()))

            # Top-Down View: only once calibration is ready
            if self.calib_data is not None and self.calib_data.is_calibrated:
                warped = self._warp_frame(frame, self.calib_data)
                if warped is not None:
                    topdown_view = warped.copy()
                    self.draw_topdown_view(topdown_view)
                    cv2.imshow('Top-Down View', topdown_view)

            if self.diagnose_hand and self.raw_straightness is not None:
                self.diag_raw_pub.publish(Float32(data=self.raw_straightness))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rate.sleep()

        cv2.destroyAllWindows()

    def cleanup(self):
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
