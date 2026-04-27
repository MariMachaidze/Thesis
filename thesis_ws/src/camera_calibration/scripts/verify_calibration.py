#!/usr/bin/env python3
"""
Calibration verification script.

Subscribes to:
  /camera/calibration  (camera_calibration/CalibrationData)
  /d435i/rgb/image_raw (sensor_msgs/Image)

On the first valid calibration message it prints:
  - 3-D camera-frame coords of all 4 paper corners
  - Plane normal vector and its L2 norm (should be 1.0)
  - dot(normal, x_axis) and dot(normal, y_axis) (should be 0.0)
  - cross(x_axis, y_axis) compared to the stored normal

It also opens a window titled "Calibration verify" showing the live
RGB stream with the four projected corners overlaid as coloured circles.

Usage (with ROS master running and workspace sourced):
    python3 verify_calibration.py

Press 'q' in the image window to quit.
"""

import sys
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from camera_calibration.msg import CalibrationData


# ── colour / label constants ──────────────────────────────────────────────────
_CORNER_COLORS = [
    (0,   255,   0),   # p1 Top-Left     green
    (0,   200, 255),   # p2 Top-Right    cyan
    (255, 100,   0),   # p3 Bottom-Left  orange
    (0,     0, 255),   # p4 Bottom-Right red
]
_CORNER_LABELS = ['1:TL', '2:TR', '3:BL', '4:BR']


class CalibrationVerifier:
    def __init__(self):
        rospy.init_node('verify_calibration', anonymous=True)

        self.bridge = CvBridge()
        self.calib: CalibrationData | None = None
        self.rgb_frame = None
        self.diagnostics_printed = False

        # Camera intrinsics — must match camera_calibration_node exactly.
        # Defaults mirror thesis_system.launch (D435i spec).
        self.fx = float(rospy.get_param('~fx', 901.473))
        self.fy = float(rospy.get_param('~fy', 899.637))
        self.cx = float(rospy.get_param('~cx', 642.351))
        self.cy = float(rospy.get_param('~cy', 349.990))

        rospy.Subscriber('/camera/calibration', CalibrationData,
                         self._calib_callback, queue_size=1)
        rospy.Subscriber('/d435i/rgb/image_raw', Image,
                         self._rgb_callback, queue_size=1)

        rospy.loginfo('verify_calibration: waiting for /camera/calibration ...')

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _v(v) -> np.ndarray:
        """geometry_msgs Vector3 / Point → numpy array."""
        return np.array([v.x, v.y, v.z], dtype=float)

    def _corners_3d(self, msg: CalibrationData):
        """
        Reconstruct 3-D camera-frame coordinates of all four paper corners.

        The CalibrationData message stores only p1 (plane_center) plus
        unit-direction axes.  p2-p4 are recovered using the paper dimensions
        stored directly in metres in the message:
            x_axis goes from p1(TL) → p2(TR) along the long  edge: paper_x_m
            y_axis goes from p1(TL) → p3(BL) along the short edge: paper_y_m
        """
        p1     = self._v(msg.plane_center)
        x_axis = self._v(msg.paper_x_axis)
        y_axis = self._v(msg.paper_y_axis)
        long_m  = msg.paper_x_m   # long edge  (TL → TR, x_axis direction)
        short_m = msg.paper_y_m   # short edge (TL → BL, y_axis direction)
        return [p1,
                p1 + x_axis * long_m,
                p1 + y_axis * short_m,
                p1 + x_axis * long_m + y_axis * short_m]

    def _project(self, pt3d) -> tuple[int, int] | None:
        """Back-project a 3-D camera-frame point to pixel coordinates."""
        X, Y, Z = pt3d
        if Z <= 0:
            return None
        return (int(X * self.fx / Z + self.cx),
                int(Y * self.fy / Z + self.cy))

    # ── diagnostics ───────────────────────────────────────────────────────────

    def _print_diagnostics(self, msg: CalibrationData):
        normal = self._v(msg.plane_normal)
        x_axis = self._v(msg.paper_x_axis)
        y_axis = self._v(msg.paper_y_axis)
        corners = self._corners_3d(msg)

        SEP = '─' * 62
        print()
        print(SEP)
        print('  CALIBRATION DIAGNOSTICS')
        print(SEP)

        # ── axis unit-vector check ────────────────────────────────────────────
        x_axis_norm = np.linalg.norm(x_axis)
        y_axis_norm = np.linalg.norm(y_axis)
        print(f'\n  Axis unit-vector check (both must be 1.0 for scaling to be exact)')
        print(f'    |x_axis| = {x_axis_norm:.8f}   ← must be 1.0')
        print(f'    |y_axis| = {y_axis_norm:.8f}   ← must be 1.0')

        # ── 3-D corner coordinates ────────────────────────────────────────────
        print('\n  3-D corner coordinates in camera frame (metres)')
        print(f'  (p1 measured; p2–p4 reconstructed: x_axis×{msg.paper_x_m:.3f} m, y_axis×{msg.paper_y_m:.3f} m)')
        names = ['p1  Top-Left ', 'p2  Top-Right', 'p3  Bot-Left ', 'p4  Bot-Right']
        for name, pt in zip(names, corners):
            print(f'    {name} :  X={pt[0]:+.4f}  Y={pt[1]:+.4f}  Z={pt[2]:+.4f}')

        # ── reconstruction distance check ─────────────────────────────────────
        p1_pt, p2_pt, p3_pt, p4_pt = corners
        d_p1_p2 = float(np.linalg.norm(p2_pt - p1_pt))
        d_p1_p3 = float(np.linalg.norm(p3_pt - p1_pt))
        d_p1_p4 = float(np.linalg.norm(p4_pt - p1_pt))
        ok_p2 = 'OK' if abs(d_p1_p2 - msg.paper_x_m) < 1e-4 else 'WRONG'
        ok_p3 = 'OK' if abs(d_p1_p3 - msg.paper_y_m) < 1e-4 else 'WRONG'
        diag  = (msg.paper_x_m**2 + msg.paper_y_m**2) ** 0.5
        print(f'\n  Reconstruction distance check')
        print(f'    |p1 → p2| = {d_p1_p2:.6f} m   (expected {msg.paper_x_m:.6f} m)  [{ok_p2}]')
        print(f'    |p1 → p3| = {d_p1_p3:.6f} m   (expected {msg.paper_y_m:.6f} m)  [{ok_p3}]')
        print(f'    |p1 → p4| = {d_p1_p4:.6f} m   (expected {diag:.6f} m, diagonal)')

        # ── plane normal ──────────────────────────────────────────────────────
        norm_len = np.linalg.norm(normal)
        print(f'\n  Plane normal vector : [{normal[0]:+.6f}, {normal[1]:+.6f}, {normal[2]:+.6f}]')
        print(f'  L2 norm             : {norm_len:.6f}   ← should be ≈ 1.0')

        # ── orthogonality checks ──────────────────────────────────────────────
        dot_nx  = float(np.dot(normal, x_axis))
        dot_ny  = float(np.dot(normal, y_axis))
        dot_xy  = float(np.dot(x_axis, y_axis))
        print(f'\n  Axis orthogonality checks')
        print(f'    dot(normal, x_axis) = {dot_nx:+.6f}   ← should be ≈  0.0')
        print(f'    dot(normal, y_axis) = {dot_ny:+.6f}   ← should be ≈  0.0')
        print(f'    dot(x_axis, y_axis) = {dot_xy:+.6f}   ← should be ≈  0.0 (right-angle paper)')

        # ── cross-product check ───────────────────────────────────────────────
        cross      = np.cross(x_axis, y_axis)
        cross_norm = np.linalg.norm(cross)
        cross_unit = cross / cross_norm if cross_norm > 1e-9 else cross
        alignment  = float(np.dot(cross_unit, normal))
        print(f'\n  cross(x_axis, y_axis)')
        print(f'    raw              : [{cross[0]:+.6f}, {cross[1]:+.6f}, {cross[2]:+.6f}]')
        print(f'    unit             : [{cross_unit[0]:+.6f}, {cross_unit[1]:+.6f}, {cross_unit[2]:+.6f}]')
        print(f'    plane normal     : [{normal[0]:+.6f}, {normal[1]:+.6f}, {normal[2]:+.6f}]')
        print(f'    dot(cross_unit, normal) = {alignment:+.6f}   ← should be ≈ ±1.0')

        # ── other info ────────────────────────────────────────────────────────
        print(f'\n  Camera tilt to paper : {np.degrees(msg.tilt_angle_rad):.2f}°')
        print(f'  Distance to paper    : {msg.distance_to_paper_m:.3f} m')
        print(f'  Paper size           : {msg.paper_x_m*1000:.0f} × {msg.paper_y_m*1000:.0f} mm  ({msg.paper_x_m:.3f} × {msg.paper_y_m:.3f} m)')
        print(SEP)
        print()

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _calib_callback(self, msg: CalibrationData):
        if not msg.is_calibrated:
            return
        self.calib = msg
        if not self.diagnostics_printed:
            self._print_diagnostics(msg)
            self.diagnostics_printed = True

    def _rgb_callback(self, msg: Image):
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            rospy.logerr(f'verify_calibration rgb_callback: {exc}')

    # ── main loop ─────────────────────────────────────────────────────────────

    def spin(self):
        rate = rospy.Rate(30)
        cv2.namedWindow('Calibration verify')

        while not rospy.is_shutdown():
            frame = self.rgb_frame
            if frame is None:
                rate.sleep()
                continue

            display = frame.copy()

            if self.calib is not None and self.calib.is_calibrated:
                corners_3d = self._corners_3d(self.calib)
                pixels = [self._project(c) for c in corners_3d]

                # Draw paper outline: TL → TR → BR → BL → close
                outline_order = [0, 1, 3, 2, 0]
                for i in range(len(outline_order) - 1):
                    a = pixels[outline_order[i]]
                    b = pixels[outline_order[i + 1]]
                    if a is not None and b is not None:
                        cv2.line(display, a, b, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw corner circles and labels
                for px, label, color in zip(pixels, _CORNER_LABELS, _CORNER_COLORS):
                    if px is None:
                        continue
                    cv2.circle(display, px, 10, color, -1)
                    cv2.circle(display, px, 10, (255, 255, 255), 2)  # white border
                    cv2.putText(display, label, (px[0] + 14, px[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

                cv2.putText(display, 'CALIBRATED', (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, 'Waiting for calibration...', (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 255), 2, cv2.LINE_AA)

            cv2.imshow('Calibration verify', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        CalibrationVerifier().spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
