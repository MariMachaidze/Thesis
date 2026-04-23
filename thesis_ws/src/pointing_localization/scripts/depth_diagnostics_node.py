#!/usr/bin/env python3
"""
Depth Diagnostics Node
Validates depth patch extraction, 3D back-projection, ray-plane intersection,
depth gradients, and angular error for the finger pointing localization system.

Topics consumed:
  /finger/analysis          — FingerAnalysis
  /camera/calibration       — CalibrationData
  /pointing/target          — PointingTarget
  /camera/depth/image_rect_raw  — sensor_msgs/Image (16UC1, depth in mm)
  /camera/color/camera_info     — sensor_msgs/CameraInfo

Topics published:
  /diag/joints_3d           — visualization_msgs/MarkerArray

ROS params:
  ~verbose_patch  (bool, default False)  — enable ASCII depth heatmaps
"""

import rospy
import numpy as np
import time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from finger_analysis.msg import FingerAnalysis
from camera_calibration.msg import CalibrationData
from pointing_localization.msg import PointingTarget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vec3_to_np(v):
    """geometry_msgs/Vector3 or Point → np.array([x,y,z])"""
    return np.array([v.x, v.y, v.z], dtype=float)


def make_sphere_marker(ns, marker_id, position, color_rgba, frame_id, stamp, radius=0.012):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = position[0]
    m.pose.position.y = position[1]
    m.pose.position.z = position[2]
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = radius
    m.color.r, m.color.g, m.color.b, m.color.a = color_rgba
    return m


def make_arrow_marker(ns, marker_id, origin, direction, length, color_rgba, frame_id, stamp,
                      shaft_diam=0.006, head_diam=0.015):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.ARROW
    m.action = Marker.ADD
    start = Point(x=origin[0],    y=origin[1],    z=origin[2])
    end   = Point(x=origin[0] + direction[0] * length,
                  y=origin[1] + direction[1] * length,
                  z=origin[2] + direction[2] * length)
    m.points = [start, end]
    m.scale.x = shaft_diam
    m.scale.y = head_diam
    m.scale.z = 0.0
    m.color.r, m.color.g, m.color.b, m.color.a = color_rgba
    return m


def ray_plane_intersection(ray_origin, ray_dir, plane_normal, plane_point):
    """Returns (point_3d, t) or (None, None)."""
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-8:
        return None, None
    t = np.dot(plane_normal, plane_point - ray_origin) / denom
    if t < 0:
        return None, None
    return ray_origin + t * ray_dir, t


def ascii_heatmap(patch, rows=5, cols=5):
    """
    Render a 2-D numpy array as a small ASCII heatmap.
    Values are normalized 0-1 and mapped to 10 grey levels.
    Zero / invalid depth cells are shown as '?'.
    """
    CHARS = " .:-=+*#%@"
    lines = []
    mn, mx = patch[patch > 0].min() if (patch > 0).any() else 0, patch.max()
    span = mx - mn if mx > mn else 1.0
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            v = patch[r, c]
            if v == 0:
                row_str += "? "
            else:
                idx = int((v - mn) / span * (len(CHARS) - 1))
                row_str += CHARS[idx] + " "
        lines.append("  " + row_str)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class DepthDiagnosticsNode:

    # Joint labels for logging
    JOINT_LABELS = {5: "MCP", 6: "PIP", 7: "DIP", 8: "TIP"}
    # Joint sphere colours (r,g,b,a)  MCP=blue, PIP=green, DIP=yellow, TIP=red
    JOINT_COLORS = {
        5: (0.2, 0.4, 1.0, 0.9),
        6: (0.2, 0.9, 0.2, 0.9),
        7: (1.0, 0.9, 0.1, 0.9),
        8: (1.0, 0.2, 0.2, 0.9),
    }

    def __init__(self):
        rospy.init_node("depth_diagnostics_node", anonymous=False)

        self.verbose_patch = rospy.get_param("~verbose_patch", False)

        # ---------- state ----------
        self.depth_image  = None        # np.ndarray uint16, depth in mm
        self.camera_info  = None        # CameraInfo
        self.calibration  = None        # CalibrationData
        self.last_target  = None        # PointingTarget
        self._last_grad_t = 0.0         # throttle for gradient check

        self.bridge = CvBridge()

        # ---------- subscribers ----------
        rospy.Subscriber("/camera/depth/image_rect_raw", Image,       self._cb_depth)
        rospy.Subscriber("/camera/color/camera_info",    CameraInfo,  self._cb_info)
        rospy.Subscriber("/camera/calibration",          CalibrationData, self._cb_calib)
        rospy.Subscriber("/pointing/target",             PointingTarget,  self._cb_target)
        rospy.Subscriber("/finger/analysis",             FingerAnalysis,  self._cb_finger)

        # ---------- publishers ----------
        self.marker_pub = rospy.Publisher("/diag/joints_3d", MarkerArray, queue_size=1)

        rospy.loginfo("[DIAG] Depth Diagnostics Node ready  (verbose_patch=%s)", self.verbose_patch)

    # ------------------------------------------------------------------
    # Callbacks — passive data stores
    # ------------------------------------------------------------------

    def _cb_depth(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
        except Exception as e:
            rospy.logerr("[DIAG] depth decode error: %s", e)

    def _cb_info(self, msg):
        self.camera_info = msg

    def _cb_calib(self, msg):
        if msg.is_calibrated:
            self.calibration = msg

    def _cb_target(self, msg):
        self.last_target = msg

    # ------------------------------------------------------------------
    # Main callback — fires on every FingerAnalysis message
    # ------------------------------------------------------------------

    def _cb_finger(self, msg):
        if self.depth_image is None:
            rospy.logwarn_once("[DIAG] Waiting for depth image…")
            return
        if self.camera_info is None:
            rospy.logwarn_once("[DIAG] Waiting for CameraInfo…")
            return

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        h, w = self.depth_image.shape[:2]
        stamp = msg.header.stamp
        frame_id = msg.header.frame_id or "camera_color_optical_frame"

        # knuckle_2d is normalised [0,1] — convert to pixels
        kp_norm = {
            5: (msg.knuckle_2d.x, msg.knuckle_2d.y),
            8: (msg.tip_2d.x,     msg.tip_2d.y),
        }
        # PIP (6) and DIP (7) pixel coords are not published; we interpolate
        # linearly between knuckle and tip as a best-effort audit position.
        for frac, idx in [(1/3, 6), (2/3, 7)]:
            kp_norm[idx] = (
                kp_norm[5][0] + frac * (kp_norm[8][0] - kp_norm[5][0]),
                kp_norm[5][1] + frac * (kp_norm[8][1] - kp_norm[5][1]),
            )

        # ---------------------------------------------------------------
        # 1. PATCH DEPTH AUDIT
        # ---------------------------------------------------------------
        joints_3d = {}   # idx → np.array([X,Y,Z]) or None

        for idx in [5, 6, 7, 8]:
            nx, ny = kp_norm[idx]
            px = int(np.clip(nx * w, 0, w - 1))
            py = int(np.clip(ny * h, 0, h - 1))

            center_depth = int(self.depth_image[py, px])

            half = 2   # 5×5 patch
            y0, y1 = max(0, py - half), min(h, py + half + 1)
            x0, x1 = max(0, px - half), min(w, px + half + 1)
            patch = self.depth_image[y0:y1, x0:x1].astype(float)

            valid = patch[patch > 0]
            if len(valid) == 0:
                patch_median = 0
                patch_std    = 0.0
                patch_range  = 0
            else:
                patch_median = int(np.median(valid))
                patch_std    = float(np.std(valid))
                patch_range  = int(valid.max() - valid.min())

            center_ok = center_depth > 0
            outlier   = center_ok and abs(center_depth - patch_median) > 30

            rospy.loginfo(
                "[PATCH] joint=%-3s  center=%5d mm  patch_median=%5d mm  "
                "std=%5.1f mm  range=%4d mm  center_zero=%s  outlier=%s",
                self.JOINT_LABELS[idx],
                center_depth, patch_median,
                patch_std, patch_range,
                "YES" if not center_ok else "no",
                "YES" if outlier else "no",
            )

            # Back-project the patch median depth to 3D
            if patch_median > 0:
                z = patch_median * 0.001
                X = (px - cx) * z / fx
                Y = (py - cy) * z / fy
                joints_3d[idx] = np.array([X, Y, z])
            else:
                joints_3d[idx] = None

            # ---------------------------------------------------------------
            # 4. DEPTH GRADIENT TEST (throttled to every 2 s, finger straight)
            # ---------------------------------------------------------------
            now = time.time()
            if (msg.is_straight
                    and self.verbose_patch
                    and (now - self._last_grad_t) >= 2.0):
                flag = "OBLIQUE GRADIENT DETECTED" if patch_range > 30 else "ok"
                print(
                    f"[GRADIENT] joint={self.JOINT_LABELS[idx]}  "
                    f"range={patch_range} mm  [{flag}]\n"
                    + ascii_heatmap(patch.astype(int))
                )
            if idx == 8 and (now - self._last_grad_t) >= 2.0 and msg.is_straight:
                self._last_grad_t = now

        # ---------------------------------------------------------------
        # 2. 3D JOINT VISUALIZER — MarkerArray
        # ---------------------------------------------------------------
        marker_array = MarkerArray()
        marker_id = 0

        # Delete all previous markers first
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        delete_all.header.stamp = stamp
        delete_all.header.frame_id = frame_id
        marker_array.markers.append(delete_all)
        marker_id += 1

        # Joint spheres — prefer the already-filtered 3D from the message
        # for the display; fall back to our back-projected values.
        msg_joints = {
            5: vec3_to_np(msg.knuckle_3d),
            8: vec3_to_np(msg.tip_3d),
        }
        for idx in [5, 8]:
            p = msg_joints[idx]
            if np.linalg.norm(p) < 1e-6 and joints_3d.get(idx) is not None:
                p = joints_3d[idx]
            if np.linalg.norm(p) < 1e-6:
                continue
            m = make_sphere_marker(
                "joints", marker_id, p,
                self.JOINT_COLORS[idx], frame_id, stamp,
            )
            marker_array.markers.append(m)
            marker_id += 1

        # Interpolated PIP / DIP spheres from our back-projected values
        for idx in [6, 7]:
            if joints_3d.get(idx) is None:
                continue
            m = make_sphere_marker(
                "joints", marker_id, joints_3d[idx],
                self.JOINT_COLORS[idx], frame_id, stamp,
            )
            marker_array.markers.append(m)
            marker_id += 1

        # Ray arrow — knuckle_3d + direction_3d × 1.5 m
        knuckle = vec3_to_np(msg.knuckle_3d)
        direction = vec3_to_np(msg.direction_3d)
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-6 and np.linalg.norm(knuckle) > 1e-6:
            direction = direction / dir_norm
            arrow = make_arrow_marker(
                "ray", marker_id, knuckle, direction,
                length=1.5,
                color_rgba=(0.0, 0.9, 1.0, 0.85),
                frame_id=frame_id,
                stamp=stamp,
            )
            marker_array.markers.append(arrow)
            marker_id += 1

        self.marker_pub.publish(marker_array)

        # ---------------------------------------------------------------
        # 3. RAY-PLANE INTERSECTION VALIDATOR
        # ---------------------------------------------------------------
        if self.calibration is not None and dir_norm > 1e-6 and np.linalg.norm(knuckle) > 1e-6:
            plane_normal = vec3_to_np(self.calibration.plane_normal)
            plane_center = vec3_to_np(self.calibration.plane_center)

            local_pt, local_t = ray_plane_intersection(knuckle, direction, plane_normal, plane_center)

            if local_pt is not None:
                rospy.loginfo(
                    "[RAY] local intersection  x=%.4f m  y=%.4f m  z=%.4f m  t=%.4f m",
                    local_pt[0], local_pt[1], local_pt[2], local_t,
                )

                # Compare against /pointing/target if we have a recent one
                if self.last_target is not None and self.last_target.is_valid:
                    # Reconstruct the published 3D point from u/v mm
                    x_axis  = vec3_to_np(self.calibration.paper_x_axis)
                    y_axis  = vec3_to_np(self.calibration.paper_y_axis)
                    pub_pt  = (plane_center
                               + x_axis * (self.last_target.u_mm / 1000.0)
                               + y_axis * (self.last_target.v_mm / 1000.0))
                    delta_m  = np.linalg.norm(local_pt - pub_pt)
                    delta_mm = delta_m * 1000.0

                    if delta_mm > 5.0:
                        rospy.logwarn(
                            "[RAY] MISMATCH vs /pointing/target  delta=%.1f mm  "
                            "(local=[%.1f, %.1f] mm  published=[%.1f, %.1f] mm)",
                            delta_mm,
                            np.dot(local_pt - plane_center, x_axis) * 1000.0,
                            np.dot(local_pt - plane_center, y_axis) * 1000.0,
                            self.last_target.u_mm,
                            self.last_target.v_mm,
                        )
                    else:
                        rospy.loginfo(
                            "[RAY] agrees with /pointing/target  delta=%.2f mm  (within 5 mm)",
                            delta_mm,
                        )

                # ---------------------------------------------------------------
                # 5. ANGULAR ERROR ESTIMATOR
                # ---------------------------------------------------------------
                pn_norm = np.linalg.norm(plane_normal)
                if pn_norm > 1e-6:
                    cos_angle = abs(np.dot(direction, plane_normal / pn_norm))
                    # angle between ray and plane normal; 0 = perpendicular incidence
                    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
                    # sensitivity: δmm per 1° of direction error at this t
                    # For small δθ: lateral displacement ≈ t * sin(δθ) ≈ t * δθ_rad
                    one_deg_mm = local_t * np.sin(np.radians(1.0)) * 1000.0
                    rospy.loginfo(
                        "[ANGLE] ray∠normal=%.2f°  "
                        "1° direction error → %.1f mm lateral shift at t=%.3f m",
                        angle_deg, one_deg_mm, local_t,
                    )
            else:
                rospy.logwarn("[RAY] no intersection (ray parallel to plane or behind origin)")

    def spin(self):
        rospy.spin()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        node = DepthDiagnosticsNode()
        node.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[DIAG] Shutting down.")
