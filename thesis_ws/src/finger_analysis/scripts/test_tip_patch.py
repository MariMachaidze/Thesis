#!/usr/bin/env python3
"""
test_tip_patch.py
-----------------
For every frame where the hand is detected, sample the depth image at all
4 index finger joints (MCP=5, PIP=6, DIP=7, TIP=8) using patch sizes
1x1 through 9x9, and print how many valid (non-zero) pixels each finds.

Run in a separate terminal while the system is running:
    source thesis_ws/devel/setup.bash
    rosrun finger_analysis test_tip_patch.py

Output (one line per joint per frame):
    [MCP] 1x1=100%  3x3=100%  5x5=100%  7x7=100%  9x9=100%  center=1450mm
    [PIP] 1x1=100%  3x3=100%  5x5=100%  7x7=100%  9x9=100%  center=1480mm
    [DIP] 1x1=100%  3x3=100%  5x5= 88%  7x7= 80%  9x9= 72%  center=1510mm
    [TIP] 1x1=  0%  3x3= 44%  5x5= 68%  7x7= 76%  9x9= 79%  center=NO DEPTH
    ---
"""

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hand_detection.msg import Hand

PATCH_SIZES = [1, 3, 5, 7, 9]
JOINTS      = {5: "MCP", 6: "PIP", 7: "DIP", 8: "TIP"}


class PatchTester:
    def __init__(self):
        rospy.init_node('test_tip_patch', anonymous=True)
        self.bridge      = CvBridge()
        self.depth_image = None

        rospy.Subscriber('/d435i/depth/image_raw', Image, self._cb_depth)
        rospy.Subscriber('/hand/detection',        Hand,  self._cb_hand)
        rospy.loginfo("[PATCH TEST] Running — keep your index finger extended")

    def _cb_depth(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        except Exception as e:
            rospy.logerr(f"Depth decode error: {e}")

    def _sample_pct(self, px, py, size):
        """Percentage of non-zero pixels in a patch centred at (px, py)."""
        h, w = self.depth_image.shape[:2]
        half = size // 2
        y0 = max(0, py - half);  y1 = min(h, py + half + 1)
        x0 = max(0, px - half);  x1 = min(w, px + half + 1)
        patch = self.depth_image[y0:y1, x0:x1]
        total = patch.size
        valid = int(np.sum(patch > 0))
        return valid / total * 100 if total > 0 else 0.0

    def _cb_hand(self, msg):
        if not msg.detected or self.depth_image is None:
            return
        if len(msg.keypoints_2d) < 9:
            return

        h, w = self.depth_image.shape[:2]

        for idx, label in JOINTS.items():
            pt = msg.keypoints_2d[idx]
            px = int(np.clip(pt.x * w, 0, w - 1))
            py = int(np.clip(pt.y * h, 0, h - 1))

            center_depth = int(self.depth_image[py, px])
            depth_str    = f"{center_depth}mm" if center_depth > 0 else "NO DEPTH"

            cols = "  ".join(f"{s}x{s}={self._sample_pct(px, py, s):3.0f}%"
                             for s in PATCH_SIZES)
            rospy.loginfo(f"[{label}] {cols}  center={depth_str}")

        rospy.loginfo("---")

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        PatchTester().spin()
    except KeyboardInterrupt:
        pass
