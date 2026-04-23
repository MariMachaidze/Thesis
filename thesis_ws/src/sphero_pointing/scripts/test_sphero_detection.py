#!/usr/bin/env python3
"""
test_sphero_detection.py
------------------------
1. Connects to the Sphero BOLT via Bluetooth
2. Sets its LED to magenta (the tracking color)
3. Opens three live windows to verify the camera can see it:
     "RGB + detection"  — camera feed with green circle on the Sphero
     "HSV mask (raw)"   — white = pixels in the magenta HSV range
     "Mask (cleaned)"   — after morphological noise removal

Run:
    source thesis_ws/devel/setup.bash
    rosrun sphero_pointing test_sphero_detection.py

Optional ROS params:
  ~sphero_name  (default "SB-FD03") — Sphero device name
  ~sat_min      (default 100)       — lower HSV saturation bound
  ~val_min      (default 100)       — lower HSV brightness bound
  ~hue_hi       (default 10)        — upper hue for the 0-side wrap range

Press 'q' to exit (LED turns off on exit).
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color


class SpheroDiag:
    MIN_BLOB_AREA = 50
    MAX_BLOB_AREA = 50000

    TRACK_COLOR = Color(r=255, g=0, b=0)

    def __init__(self):
        rospy.init_node('test_sphero_detection', anonymous=True)
        self.bridge = CvBridge()
        self.frame  = None

        sat_min = rospy.get_param('~sat_min', 100)
        val_min = rospy.get_param('~val_min', 100)

        # Red wraps around the hue wheel at 0/180, so two ranges needed
        self.lower1 = np.array([170, sat_min, val_min])
        self.upper1 = np.array([180, 255,     255    ])
        self.lower2 = np.array([0,   sat_min, val_min])
        self.upper2 = np.array([10,  255,     255    ])

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        rospy.Subscriber('/d435i/rgb/image_raw', Image, self._cb, queue_size=1)

    def _cb(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"imgmsg_to_cv2: {e}")

    def _detect(self, bgr):
        hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower1, self.upper1)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        raw   = cv2.bitwise_or(mask1, mask2)
        clean = cv2.morphologyEx(raw,   cv2.MORPH_OPEN,  self.kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [(c, cv2.contourArea(c)) for c in contours
                 if self.MIN_BLOB_AREA <= cv2.contourArea(c) <= self.MAX_BLOB_AREA]

        centroid = None
        area     = 0
        if valid:
            largest, area = max(valid, key=lambda x: x[1])
            M = cv2.moments(largest)
            if M["m00"] > 0:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        all_areas = sorted([int(cv2.contourArea(c)) for c in contours], reverse=True)
        return centroid, int(area), raw, clean, all_areas

    def spin(self):
        cv2.namedWindow("RGB + detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV mask (raw)",  cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask (cleaned)",  cv2.WINDOW_NORMAL)

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.frame is None:
                rate.sleep()
                continue

            bgr = self.frame.copy()
            centroid, area, raw_mask, clean_mask, all_areas = self._detect(bgr)

            h, w = bgr.shape[:2]

            if centroid is not None:
                cx, cy = centroid
                cv2.circle(bgr, (cx, cy), 20, (0, 255, 0), 3)
                cv2.circle(bgr, (cx, cy),  4, (0, 255, 0), -1)
                cv2.putText(bgr,
                            f"DETECTED  pixel=({cx},{cy})  area={area}px",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                rospy.loginfo_throttle(2,
                    "[SPHERO DIAG] Detected at pixel (%d, %d), blob area=%d px", cx, cy, area)
            else:
                cv2.putText(bgr, "NOT DETECTED",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                if all_areas:
                    note = f"blobs out of area range: {all_areas[:5]}"
                else:
                    note = "no magenta blobs found at all"
                cv2.putText(bgr, note,
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)
                rospy.logwarn_throttle(3, "[SPHERO DIAG] %s", note)

            cv2.putText(bgr,
                        f"H:[{self.lower1[0]}-{self.upper1[0]} | {self.lower2[0]}-{self.upper2[0]}]  S>={self.lower1[1]}  V>={self.lower1[2]}",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Draw centroid on the cleaned mask so you can see blob + position together
            mask_vis = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
            if centroid is not None:
                cx, cy = centroid
                cv2.circle(mask_vis, (cx, cy), 20, (0, 255, 0), 3)
                cv2.circle(mask_vis, (cx, cy),  4, (0, 255, 0), -1)
                cv2.putText(mask_vis, f"({cx},{cy})",
                            (cx + 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("RGB + detection", bgr)
            cv2.imshow("HSV mask (raw)",  raw_mask)
            cv2.imshow("Mask (cleaned)",  mask_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    # ── Step 1: connect to Sphero and set LED to magenta ─────────────────────
    sphero_name = None
    # Peek at ROS param before init_node (use sys.argv parsing isn't needed —
    # just read it after init inside SpheroDiag, but we need it here first).
    # Simple fallback: read from environment or hardcode name.
    import sys
    name_arg = [a for a in sys.argv if a.startswith('_sphero_name:=') or a.startswith('__sphero_name:=')]
    if name_arg:
        sphero_name = name_arg[0].split(':=', 1)[1]
    else:
        sphero_name = 'SB-FD03'  # default — change if your Sphero has a different name

    print(f"[SPHERO DIAG] Scanning for Sphero: {sphero_name} ...")
    try:
        toy = scanner.find_toy(toy_name=sphero_name)
    except Exception as e:
        print(f"[SPHERO DIAG] Scan failed: {e}")
        sys.exit(1)

    print(f"[SPHERO DIAG] Connected to {toy.name}!")

    try:
        with SpheroEduAPI(toy) as droid:
            # Set LED to red so camera detection can find it
            droid.set_main_led(SpheroDiag.TRACK_COLOR)
            print("[SPHERO DIAG] LED set to red — starting camera detection...")

            # ── Step 2: run the ROS detection loop ───────────────────────────
            try:
                SpheroDiag().spin()
            except KeyboardInterrupt:
                pass

            # Turn off LED on clean exit
            droid.set_main_led(Color(r=0, g=0, b=0))
            print("[SPHERO DIAG] LED off — disconnected.")

    except Exception as e:
        print(f"[SPHERO DIAG] Sphero error: {e}")
        import traceback
        traceback.print_exc()
