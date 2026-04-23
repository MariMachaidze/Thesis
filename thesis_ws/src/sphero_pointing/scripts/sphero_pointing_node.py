#!/usr/bin/env python3
"""
Sphero Pointing Controller Node
Moves Sphero BOLT to where the user points on the paper.

Pipeline:
  /pointing/target  -> target location on paper (u,v normalized 0-1)
  -> compute heading + duration from calibrated velocity
  -> droid.roll(heading, speed, duration)  [single timed move per target]

Velocity calibration (measured):
  speed=40, 1.10 s  =>  ~20 cm
  => cm_per_sec = 20 / 1.10 = 18.18 cm/s at speed 40

The Sphero's position is NOT tracked by camera.
It starts at the center of the paper (0.5, 0.5) and the
estimated position is updated after each completed move.
"""

import rospy
import numpy as np
import math
import threading

from pointing_localization.msg import PointingTarget

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# ── Commented out: camera-based Sphero detection ─────────────────────────────
# import cv2
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from camera_calibration.msg import CalibrationData
# ─────────────────────────────────────────────────────────────────────────────


class SpheroPointingNode:
    TRACK_COLOR   = Color(r=0,   g=0,   b=255)   # blue = idle/ready
    MOVING_COLOR  = Color(r=255, g=200, b=0  )   # yellow  = rolling

    def __init__(self):
        rospy.init_node('sphero_pointing_node', anonymous=False)

        # --- Motion parameters ---
        # Measured: speed=40 for 1.10 s travels ~20 cm
        self.roll_speed        = int(rospy.get_param('~roll_speed',        40  ))
        self.cm_per_sec        = float(rospy.get_param('~cm_per_sec',      18.18))  # at roll_speed
        self.arrival_threshold_cm = float(rospy.get_param('~arrival_threshold_cm', 2.0))

        # Paper physical dimensions (must match camera_calibration params)
        self.paper_x_cm = float(rospy.get_param('~paper_x_cm', 84.1))
        self.paper_y_cm = float(rospy.get_param('~paper_y_cm', 59.4))

        # Heading offset: how many degrees Sphero heading-0 is rotated
        # relative to paper "up" (decreasing v direction).
        # Tune this if the Sphero goes in the wrong direction.
        self.heading_offset_deg = float(rospy.get_param('~heading_offset', 0.0))

        self.target_hold_time  = float(rospy.get_param('~target_hold_time', 0.5))
        self.sphero_name       = rospy.get_param('~sphero_name', '')

        # --- State ---
        self.sphero_uv        = np.array([0.5, 0.5])  # assumed start = center
        self.target_uv        = None
        self.target_stamp     = None
        self.stable_target_uv = None
        self.is_moving        = False

        self.lock = threading.Lock()
        rospy.loginfo(
            "Sphero Pointing Node: timed-roll mode\n"
            "  speed=%d  cm/s=%.2f  paper=%.1f×%.1f cm  heading_offset=%.1f°",
            self.roll_speed, self.cm_per_sec,
            self.paper_x_cm, self.paper_y_cm, self.heading_offset_deg
        )

    # ================================================================
    #  ROS callback — pointing target
    # ================================================================
    def target_callback(self, msg):
        """Accept a new target once it has been stable for target_hold_time."""
        if not msg.is_valid:
            return

        with self.lock:
            new_uv = (msg.u_normalized, msg.v_normalized)
            now    = rospy.get_time()

            if self.target_uv is not None:
                d = math.sqrt((new_uv[0] - self.target_uv[0]) ** 2 +
                              (new_uv[1] - self.target_uv[1]) ** 2)
                if d > 0.03:
                    self.target_stamp = now
            else:
                self.target_stamp = now

            self.target_uv = new_uv

            if self.target_stamp is not None and (now - self.target_stamp) >= self.target_hold_time:
                self.stable_target_uv = new_uv

    # ================================================================
    #  Control loop — one timed roll per stable target
    # ================================================================
    def control_loop(self, droid):
        rate = rospy.Rate(10)   # check for new targets at 10 Hz

        while not rospy.is_shutdown():
            with self.lock:
                target  = self.stable_target_uv
                current = self.sphero_uv.copy()

            if target is None or self.is_moving:
                rate.sleep()
                continue

            # Distance in physical cm
            du = target[0] - current[0]
            dv = target[1] - current[1]
            dx_cm = du * self.paper_x_cm
            dy_cm = dv * self.paper_y_cm
            dist_cm = math.sqrt(dx_cm ** 2 + dy_cm ** 2)

            if dist_cm < self.arrival_threshold_cm:
                # Already there — consume the target and wait for the next one
                with self.lock:
                    self.sphero_uv        = np.array(target)
                    self.stable_target_uv = None
                rospy.loginfo("Target within threshold (%.1f cm) — no move needed", dist_cm)
                rate.sleep()
                continue

            # Heading: paper angle (0 = up / decreasing v, CW positive)
            angle_paper = math.degrees(math.atan2(du, -dv)) % 360
            heading     = int(round(angle_paper - self.heading_offset_deg)) % 360

            # Duration from calibrated velocity
            duration = dist_cm / self.cm_per_sec

            rospy.loginfo(
                "Roll → target=(%.3f,%.3f)  heading=%d°  speed=%d  dist=%.1fcm  duration=%.2fs",
                target[0], target[1], heading, self.roll_speed, dist_cm, duration
            )

            # Execute move (blocks for `duration` seconds)
            self.is_moving = True
            droid.set_main_led(self.MOVING_COLOR)
            droid.roll(heading, self.roll_speed, duration)
            droid.set_speed(0)
            droid.set_main_led(self.TRACK_COLOR)
            self.is_moving = False

            # Update estimated position and clear target
            with self.lock:
                self.sphero_uv        = np.array(target)
                self.stable_target_uv = None   # wait for next stable target

            rospy.loginfo("Arrived at (%.3f, %.3f)", target[0], target[1])

    # ================================================================
    #  Main entry
    # ================================================================
    def run(self):
        rospy.loginfo("Scanning for Sphero BOLT...")
        toy = scanner.find_toy(toy_name=self.sphero_name) if self.sphero_name else scanner.find_toy()
        rospy.loginfo("Connected: %s", toy.name)

        with SpheroEduAPI(toy) as droid:
            droid.set_main_led(self.TRACK_COLOR)
            droid.reset_aim()   # heading 0 = current facing direction

            rospy.Subscriber('/pointing/target', PointingTarget, self.target_callback, queue_size=1)

            rospy.loginfo(
                "ACTIVE — Sphero starts at center (0.5, 0.5)\n"
                "  Point at the paper and hold to move the Sphero."
            )

            self.control_loop(droid)

            droid.set_speed(0)
            droid.set_main_led(Color(r=0, g=0, b=0))
            rospy.loginfo("Shutdown complete")


if __name__ == '__main__':
    try:
        node = SpheroPointingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
