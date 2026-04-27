#!/usr/bin/env python3
"""
Sphero Pointing Controller Node — closed-loop with camera feedback.

Continuously monitors /sphero/detection and drives the Sphero toward the
pointed target in short roll steps, re-reading camera position after each
step and correcting course until within arrival_threshold_cm.

Startup auto-calibration:
  - Reads a stable camera position (waits for consistent detections)
  - Rolls heading=0 for calib_dist_cm
  - Measures actual direction → sets heading_offset_deg automatically
"""

import rospy
import numpy as np
import math
import threading

from geometry_msgs.msg import Point
from pointing_localization.msg import PointingTarget

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color


class SpheroPointingNode:
    TRACK_COLOR  = Color(r=0,   g=0,   b=255)
    MOVING_COLOR = Color(r=255, g=200, b=0  )
    CALIB_COLOR  = Color(r=255, g=100, b=0  )

    def __init__(self):
        rospy.init_node('sphero_pointing_node', anonymous=False)

        self.roll_speed           = int(rospy.get_param('~roll_speed',           40   ))
        self.cm_per_sec           = float(rospy.get_param('~cm_per_sec',         18.18))
        self.arrival_threshold_cm = float(rospy.get_param('~arrival_threshold_cm', 2.0))

        self.paper_x_cm = float(rospy.get_param('~paper_x_cm', 85.0))
        self.paper_y_cm = float(rospy.get_param('~paper_y_cm', 60.0))

        self.target_hold_time = float(rospy.get_param('~target_hold_time', 0.5))
        self.sphero_name      = rospy.get_param('~sphero_name', '')

        self.use_compass         = rospy.get_param('~use_compass', False)
        self.paper_north_heading = int(rospy.get_param('~paper_north_heading', 0))
        self.heading_offset_deg  = float(rospy.get_param('~heading_offset', 0.0))
        self.calib_dist_cm       = float(rospy.get_param('~calib_dist_cm', 10.0))

        # Max duration of a single roll step — shorter = more responsive
        self.max_step_s = float(rospy.get_param('~max_step_s', 0.4))
        # Brief pause after each step so Sphero stops before camera reads
        self.step_pause_s = float(rospy.get_param('~step_pause_s', 0.15))

        self.cm_per_sec = 0.0   # measured during auto heading calibration

        self.sphero_uv        = np.array([0.5, 0.5])
        self.target_uv        = None
        self.target_stamp     = None
        self.stable_target_uv = None

        self.camera_uv    = None
        self.camera_stamp = None

        self.lock = threading.Lock()

        rospy.loginfo(
            "Sphero Pointing Node ready\n"
            "  speed=%d  cm/s=%.2f  paper=%.1f×%.1f cm  max_step=%.2fs",
            self.roll_speed, self.cm_per_sec,
            self.paper_x_cm, self.paper_y_cm, self.max_step_s
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def target_callback(self, msg):
        if not msg.is_valid:
            return
        with self.lock:
            new_uv = (float(np.clip(msg.u_normalized, 0.0, 1.0)),
                      float(np.clip(msg.v_normalized, 0.0, 1.0)))
            now    = rospy.get_time()
            if self.target_uv is not None:
                d = math.hypot(new_uv[0] - self.target_uv[0], new_uv[1] - self.target_uv[1])
                if d > 0.03:
                    self.target_stamp = now
            else:
                self.target_stamp = now
            self.target_uv = new_uv
            if self.target_stamp is not None and (now - self.target_stamp) >= self.target_hold_time:
                self.stable_target_uv = new_uv

    def detection_callback(self, msg):
        with self.lock:
            self.camera_uv    = (float(msg.x), float(msg.y))
            self.camera_stamp = rospy.get_time()

    # ── Camera helpers ────────────────────────────────────────────────────────

    def _latest_camera_pos(self, max_age=0.5):
        """Return camera (u,v) if fresh enough, else None. Must hold lock."""
        if self.camera_uv is None or self.camera_stamp is None:
            return None
        if rospy.get_time() - self.camera_stamp > max_age:
            return None
        return self.camera_uv

    def _wait_for_detection(self, timeout=3.0, max_age=0.5):
        """Block until a fresh detection arrives. Returns (u,v) or None."""
        deadline = rospy.get_time() + timeout
        while rospy.get_time() < deadline and not rospy.is_shutdown():
            with self.lock:
                pos = self._latest_camera_pos(max_age=max_age)
            if pos is not None:
                return pos
            rospy.sleep(0.05)
        return None

    def _wait_for_stable_pos(self, timeout=15.0, n_needed=5, radius_cm=2.5):
        """
        Wait until n_needed consecutive fresh detections agree within radius_cm.
        Prevents using noisy or mid-motion readings as position reference.
        """
        rospy.loginfo("Waiting for stable camera position (%d samples within %.1f cm)…",
                      n_needed, radius_cm)
        deadline  = rospy.get_time() + timeout
        history   = []
        prev_stamp = None

        while rospy.get_time() < deadline and not rospy.is_shutdown():
            with self.lock:
                pos   = self._latest_camera_pos(max_age=0.5)
                stamp = self.camera_stamp

            if pos is not None and stamp != prev_stamp:
                prev_stamp = stamp
                if history:
                    cu, cv = np.mean(history, axis=0)
                    d_cm = math.sqrt(((pos[0] - cu) * self.paper_x_cm) ** 2 +
                                     ((pos[1] - cv) * self.paper_y_cm) ** 2)
                    if d_cm > radius_cm:
                        history = []
                history.append(pos)
                if len(history) >= n_needed:
                    result = tuple(np.mean(history, axis=0))
                    rospy.loginfo("Stable: (%.3f, %.3f) confirmed", result[0], result[1])
                    return result

            rospy.sleep(0.1)
        return None

    def _read_pos_after_stop(self):
        """
        Read camera position after a roll step finishes.
        Waits step_pause_s for the Sphero to coast to a stop, then takes
        the very next fresh detection (not one cached from during the roll).
        """
        rospy.sleep(self.step_pause_s)
        # Invalidate stamp so we block until the camera captures a new frame
        with self.lock:
            self.camera_stamp = None
        return self._wait_for_detection(timeout=1.5)

    # ── Auto heading calibration ───────────────────────────────────────────────

    def _auto_calibrate_heading(self, droid):
        print("\n" + "="*56)
        print("  AUTO HEADING CALIBRATION")
        print("  Camera will measure actual roll direction…")

        start_pos = self._wait_for_stable_pos(timeout=15.0)
        if start_pos is None:
            rospy.logwarn("Cannot confirm Sphero position — skipping auto-calibration. "
                          "heading_offset stays %.1f°", self.heading_offset_deg)
            print("  WARNING: camera not stable — skipped.")
            print("="*56 + "\n")
            return

        rospy.loginfo("Start (%.3f,%.3f) confirmed — rolling heading=0 for %.1fs…",
                      start_pos[0], start_pos[1], self.max_step_s)

        duration = self.max_step_s
        droid.set_main_led(self.CALIB_COLOR)
        droid.roll(0, self.roll_speed, duration)
        droid.set_speed(0)

        end_pos = self._read_pos_after_stop()
        droid.set_main_led(self.TRACK_COLOR)

        if end_pos is None:
            rospy.logwarn("Camera lost Sphero after calibration roll — skipping.")
            print("  WARNING: camera lost Sphero — skipped.")
            print("="*56 + "\n")
            return

        du = end_pos[0] - start_pos[0]
        dv = end_pos[1] - start_pos[1]
        dist_cm = math.sqrt((du * self.paper_x_cm) ** 2 + (dv * self.paper_y_cm) ** 2)

        if dist_cm < 2.0:
            rospy.logwarn("Barely moved (%.1f cm) — calibration unreliable. heading_offset stays %.1f°",
                          dist_cm, self.heading_offset_deg)
            print("  WARNING: moved only %.1f cm — skipped." % dist_cm)
            print("="*56 + "\n")
            return

        self.heading_offset_deg = math.degrees(math.atan2(du, -dv)) % 360
        self.cm_per_sec         = dist_cm / self.max_step_s
        with self.lock:
            self.sphero_uv = np.array(end_pos)

        rospy.loginfo("Calibration done: %.1f cm  →  heading_offset=%.1f°  cm/s=%.1f",
                      dist_cm, self.heading_offset_deg, self.cm_per_sec)
        print("  start=(%.3f,%.3f)  end=(%.3f,%.3f)  dist=%.1fcm" % (
              start_pos[0], start_pos[1], end_pos[0], end_pos[1], dist_cm))
        print("  heading_offset = %.1f°" % self.heading_offset_deg)
        print("  Sphero is ready.")
        print("="*56 + "\n")

    # ── Closed-loop control ───────────────────────────────────────────────────

    def control_loop(self, droid):
        """
        Single-shot movement per target.

        When a stable pointing target arrives, computes exact roll duration
        from the cm_per_sec measured at calibration, rolls once, then reads
        camera to confirm where the Sphero landed. Waits for the next target.
        """
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            with self.lock:
                target = self.stable_target_uv
                cam    = self._latest_camera_pos(max_age=1.0)
                if cam is not None:
                    self.sphero_uv = np.array(cam)
                current = self.sphero_uv.copy()

            if target is None:
                rate.sleep()
                continue

            du      = target[0] - current[0]
            dv      = target[1] - current[1]
            dist_cm = math.sqrt((du * self.paper_x_cm) ** 2 + (dv * self.paper_y_cm) ** 2)

            if dist_cm < self.arrival_threshold_cm:
                with self.lock:
                    self.stable_target_uv = None
                rospy.loginfo("Already within threshold (%.1f cm) — no move", dist_cm)
                droid.set_main_led(self.TRACK_COLOR)
                rate.sleep()
                continue

            angle_paper = math.degrees(math.atan2(du, -dv)) % 360
            heading     = int(round(self.heading_offset_deg - angle_paper)) % 360

            # Single roll: duration computed from measured speed
            if self.cm_per_sec > 0:
                duration = dist_cm / self.cm_per_sec
            else:
                duration = self.max_step_s   # fallback if calibration skipped

            rospy.loginfo("Roll → target=(%.3f,%.3f)  from=(%.3f,%.3f)  "
                          "heading=%d°  dist=%.1fcm  dur=%.2fs",
                          target[0], target[1], current[0], current[1],
                          heading, dist_cm, duration)

            droid.set_main_led(self.MOVING_COLOR)
            droid.roll(heading, self.roll_speed, duration)
            droid.set_speed(0)

            new_pos = self._read_pos_after_stop()
            droid.set_main_led(self.TRACK_COLOR)

            with self.lock:
                if new_pos is not None:
                    self.sphero_uv = np.array(new_pos)
                    err_mm = math.sqrt(((new_pos[0] - target[0]) * self.paper_x_cm * 10) ** 2 +
                                       ((new_pos[1] - target[1]) * self.paper_y_cm * 10) ** 2)
                    rospy.loginfo("Landed (%.3f,%.3f)  err=%.0f mm", new_pos[0], new_pos[1], err_mm)
                else:
                    rospy.logwarn("No camera confirmation after roll")
                self.stable_target_uv = None   # done — wait for next target

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self):
        rospy.loginfo("Scanning for Sphero BOLT…")
        toy = scanner.find_toy(toy_name=self.sphero_name) if self.sphero_name else scanner.find_toy()
        rospy.loginfo("Connected: %s", toy.name)

        with SpheroEduAPI(toy) as droid:
            droid.set_main_led(self.TRACK_COLOR)

            rospy.Subscriber('/pointing/target',  PointingTarget, self.target_callback,    queue_size=1)
            rospy.Subscriber('/sphero/detection', Point,          self.detection_callback, queue_size=1)

            if self.use_compass:
                print("\n" + "="*56)
                print("  HEADING CALIBRATION — compass mode")
                droid.calibrate(self.paper_north_heading)
                print(f"  paper-up = {self.paper_north_heading}°")
                droid.set_main_led(Color(r=0, g=200, b=0))
                rospy.sleep(0.5)
                droid.set_main_led(self.TRACK_COLOR)
                print("="*56 + "\n")
            else:
                self._auto_calibrate_heading(droid)

            rospy.loginfo(
                "ACTIVE — heading_offset=%.1f°  max_step=%.2fs\n"
                "  Point at the paper and hold to move the Sphero.",
                self.heading_offset_deg, self.max_step_s
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
        rospy.loginfo("Shutting down…")
