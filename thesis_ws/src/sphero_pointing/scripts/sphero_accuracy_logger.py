#!/usr/bin/env python3
"""
Sphero Accuracy Logger

Time-aligns commanded pointing targets with camera-observed Sphero positions
so the open-loop controller can be validated against ground truth.

Subscribes:
  /pointing/target    pointing_localization/PointingTarget   (commanded u,v)
  /sphero/detection   geometry_msgs/Point                    (actual u,v from camera)

Every /sphero/detection sample is paired with the most recent /pointing/target
and appended to a CSV. A live error readout prints once per second.

Output CSV columns:
  t, cmd_u, cmd_v, actual_u, actual_v, error_mm

Usage:
  rosrun sphero_pointing sphero_accuracy_logger.py
  rosrun sphero_pointing sphero_accuracy_logger.py _output_file:=~/run.csv
  rosrun sphero_pointing sphero_accuracy_logger.py _paper_x_mm:=850 _paper_y_mm:=600

Ctrl+C saves and exits.
"""

import csv
import math
import os
import rospy

from geometry_msgs.msg import Point
from pointing_localization.msg import PointingTarget


class SpheroAccuracyLogger:
    def __init__(self):
        rospy.init_node('sphero_accuracy_logger', anonymous=True)

        ts = int(rospy.get_time())
        default_path = os.path.expanduser(f'~/Desktop/thesis/sphero_accuracy_{ts}.csv')
        self.output_file = os.path.expanduser(
            rospy.get_param('~output_file', default_path)
        )

        # Paper size in mm — used to turn normalized error into physical distance.
        # Keep in sync with camera_calibration paper_x_m / paper_y_m.
        self.paper_x_mm = float(rospy.get_param('~paper_x_mm', 850.0))
        self.paper_y_mm = float(rospy.get_param('~paper_y_mm', 600.0))

        self.print_interval = float(rospy.get_param('~print_interval', 1.0))

        self.cmd = None    # (t, u, v) — latest valid command
        self.rows = []     # one entry per /sphero/detection sample

        rospy.Subscriber('/pointing/target', PointingTarget, self._cmd_cb,    queue_size=50)
        rospy.Subscriber('/sphero/detection', Point,         self._actual_cb, queue_size=50)

        rospy.Timer(rospy.Duration(self.print_interval), self._print_live)

        rospy.loginfo(
            "Sphero Accuracy Logger ready\n"
            f"  output_file = {self.output_file}\n"
            f"  paper       = {self.paper_x_mm:.0f} x {self.paper_y_mm:.0f} mm"
        )

    def _cmd_cb(self, msg):
        if not msg.is_valid:
            return
        t = msg.header.stamp.to_sec() if msg.header.stamp.secs != 0 else rospy.get_time()
        self.cmd = (t, float(msg.u_normalized), float(msg.v_normalized))

    def _actual_cb(self, msg):
        t = rospy.get_time()
        u = float(msg.x)
        v = float(msg.y)

        if self.cmd is None:
            self.rows.append((t, None, None, u, v, None))
            return

        cmd_u, cmd_v = self.cmd[1], self.cmd[2]
        err_mm = math.sqrt(((u - cmd_u) * self.paper_x_mm) ** 2 +
                           ((v - cmd_v) * self.paper_y_mm) ** 2)
        self.rows.append((t, cmd_u, cmd_v, u, v, err_mm))

    def _print_live(self, _event):
        if not self.rows:
            return
        t, cmd_u, cmd_v, u, v, err = self.rows[-1]
        if err is None:
            rospy.loginfo(f"[ACC] actual=({u:.3f},{v:.3f})  cmd=— (no command yet)")
        else:
            rospy.loginfo(
                f"[ACC] cmd=({cmd_u:.3f},{cmd_v:.3f})  "
                f"actual=({u:.3f},{v:.3f})  err={err:.1f} mm"
            )

    def save(self):
        if not self.rows:
            rospy.logwarn("No samples collected — nothing saved.")
            return
        out_dir = os.path.dirname(self.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.output_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'cmd_u', 'cmd_v', 'actual_u', 'actual_v', 'error_mm'])
            for t, cmd_u, cmd_v, u, v, err in self.rows:
                w.writerow([
                    round(t, 4),
                    '' if cmd_u is None else round(cmd_u, 5),
                    '' if cmd_v is None else round(cmd_v, 5),
                    round(u, 5),
                    round(v, 5),
                    '' if err   is None else round(err,   1),
                ])
        rospy.loginfo(f"Saved {len(self.rows)} rows → {self.output_file}")

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    node = SpheroAccuracyLogger()
    try:
        node.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    finally:
        node.save()
