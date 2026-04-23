#!/usr/bin/env python3
"""
Straightness data logger.

Subscribes to /finger/analysis and writes one CSV row per message:
  time_s, raw, ema, one_euro, is_straight

Usage (with system running):
  python3 straightness_logger.py
  Point at A ... stop pointing ... point at B ... Ctrl+C to save.

Output file: ~/straightness_log.csv  (override with _output_file:=path)
"""

import os
import csv
import rospy
from finger_analysis.msg import FingerAnalysis


class StraightnessLogger:
    def __init__(self):
        rospy.init_node('straightness_logger', anonymous=True)

        self.output_file = rospy.get_param(
            '~output_file', os.path.expanduser('~/straightness_log.csv'))

        self.rows = []
        self.t0 = None

        rospy.Subscriber('/finger/analysis', FingerAnalysis, self._finger_cb)

        rospy.loginfo(f"Logger ready — will save to: {self.output_file}")
        rospy.loginfo("Point at A, stop pointing, point at B, then Ctrl+C.")

    def _finger_cb(self, msg):
        t = msg.header.stamp.to_sec()
        if self.t0 is None:
            self.t0 = t

        self.rows.append({
            'time_s':      round(t - self.t0, 4),
            'raw':         round(msg.straightness_raw, 6),
            'ema':         round(msg.straightness_ema, 6),
            'one_euro':    round(msg.straightness_one_euro, 6),
            'is_straight': int(msg.is_straight),
        })

    def save(self):
        if not self.rows:
            rospy.logwarn("No data collected — nothing saved.")
            return
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['time_s', 'raw', 'ema', 'one_euro', 'is_straight'])
            writer.writeheader()
            writer.writerows(self.rows)
        rospy.loginfo(f"Saved {len(self.rows)} rows → {self.output_file}")

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    node = StraightnessLogger()
    try:
        node.spin()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        node.save()
