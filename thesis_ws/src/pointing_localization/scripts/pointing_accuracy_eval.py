#!/usr/bin/env python3
"""
Pointing accuracy evaluation — gesture-triggered recording.
Order: height (0 → 25 → 50 cm) x target (1→5) x 5 trials.

Workflow per trial:
  1. Script shows which target and trial to do.
  2. Extend index finger and point at the target — recording starts.
  3. Hold as long as you like, then OPEN hand — trial ends.
  4. Results printed → next trial starts automatically.

Results saved to ~/Desktop/thesis/accuracy_<timestamp>.csv
"""

import rospy
import csv
import time
import os
import numpy as np
from finger_analysis.msg import FingerAnalysis
from pointing_localization.msg import PointingTarget

# Ground truth (u, v) normalized
TARGETS = {
    1: (0.100, 0.167),
    2: (0.300, 0.500),
    3: (0.500, 0.833),
    4: (0.700, 0.167),
    5: (0.900, 0.833),
}

HEIGHTS      = [0, 25, 50]   # cm above paper
N_TRIALS     = 5
MIN_SAMPLES  = 8             # minimum samples for a valid trial after trimming
TRIM_START   = 8             # drop first N samples — hand moving into position (~0.25 s)
TRIM_END     = 5             # drop last  N samples — hand moving away           (~0.17 s)


class PointingEvaluator:
    def __init__(self):
        rospy.init_node('pointing_evaluator', anonymous=False)
        self._is_straight = False
        self._collecting  = False
        self._samples     = []

        rospy.Subscriber('/finger/analysis',  FingerAnalysis,  self._finger_cb)
        rospy.Subscriber('/pointing/target',   PointingTarget,  self._target_cb)
        rospy.sleep(0.5)   # let subscribers connect

    # ------------------------------------------------------------------
    def _finger_cb(self, msg):
        self._is_straight = msg.is_straight

    def _target_cb(self, msg):
        if self._collecting and msg.is_valid:
            latency_ms = (rospy.Time.now() - msg.header.stamp).to_sec() * 1000.0
            self._samples.append((msg.u_normalized, msg.v_normalized, latency_ms))

    # ------------------------------------------------------------------
    def _run_trial(self):
        """
        Block until one complete point-then-open gesture is captured.
        Returns the collected samples list.
        """
        while not rospy.is_shutdown():
            print("  → Extend finger and point at the target ...")

            # Wait for finger to become straight
            while not rospy.is_shutdown() and not self._is_straight:
                rospy.sleep(0.01)

            # Finger is straight — start collecting
            self._samples    = []
            self._collecting = True
            print("  RECORDING ... open hand to stop", end='', flush=True)

            # Collect while finger stays straight
            while not rospy.is_shutdown() and self._is_straight:
                rospy.sleep(0.01)

            # Finger opened — stop collecting
            self._collecting = False
            samples = list(self._samples)

            # Trim edges — discard hand moving in/out
            trimmed = samples[TRIM_START : len(samples) - TRIM_END if TRIM_END else None]
            print(f"  ({len(samples)} raw → {len(trimmed)} after trimming edges)")

            if len(trimmed) >= MIN_SAMPLES:
                return trimmed

            print(f"  Too short after trimming (need {MIN_SAMPLES}+), try again.")

        return []

    # ------------------------------------------------------------------
    def run(self):
        ts        = int(time.time())
        path      = os.path.expanduser(f'~/Desktop/thesis/accuracy_{ts}.csv')
        path_raw  = os.path.expanduser(f'~/Desktop/thesis/accuracy_{ts}_raw.csv')

        raw_file  = open(path_raw, 'w', newline='')
        raw_w     = csv.writer(raw_file)
        raw_w.writerow(['target', 'height_cm', 'trial', 'sample_i',
                        'u', 'v', 'latency_ms'])

        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['target', 'height_cm', 'trial',
                        'u_gt', 'v_gt',
                        'u_mean', 'v_mean',
                        'u_std',  'v_std',
                        'n_samples', 'error_mm',
                        'latency_mean_ms', 'latency_std_ms'])

            total = len(TARGETS) * len(HEIGHTS) * N_TRIALS
            done  = 0

            for height in HEIGHTS:
                print(f"\n{'='*56}")
                print(f"  SET HAND HEIGHT TO  {height} cm  ABOVE THE PAPER")
                input(f"  Press ENTER when ready → ")

                for target_id, (u_gt, v_gt) in TARGETS.items():
                    for trial in range(1, N_TRIALS + 1):
                        done += 1
                        print(f"\n  {'─'*50}")
                        print(f"  TARGET {target_id}/5  |  Trial {trial}/{N_TRIALS}  |  {done}/{total} total")
                        print(f"  Height {height} cm  |  u_gt={u_gt:.3f}  v_gt={v_gt:.3f}")

                        samples = self._run_trial()

                        if not samples:
                            w.writerow([target_id, height, trial,
                                        u_gt, v_gt,
                                        '', '', '', '', 0, '', '', ''])
                            f.flush()
                            continue

                        us  = np.array([s[0] for s in samples])
                        vs  = np.array([s[1] for s in samples])
                        ls  = np.array([s[2] for s in samples])
                        u_m = float(us.mean());  v_m = float(vs.mean())
                        u_s = float(us.std());   v_s = float(vs.std())
                        l_m = float(ls.mean());  l_s = float(ls.std())
                        err = float(np.sqrt(((u_m - u_gt) * 850) ** 2 +
                                            ((v_m - v_gt) * 600) ** 2))

                        print(f"  u = {u_m:.4f} ± {u_s:.4f}")
                        print(f"  v = {v_m:.4f} ± {v_s:.4f}")
                        print(f"  error   = {err:.1f} mm")
                        print(f"  latency = {l_m:.1f} ± {l_s:.1f} ms")

                        w.writerow([target_id, height, trial,
                                    u_gt, v_gt,
                                    round(u_m, 4), round(v_m, 4),
                                    round(u_s, 4), round(v_s, 4),
                                    len(samples),  round(err,  1),
                                    round(l_m, 1), round(l_s, 1)])
                        f.flush()

                        # Save raw samples
                        for i, s in enumerate(samples):
                            raw_w.writerow([target_id, height, trial, i,
                                            round(s[0], 5), round(s[1], 5), round(s[2], 2)])
                        raw_file.flush()

        raw_file.close()
        print(f"\n  All done.")
        print(f"  Summary : {path}")
        print(f"  Raw     : {path_raw}\n")


if __name__ == '__main__':
    try:
        PointingEvaluator().run()
    except rospy.ROSInterruptException:
        pass
