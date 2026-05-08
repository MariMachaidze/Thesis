#!/usr/bin/env python3
"""
analyze_session.py — Post-process a rosbag session from the pointing teleoperation thesis.

Usage:
    python3 analyze_session.py <path_to_bag_file> [--output results.csv]

Requires ROS to be sourced:
    source /opt/ros/one/setup.bash

Trial definition:
    Each time the user's pointing target moves to a NEW grid square, a new trial begins.
    Staying on the same square (even for several seconds) counts as one trial.
    A trial succeeds if the Sphero's center enters the target square before the next trial.

Metrics per trial:
    - sq_x/sq_y        : target square center in cm
    - t_start          : ROS timestamp when user first pointed at this square
    - arrived          : 1 if Sphero entered the square during this trial, else 0
    - latency_s        : seconds from t_start to first arrival (empty if not arrived)
    - robot_x_cm/y_cm  : Sphero position at arrival (or last known in trial window)
"""

import sys
import csv
import math
import argparse
from collections import defaultdict

sys.path.insert(0, '/opt/ros/one/lib/python3/dist-packages')

import numpy as np

try:
    import rosbag
except Exception as e:
    print(f"ERROR importing rosbag: {e}")
    sys.exit(1)

# ── Experiment parameters ─────────────────────────────────────────────────────
PAPER_X_CM = 85.0
PAPER_Y_CM = 60.0
SQUARE_CM  = 10.0
BORDER_CM  = 5.0

# ── Topics ────────────────────────────────────────────────────────────────────
TOPIC_POINTING  = '/pointing/target'
TOPIC_DETECTION = '/sphero/detection'


# ── Grid helpers ──────────────────────────────────────────────────────────────

def build_grid():
    centers = []
    x = BORDER_CM + SQUARE_CM / 2.0
    while x <= PAPER_X_CM - BORDER_CM - SQUARE_CM / 2.0 + 1e-9:
        y = BORDER_CM + SQUARE_CM / 2.0
        while y <= PAPER_Y_CM - BORDER_CM - SQUARE_CM / 2.0 + 1e-9:
            centers.append((round(x, 1), round(y, 1)))
            y += SQUARE_CM
        x += SQUARE_CM
    return centers


def nearest_square(px_cm, py_cm, grid):
    best, best_d = grid[0], float('inf')
    for cx, cy in grid:
        d = math.hypot(px_cm - cx, py_cm - cy)
        if d < best_d:
            best_d, best = d, (cx, cy)
    return best


def in_square(rx, ry, sq_x, sq_y):
    return (abs(rx - sq_x) <= SQUARE_CM / 2.0 and
            abs(ry - sq_y) <= SQUARE_CM / 2.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', help='Path to .bag file')
    ap.add_argument('--output', default='session_results.csv', help='Output CSV filename')
    args = ap.parse_args()

    grid = build_grid()
    print(f"Grid: {len(grid)} squares  "
          f"({PAPER_X_CM:.0f}×{PAPER_Y_CM:.0f} cm paper, "
          f"{SQUARE_CM:.0f} cm squares, {BORDER_CM:.0f} cm border)")

    # ── Load messages ──────────────────────────────────────────────────────────
    pointing_events  = []   # (t_sec, sq_x, sq_y) — one entry per valid pointing msg
    detection_events = []   # (t_sec, rx_cm, ry_cm)

    print(f"Reading {args.bag} …")
    try:
        bag = rosbag.Bag(args.bag, 'r')
    except Exception as e:
        print(f"ERROR opening bag: {e}")
        sys.exit(1)

    with bag:
        for topic, msg, t in bag.read_messages(
                topics=[TOPIC_POINTING, TOPIC_DETECTION]):
            ts = t.to_sec()
            if topic == TOPIC_POINTING and msg.is_valid:
                px = msg.u_normalized * PAPER_X_CM
                py = msg.v_normalized * PAPER_Y_CM
                sq = nearest_square(px, py, grid)
                pointing_events.append((ts, sq[0], sq[1]))
            elif topic == TOPIC_DETECTION:
                detection_events.append((ts, msg.x * PAPER_X_CM, msg.y * PAPER_Y_CM))

    print(f"  Pointing events : {len(pointing_events)} valid messages")
    print(f"  Detection events: {len(detection_events)} messages")

    if not pointing_events:
        print("No pointing events — nothing to analyse.")
        return

    # ── Segment into trials ────────────────────────────────────────────────────
    # New trial starts whenever pointing moves to a different grid square.
    # Staying on the same square (even for seconds) = one trial.
    trials = []   # [{'sq_x', 'sq_y', 't_start', 't_end'}, ...]

    cur_sq  = (pointing_events[0][1], pointing_events[0][2])
    t_start = pointing_events[0][0]

    for ts, sq_x, sq_y in pointing_events[1:]:
        if (sq_x, sq_y) != cur_sq:
            trials.append({'sq_x': cur_sq[0], 'sq_y': cur_sq[1],
                           't_start': t_start, 't_end': ts})
            cur_sq  = (sq_x, sq_y)
            t_start = ts

    # Final trial ends at the last detection or pointing timestamp
    t_bag_end = max(
        detection_events[-1][0] if detection_events else 0,
        pointing_events[-1][0]
    )
    trials.append({'sq_x': cur_sq[0], 'sq_y': cur_sq[1],
                   't_start': t_start, 't_end': t_bag_end})

    print(f"  Trials segmented: {len(trials)}")

    # ── Evaluate each trial ────────────────────────────────────────────────────
    rows    = []
    det_ptr = 0   # monotonically-advancing pointer into detection_events

    for i, trial in enumerate(trials):
        sq_x, sq_y = trial['sq_x'], trial['sq_y']
        t_start    = trial['t_start']
        t_end      = trial['t_end']

        # Advance pointer to t_start
        while det_ptr < len(detection_events) and detection_events[det_ptr][0] < t_start:
            det_ptr += 1

        arrived   = False
        arrival_t = None
        robot_x = robot_y = None

        j = det_ptr
        while j < len(detection_events) and detection_events[j][0] <= t_end:
            ts, rx, ry = detection_events[j]
            robot_x, robot_y = rx, ry   # keep last known position in window
            if not arrived and in_square(rx, ry, sq_x, sq_y):
                arrived   = True
                arrival_t = ts
            j += 1

        latency = round(arrival_t - t_start, 3) if arrived else ''

        rows.append({
            'trial':      i + 1,
            'sq_x':       sq_x,
            'sq_y':       sq_y,
            't_start':    round(t_start, 3),
            'arrived':    int(arrived),
            'latency_s':  latency,
            'robot_x_cm': round(robot_x, 1) if robot_x is not None else '',
            'robot_y_cm': round(robot_y, 1) if robot_y is not None else '',
        })

    # ── Overall summary ────────────────────────────────────────────────────────
    n         = len(rows)
    successes = sum(r['arrived'] for r in rows)
    lats      = [r['latency_s'] for r in rows if r['latency_s'] != '']

    print(f"\n{'='*58}")
    print(f"  RESULTS — {n} trials")
    print(f"{'='*58}")
    print(f"  Success rate    : {successes}/{n} = {100*successes/n:.1f}%")
    if lats:
        print(f"  Arrival latency : {np.mean(lats):.2f} ± {np.std(lats):.2f} s")
    else:
        print(f"  Arrival latency : N/A")
    print(f"{'='*58}\n")

    # ── Per-square breakdown ───────────────────────────────────────────────────
    sq_stats = defaultdict(lambda: {'n': 0, 'ok': 0, 'lats': []})
    for r in rows:
        key = (r['sq_x'], r['sq_y'])
        sq_stats[key]['n']  += 1
        sq_stats[key]['ok'] += r['arrived']
        if r['latency_s'] != '':
            sq_stats[key]['lats'].append(r['latency_s'])

    xs = sorted(set(cx for cx, cy in grid))
    ys = sorted(set(cy for cx, cy in grid))

    print(f"  Per-square breakdown  ({len(grid)} squares, {len(sq_stats)} visited)")
    print(f"  {'x_cm':>5} {'y_cm':>5} {'trials':>7} {'ok':>4} {'rate':>6}  {'avg_s':>6}  {'std_s':>6}")
    print(f"  {'-'*50}")
    for sq_x in xs:
        for sq_y in ys:
            d = sq_stats.get((sq_x, sq_y), {'n': 0, 'ok': 0, 'lats': []})
            if d['n']:
                rate  = f"{100*d['ok']//d['n']:3.0f}%"
                avg_l = f"{np.mean(d['lats']):.2f}" if d['lats'] else "  —"
                std_l = f"{np.std(d['lats']):.2f}"  if d['lats'] else "  —"
            else:
                rate = avg_l = std_l = "  —"
            print(f"  {sq_x:5.0f} {sq_y:5.0f} {d['n']:7}  {d['ok']:4}  {rate:>5}  {avg_l:>6}  {std_l:>6}")
    print()

    # 2-D grid map: ok/trials
    cell_w = 7
    print(f"  Grid map  (ok/trials  ·=never targeted)   y↑  x→")
    hdr = "        " + "".join(f"x={x:2.0f}  " for x in xs)
    print("  " + hdr)
    for y in reversed(ys):
        row = f"  y={y:2.0f}  "
        for x in xs:
            d = sq_stats.get((x, y), {'n': 0, 'ok': 0, 'lats': []})
            cell = f"{d['ok']}/{d['n']}" if d['n'] else "·"
            row += cell.center(cell_w)
        print(row)
    print()

    # ── Write CSV ──────────────────────────────────────────────────────────────
    fields = ['trial', 'sq_x', 'sq_y', 't_start', 'arrived', 'latency_s',
              'robot_x_cm', 'robot_y_cm']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
