#!/usr/bin/env python3
"""
Offline analysis of a straightness_log.csv file.

Usage:
  python3 analyze_straightness.py [path/to/straightness_log.csv]
  (defaults to ~/straightness_log.csv)

Produces:
  - Terminal: stats for raw, EMA and One Euro
  - Plot: raw scatter + EMA line + One Euro line + threshold lines + is_straight shading
  - PNG: saved next to the CSV file
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Threshold values — keep in sync with thesis_system.launch ────────────────
ON_THRESHOLD  = 0.65
OFF_THRESHOLD = 0.45


def load_csv(path):
    times, raws, emas, one_euros, states = [], [], [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        has_one_euro = 'one_euro' in reader.fieldnames
        for row in reader:
            times.append(float(row['time_s']))
            raws.append(float(row['raw']) if row['raw'] else None)
            emas.append(float(row['ema']))
            one_euros.append(float(row['one_euro']) if has_one_euro else None)
            states.append(int(row['is_straight']))
    return times, raws, emas, one_euros, states


def print_stats(times, raws, emas, one_euros, states):
    valid_raws = [r for r in raws if r is not None]
    valid_oes  = [v for v in one_euros if v is not None]
    duration   = times[-1] - times[0] if len(times) > 1 else 0
    dt         = duration / max(len(times) - 1, 1)

    print()
    print(f"  Duration        : {duration:.1f} s   ({len(times)} frames, ~{1/dt:.0f} Hz)")
    print(f"  Frames w/ raw   : {len(valid_raws)} / {len(times)}")
    print()
    if valid_raws:
        print(f"  RAW       mean={np.mean(valid_raws):.3f}  std={np.std(valid_raws):.3f}"
              f"  min={np.min(valid_raws):.3f}  max={np.max(valid_raws):.3f}")
    print(f"  EMA       mean={np.mean(emas):.3f}  std={np.std(emas):.3f}"
          f"  min={np.min(emas):.3f}  max={np.max(emas):.3f}")
    if valid_oes:
        print(f"  One Euro  mean={np.mean(valid_oes):.3f}  std={np.std(valid_oes):.3f}"
              f"  min={np.min(valid_oes):.3f}  max={np.max(valid_oes):.3f}")

    # Lag: frames EMA/OE stays above OFF_THRESHOLD after raw drops below it
    def lag_analysis(signal, label):
        lags = []
        for i in range(len(raws) - 1):
            if raws[i] is not None and raws[i] < OFF_THRESHOLD:
                j = i
                while j < len(signal) and signal[j] > OFF_THRESHOLD:
                    j += 1
                if j > i:
                    lags.append(j - i)
        if lags:
            print(f"\n  {label} lag (raw<{OFF_THRESHOLD} → {label}<{OFF_THRESHOLD}):")
            print(f"    max  = {max(lags)} frames  (~{max(lags)*dt:.2f} s)")
            print(f"    mean = {np.mean(lags):.1f} frames  (~{np.mean(lags)*dt:.2f} s)")
        else:
            print(f"\n  {label}: never lagged above {OFF_THRESHOLD} during raw dips")

    lag_analysis(emas, "EMA")
    if valid_oes:
        lag_analysis(valid_oes, "One Euro")

    transitions  = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
    pct_straight = 100 * sum(states) / len(states)
    print(f"\n  is_straight transitions : {transitions}")
    print(f"  Time spent straight     : {pct_straight:.1f}%")
    print()


def plot(times, raws, emas, one_euros, states, out_png):
    fig, ax = plt.subplots(figsize=(15, 6))

    # Shade is_straight=True regions
    in_seg, seg_start = False, None
    for t, s in zip(times, states):
        if s and not in_seg:
            seg_start, in_seg = t, True
        elif not s and in_seg:
            ax.axvspan(seg_start, t, alpha=0.13, color='limegreen')
            in_seg = False
    if in_seg:
        ax.axvspan(seg_start, times[-1], alpha=0.13, color='limegreen')

    # Raw scatter
    raw_t = [t for t, r in zip(times, raws) if r is not None]
    raw_v = [r for r in raws if r is not None]
    ax.scatter(raw_t, raw_v, s=7, color='steelblue', alpha=0.45, zorder=3, label='raw')

    # EMA line
    ax.plot(times, emas, color='darkorange', linewidth=2.0, zorder=4, label='EMA')

    # One Euro line (only if data present and non-zero)
    valid_oes = [v for v in one_euros if v is not None]
    if valid_oes and any(v > 0 for v in valid_oes):
        oe_t = [t for t, v in zip(times, one_euros) if v is not None]
        oe_v = [v for v in one_euros if v is not None]
        ax.plot(oe_t, oe_v, color='mediumvioletred', linewidth=2.0,
                linestyle='-', zorder=5, label='One Euro')

    # Threshold lines
    ax.axhline(ON_THRESHOLD,  color='green', linestyle='--', linewidth=1.2,
               label=f'on-threshold  ({ON_THRESHOLD})')
    ax.axhline(OFF_THRESHOLD, color='red',   linestyle='--', linewidth=1.2,
               label=f'off-threshold ({OFF_THRESHOLD})')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Straightness score', fontsize=12)
    ax.set_title('Raw vs EMA vs One Euro  —  A → gap → B', fontsize=13)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(True, alpha=0.3)

    straight_patch = mpatches.Patch(color='limegreen', alpha=0.4, label='is_straight = True')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [straight_patch], loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"  Plot saved → {out_png}")
    plt.show()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/straightness_log.csv')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    times, raws, emas, one_euros, states = load_csv(path)
    print_stats(times, raws, emas, one_euros, states)

    out_png = os.path.splitext(path)[0] + '.png'
    plot(times, raws, emas, one_euros, states, out_png)


if __name__ == '__main__':
    main()
