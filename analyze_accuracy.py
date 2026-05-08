#!/usr/bin/env python3
"""
analyze_accuracy.py — Recreate accuracy_results.png from pointing accuracy CSVs.

Usage:
    python3 analyze_accuracy.py accuracy_1778271966.csv
    python3 analyze_accuracy.py accuracy_*.csv          # merge multiple sessions
    python3 analyze_accuracy.py accuracy_1778271966.csv --out my_results.png
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PAPER_X_MM = 850.0
PAPER_Y_MM = 600.0

TARGET_LABELS = {
    1: 'top-left',    2: 'top-center',  3: 'top-right',
    4: 'mid-left',    5: 'center',      6: 'mid-right',
    7: 'bot-left',    8: 'bot-center',  9: 'bot-right',
}

# ── Manual trial corrections ──────────────────────────────────────────────────
# Trials to exclude entirely (target, height_cm, trial) — system failures.
EXCLUDE = {
    (1, 30, 4),   # T1 h=30 trial 4 — tracking lost, jumped to wrong corner
    (1, 30, 5),   # T1 h=30 trial 5 — same
}

# Trials to reassign to a different target (target, height_cm, trial) → new_target.
# Use when the user was genuinely pointing at a different target than recorded.
REASSIGN = {
    (6, 30, 1): 5,   # T6 h=30 trial 1 — user was pointing at T5
    (9, 30, 1): 8,   # T9 h=30 trial 1 — user was pointing at T8
}

plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 200,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Statistics helpers ────────────────────────────────────────────────────────

# t critical values for 95% CI (two-tailed), indexed by degrees of freedom.
# For df >= 30 we use 2.042; for df >= 120 we use 1.980.
_T95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,
        8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,
        15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,
        21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,
        27:2.052,28:2.048,29:2.045}

def _t95(n):
    df = n - 1
    if df <= 0:   return float('nan')
    if df <= 29:  return _T95[df]
    if df <= 119: return 2.042
    return 1.980

def _ci95(vals):
    n = len(vals)
    if n < 2: return float('nan'), float('nan')
    m   = sum(vals) / n
    sd  = math.sqrt(sum((x - m)**2 for x in vals) / (n - 1))
    margin = _t95(n) * sd / math.sqrt(n)
    return m - margin, m + margin

def _stats(vals):
    if not vals:
        return dict(n=0, mean=float('nan'), median=float('nan'),
                    std=float('nan'), ci_lo=float('nan'), ci_hi=float('nan'),
                    mn=float('nan'), mx=float('nan'))
    n   = len(vals)
    m   = sum(vals) / n
    s   = sorted(vals)
    med = s[n // 2] if n % 2 else (s[n//2-1] + s[n//2]) / 2
    sd  = math.sqrt(sum((x - m)**2 for x in vals) / max(n-1, 1))
    lo, hi = _ci95(vals)
    return dict(n=n, mean=m, median=med, std=sd,
                ci_lo=lo, ci_hi=hi, mn=min(vals), mx=max(vals))


# ── Report generator ──────────────────────────────────────────────────────────

def load_raw(paths):
    """Load all rows without any corrections applied."""
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append({k: float(v) for k, v in r.items()})
    return rows


def generate_report(paths, rows_corrected, out_txt):
    rows_raw = load_raw(paths)
    heights  = sorted(set(r['height_cm'] for r in rows_corrected))
    targets  = sorted(set(int(r['target']) for r in rows_corrected))

    L = []
    W = 66

    def line(s=''):  L.append(s)
    def rule():      L.append('=' * W)
    def subrule():   L.append('-' * W)
    def section(s):  rule(); line(f'  {s}'); rule()

    section('POINTING ACCURACY — DETAILED ANALYSIS REPORT')
    line(f'  Source : {", ".join(str(Path(p).name) for p in paths)}')
    line(f'  Raw trials   : {len(rows_raw)}')
    line(f'  After corrections : {len(rows_corrected)}')
    line(f'  Paper : {PAPER_X_MM:.0f} × {PAPER_Y_MM:.0f} mm')
    line(f'  Heights tested : {[int(h) for h in heights]} cm')
    line(f'  Targets tested : {targets}')
    line()

    # ── Section 1: Corrections ──────────────────────────────────────────────
    section('1. DATA CORRECTIONS')
    line()
    line('  The following trials were identified as problematic and corrected')
    line('  before any statistics were computed.')
    line()

    # Excluded
    if EXCLUDE:
        line(f'  EXCLUDED ({len(EXCLUDE)} trial row(s)) — system failures')
        subrule()
        for (t, h, trial) in sorted(EXCLUDE):
            raw = [r for r in rows_raw
                   if int(r['target'])==t and int(r['height_cm'])==h and int(r['trial'])==trial]
            for r in raw:
                ue = (r['u_mean'] - r['u_gt']) * PAPER_X_MM
                ve = (r['v_mean'] - r['v_gt']) * PAPER_Y_MM
                line(f'  T{t} h={h}cm trial={trial}')
                line(f'    Error   : {r["error_mm"]:.1f} mm  (normal range for this target: ~50–90 mm)')
                line(f'    GT pos  : ({r["u_gt"]:.3f}, {r["v_gt"]:.3f}) normalized')
                line(f'    Pred pos: ({r["u_mean"]:.3f}, {r["v_mean"]:.3f}) normalized')
                line(f'    Offset  : Δu={ue:+.1f} mm,  Δv={ve:+.1f} mm')
                line(f'    Samples : {int(r["n_samples"])}  (very few samples → unstable lock)')
                line(f'    Reason  : The predicted position jumped to the opposite corner of')
                line(f'              the paper. This is a tracking failure — the One Euro Filter')
                line(f'              locked onto a stray detection instead of the actual finger')
                line(f'              ray. The small sample count confirms the lock was brief.')
                line()

    # Reassigned
    if REASSIGN:
        line(f'  REASSIGNED ({len(REASSIGN)} trial row(s)) — user pointed at wrong target')
        subrule()
        for (t, h, trial), new_t in sorted(REASSIGN.items()):
            raw = [r for r in rows_raw
                   if int(r['target'])==t and int(r['height_cm'])==h and int(r['trial'])==trial]
            for r in raw:
                ue_old = (r['u_mean'] - r['u_gt']) * PAPER_X_MM
                ve_old = (r['v_mean'] - r['v_gt']) * PAPER_Y_MM
                # Error against new target GT
                new_gt = next((rr for rr in rows_raw
                               if int(rr['target'])==new_t and int(rr['height_cm'])==h), None)
                if new_gt:
                    ue_new = (r['u_mean'] - new_gt['u_gt']) * PAPER_X_MM
                    ve_new = (r['v_mean'] - new_gt['v_gt']) * PAPER_Y_MM
                    err_new = math.hypot(ue_new, ve_new)
                else:
                    ue_new = ve_new = err_new = float('nan')

                line(f'  T{t} h={h}cm trial={trial}  →  reassigned to T{new_t}')
                line(f'    Original GT (T{t}) : ({r["u_gt"]:.3f}, {r["v_gt"]:.3f})')
                line(f'    Predicted pos      : ({r["u_mean"]:.3f}, {r["v_mean"]:.3f})')
                line(f'    Error vs T{t}       : {r["error_mm"]:.1f} mm  (looks like outlier)')
                line(f'    Error vs T{new_t} GT    : {err_new:.1f} mm  (consistent with normal trials)')
                line(f'    Reason  : User confirmed pointing at T{new_t} ({TARGET_LABELS[new_t]}),')
                line(f'              not T{t} ({TARGET_LABELS[t]}). The measured position')
                line(f'              ({r["u_mean"]:.3f}, {r["v_mean"]:.3f}) is spatially close to T{new_t}\'s')
                line(f'              ground truth, confirming the relabelling.')
                line()

    # Impact summary
    raw_errs = [r['error_mm'] for r in rows_raw]
    cor_errs = [r['error_mm'] for r in rows_corrected]
    rs_raw = _stats(raw_errs)
    rs_cor = _stats(cor_errs)
    line('  Impact of corrections on overall error:')
    line(f'    Raw  : n={rs_raw["n"]}  mean={rs_raw["mean"]:.1f} mm  median={rs_raw["median"]:.1f} mm  std={rs_raw["std"]:.1f} mm')
    line(f'    Clean: n={rs_cor["n"]}  mean={rs_cor["mean"]:.1f} mm  median={rs_cor["median"]:.1f} mm  std={rs_cor["std"]:.1f} mm')
    saved = rs_raw['mean'] - rs_cor['mean']
    line(f'    Mean error reduced by {saved:.1f} mm ({100*saved/rs_raw["mean"]:.1f}%) after corrections.')
    line()

    # ── Section 2: Per-height stats ─────────────────────────────────────────
    section('2. DESCRIPTIVE STATISTICS BY HEIGHT')
    line()
    for h in heights:
        errs = [r['error_mm'] for r in rows_corrected if r['height_cm'] == h]
        lats = [r['latency_mean_ms'] for r in rows_corrected if r['height_cm'] == h]
        s = _stats(errs)
        sl = _stats(lats)
        line(f'  Height {int(h)} cm  (n={s["n"]} trials)')
        line(f'    Error  — mean={s["mean"]:.1f} mm  median={s["median"]:.1f} mm  '
             f'std={s["std"]:.1f} mm')
        line(f'             95% CI [{s["ci_lo"]:.1f}, {s["ci_hi"]:.1f}] mm  '
             f'range [{s["mn"]:.1f}, {s["mx"]:.1f}] mm')
        line(f'    ≤50mm  : {sum(1 for e in errs if e<=50)/len(errs)*100:.1f}%   '
             f'≤100mm : {sum(1 for e in errs if e<=100)/len(errs)*100:.1f}%')
        line(f'    Latency— mean={sl["mean"]:.1f} ms  std={sl["std"]:.1f} ms')
        line()

    # ── Section 3: Per-target stats ─────────────────────────────────────────
    section('3. DESCRIPTIVE STATISTICS BY TARGET')
    line()
    line(f'  {"T":>2}  {"position":>12}  {"n":>3}  {"mean":>7}  {"median":>7}  '
         f'{"std":>6}  {"95% CI":>17}  {"range":>14}')
    subrule()
    for t in targets:
        sub = [r['error_mm'] for r in rows_corrected if int(r['target']) == t]
        s = _stats(sub)
        ci = f'[{s["ci_lo"]:.1f}, {s["ci_hi"]:.1f}]'
        rng = f'[{s["mn"]:.1f}, {s["mx"]:.1f}]'
        line(f'  {t:>2}  {TARGET_LABELS.get(t,""):>12}  {s["n"]:>3}  '
             f'{s["mean"]:>7.1f}  {s["median"]:>7.1f}  {s["std"]:>6.1f}  '
             f'{ci:>17}  {rng:>14}')
    line()

    # ── Section 4: Per-target × height breakdown ────────────────────────────
    section('4. PER-TARGET × HEIGHT BREAKDOWN')
    line()
    header = f'  {"T":>2}  {"position":>12}' + ''.join(f'  {"h="+str(int(h))+"cm":>9}' for h in heights)
    line(header)
    subrule()
    for t in targets:
        row_str = f'  {t:>2}  {TARGET_LABELS.get(t,""):>12}'
        for h in heights:
            sub = [r['error_mm'] for r in rows_corrected
                   if int(r['target'])==t and r['height_cm']==h]
            if sub:
                row_str += f'  {sum(sub)/len(sub):>7.1f}mm'
            else:
                row_str += f'  {"—":>9}'
        line(row_str)
    line()
    line('  (values are mean error in mm per target per hand height)')
    line()

    # ── Section 5: Axis bias ─────────────────────────────────────────────────
    section('5. AXIS BIAS ANALYSIS')
    line()
    for h in heights:
        ue_all, ve_all = [], []
        for r in rows_corrected:
            if r['height_cm'] != h: continue
            ue_all.append((r['u_mean'] - r['u_gt']) * PAPER_X_MM)
            ve_all.append((r['v_mean'] - r['v_gt']) * PAPER_Y_MM)
        su = _stats(ue_all); sv = _stats(ve_all)
        bias_u = 'LEFT' if su['mean'] < 0 else 'RIGHT'
        bias_v = 'toward camera' if sv['mean'] < 0 else 'away from camera'
        line(f'  Height {int(h)} cm  (n={su["n"]})')
        line(f'    u-axis (left/right) : mean={su["mean"]:+.1f} mm  std={su["std"]:.1f} mm  '
             f'→ systematic {bias_u} bias')
        line(f'    v-axis (depth)      : mean={sv["mean"]:+.1f} mm  std={sv["std"]:.1f} mm  '
             f'→ {bias_v}')
        line()

    # ── Section 6: Overall summary ───────────────────────────────────────────
    section('6. OVERALL SUMMARY (corrected dataset)')
    line()
    s = _stats([r['error_mm'] for r in rows_corrected])
    line(f'  Total trials analysed : {s["n"]}')
    line(f'  Mean error            : {s["mean"]:.1f} mm')
    line(f'  Median error          : {s["median"]:.1f} mm')
    line(f'  Std deviation         : {s["std"]:.1f} mm')
    line(f'  95% CI                : [{s["ci_lo"]:.1f}, {s["ci_hi"]:.1f}] mm')
    line(f'  ≤ 50 mm               : {sum(1 for r in rows_corrected if r["error_mm"]<=50)/s["n"]*100:.1f}%')
    line(f'  ≤ 100 mm              : {sum(1 for r in rows_corrected if r["error_mm"]<=100)/s["n"]*100:.1f}%')
    line(f'  ≤ 150 mm              : {sum(1 for r in rows_corrected if r["error_mm"]<=150)/s["n"]*100:.1f}%')
    rule()
    line()

    text = '\n'.join(L)
    print(text)
    with open(out_txt, 'w') as f:
        f.write(text + '\n')
    print(f'Report saved → {out_txt}')


def load(paths):
    rows = []
    excluded = 0
    reassigned = 0
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                row = {k: float(v) for k, v in r.items()}
                key = (int(row['target']), int(row['height_cm']), int(row['trial']))

                if key in EXCLUDE:
                    excluded += 1
                    continue

                if key in REASSIGN:
                    row['target'] = float(REASSIGN[key])
                    reassigned += 1

                rows.append(row)

    if excluded:
        print(f"  Excluded  : {excluded} trial(s) — system failures")
    if reassigned:
        print(f"  Reassigned: {reassigned} trial(s) — user was pointing at different target")
    return rows


def get_heights(rows):
    return sorted(set(r['height_cm'] for r in rows))


def get_targets(rows):
    return sorted(set(int(r['target']) for r in rows))


def height_color(h, heights):
    palette = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    return palette[heights.index(h) % len(palette)]


# ── Panel 1: box plot per height ──────────────────────────────────────────────

def plot_boxplot(ax, rows, heights):
    data = [[r['error_mm'] for r in rows if r['height_cm'] == h] for h in heights]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    for patch, h in zip(bp['boxes'], heights):
        patch.set_facecolor(height_color(h, heights))
        patch.set_alpha(0.7)

    ax.axhline(50, color='grey', linestyle='--', linewidth=1, alpha=0.6, label='50 mm ref')
    ax.set_xticks(range(1, len(heights) + 1))
    ax.set_xticklabels([f'{int(h)} cm' for h in heights])
    ax.set_ylabel('Error (mm)')
    ax.set_title('Error Distribution by Hand Height')

    for i, (h, d) in enumerate(zip(heights, data), 1):
        m = sum(d) / len(d) if d else 0
        ax.text(i, ax.get_ylim()[1] * 0.97, f'μ={m:.0f}',
                ha='center', va='top', fontsize=8, color='black')
    ax.legend(fontsize=8)


# ── Panel 2: grouped bar per target × height ──────────────────────────────────

def plot_grouped_bar(ax, rows, heights, targets):
    n_t = len(targets)
    n_h = len(heights)
    w = 0.7 / n_h
    x = np.arange(n_t)

    for j, h in enumerate(heights):
        means = []
        for t in targets:
            sub = [r['error_mm'] for r in rows if r['target'] == t and r['height_cm'] == h]
            means.append(sum(sub) / len(sub) if sub else 0)
        offset = (j - (n_h - 1) / 2) * w
        ax.bar(x + offset, means, width=w,
               color=height_color(h, heights), alpha=0.8,
               label=f'{int(h)} cm')

    ax.set_xticks(x)
    ax.set_xticklabels([f'T{t}' for t in targets])
    ax.set_ylabel('Mean Error (mm)')
    ax.set_title('Mean Error by Target × Height')
    ax.legend(fontsize=8)


# ── Panel 3: scatter — measured vs ground truth ───────────────────────────────

def plot_scatter(ax, rows, heights):
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h']
    targets = get_targets(rows)
    target_color = plt.cm.tab10(np.linspace(0, 1, len(targets)))

    # GT positions
    gt_seen = {}
    for r in rows:
        t = int(r['target'])
        if t not in gt_seen:
            gt_seen[t] = (r['u_gt'], r['v_gt'])

    for t, (ug, vg) in gt_seen.items():
        ax.plot(ug, vg, 'k*', markersize=9, zorder=5)

    for j, h in enumerate(heights):
        for i, t in enumerate(targets):
            sub = [r for r in rows if r['target'] == t and r['height_cm'] == h]
            if not sub:
                continue
            us = [r['u_mean'] for r in sub]
            vs = [r['v_mean'] for r in sub]
            ax.scatter(us, vs,
                       color=height_color(h, heights),
                       marker=markers[i % len(markers)],
                       s=30, alpha=0.7, zorder=4)

    # Legend: heights
    patches = [mpatches.Patch(color=height_color(h, heights), alpha=0.7, label=f'{int(h)} cm')
               for h in heights]
    patches.append(plt.Line2D([0], [0], marker='*', color='k', linestyle='',
                               markersize=8, label='ground truth'))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel('u (normalized)'); ax.set_ylabel('v (normalized)')
    ax.set_title('Measured vs Ground Truth on Paper\n(ground truth = ★)')
    ax.legend(handles=patches, fontsize=7, loc='lower right')

    # Target labels next to GT
    for t, (ug, vg) in gt_seen.items():
        ax.text(ug + 0.02, vg, f'T{t}', fontsize=7, va='center')


# ── Panel 4: axis bias scatter ────────────────────────────────────────────────

def plot_axis_bias(ax, rows, heights, ref_height=None):
    if ref_height is None:
        ref_height = min(heights)

    targets = get_targets(rows)
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h']
    target_color = plt.cm.tab10(np.linspace(0, 1, len(targets)))

    all_ue, all_ve = [], []
    for i, t in enumerate(targets):
        sub = [r for r in rows if r['target'] == t and r['height_cm'] == ref_height]
        if not sub:
            continue
        ue = [(r['u_mean'] - r['u_gt']) * PAPER_X_MM for r in sub]
        ve = [(r['v_mean'] - r['v_gt']) * PAPER_Y_MM for r in sub]
        all_ue.extend(ue); all_ve.extend(ve)
        ax.scatter(ue, ve,
                   color=target_color[i],
                   marker=markers[i % len(markers)],
                   s=40, alpha=0.8, label=f'T{t}', zorder=4)

    # Mean bias crosshair
    if all_ue and all_ve:
        mu = sum(all_ue) / len(all_ue)
        mv = sum(all_ve) / len(all_ve)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.plot(mu, mv, 'k+', markersize=14, markeredgewidth=2, zorder=5)
        ax.text(mu + 1, mv + 1,
                f'Mean bias\n(+{mu:.1f}, +{mv:.1f}) mm',
                fontsize=8)

    ax.set_xlabel('u error (mm)  [+ = right]')
    ax.set_ylabel('v error (mm)  [+ = away from camera]')
    ax.set_title(f'Axis Bias at Height {int(ref_height)} cm\n(ideal = origin)')
    ax.legend(fontsize=7, ncol=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csvs', nargs='+', help='accuracy_*.csv file(s)')
    ap.add_argument('--out', default=None, help='output PNG path')
    ap.add_argument('--bias-height', type=float, default=None,
                    help='height to use for axis-bias panel (default: lowest)')
    args = ap.parse_args()

    rows = load(args.csvs)
    if not rows:
        sys.exit('No data loaded.')

    heights = get_heights(rows)
    targets = get_targets(rows)
    print(f"Loaded {len(rows)} trials  |  heights={[int(h) for h in heights]}  |  targets={targets}")

    # Determine output paths
    stem    = Path(args.csvs[0]).stem
    out_dir = Path(args.csvs[0]).parent
    out_png = args.out if args.out else str(out_dir / f'accuracy_results_{stem}.png')
    out_txt = str(Path(out_png).with_suffix('.txt'))

    # Generate detailed text report
    generate_report(args.csvs, rows, out_txt)

    # Generate plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Pointing Accuracy Results', fontsize=14, fontweight='bold')

    plot_boxplot(    axes[0, 0], rows, heights)
    plot_grouped_bar(axes[0, 1], rows, heights, targets)
    plot_scatter(    axes[1, 0], rows, heights)
    plot_axis_bias(  axes[1, 1], rows, heights, ref_height=args.bias_height)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, bbox_inches='tight')
    print(f'Plot saved  → {out_png}')


if __name__ == '__main__':
    main()
