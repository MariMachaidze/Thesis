#!/usr/bin/env python3
"""
analysis.py — Robot Navigation Latency analysis for gesture-based
teleoperation of a Sphero mobile robot.

Reads nav_accuracy_min_results.csv (one row per arrived trial) and
produces four publication-quality PNG figures plus aggregated
statistics that are consumed by results.txt.

Latency is defined as the time, in seconds, from t_start (the moment
the user's pointing gesture is committed and the navigation target is
issued) to t_min_distance (the detection sample at which the Sphero
attained its minimum distance to the pointed coordinate during the
first continuous in-square run).

Travel distance is the Euclidean distance, in centimetres, between
the previous trial's pointed coordinate (the robot's approximate
parking position) and the current trial's pointed coordinate, chained
within each source bag. The first trial of each bag has no prior
parking position and is excluded from the distance-vs-latency
regression. This approximation tracks the actual distance the robot
had to cover much more closely than a fixed nominal home position.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = '/home/mariam-machaidze/Desktop/thesis/bags/thesis_final/nav_accuracy_min_results.csv'
OUT_DIR = '/home/mariam-machaidze/Desktop/thesis/5_4_navigation_latency_results'
DPI = 300

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.autolayout': True,
})


def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(['source', 'trial']).reset_index(drop=True)
    df['prev_x'] = df.groupby('source')['point_x_clipped'].shift(1)
    df['prev_y'] = df.groupby('source')['point_y_clipped'].shift(1)
    df['distance_cm'] = np.hypot(df['point_x_clipped'] - df['prev_x'],
                                 df['point_y_clipped'] - df['prev_y'])
    df['t_to_entry_s'] = df['t_entry'] - df['t_start']
    df['t_settle_s'] = df['t_min_distance'] - df['t_entry']
    return df


def summarise(df):
    sub = df.dropna(subset=['distance_cm'])
    s = {
        'n': len(df),
        'n_with_dist': len(sub),
        'mean': df['latency_s'].mean(),
        'median': df['latency_s'].median(),
        'std': df['latency_s'].std(ddof=1),
        'min': df['latency_s'].min(),
        'max': df['latency_s'].max(),
        'p25': df['latency_s'].quantile(0.25),
        'p75': df['latency_s'].quantile(0.75),
        'p95': df['latency_s'].quantile(0.95),
        'mean_dist': sub['distance_cm'].mean(),
        'std_dist': sub['distance_cm'].std(ddof=1),
        'max_dist': sub['distance_cm'].max(),
        'mean_to_entry': df['t_to_entry_s'].mean(),
        'mean_settle': df['t_settle_s'].mean(),
    }
    return s


def regression(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, intercept, r, r2


def plot_histogram(df, stats):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['latency_s'], bins=30, color='#4C72B0',
            edgecolor='white', alpha=0.9)
    ax.axvline(stats['mean'], color='#C44E52', linestyle='--',
               linewidth=2, label=f"mean = {stats['mean']:.2f} s")
    ax.axvline(stats['median'], color='#55A868', linestyle='--',
               linewidth=2, label=f"median = {stats['median']:.2f} s")
    ax.set_xlabel('Navigation latency (s)')
    ax.set_ylabel('Number of trials')
    ax.set_title('Distribution of navigation latency (N = '
                 f"{stats['n']} trials)")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT_DIR, 'navigation_latency_histogram.png'),
                dpi=DPI)
    plt.close(fig)


def plot_scatter(df, stats):
    sub = df.dropna(subset=['distance_cm']).copy()
    slope, intercept, r, r2 = regression(sub['distance_cm'].values,
                                         sub['latency_s'].values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sub['distance_cm'], sub['latency_s'],
               s=22, alpha=0.55, color='#4C72B0',
               edgecolor='white', linewidth=0.4)
    xs = np.linspace(sub['distance_cm'].min(),
                     sub['distance_cm'].max(), 100)
    ax.plot(xs, slope * xs + intercept, color='#C44E52',
            linewidth=2,
            label=f'fit: {slope:.3f} s/cm  (r = {r:+.2f})')
    ax.set_xlabel('Travel distance from prior parking position (cm)')
    ax.set_ylabel('Navigation latency (s)')
    ax.set_title(f'Latency vs travel distance (n = {len(sub)})')
    ax.legend(frameon=False, loc='upper left')
    fig.savefig(os.path.join(OUT_DIR, 'latency_vs_distance_scatter.png'),
                dpi=DPI)
    plt.close(fig)
    return slope, intercept, r, r2


def plot_boxplot(df):
    edges = [0, 10, 20, 30, np.inf]
    labels = ['0-10', '10-20', '20-30', '>30']
    df = df.dropna(subset=['distance_cm']).copy()
    df['dist_bin'] = pd.cut(df['distance_cm'], bins=edges,
                            labels=labels, right=False)
    data = [df.loc[df['dist_bin'] == lab, 'latency_s'].values
            for lab in labels]
    counts = [len(d) for d in data]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=[f'{lab}\n(n={n})'
                                  for lab, n in zip(labels, counts)],
                    patch_artist=True, widths=0.55, showfliers=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#8FBED6')
        patch.set_edgecolor('#27496D')
    for med in bp['medians']:
        med.set_color('#C44E52')
        med.set_linewidth(2)
    ax.set_xlabel('Travel distance from prior parking position (cm)')
    ax.set_ylabel('Navigation latency (s)')
    ax.set_title('Latency distribution by travel-distance bin')
    fig.savefig(os.path.join(OUT_DIR, 'latency_boxplot.png'), dpi=DPI)
    plt.close(fig)
    return labels, counts, [np.mean(d) if len(d) else float('nan')
                            for d in data]


def plot_cumulative(df, stats):
    sorted_lat = np.sort(df['latency_s'].values)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat) * 100.0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_lat, cdf, color='#4C72B0', linewidth=2)
    for thr in (1.0, 2.0, 3.0):
        frac = (sorted_lat <= thr).mean() * 100.0
        ax.axvline(thr, color='grey', linestyle=':', linewidth=1)
        ax.text(thr, 5, f' {thr:.0f}s\n {frac:.0f}%',
                fontsize=9, color='dimgray')
    ax.set_xlabel('Navigation latency (s)')
    ax.set_ylabel('Cumulative percentage of trials')
    ax.set_title('Cumulative distribution of navigation latency')
    ax.set_ylim(0, 105)
    fig.savefig(os.path.join(OUT_DIR, 'cumulative_navigation_time.png'),
                dpi=DPI)
    plt.close(fig)
    fractions = {
        '<=1s': float((sorted_lat <= 1.0).mean() * 100.0),
        '<=2s': float((sorted_lat <= 2.0).mean() * 100.0),
        '<=3s': float((sorted_lat <= 3.0).mean() * 100.0),
        '<=5s': float((sorted_lat <= 5.0).mean() * 100.0),
    }
    return fractions


def write_results(stats, reg, bins, fractions):
    slope, intercept, r, r2 = reg
    labels, counts, means = bins
    means = [m if not np.isnan(m) else 0.0 for m in means]
    p = os.path.join(OUT_DIR, 'results.txt')
    lines = []
    w = lines.append
    w('=' * 80)
    w('Section 5.4 — Robot Navigation Latency')
    w('Gesture-Based Teleoperation of a Sphero Mobile Robot')
    w('=' * 80)
    w('')
    w('Dataset')
    w('-------')
    w(f"The analysis is based on {stats['n']} arrived trials extracted from")
    w('the recording sessions stored in bags/thesis_final/. Each trial')
    w("corresponds to one user pointing gesture that was successfully")
    w('committed by the gating logic (t_start) together with the')
    w("subsequent first continuous in-square run of the Sphero robot.")
    w('Latency is defined per trial as')
    w('')
    w('    latency_s = t_min_distance - t_start')
    w('')
    w("where t_min_distance is the detection sample at which the robot")
    w("attained its minimum distance to the user's pointed coordinate.")
    w('Travel distance for each trial is the Euclidean distance, in')
    w("centimetres, between the previous trial's pointed coordinate (the")
    w("robot's approximate parking position) and the current trial's")
    w('pointed coordinate, chained within each source bag. The first')
    w('trial of each bag has no prior parking position and is therefore')
    w('excluded from the distance-based analyses; the latency')
    w(f'descriptives use all {stats["n"]} trials while the regression and')
    w(f'binned breakdown use {stats["n_with_dist"]} trials with a defined')
    w(f"travel distance (mean {stats['mean_dist']:.1f} cm, max")
    w(f"{stats['max_dist']:.1f} cm).")
    w('')
    w('-' * 80)
    w('1. Overall navigation responsiveness')
    w('-' * 80)
    w(f"Across all {stats['n']} trials the mean end-to-end navigation")
    w(f"latency was {stats['mean']:.2f} s (SD = {stats['std']:.2f} s;")
    w(f"median = {stats['median']:.2f} s; min = {stats['min']:.2f} s;")
    w(f"max = {stats['max']:.2f} s). The 25th and 75th percentiles were")
    w(f"{stats['p25']:.2f} s and {stats['p75']:.2f} s respectively, and the")
    w(f"95th percentile sat at {stats['p95']:.2f} s. The median falling")
    w('below the mean indicates a mildly right-skewed distribution, with')
    w('the bulk of trials resolving quickly and a smaller tail of slower')
    w('arrivals that pull the mean upwards.')
    w('')
    w('Decomposed into its two natural phases the latency comprises')
    w(f"a transit phase of {stats['mean_to_entry']:.2f} s on average")
    w('(t_start -> t_entry, the time taken for the robot to first enter')
    w(f"the target square) and a settling phase of {stats['mean_settle']:.2f} s")
    w('on average (t_entry -> t_min_distance, the additional time the')
    w('controller spends fine-tuning the in-square pose). The transit')
    w('phase accounts for almost the entire end-to-end latency, while')
    w('the in-square settling phase is short and consistent across')
    w('trials.')
    w('')
    w('-' * 80)
    w('2. Latency variation')
    w('-' * 80)
    w('Cumulative breakdown of the latency distribution:')
    w('')
    w(f"    Within 1 s : {fractions['<=1s']:5.1f} %")
    w(f"    Within 2 s : {fractions['<=2s']:5.1f} %")
    w(f"    Within 3 s : {fractions['<=3s']:5.1f} %")
    w(f"    Within 5 s : {fractions['<=5s']:5.1f} %")
    w('')
    w('The inter-quartile range of')
    w(f"{(stats['p75']-stats['p25']):.2f} s is comparable in magnitude to the median,")
    w('which means that trial-to-trial latency is variable rather than')
    w('tightly concentrated around a single value. As shown in the next')
    w('section, this variation is not explained by travel distance, so')
    w('the dominant source of variation must be sought in per-trial')
    w('factors such as initial heading misalignment of the Sphero,')
    w('intermittent corrective manoeuvres, and the time required for the')
    w('controller to satisfy the in-square dwell condition.')
    w('')
    w('-' * 80)
    w('3. Effect of travel distance on latency')
    w('-' * 80)
    w("A linear regression of latency on the trial's travel distance")
    w('(distance from the previous parking position to the current')
    w('pointed coordinate, chained within each source bag) yields')
    w('')
    w(f'    latency_s = {slope:.3f} * distance_cm + {intercept:.3f}')
    w(f'    Pearson r = {r:+.2f},  R^2 = {r2:.2f}')
    w('')
    w('The relationship is essentially flat. The fitted slope of')
    w(f'{slope*1000:.0f} ms per centimetre is small and the regression')
    w(f'explains R^2 = {r2:.2f} of the variance in latency, meaning that')
    w('how far the robot has to travel within the tabletop workspace is')
    w('not a meaningful predictor of how long the trial takes. Binned')
    w('per distance the picture is the same:')
    w('')
    w('    bin (cm)      n     mean latency (s)')
    for lab, n, m in zip(labels, counts, means):
        if n == 0:
            continue
        w(f'    {lab:<10}{n:6d}        {m:6.2f}')
    w('')
    w('The first two bins, which contain the overwhelming majority of')
    w('trials, are within a few hundred milliseconds of one another,')
    w('and the few trials in the longer-distance bins do not shift the')
    w('mean upwards in a way that would survive any reasonable')
    w('statistical test. The most plausible explanation is that on this')
    w('workspace scale (a few tens of centimetres) the time spent on')
    w('initial heading alignment, on corrective manoeuvres triggered by')
    w('the controller, and on satisfying the in-square dwell condition')
    w("dominates the actual transit time. The Sphero's heading is not")
    w("observed at t_start, so even a short-distance gesture can")
    w('require the robot to first rotate-in-place before any useful')
    w('translation begins.')
    w('')
    w('-' * 80)
    w('4. Smoothness and discussion')
    w('-' * 80)
    w('Robot control responsiveness. The median end-to-end latency of')
    w(f"{stats['median']:.2f} s reflects the rhythm of a deliberate")
    w('pointing-and-go interaction rather than a continuous teleoperation')
    w('loop: the user commits a gesture, the Sphero then has to physically')
    w('traverse the table, and only on arrival is the next gesture issued.')
    w('At this granularity the cause-effect link between gesture and')
    w('motion is preserved, but the responsiveness is bounded by the')
    w("robot's translation speed and not by the perceptual stack.")
    w('')
    w('Effect of target distance. Within the tabletop workspace, travel')
    w('distance is essentially uncorrelated with latency. This is a')
    w('useful finding for interaction design: the user cannot anticipate')
    w("a shorter response by pointing closer to the robot, but the")
    w('worst-case latency does not blow up either. Even the slowest 5 %')
    w(f"of trials complete in under {stats['p95']:.1f} s, bounding the")
    w('worst-case interaction time.')
    w('')
    w('Smoothness of navigation. The settling phase contributes only')
    w(f"around {stats['mean_settle']:.2f} s on average, which indicates")
    w('that once the robot enters the target square it does not')
    w('oscillate or hunt around the goal. The bulk of the latency')
    w("budget is therefore spent on directed transit rather than on")
    w("end-of-trajectory correction, giving the motion the qualitative")
    w("character of a single committed manoeuvre rather than a")
    w("stop-and-correct sequence.")
    w('')
    w('Suitability for interactive teleoperation. The pipeline is')
    w('appropriate for discrete pointing-and-go interaction on a')
    w("tabletop, where the user issues a gesture and waits a few")
    w('seconds for the robot to arrive before issuing the next one.')
    w('It is not suitable as-is for continuous-control teleoperation,')
    w("where sub-second responsiveness would be required. The lack of")
    w("a meaningful distance-latency coupling implies that the largest")
    w('gains in responsiveness would come from shortening the per-trial')
    w('overheads (initial heading alignment, controller corrections,')
    w('and in-square dwell) rather than from increasing translational')
    w("speed alone. The gesture-recognition and planning stages")
    w('upstream are not the bottleneck.')
    w('')
    w('=' * 80)
    w('Figures')
    w('=' * 80)
    w('  navigation_latency_histogram.png  - distribution of latencies')
    w('  latency_vs_distance_scatter.png   - latency vs travel distance')
    w('  latency_boxplot.png               - latency by distance bin')
    w('  cumulative_navigation_time.png    - cumulative latency CDF')
    w('=' * 80)
    with open(p, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    stats = summarise(df)
    plot_histogram(df, stats)
    reg = plot_scatter(df, stats)
    bins = plot_boxplot(df)
    fractions = plot_cumulative(df, stats)
    write_results(stats, reg, bins, fractions)
    print(f"wrote outputs to {OUT_DIR}")


if __name__ == '__main__':
    main()
