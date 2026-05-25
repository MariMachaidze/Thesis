#!/usr/bin/env python3
"""
analysis.py — Robot Navigation Accuracy analysis for gesture-based teleoperation.

Reads nav_accuracy_min_results.csv (one row per arrived trial) and produces
six publication-quality PNG figures plus aggregated statistics that are
consumed by results.txt.

Conventions
-----------
* All spatial quantities are expressed in centimetres (cm).
* "Success" is defined as a final in-square minimum distance <= 5 cm
  from the user's pointed coordinate. This matches the threshold used
  in the upstream summary and represents the radius inside which the
  Sphero is considered to have reached the user's intended target.
* Latency is the time, in seconds, from t_start (gesture committed) to
  the detection sample at which the minimum distance was achieved.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = '/home/mariam-machaidze/Desktop/thesis/bags/thesis_final/nav_accuracy_min_results.csv'
OUT_DIR = '/home/mariam-machaidze/Desktop/thesis/5_2_robot_navigation_results'
SUCCESS_THRESHOLD_CM = 5.0
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
    df['success'] = df['nav_accuracy_cm'] <= SUCCESS_THRESHOLD_CM
    df['target_error_cm'] = np.hypot(df['sq_x'] - df['point_x_clipped'],
                                     df['sq_y'] - df['point_y_clipped'])
    return df


def fig_error_histogram(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = df['nav_accuracy_cm'].values
    ax.hist(vals, bins=30, color='#3a6ea5', edgecolor='white', alpha=0.9)
    ax.axvline(vals.mean(), color='#c0392b', linestyle='--', linewidth=1.5,
               label=f'Mean = {vals.mean():.2f} cm')
    ax.axvline(np.median(vals), color='#27ae60', linestyle='--', linewidth=1.5,
               label=f'Median = {np.median(vals):.2f} cm')
    ax.axvline(SUCCESS_THRESHOLD_CM, color='#7f8c8d', linestyle=':', linewidth=1.5,
               label=f'Success threshold = {SUCCESS_THRESHOLD_CM:.0f} cm')
    ax.set_xlabel('Navigation accuracy (cm)')
    ax.set_ylabel('Number of trials')
    ax.set_title('Robot navigation accuracy — minimum in-square distance')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT_DIR, 'navigation_error_histogram.png'), dpi=DPI)
    plt.close(fig)


def fig_success_rate_bar(df):
    thresholds = [2.0, 5.0, 8.0]
    rates = [(df['nav_accuracy_cm'] <= t).mean() * 100 for t in thresholds]
    fail_rate = (df['nav_accuracy_cm'] > SUCCESS_THRESHOLD_CM).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [f'≤ {t:.0f} cm' for t in thresholds] + [f'> {SUCCESS_THRESHOLD_CM:.0f} cm (fail)'],
        rates + [fail_rate],
        color=['#2ecc71', '#27ae60', '#16a085', '#c0392b'],
        edgecolor='white',
    )
    for b, v in zip(bars, rates + [fail_rate]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Trials (%)')
    ax.set_ylim(0, 110)
    ax.set_title(f'Navigation success rate across thresholds (n = {len(df)})')
    fig.savefig(os.path.join(OUT_DIR, 'navigation_success_rate_bar.png'), dpi=DPI)
    plt.close(fig)


def fig_latency_histogram(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = df['latency_s'].values
    ax.hist(vals, bins=30, color='#8e44ad', edgecolor='white', alpha=0.9)
    ax.axvline(vals.mean(), color='#c0392b', linestyle='--', linewidth=1.5,
               label=f'Mean = {vals.mean():.2f} s')
    ax.axvline(np.median(vals), color='#27ae60', linestyle='--', linewidth=1.5,
               label=f'Median = {np.median(vals):.2f} s')
    ax.set_xlabel('Navigation latency (s)')
    ax.set_ylabel('Number of trials')
    ax.set_title('Latency from gesture commit (t_start) to min-distance sample')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT_DIR, 'navigation_latency_histogram.png'), dpi=DPI)
    plt.close(fig)


def fig_robot_trajectory_plot(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    df_sorted = df.sort_values(['source', 't_start'])
    for src, sub in df_sorted.groupby('source'):
        ax.plot(sub['point_x_clipped'], sub['point_y_clipped'],
                marker='o', markersize=3, linewidth=0.6, alpha=0.45)
    ax.scatter(df['sq_x'], df['sq_y'], marker='s', s=55,
               facecolors='none', edgecolors='#2c3e50', linewidths=0.8,
               label='Target square centres')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_xlim(0, 85)
    ax.set_ylim(0, 60)
    ax.set_aspect('equal')
    ax.set_title('Per-bag robot navigation paths (pointed coordinates in order of arrival)')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.savefig(os.path.join(OUT_DIR, 'robot_trajectory_plot.png'), dpi=DPI)
    plt.close(fig)


def fig_trajectory_error_over_time(df):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for src, sub in df.groupby('source'):
        sub = sub.sort_values('t_start').reset_index(drop=True)
        t_rel = sub['t_start'] - sub['t_start'].iloc[0]
        ax.plot(t_rel, sub['nav_accuracy_cm'], marker='o', markersize=3,
                linewidth=0.7, alpha=0.55)
    ax.axhline(SUCCESS_THRESHOLD_CM, color='#7f8c8d', linestyle=':', linewidth=1.2,
               label=f'Success threshold = {SUCCESS_THRESHOLD_CM:.0f} cm')
    ax.set_xlabel('Time since first trial in session (s)')
    ax.set_ylabel('Navigation accuracy (cm)')
    ax.set_title('Trajectory error over time within each recording session')
    ax.legend(frameon=False)
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.savefig(os.path.join(OUT_DIR, 'trajectory_error_over_time.png'), dpi=DPI)
    plt.close(fig)


def fig_target_vs_robot_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ok = df[df['success']]
    bad = df[~df['success']]
    ax.scatter(ok['sq_x'], ok['point_x_clipped'], color='#27ae60', s=22,
               alpha=0.7, label=f'x within {SUCCESS_THRESHOLD_CM:.0f} cm')
    ax.scatter(bad['sq_x'], bad['point_x_clipped'], color='#c0392b', s=22,
               alpha=0.7, label=f'x outside {SUCCESS_THRESHOLD_CM:.0f} cm')
    ax.scatter(ok['sq_y'], ok['point_y_clipped'], color='#27ae60', s=22,
               alpha=0.35, marker='^')
    ax.scatter(bad['sq_y'], bad['point_y_clipped'], color='#c0392b', s=22,
               alpha=0.35, marker='^')
    lo, hi = 0, 85
    ax.plot([lo, hi], [lo, hi], color='#2c3e50', linewidth=1.0,
            linestyle='--', label='Ideal (target = robot)')
    ax.set_xlabel('Target coordinate (cm)')
    ax.set_ylabel('Robot-reached coordinate (cm)')
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect('equal')
    ax.set_title('Target vs robot-reached coordinates (circles: x, triangles: y)')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.savefig(os.path.join(OUT_DIR, 'target_vs_robot_scatter.png'), dpi=DPI)
    plt.close(fig)


def summary_stats(df):
    s = {
        'n': len(df),
        'mean_err': df['nav_accuracy_cm'].mean(),
        'std_err': df['nav_accuracy_cm'].std(),
        'med_err': df['nav_accuracy_cm'].median(),
        'min_err': df['nav_accuracy_cm'].min(),
        'max_err': df['nav_accuracy_cm'].max(),
        'mean_lat': df['latency_s'].mean(),
        'std_lat': df['latency_s'].std(),
        'med_lat': df['latency_s'].median(),
        'min_lat': df['latency_s'].min(),
        'max_lat': df['latency_s'].max(),
        'within_2': (df['nav_accuracy_cm'] <= 2.0).mean() * 100,
        'within_5': (df['nav_accuracy_cm'] <= 5.0).mean() * 100,
        'within_8': (df['nav_accuracy_cm'] <= 8.0).mean() * 100,
        'fail_pct': (df['nav_accuracy_cm'] > SUCCESS_THRESHOLD_CM).mean() * 100,
        'sessions': df['source'].nunique(),
        'mean_samples': df['in_square_samples'].mean(),
    }
    return s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    fig_error_histogram(df)
    fig_success_rate_bar(df)
    fig_latency_histogram(df)
    fig_robot_trajectory_plot(df)
    fig_trajectory_error_over_time(df)
    fig_target_vs_robot_scatter(df)
    s = summary_stats(df)
    print('--- Robot Navigation Accuracy summary ---')
    for k, v in s.items():
        print(f'{k:>14}: {v:.3f}' if isinstance(v, float) else f'{k:>14}: {v}')
    print(f'\nWrote 6 PNG figures to: {OUT_DIR}')


if __name__ == '__main__':
    main()
