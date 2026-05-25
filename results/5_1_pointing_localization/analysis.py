"""
Pointing Localization Accuracy Analysis
=======================================

Statistical and figure-generation script for Chapter 5.1 of the thesis.

Two input files are used:
  - pointing_results.csv     : one row per (target, height, trial) with the
                               aggregated estimate and error.
  - pointing_results_raw.csv : the per-sample stream produced during each
                               trial (used here for the latency histogram).

The pointing workspace is the printed reference sheet, which physically
measures 850 mm (width, u-axis) by 600 mm (height, v-axis). All error
metrics in this script are expressed in millimetres.

Two data corrections are applied before any statistics are computed,
matching the corrections documented in the experiment log:

  - EXCLUDED: T1, h=30 cm, trial=4 and trial=5
      Both trials show the One-Euro filter locking onto a stray
      detection on the opposite side of the workspace (errors > 500 mm
      with very few samples). They are tracking failures, not
      representative localization measurements, and are removed.

  - REASSIGNED: T6 trial=1 at h=30 cm  -> T5 ground truth
                T9 trial=1 at h=30 cm  -> T8 ground truth
      For these two trials the user pointed at the neighbouring target
      by mistake. The measured positions are spatially consistent with
      T5 and T8 respectively, so the rows are kept but the ground-truth
      label and resulting error are recomputed against the correct
      target.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_AGG = os.path.join(HERE, "pointing_results.csv")
CSV_RAW = os.path.join(HERE, "pointing_results_raw.csv")
OUT_DIR = HERE
DPI = 300

# Physical paper dimensions used for the experiment.
WORKSPACE_WIDTH_MM = 850.0   # u-axis (left/right)
WORKSPACE_HEIGHT_MM = 600.0  # v-axis (near/far from camera)

# Ground-truth coordinates of the 9 targets (normalized on the paper)
GT_BY_TARGET = {
    1: (0.206, 0.222), 2: (0.500, 0.222), 3: (0.794, 0.222),
    4: (0.206, 0.500), 5: (0.500, 0.500), 6: (0.794, 0.500),
    7: (0.206, 0.778), 8: (0.500, 0.778), 9: (0.794, 0.778),
}

os.makedirs(OUT_DIR, exist_ok=True)


def euclid_mm(u1, v1, u2, v2):
    """Euclidean distance between two normalized points, in mm."""
    du = (u1 - u2) * WORKSPACE_WIDTH_MM
    dv = (v1 - v2) * WORKSPACE_HEIGHT_MM
    return float(np.sqrt(du * du + dv * dv))


# --------------------------------------------------------------------------
# Load and clean the aggregated data
# --------------------------------------------------------------------------
df = pd.read_csv(CSV_AGG)
n_raw = len(df)

# -- Exclusions (tracking failures) ----------------------------------------
exclude_mask = (
    (df["target"] == 1) & (df["height_cm"] == 30) &
    (df["trial"].isin([4, 5]))
)
df = df.loc[~exclude_mask].copy()

# -- Reassignments (user pointed at neighbouring target) -------------------
reassignments = [
    {"orig": 6, "new": 5, "height": 30, "trial": 1},
    {"orig": 9, "new": 8, "height": 30, "trial": 1},
]
for r in reassignments:
    sel = (
        (df["target"] == r["orig"]) &
        (df["height_cm"] == r["height"]) &
        (df["trial"] == r["trial"])
    )
    if sel.any():
        u_new, v_new = GT_BY_TARGET[r["new"]]
        idx = df.index[sel][0]
        u_est = df.at[idx, "u_mean"]
        v_est = df.at[idx, "v_mean"]
        df.at[idx, "target"] = r["new"]
        df.at[idx, "u_gt"] = u_new
        df.at[idx, "v_gt"] = v_new
        df.at[idx, "error_mm"] = euclid_mm(u_est, v_est, u_new, v_new)

df = df.sort_values(["target", "height_cm", "trial"]).reset_index(drop=True)
n_clean = len(df)

# Convert coordinates to millimetres for plotting
df["x_gt_mm"]  = df["u_gt"]   * WORKSPACE_WIDTH_MM
df["y_gt_mm"]  = df["v_gt"]   * WORKSPACE_HEIGHT_MM
df["x_est_mm"] = df["u_mean"] * WORKSPACE_WIDTH_MM
df["y_est_mm"] = df["v_mean"] * WORKSPACE_HEIGHT_MM

errors = df["error_mm"].values

# --------------------------------------------------------------------------
# A. Overall localization statistics
# --------------------------------------------------------------------------
overall_stats = {
    "n":      int(len(errors)),
    "mean":   float(np.mean(errors)),
    "std":    float(np.std(errors, ddof=1)),
    "min":    float(np.min(errors)),
    "max":    float(np.max(errors)),
    "median": float(np.median(errors)),
}

# --------------------------------------------------------------------------
# B. Error grouped by hand height
# --------------------------------------------------------------------------
height_stats = df.groupby("height_cm")["error_mm"].agg(
    ["mean", "std", "min", "max", "median", "count"]
)

# --------------------------------------------------------------------------
# C. Spatial error analysis across the 3x3 grid
# --------------------------------------------------------------------------
target_stats = df.groupby("target")["error_mm"].agg(["mean", "std", "count"]).reset_index()
spatial_grid = np.zeros((3, 3))
for t in range(1, 10):
    row = (t - 1) // 3
    col = (t - 1) % 3
    spatial_grid[row, col] = target_stats.loc[
        target_stats["target"] == t, "mean"
    ].values[0]

# --------------------------------------------------------------------------
# Latency statistics (from per-sample raw data when available)
# --------------------------------------------------------------------------
try:
    raw = pd.read_csv(CSV_RAW)
    # Drop samples belonging to the excluded trials
    raw = raw.merge(
        df[["target", "height_cm", "trial"]].drop_duplicates(),
        on=["target", "height_cm", "trial"], how="inner"
    )
    latency_samples = raw["latency_ms"].values
except Exception:
    latency_samples = df["latency_mean_ms"].values

latency_stats = {
    "mean":   float(np.mean(latency_samples)),
    "std":    float(np.std(latency_samples, ddof=1)),
    "min":    float(np.min(latency_samples)),
    "max":    float(np.max(latency_samples)),
    "median": float(np.median(latency_samples)),
}

# --------------------------------------------------------------------------
# Common matplotlib styling
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
})

# --------------------------------------------------------------------------
# Figure 1: Overall error histogram
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(errors, bins=20, color="#3a6ea5", edgecolor="black", alpha=0.85)
ax.axvline(overall_stats["mean"], color="red", linestyle="--", linewidth=2,
           label=f"Mean = {overall_stats['mean']:.1f} mm")
ax.axvline(overall_stats["median"], color="orange", linestyle=":", linewidth=2,
           label=f"Median = {overall_stats['median']:.1f} mm")
ax.set_xlabel("Localization Error (mm)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Pointing Localization Errors")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "overall_error_histogram.png"), dpi=DPI)
plt.close()

# --------------------------------------------------------------------------
# Figure 2: Boxplot of error by hand height
# --------------------------------------------------------------------------
heights = sorted(df["height_cm"].unique())
data_by_height = [df.loc[df["height_cm"] == h, "error_mm"].values for h in heights]

fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot(data_by_height, tick_labels=[f"{h} cm" for h in heights],
                patch_artist=True, widths=0.5)
colors = ["#9ec5e8", "#5b9bd5", "#2a5a8a"]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
ax.set_xlabel("Hand Height Above Workspace")
ax.set_ylabel("Localization Error (mm)")
ax.set_title("Localization Error by Hand Height")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hand_height_boxplot.png"), dpi=DPI)
plt.close()

# --------------------------------------------------------------------------
# Figure 3: Bar chart of mean error per hand height
# --------------------------------------------------------------------------
means = [height_stats.loc[h, "mean"] for h in heights]
stds = [height_stats.loc[h, "std"] for h in heights]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(heights))
bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
              edgecolor="black", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([f"{h} cm" for h in heights])
ax.set_xlabel("Hand Height Above Workspace")
ax.set_ylabel("Mean Localization Error (mm)")
ax.set_title("Mean Localization Error per Hand Height")
ax.grid(True, axis="y", alpha=0.3)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{m:.1f}",
            ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hand_height_bar_chart.png"), dpi=DPI)
plt.close()

# --------------------------------------------------------------------------
# Figure 4: Spatial error heatmap over 3x3 grid
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(spatial_grid, cmap="YlOrRd", aspect="equal")
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["Left", "Center", "Right"])
ax.set_yticklabels(["Top", "Middle", "Bottom"])
ax.set_xlabel("Workspace Column")
ax.set_ylabel("Workspace Row")
ax.set_title("Mean Localization Error Across the 3x3 Target Grid")
for i in range(3):
    for j in range(3):
        target_idx = i * 3 + j + 1
        ax.text(j, i, f"T{target_idx}\n{spatial_grid[i, j]:.1f} mm",
                ha="center", va="center",
                color="black", fontsize=10, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Mean Error (mm)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spatial_error_heatmap.png"), dpi=DPI)
plt.close()

# --------------------------------------------------------------------------
# Figure 5: Scatter of estimated vs ground-truth positions
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

gt_unique = df[["x_gt_mm", "y_gt_mm"]].drop_duplicates()
ax.scatter(gt_unique["x_gt_mm"], gt_unique["y_gt_mm"],
           s=200, marker="*", color="black",
           label="Ground-truth targets", zorder=3)

height_colors = {10: "#1f77b4", 30: "#ff7f0e", 50: "#2ca02c"}
for h in heights:
    sub = df[df["height_cm"] == h]
    ax.scatter(sub["x_est_mm"], sub["y_est_mm"],
               s=40, alpha=0.7,
               color=height_colors[h], edgecolor="black", linewidth=0.4,
               label=f"Estimate ({h} cm)")

for _, row in df.iterrows():
    ax.plot([row["x_gt_mm"], row["x_est_mm"]],
            [row["y_gt_mm"], row["y_est_mm"]],
            color="grey", alpha=0.25, linewidth=0.7, zorder=1)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_title("Estimated vs Ground-Truth Pointing Locations")
ax.set_xlim(0, WORKSPACE_WIDTH_MM)
ax.set_ylim(0, WORKSPACE_HEIGHT_MM)
ax.invert_yaxis()  # image-coordinate convention (origin at top-left)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "localization_error_scatter.png"), dpi=DPI)
plt.close()

# --------------------------------------------------------------------------
# Figure 6: Latency distribution
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(latency_samples, bins=30, color="#6aab6a", edgecolor="black", alpha=0.85)
ax.axvline(latency_stats["mean"], color="red", linestyle="--", linewidth=2,
           label=f"Mean = {latency_stats['mean']:.1f} ms")
ax.set_xlabel("Localization Latency (ms)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Pointing Localization Latency")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latency_distribution.png"), dpi=DPI)
plt.close()

# --------------------------------------------------------------------------
# Numeric summary to stdout
# --------------------------------------------------------------------------
print("=" * 60)
print("Pointing Localization Accuracy - Summary (corrected)")
print("=" * 60)
print(f"Raw rows   : {n_raw}")
print(f"After corr : {n_clean} (excluded 2 tracking failures, "
      f"reassigned 2 mislabelled trials)")
print(f"Mean error   : {overall_stats['mean']:.2f} mm")
print(f"Std  error   : {overall_stats['std']:.2f} mm")
print(f"Median error : {overall_stats['median']:.2f} mm")
print(f"Min  / Max   : {overall_stats['min']:.2f} / {overall_stats['max']:.2f} mm")
print()
print("Error by hand height:")
print(height_stats.round(2))
print()
print("Error by target (3x3 grid):")
print(target_stats.round(2))
print()
print(f"Latency (per-sample, N={len(latency_samples)}): "
      f"mean={latency_stats['mean']:.1f} ms, "
      f"std={latency_stats['std']:.1f} ms, "
      f"range=[{latency_stats['min']:.1f}, {latency_stats['max']:.1f}] ms")
print()
print(f"Outputs saved to: {OUT_DIR}")
