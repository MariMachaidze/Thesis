"""
Section 5.5 -- Navigation Success Rate analysis
Success criterion (from thesis_ws/analyze_session.py):
    A trial succeeds if the Sphero's centre eventually enters the
    requested 10x10 cm target square after t_start. The per-trial
    `arrived` flag in the aggregated session results is the
    operational success label.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = (Path(__file__).resolve().parent.parent
            / "bags" / "thesis_final" / "aggregate_results_deduped.csv")
OUT_DIR = Path(__file__).resolve().parent

SQUARE_CM = 10.0
WORKSPACE_X = (5.0, 80.0)
WORKSPACE_Y = (5.0, 55.0)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["arrived"] = df["arrived"].astype(bool)
    df["robot_x_cm"] = pd.to_numeric(df["robot_x_cm"], errors="coerce")
    df["robot_y_cm"] = pd.to_numeric(df["robot_y_cm"], errors="coerce")
    df["centre_err_cm"] = np.hypot(df["robot_x_cm"] - df["sq_x"],
                                   df["robot_y_cm"] - df["sq_y"])
    return df


def assign_region(row):
    x, y = row["sq_x"], row["sq_y"]
    x_third = (WORKSPACE_X[1] - WORKSPACE_X[0]) / 3.0
    y_half = (WORKSPACE_Y[1] - WORKSPACE_Y[0]) / 2.0
    if x < WORKSPACE_X[0] + x_third:
        col = "Left"
    elif x < WORKSPACE_X[0] + 2 * x_third:
        col = "Centre"
    else:
        col = "Right"
    row_label = "Near" if y < WORKSPACE_Y[0] + y_half else "Far"
    return f"{row_label}-{col}"


REGION_ORDER = [
    "Near-Left", "Near-Centre", "Near-Right",
    "Far-Left", "Far-Centre", "Far-Right",
]


def plot_overall_success_rate(df: pd.DataFrame) -> None:
    n_total = len(df)
    n_succ = int(df["arrived"].sum())
    n_fail = n_total - n_succ
    succ_pct = 100.0 * n_succ / n_total
    fail_pct = 100.0 - succ_pct

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Arrived (inside 10x10 cm square)", "Missed"],
        [succ_pct, fail_pct],
        color=["#2a9d8f", "#e76f51"],
        edgecolor="black",
        width=0.6,
    )
    for bar, pct, count in zip(bars, [succ_pct, fail_pct], [n_succ, n_fail]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.5,
            f"{pct:.1f}%\n(n={count})",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Proportion of trials (%)")
    ax.set_title(f"Overall navigation success rate (N = {n_total} trials)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overall_success_rate_bar.png")
    plt.close(fig)


def plot_success_by_region(df: pd.DataFrame) -> None:
    df = df.copy()
    df["region"] = df.apply(assign_region, axis=1)
    grouped = df.groupby("region")["arrived"].agg(["sum", "count"])
    grouped["rate"] = 100.0 * grouped["sum"] / grouped["count"]
    grouped = grouped.reindex(REGION_ORDER)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(grouped["rate"] / 100.0)
    bars = ax.bar(grouped.index, grouped["rate"], color=colors, edgecolor="black")
    overall = 100.0 * df["arrived"].mean()
    ax.axhline(overall, color="black", linestyle="--", linewidth=1,
               label=f"Overall {overall:.1f}%")
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        if pd.isna(row["count"]):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.2,
            f"{row['rate']:.1f}%\n({int(row['sum'])}/{int(row['count'])})",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Workspace region (target square location)")
    ax.set_title("Navigation success rate by workspace region")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "success_rate_by_region.png")
    plt.close(fig)


def plot_success_vs_error_scatter(df: pd.DataFrame) -> None:
    succ = df[df["arrived"] & df["centre_err_cm"].notna()]
    fail = df[~df["arrived"] & df["centre_err_cm"].notna()]

    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    ax.scatter(succ["centre_err_cm"],
               np.ones(len(succ)) + rng.uniform(-0.05, 0.05, len(succ)),
               color="#2a9d8f", alpha=0.55, s=32, edgecolor="black",
               linewidth=0.3, label=f"Arrived (n={len(succ)})")
    ax.scatter(fail["centre_err_cm"],
               np.zeros(len(fail)) + rng.uniform(-0.05, 0.05, len(fail)),
               color="#e76f51", alpha=0.85, s=48, edgecolor="black",
               linewidth=0.4, label=f"Missed (n={len(fail)})")
    # Maximum in-square distance from the square centre is sqrt(2)*SQUARE_CM/2
    ax.axvline(SQUARE_CM / 2.0 * np.sqrt(2), color="black",
               linestyle="--", linewidth=1,
               label=f"Square half-diagonal ({SQUARE_CM/2*np.sqrt(2):.2f} cm)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Missed", "Arrived"])
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("Distance from robot to target-square centre (cm)")
    ax.set_title("Outcome vs. final robot-to-target distance")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "success_vs_error_scatter.png")
    plt.close(fig)


def plot_workspace_heatmap(df: pd.DataFrame) -> None:
    x_edges = np.arange(WORKSPACE_X[0], WORKSPACE_X[1] + 0.001, SQUARE_CM)
    y_edges = np.arange(WORKSPACE_Y[0], WORKSPACE_Y[1] + 0.001, SQUARE_CM)

    succ_count, _, _ = np.histogram2d(
        df.loc[df["arrived"], "sq_x"], df.loc[df["arrived"], "sq_y"],
        bins=[x_edges, y_edges],
    )
    total_count, _, _ = np.histogram2d(
        df["sq_x"], df["sq_y"], bins=[x_edges, y_edges],
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(total_count > 0, 100.0 * succ_count / total_count, np.nan)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        rate.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
        cmap="RdYlGn",
        vmin=0, vmax=100,
    )
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            if total_count[i, j] == 0:
                continue
            cx = 0.5 * (x_edges[i] + x_edges[i + 1])
            cy = 0.5 * (y_edges[j] + y_edges[j + 1])
            ax.text(cx, cy,
                    f"{rate[i, j]:.0f}%\n{int(succ_count[i, j])}/{int(total_count[i, j])}",
                    ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Success rate (%)")
    ax.set_xlabel("Target x (cm)")
    ax.set_ylabel("Target y (cm)")
    ax.set_title("Workspace success-rate heatmap (10 cm target squares)")
    ax.set_xticks(x_edges)
    ax.set_yticks(y_edges)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "workspace_success_heatmap.png")
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    n_total = len(df)
    n_succ = int(df["arrived"].sum())
    succ_rate = 100.0 * n_succ / n_total

    df_r = df.copy()
    df_r["region"] = df_r.apply(assign_region, axis=1)
    region_stats = df_r.groupby("region")["arrived"].agg(["sum", "count", "mean"])
    region_stats["rate_pct"] = 100.0 * region_stats["mean"]
    region_stats = region_stats.reindex(REGION_ORDER)

    succ_err = df.loc[df["arrived"], "centre_err_cm"].dropna()
    fail_err = df.loc[~df["arrived"], "centre_err_cm"].dropna()

    lines = []
    lines.append("Section 5.5 -- Navigation Success Rate (summary)")
    lines.append("=" * 70)
    lines.append("Success criterion       : Sphero centre entered the 10x10 cm")
    lines.append("                          target square after t_start")
    lines.append(f"Trials                  : {n_total}  "
                 f"(deduplicated across {df['source'].nunique()} sessions)")
    lines.append(f"Arrived                 : {n_succ}")
    lines.append(f"Missed                  : {n_total - n_succ}")
    lines.append(f"Overall success rate    : {succ_rate:.2f} %")
    lines.append("")
    lines.append("Success rate by region:")
    for region, row in region_stats.iterrows():
        if pd.isna(row["count"]):
            continue
        lines.append(f"  {region:<12s}: {row['rate_pct']:5.1f} %  "
                     f"({int(row['sum']):3d}/{int(row['count']):3d})")
    lines.append("")
    lines.append("Robot-to-target-centre distance:")
    if len(succ_err):
        lines.append(f"  Arrived trials        : mean {succ_err.mean():.2f} cm, "
                     f"SD {succ_err.std():.2f}, max {succ_err.max():.2f}")
    if len(fail_err):
        lines.append(f"  Missed trials         : mean {fail_err.mean():.2f} cm, "
                     f"SD {fail_err.std():.2f}, max {fail_err.max():.2f}")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")


def main():
    df = load_data()
    plot_overall_success_rate(df)
    plot_success_by_region(df)
    plot_success_vs_error_scatter(df)
    plot_workspace_heatmap(df)
    write_summary(df)
    print(f"Wrote figures and summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
