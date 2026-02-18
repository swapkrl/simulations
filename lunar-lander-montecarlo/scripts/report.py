"""
Monte Carlo Statistical Report Generator
==========================================
Reads Monte Carlo results CSV and generates:
1. Histogram of touchdown velocity
2. Histogram of remaining propellant
3. 3σ confidence interval analysis
4. Pass/fail validation against 99.7% threshold

Usage:
    python scripts/report.py [--csv output/monte_carlo_results.csv]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def load_results(csv_path: str) -> pd.DataFrame:
    """Load Monte Carlo results."""
    if not os.path.exists(csv_path):
        print(f"ERROR: Results file not found: {csv_path}")
        print("Run Monte Carlo first: python src/montecarlo.py")
        sys.exit(1)
    return pd.read_csv(csv_path)


def generate_report(df: pd.DataFrame, output_dir: str):
    """Generate the full statistical report with plots."""
    os.makedirs(output_dir, exist_ok=True)

    speeds = df["touchdown_speed"].values
    fuel = df["fuel_remaining_kg"].values
    n = len(df)

    n_safe = np.sum(speeds < 2.0)
    pct_safe = 100 * n_safe / n
    passed = pct_safe >= 99.7

    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    three_sigma_speed = mean_speed + 3 * std_speed

    mean_fuel = np.mean(fuel)
    std_fuel = np.std(fuel)
    three_sigma_fuel_low = mean_fuel - 3 * std_fuel

    # ===================================================================
    # Figure: 2x2 statistical dashboard
    # ===================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Lunar Lander Monte Carlo — Statistical Report",
                 fontsize=18, fontweight="bold", color="white", y=0.98)

    for ax in axes.flat:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    # --- Touchdown Speed Histogram ---
    ax = axes[0, 0]
    counts, bins, patches = ax.hist(speeds, bins=80, color="#339af0",
                                     edgecolor="#1a1a2e", alpha=0.9)
    ax.axvline(x=2.0, color="#ff6b6b", linewidth=2, linestyle="--",
               label="Limit: 2.0 m/s")
    ax.axvline(x=mean_speed, color="#51cf66", linewidth=1.5, linestyle="-",
               label=f"Mean: {mean_speed:.3f} m/s")
    ax.axvline(x=three_sigma_speed, color="#ffd43b", linewidth=1.5,
               linestyle="--", label=f"3σ: {three_sigma_speed:.3f} m/s")
    ax.set_xlabel("Touchdown Speed (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Touchdown Speed Distribution", fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white",
              fontsize=8)

    # --- Fuel Remaining Histogram ---
    ax = axes[0, 1]
    ax.hist(fuel, bins=80, color="#51cf66", edgecolor="#1a1a2e", alpha=0.9)
    ax.axvline(x=mean_fuel, color="#339af0", linewidth=1.5, linestyle="-",
               label=f"Mean: {mean_fuel:.1f} kg")
    ax.axvline(x=three_sigma_fuel_low, color="#ffd43b", linewidth=1.5,
               linestyle="--", label=f"3σ low: {three_sigma_fuel_low:.1f} kg")
    ax.axvline(x=0, color="#ff6b6b", linewidth=2, linestyle="--",
               label="Zero fuel")
    ax.set_xlabel("Fuel Remaining (kg)")
    ax.set_ylabel("Count")
    ax.set_title("Remaining Propellant Distribution", fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white",
              fontsize=8)

    # --- Vertical vs Horizontal Velocity Scatter ---
    ax = axes[1, 0]
    v_vert = df["touchdown_v_vertical"].values
    v_horiz = df["touchdown_v_horizontal"].values
    scatter = ax.scatter(v_horiz, v_vert, s=1, c=speeds, cmap="coolwarm",
                         alpha=0.5, vmin=0, vmax=min(3.0, np.max(speeds)))
    circle = plt.Circle((0, 0), 2.0, fill=False, color="#ff6b6b",
                         linewidth=2, linestyle="--", label="2 m/s limit")
    ax.add_patch(circle)
    ax.set_xlabel("Horizontal Velocity (m/s)")
    ax.set_ylabel("Vertical Velocity (m/s)")
    ax.set_title("Touchdown Velocity Components", fontweight="bold")
    ax.set_xlim(-0.5, max(3.0, np.percentile(v_horiz, 99.5)))
    ax.set_ylim(-0.5, max(3.0, np.percentile(v_vert, 99.5)))
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Total Speed (m/s)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # --- Summary Statistics Text ---
    ax = axes[1, 1]
    ax.axis("off")

    status_color = "#51cf66" if passed else "#ff6b6b"
    status_text = "PASS ✓" if passed else "FAIL ✗"

    summary_text = (
        f"Monte Carlo Summary\n"
        f"{'─' * 35}\n"
        f"Total Runs:        {n:,}\n"
        f"Safe Landings:     {n_safe:,} ({pct_safe:.2f}%)\n"
        f"\n"
        f"Touchdown Speed:\n"
        f"  Mean:            {mean_speed:.4f} m/s\n"
        f"  Std Dev:         {std_speed:.4f} m/s\n"
        f"  3σ Upper:        {three_sigma_speed:.4f} m/s\n"
        f"  Max:             {np.max(speeds):.4f} m/s\n"
        f"\n"
        f"Fuel Remaining:\n"
        f"  Mean:            {mean_fuel:.1f} kg\n"
        f"  Std Dev:         {std_fuel:.1f} kg\n"
        f"  3σ Lower:        {three_sigma_fuel_low:.1f} kg\n"
        f"  Min:             {np.min(fuel):.1f} kg\n"
        f"\n"
        f"{'─' * 35}\n"
        f"Validation (>99.7% < 2 m/s):\n"
        f"  Result: {pct_safe:.2f}% — {status_text}"
    )

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, color="white", fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#16213e",
                      edgecolor=status_color, linewidth=2))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "monte_carlo_report.png")
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

    # ===================================================================
    # Print text report
    # ===================================================================
    print(f"\n{'='*60}")
    print(f" MONTE CARLO VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"  Total runs:              {n:,}")
    print(f"  Safe landings (< 2 m/s): {n_safe:,} ({pct_safe:.2f}%)")
    print(f"  Mean touchdown speed:    {mean_speed:.4f} m/s")
    print(f"  3σ touchdown speed:      {three_sigma_speed:.4f} m/s")
    print(f"  Mean fuel remaining:     {mean_fuel:.1f} kg")
    print(f"  3σ fuel lower bound:     {three_sigma_fuel_low:.1f} kg")
    print(f"")
    print(f"  VALIDATION: >99.7% safe? {status_text}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Monte Carlo statistical report")
    parser.add_argument("--csv", default="output/monte_carlo_results.csv",
                        help="Path to Monte Carlo results CSV")
    parser.add_argument("--output", default="output",
                        help="Output directory for plots")
    args = parser.parse_args()

    print("Loading Monte Carlo results...")
    df = load_results(args.csv)
    print(f"Loaded {len(df):,} runs")

    print("Generating report...")
    generate_report(df, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
