"""
N-Body Orbital Propagator — 3D Trajectory Visualization
========================================================
Reads the CSV telemetry output and renders:
1. 3D orbital trajectory with maneuver points highlighted
2. Altitude vs. time plot
3. Speed vs. time plot

Usage:
    python scripts/visualize.py [--csv output/trajectory.csv]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
R_EARTH_KM = 6371.0
R_GEO_KM = 42164.0 - R_EARTH_KM  # GEO altitude


def load_telemetry(csv_path: str) -> pd.DataFrame:
    """Load trajectory CSV telemetry data."""
    if not os.path.exists(csv_path):
        print(f"ERROR: Telemetry file not found: {csv_path}")
        print("Run the C++ propagator first to generate output/trajectory.csv")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["time_hr"] = df["time_s"] / 3600.0
    df["time_days"] = df["time_s"] / 86400.0
    df["x_km"] = df["x_m"] / 1000.0
    df["y_km"] = df["y_m"] / 1000.0
    df["z_km"] = df["z_m"] / 1000.0
    return df


def plot_3d_trajectory(df: pd.DataFrame, output_dir: str):
    """Plot 3D trajectory with Earth sphere and maneuver markers."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Draw Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xe = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    ye = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    ze = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xe, ye, ze, alpha=0.3, color="dodgerblue", label="Earth")

    # Plot trajectory
    ax.plot(df["x_km"], df["y_km"], df["z_km"],
            linewidth=0.8, color="white", alpha=0.9, label="Trajectory")

    # Highlight maneuver points
    maneuvers = df[df["maneuver"] == 1]
    if len(maneuvers) > 0:
        ax.scatter(maneuvers["x_km"], maneuvers["y_km"], maneuvers["z_km"],
                   color="red", s=100, marker="^", depthshade=False,
                   label=f"Δv Burns ({len(maneuvers)})", zorder=5)

    # Mark start and end
    ax.scatter([df["x_km"].iloc[0]], [df["y_km"].iloc[0]], [df["z_km"].iloc[0]],
               color="lime", s=80, marker="o", depthshade=False, label="Start (LEO)")
    ax.scatter([df["x_km"].iloc[-1]], [df["y_km"].iloc[-1]], [df["z_km"].iloc[-1]],
               color="gold", s=80, marker="*", depthshade=False, label="End (GEO)")

    # Draw GEO orbit ring
    theta = np.linspace(0, 2 * np.pi, 200)
    geo_r = 42164.0  # GEO radius in km
    ax.plot(geo_r * np.cos(theta), geo_r * np.sin(theta),
            np.zeros_like(theta), "--", color="yellow", alpha=0.4, linewidth=1,
            label="GEO orbit")

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("N-Body Orbital Propagation — LEO to GEO Hohmann Transfer",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # Dark background
    ax.set_facecolor("black")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.title.set_color("white")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_3d.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_dir}/trajectory_3d.png")


def plot_altitude_and_speed(df: pd.DataFrame, output_dir: str):
    """Plot altitude and orbital speed vs. time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    # Altitude plot
    axes[0].plot(df["time_days"], df["altitude_km"], linewidth=0.6,
                 color="cyan", alpha=0.9)
    axes[0].axhline(y=400, color="lime", linestyle="--", alpha=0.5, label="LEO (400 km)")
    axes[0].axhline(y=R_GEO_KM, color="gold", linestyle="--", alpha=0.5,
                    label=f"GEO ({R_GEO_KM:.0f} km)")

    # Mark maneuvers
    maneuvers = df[df["maneuver"] == 1]
    if len(maneuvers) > 0:
        axes[0].scatter(maneuvers["time_days"], maneuvers["altitude_km"],
                        color="red", s=60, marker="^", zorder=5, label="Δv Burns")

    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Orbital Altitude Over Time", fontweight="bold")
    axes[0].legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333",
                   labelcolor="white")
    axes[0].grid(True, alpha=0.2)

    # Speed plot
    axes[1].plot(df["time_days"], df["speed_ms"], linewidth=0.6,
                 color="orange", alpha=0.9)
    if len(maneuvers) > 0:
        axes[1].scatter(maneuvers["time_days"], maneuvers["speed_ms"],
                        color="red", s=60, marker="^", zorder=5, label="Δv Burns")

    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_title("Orbital Speed Over Time", fontweight="bold")
    axes[1].legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333",
                   labelcolor="white")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "altitude_speed.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_dir}/altitude_speed.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize N-Body orbital trajectory")
    parser.add_argument("--csv", default="output/trajectory.csv",
                        help="Path to trajectory CSV file")
    parser.add_argument("--output", default="output",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading telemetry data...")
    df = load_telemetry(args.csv)
    print(f"Loaded {len(df)} data points over {df['time_days'].iloc[-1]:.1f} days")

    # Summary statistics
    maneuver_count = df["maneuver"].sum()
    print(f"Maneuvers detected: {int(maneuver_count)}")
    print(f"Initial altitude: {df['altitude_km'].iloc[0]:.1f} km")
    print(f"Final altitude:   {df['altitude_km'].iloc[-1]:.1f} km")

    print("\nGenerating 3D trajectory plot...")
    plot_3d_trajectory(df, args.output)

    print("Generating altitude & speed plots...")
    plot_altitude_and_speed(df, args.output)

    print("\nDone! All plots saved to:", args.output)


if __name__ == "__main__":
    main()
