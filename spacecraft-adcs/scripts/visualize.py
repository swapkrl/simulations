"""
ADCS Visualization Script
==========================
Plots the simulation results:
1. Angular velocity components vs. time
2. Quaternion components vs. time
3. Reaction wheel speeds vs. time
4. Control torque vs. time

Usage:
    python scripts/visualize.py [--csv output/adcs_telemetry.csv]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """Load ADCS telemetry data."""
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        print("Run simulation first: python src/simulation.py")
        sys.exit(1)
    return pd.read_csv(csv_path)


def create_plots(df: pd.DataFrame, output_dir: str):
    """Generate all ADCS plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Color scheme
    colors = {
        "x": "#ff6b6b",
        "y": "#51cf66",
        "z": "#339af0",
        "q0": "#ffd43b",
        "q1": "#ff6b6b",
        "q2": "#51cf66",
        "q3": "#339af0",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Spacecraft ADCS — Tumble Stabilization",
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    for ax in axes.flat:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(True, alpha=0.15, color="white")

    # --- Angular Velocity ---
    ax = axes[0, 0]
    ax.plot(df["time_s"], df["wx_rads"], color=colors["x"], linewidth=1, label="ωx")
    ax.plot(df["time_s"], df["wy_rads"], color=colors["y"], linewidth=1, label="ωy")
    ax.plot(df["time_s"], df["wz_rads"], color=colors["z"], linewidth=1, label="ωz")
    ax.axhline(y=0, color="white", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("Angular Velocity Convergence", fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=9)

    # --- Quaternion Components ---
    ax = axes[0, 1]
    ax.plot(df["time_s"], df["q0"], color=colors["q0"], linewidth=1, label="q₀ (scalar)")
    ax.plot(df["time_s"], df["q1"], color=colors["q1"], linewidth=1, label="q₁")
    ax.plot(df["time_s"], df["q2"], color=colors["q2"], linewidth=1, label="q₂")
    ax.plot(df["time_s"], df["q3"], color=colors["q3"], linewidth=1, label="q₃")
    ax.axhline(y=1, color="white", linestyle="--", alpha=0.3, label="Target q₀")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Quaternion Component")
    ax.set_title("Quaternion Convergence to Target", fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)

    # --- Wheel Speeds ---
    ax = axes[1, 0]
    ax.plot(df["time_s"], df["wheel_x_rpm"], color=colors["x"], linewidth=1, label="Wheel X")
    ax.plot(df["time_s"], df["wheel_y_rpm"], color=colors["y"], linewidth=1, label="Wheel Y")
    ax.plot(df["time_s"], df["wheel_z_rpm"], color=colors["z"], linewidth=1, label="Wheel Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wheel Speed (RPM)")
    ax.set_title("Reaction Wheel Speeds", fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=9)

    # --- Control Torque ---
    ax = axes[1, 1]
    ax.plot(df["time_s"], df["Tx_actual"], color=colors["x"], linewidth=1, label="Tx")
    ax.plot(df["time_s"], df["Ty_actual"], color=colors["y"], linewidth=1, label="Ty")
    ax.plot(df["time_s"], df["Tz_actual"], color=colors["z"], linewidth=1, label="Tz")
    ax.axhline(y=0, color="white", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title("Applied Control Torques", fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "adcs_plots.png")
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ADCS simulation results")
    parser.add_argument("--csv", default="output/adcs_telemetry.csv",
                        help="Path to ADCS telemetry CSV")
    parser.add_argument("--output", default="output",
                        help="Output directory for plots")
    args = parser.parse_args()

    print("Loading ADCS telemetry...")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} data points")

    print("Generating plots...")
    create_plots(df, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
