"""
Monte Carlo Simulation Wrapper
================================
Runs the lunar descent simulation 10,000 times with randomized
dispersions using multiprocessing for parallel execution.

Collects statistics on touchdown velocity, fuel remaining,
and mission success rate.
"""

import os
import sys
import json
import time
import csv
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import run_single_simulation


def load_dispersions(config_path: str) -> dict:
    """Load dispersion configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def sample_dispersions(dispersion_config: dict,
                       rng: np.random.Generator) -> dict:
    """
    Sample a single set of dispersions from the configuration.

    Each parameter is sampled from N(mean, sigma^2).

    Parameters
    ----------
    dispersion_config : dict with keys like "alt_bias_sigma", etc.
    rng : Random number generator

    Returns
    -------
    dispersions : dict of sampled dispersion values
    """
    d = {}

    # Initial altitude bias [m]
    d["alt_bias"] = rng.normal(0, dispersion_config.get("alt_bias_sigma", 100.0))

    # Initial velocity biases [m/s]
    d["v_vert_bias"] = rng.normal(0, dispersion_config.get("v_vert_bias_sigma", 3.0))
    d["v_horiz_bias"] = rng.normal(0, dispersion_config.get("v_horiz_bias_sigma", 5.0))

    # Thrust scaling factor (multiplicative)
    d["thrust_scale"] = rng.normal(1.0, dispersion_config.get("thrust_scale_sigma", 0.02))

    # Per-step thrust noise sigma
    d["sigma_thrust"] = dispersion_config.get("thrust_noise_sigma", 0.01)

    # Isp bias [s]
    d["Isp_bias"] = rng.normal(0, dispersion_config.get("Isp_bias_sigma", 3.0))

    # Mass bias [kg]
    d["mass_bias"] = rng.normal(0, dispersion_config.get("mass_bias_sigma", 5.0))

    # Sensor noise levels
    d["sigma_alt"] = abs(rng.normal(
        dispersion_config.get("sigma_alt_mean", 5.0),
        dispersion_config.get("sigma_alt_sigma", 1.0)
    ))
    d["sigma_vel"] = abs(rng.normal(
        dispersion_config.get("sigma_vel_mean", 0.3),
        dispersion_config.get("sigma_vel_sigma", 0.05)
    ))

    return d


def run_dispersed_sim(run_id: int, dispersion_config: dict) -> dict:
    """
    Run a single dispersed simulation.

    Parameters
    ----------
    run_id : Integer run identifier (used as seed offset)
    dispersion_config : Dispersion configuration dict

    Returns
    -------
    result : dict with touchdown conditions + run_id
    """
    rng = np.random.default_rng(seed=run_id + 42)
    dispersions = sample_dispersions(dispersion_config, rng)

    result = run_single_simulation(dispersions=dispersions, seed=run_id)
    result["run_id"] = run_id
    return result


def run_monte_carlo(n_runs: int = 10000,
                    config_path: str = None,
                    n_workers: int = None) -> list:
    """
    Execute the Monte Carlo campaign.

    Parameters
    ----------
    n_runs : Number of simulation runs
    config_path : Path to dispersions.json
    n_workers : Number of parallel workers (default: CPU count)

    Returns
    -------
    results : list of dicts with touchdown conditions
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # Load dispersion config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__),
                                    "..", "config", "dispersions.json")

    if os.path.exists(config_path):
        dispersion_config = load_dispersions(config_path)
        print(f"Loaded dispersions from: {config_path}")
    else:
        print(f"WARNING: {config_path} not found. Using defaults.")
        dispersion_config = {}

    print(f"\n{'='*60}")
    print(f" Lunar Lander Monte Carlo Simulation")
    print(f"{'='*60}")
    print(f"  Runs:     {n_runs:,}")
    print(f"  Workers:  {n_workers}")
    print(f"  Config:   {config_path}")
    print(f"{'='*60}\n")

    # Run simulations in parallel
    worker_fn = partial(run_dispersed_sim, dispersion_config=dispersion_config)

    start_time = time.time()

    with Pool(processes=n_workers) as pool:
        results = []
        total = n_runs
        chunk_size = max(1, total // 100)

        for i, result in enumerate(pool.imap_unordered(worker_fn,
                                                        range(n_runs),
                                                        chunksize=chunk_size)):
            results.append(result)
            if (i + 1) % (total // 10) == 0 or i == total - 1:
                pct = 100 * (i + 1) / total
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  Progress: {pct:5.1f}% ({i+1:,}/{total:,}) "
                      f"| {rate:.0f} runs/s | ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\nCompleted {n_runs:,} runs in {elapsed:.1f}s "
          f"({n_runs/elapsed:.0f} runs/s)")

    return results


def save_results(results: list, output_dir: str):
    """Save Monte Carlo results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "monte_carlo_results.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "touchdown_v_vertical", "touchdown_v_horizontal",
            "touchdown_speed", "fuel_remaining_kg", "time_of_flight_s",
            "landed", "fuel_exhausted"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {filepath}")


def print_statistics(results: list):
    """Print Monte Carlo statistical summary."""
    speeds = np.array([r["touchdown_speed"] for r in results])
    fuel = np.array([r["fuel_remaining_kg"] for r in results])
    landed = np.array([r["landed"] for r in results])
    v_vert = np.array([r["touchdown_v_vertical"] for r in results])
    v_horiz = np.array([r["touchdown_v_horizontal"] for r in results])

    n = len(results)
    n_safe = np.sum(speeds < 2.0)
    n_landed = np.sum(landed)

    print(f"\n{'='*60}")
    print(f" MONTE CARLO STATISTICAL REPORT")
    print(f"{'='*60}")
    print(f"  Total runs:       {n:,}")
    print(f"  Landed:           {n_landed:,} ({100*n_landed/n:.2f}%)")
    print(f"  Safe (< 2 m/s):   {n_safe:,} ({100*n_safe/n:.2f}%)")
    print(f"")
    print(f"  Touchdown Speed [m/s]:")
    print(f"    Mean:           {np.mean(speeds):.4f}")
    print(f"    Std:            {np.std(speeds):.4f}")
    print(f"    Median:         {np.median(speeds):.4f}")
    print(f"    Min:            {np.min(speeds):.4f}")
    print(f"    Max:            {np.max(speeds):.4f}")
    print(f"    3-sigma upper:  {np.mean(speeds) + 3*np.std(speeds):.4f}")
    print(f"")
    print(f"  Vertical Velocity [m/s]:")
    print(f"    Mean:           {np.mean(v_vert):.4f}")
    print(f"    3-sigma:        {np.mean(v_vert) + 3*np.std(v_vert):.4f}")
    print(f"")
    print(f"  Fuel Remaining [kg]:")
    print(f"    Mean:           {np.mean(fuel):.2f}")
    print(f"    Std:            {np.std(fuel):.2f}")
    print(f"    Min:            {np.min(fuel):.2f}")
    print(f"    3-sigma lower:  {np.mean(fuel) - 3*np.std(fuel):.2f}")
    print(f"")
    print(f"  VALIDATION: >99.7% safe touchdown (< 2 m/s)?")
    pct_safe = 100 * n_safe / n
    passed = pct_safe >= 99.7
    print(f"    Result: {pct_safe:.2f}% -- {'PASS [OK]' if passed else 'FAIL [X]'}")
    print(f"{'='*60}")

    return {
        "n_runs": n,
        "pct_safe": pct_safe,
        "mean_speed": float(np.mean(speeds)),
        "std_speed": float(np.std(speeds)),
        "three_sigma_speed": float(np.mean(speeds) + 3*np.std(speeds)),
        "mean_fuel": float(np.mean(fuel)),
        "passed": passed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Lunar Lander Monte Carlo Simulation")
    parser.add_argument("-n", "--num-runs", type=int, default=10000,
                        help="Number of Monte Carlo runs")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("-c", "--config",
                        default=None,
                        help="Path to dispersions config JSON")
    parser.add_argument("-o", "--output", default="output",
                        help="Output directory")
    args = parser.parse_args()

    # Run Monte Carlo
    results = run_monte_carlo(
        n_runs=args.num_runs,
        config_path=args.config,
        n_workers=args.workers,
    )

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", args.output)
    save_results(results, output_dir)

    # Print statistics
    stats = print_statistics(results)

    # Save statistics summary
    stats_path = os.path.join(output_dir, "statistics_summary.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")
