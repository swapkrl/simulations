"""
Single-Run Lunar Descent Simulation
=====================================
Integrates the truth model, navigation model, and guidance model
for a single powered-descent trajectory.

Returns touchdown conditions (velocity, fuel remaining, etc.).
"""

import numpy as np
from physics import LanderPhysics
from guidance import DescentGuidance
from navigation import NavigationModel


def run_single_simulation(dispersions: dict = None,
                          seed: int = None,
                          return_trajectory: bool = False) -> dict:
    """
    Run a single lunar descent simulation with optional dispersions.

    Parameters
    ----------
    dispersions : dict
        Dispersion parameters with keys:
        - alt_bias: initial altitude bias [m]
        - v_vert_bias: initial vertical velocity bias [m/s]
        - v_horiz_bias: initial horizontal velocity bias [m/s]
        - thrust_scale: multiplicative thrust scaling factor
        - Isp_bias: Isp bias [s]
        - sigma_alt: altimeter noise σ [m]
        - sigma_vel: velocity sensor noise σ [m/s]
        - mass_bias: initial mass bias [kg]
    seed : int
        Random seed for reproducibility
    return_trajectory : bool
        If True, return full trajectory history

    Returns
    -------
    result : dict with touchdown conditions
    """
    rng = np.random.default_rng(seed)

    if dispersions is None:
        dispersions = {}

    # ===================================================================
    # Apply dispersions to nominal parameters
    # ===================================================================

    # Nominal parameters
    alt_init = 2000.0 + dispersions.get("alt_bias", 0.0)
    v_vert_init = -60.0 + dispersions.get("v_vert_bias", 0.0)
    v_horiz_init = 10.0 + dispersions.get("v_horiz_bias", 0.0)
    thrust_scale = dispersions.get("thrust_scale", 1.0)
    Isp = 311.0 + dispersions.get("Isp_bias", 0.0)
    mass_bias = dispersions.get("mass_bias", 0.0)

    sigma_alt = dispersions.get("sigma_alt", 5.0)
    sigma_vel = dispersions.get("sigma_vel", 0.3)

    # ===================================================================
    # Initialize models
    # ===================================================================
    physics = LanderPhysics(
        dry_mass=1000.0,
        fuel_mass=2000.0 + mass_bias,
        max_thrust=15000.0,
        Isp=Isp,
    )

    guidance = DescentGuidance()

    nav = NavigationModel(
        sigma_alt=sigma_alt,
        sigma_vel=sigma_vel,
        rng=rng,
    )

    # Initial state
    state = physics.initial_state(
        altitude=alt_init,
        downrange=0.0,
        v_vertical=v_vert_init,
        v_horizontal=v_horiz_init,
    )

    # ===================================================================
    # Simulation loop
    # ===================================================================
    dt = 0.5           # Time step [s]
    max_time = 1200.0  # Maximum simulation time [s]
    t = 0.0

    trajectory = [] if return_trajectory else None

    while t < max_time:
        # Check for landing
        if physics.has_landed(state):
            break

        # Navigation: corrupt truth state
        nav_state = nav.corrupt_state(state)

        # Guidance: compute thrust command
        thrust_cmd, angle_cmd = guidance.compute_command(
            nav_state, physics.max_thrust, state[4]
        )

        # Apply thrust dispersion (engine performance variance)
        actual_thrust = thrust_cmd * thrust_scale

        # Apply thrust noise per step
        thrust_noise = rng.normal(1.0, dispersions.get("sigma_thrust", 0.01))
        actual_thrust *= thrust_noise

        # Propagate physics
        state = physics.rk4_step(state, dt, actual_thrust, angle_cmd)

        t += dt

        # Record trajectory
        if return_trajectory:
            trajectory.append({
                "time": t,
                "altitude": state[0],
                "downrange": state[1],
                "v_vertical": state[2],
                "v_horizontal": state[3],
                "mass": state[4],
                "thrust": actual_thrust,
                "angle_deg": np.degrees(angle_cmd),
                "fuel": physics.fuel_remaining(state),
            })

    # ===================================================================
    # Touchdown conditions
    # ===================================================================
    touchdown_v_vert = abs(state[2])
    touchdown_v_horiz = abs(state[3])
    touchdown_speed = np.sqrt(state[2]**2 + state[3]**2)
    fuel_remaining = physics.fuel_remaining(state)

    result = {
        "touchdown_v_vertical": touchdown_v_vert,
        "touchdown_v_horizontal": touchdown_v_horiz,
        "touchdown_speed": touchdown_speed,
        "fuel_remaining_kg": fuel_remaining,
        "time_of_flight_s": t,
        "landed": state[0] <= 0.0,
        "fuel_exhausted": fuel_remaining <= 0.0,
    }

    if return_trajectory:
        result["trajectory"] = trajectory

    return result


if __name__ == "__main__":
    # Run a single nominal simulation
    print("Running nominal descent simulation...")
    result = run_single_simulation(return_trajectory=True)

    print(f"\n--- Touchdown Conditions ---")
    print(f"  Vertical velocity:   {result['touchdown_v_vertical']:.2f} m/s")
    print(f"  Horizontal velocity: {result['touchdown_v_horizontal']:.2f} m/s")
    print(f"  Total speed:         {result['touchdown_speed']:.2f} m/s")
    print(f"  Fuel remaining:      {result['fuel_remaining_kg']:.1f} kg")
    print(f"  Time of flight:      {result['time_of_flight_s']:.1f} s")
    print(f"  Landed:              {result['landed']}")
    print(f"  Safe (< 2 m/s):      {result['touchdown_speed'] < 2.0}")
