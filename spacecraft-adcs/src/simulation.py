"""
ADCS Simulation Main Loop
==========================
Simulates a tumbling spacecraft being stabilized by a closed-loop
quaternion PD controller driving a 3-axis reaction wheel array.

The spacecraft starts with a random angular velocity and
the controller drives it to the target attitude (identity quaternion).
"""

import os
import sys
import numpy as np
import csv

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamics import RigidBodyDynamics
from controller import PDController, ReactionWheelArray
from quaternion_utils import quat_normalize, quat_from_axis_angle


def run_simulation(config: dict = None) -> dict:
    """
    Run the ADCS simulation.

    Parameters
    ----------
    config : dict, optional
        Simulation configuration overrides.

    Returns
    -------
    results : dict with time histories
    """
    if config is None:
        config = {}

    # ===================================================================
    # Configuration
    # ===================================================================

    # Spacecraft inertia tensor [kg*m^2] - non-symmetric (with products of inertia)
    I_body = config.get("I_body", np.array([
        [ 50.0,  -2.0,   1.0],
        [ -2.0,  40.0,  -1.5],
        [  1.0,  -1.5,  35.0]
    ]))

    # Initial angular velocity [rad/s] - tumbling
    omega_init = config.get("omega_init", np.array([0.15, -0.10, 0.08]))

    # Target attitude - identity quaternion [1, 0, 0, 0]
    q_target = config.get("q_target", np.array([1.0, 0.0, 0.0, 0.0]))

    # Initial attitude - 60 deg rotation about [1,1,1] axis
    q_init = config.get("q_init",
                        quat_from_axis_angle(np.array([1, 1, 1]),
                                             np.deg2rad(60)))

    # Controller gains
    Kp = config.get("Kp", 5.0)
    Kd = config.get("Kd", 40.0)

    # Reaction wheel parameters
    I_wheel = config.get("I_wheel", 0.1)         # [kg*m^2]
    max_rpm = config.get("max_rpm", 6000.0)       # [RPM]
    max_torque = config.get("max_torque", 0.5)    # [N*m]

    # Simulation parameters
    dt = config.get("dt", 0.01)                   # Time step [s]
    t_total = config.get("t_total", 300.0)        # Total sim time [s]

    # ===================================================================
    # Initialize Models
    # ===================================================================
    dynamics = RigidBodyDynamics(I_body)
    controller = PDController(Kp=Kp, Kd=Kd)
    wheels = ReactionWheelArray(I_wheel=I_wheel, max_rpm=max_rpm,
                                max_torque=max_torque)

    # State vector: [q0, q1, q2, q3, wx, wy, wz]
    state = np.concatenate([quat_normalize(q_init), omega_init])

    # ===================================================================
    # Storage
    # ===================================================================
    n_steps = int(t_total / dt)
    record_interval = max(1, n_steps // 5000)  # ~5000 data points max

    times = []
    quaternions = []
    angular_velocities = []
    wheel_speeds = []
    torques_cmd = []
    torques_actual = []

    # ===================================================================
    # Main Loop
    # ===================================================================
    print(f"ADCS Simulation")
    print(f"  Inertia tensor diagonal: [{I_body[0,0]:.1f}, {I_body[1,1]:.1f}, {I_body[2,2]:.1f}] kg*m^2")
    print(f"  Initial omega: {omega_init} rad/s")
    print(f"  Target quaternion: {q_target}")
    print(f"  Controller: Kp={Kp}, Kd={Kd}")
    print(f"  Wheel: I={I_wheel} kg*m^2, max={max_rpm} RPM, max_T={max_torque} N*m")
    print(f"  dt={dt} s, total={t_total} s, steps={n_steps}")
    print(f"  Running...", end="", flush=True)

    t = 0.0
    for step in range(n_steps):
        q = state[0:4]
        omega = state[4:7]

        # --- Controller ---
        T_cmd = controller.compute_torque(q, q_target, omega)

        # --- Actuator ---
        T_actual = wheels.apply_torque(T_cmd, dt)

        # --- Record ---
        if step % record_interval == 0:
            times.append(t)
            quaternions.append(q.copy())
            angular_velocities.append(omega.copy())
            wheel_speeds.append(wheels.get_wheel_speeds_rpm().copy())
            torques_cmd.append(T_cmd.copy())
            torques_actual.append(T_actual.copy())

        # --- Integrate dynamics ---
        state = dynamics.rk4_step(state, T_actual, dt)

        t += dt

        # Progress
        if step % (n_steps // 10) == 0:
            print(".", end="", flush=True)

    print(" Done!")

    # Record final state
    times.append(t)
    quaternions.append(state[0:4].copy())
    angular_velocities.append(state[4:7].copy())
    wheel_speeds.append(wheels.get_wheel_speeds_rpm().copy())
    torques_cmd.append(np.zeros(3))
    torques_actual.append(np.zeros(3))

    # ===================================================================
    # Results
    # ===================================================================
    results = {
        "time": np.array(times),
        "quaternion": np.array(quaternions),
        "omega": np.array(angular_velocities),
        "wheel_rpm": np.array(wheel_speeds),
        "torque_cmd": np.array(torques_cmd),
        "torque_actual": np.array(torques_actual),
        "q_target": q_target,
    }

    # Print summary
    q_final = state[0:4]
    omega_final = state[4:7]
    omega_mag = np.linalg.norm(omega_final)
    q_err_scalar = abs(np.dot(q_final, q_target))

    print(f"\n--- Final State ---")
    print(f"  Quaternion: [{q_final[0]:.6f}, {q_final[1]:.6f}, "
          f"{q_final[2]:.6f}, {q_final[3]:.6f}]")
    print(f"  Angular vel: [{omega_final[0]:.6e}, {omega_final[1]:.6e}, "
          f"{omega_final[2]:.6e}] rad/s")
    print(f"  |omega|: {omega_mag:.6e} rad/s")
    print(f"  Attitude error (|q.q_t|): {q_err_scalar:.8f} (1.0 = perfect)")
    print(f"  Wheel speeds: {wheels.get_wheel_speeds_rpm()} RPM")

    converged = omega_mag < 1e-3 and q_err_scalar > 0.999
    print(f"  CONVERGED: {'YES [OK]' if converged else 'NO [FAIL]'}")

    return results


def save_results_csv(results: dict, filepath: str):
    """Save simulation results to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "q0", "q1", "q2", "q3",
            "wx_rads", "wy_rads", "wz_rads",
            "wheel_x_rpm", "wheel_y_rpm", "wheel_z_rpm",
            "Tx_cmd", "Ty_cmd", "Tz_cmd",
            "Tx_actual", "Ty_actual", "Tz_actual"
        ])

        for i in range(len(results["time"])):
            row = [
                results["time"][i],
                *results["quaternion"][i],
                *results["omega"][i],
                *results["wheel_rpm"][i],
                *results["torque_cmd"][i],
                *results["torque_actual"][i],
            ]
            writer.writerow(row)

    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    results = run_simulation()

    # Save to CSV
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    csv_path = os.path.join(output_dir, "adcs_telemetry.csv")
    save_results_csv(results, csv_path)

    print("\nVisualize: python scripts/visualize.py")
