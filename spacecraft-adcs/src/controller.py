"""
Quaternion-Based PD Controller and Reaction Wheel Model
========================================================
Implements a Proportional-Derivative (PD) attitude controller
that computes commanded torques, and a 3-axis reaction wheel
array with saturation limits.

Control Law:
    T_cmd = -Kp * q_err_vec - Kd * omega

    where q_err_vec is the vector part of the error quaternion
    and omega is the current angular velocity.

Reaction Wheel Model:
    L_rw = I_rw * omega_rw
    Torque on spacecraft = -d(L_rw)/dt
    Saturation: |omega_rw| <= omega_max, |T_rw| <= T_max
"""

import numpy as np
from quaternion_utils import quat_error


class PDController:
    """
    Quaternion-based Proportional-Derivative (PD) attitude controller.

    Computes commanded torque to drive the spacecraft attitude
    towards a target quaternion while damping angular rates.
    """

    def __init__(self, Kp: float = 10.0, Kd: float = 50.0):
        """
        Parameters
        ----------
        Kp : Proportional gain [N*m/rad]
        Kd : Derivative gain [N*m*s/rad]
        """
        self.Kp = Kp
        self.Kd = Kd

    def compute_torque(self, q_current: np.ndarray, q_target: np.ndarray,
                       omega: np.ndarray) -> np.ndarray:
        """
        Compute the commanded control torque.

        T_cmd = -Kp * q_err_vec - Kd * omega

        Parameters
        ----------
        q_current : (4,) current attitude quaternion
        q_target : (4,) target attitude quaternion
        omega : (3,) current angular velocity [rad/s]

        Returns
        -------
        torque_cmd : (3,) commanded torque vector [N*m]
        """
        # Compute quaternion error
        q_err = quat_error(q_current, q_target)

        # Extract vector part (proportional to rotation error for small angles)
        q_err_vec = q_err[1:4]

        # PD control law
        torque_cmd = -self.Kp * q_err_vec - self.Kd * omega

        return torque_cmd


class ReactionWheelArray:
    """
    3-axis reaction wheel array model with saturation.

    Each wheel is aligned along a body axis (x, y, z).
    The wheel applies torque to the spacecraft by changing
    its angular momentum.

    Constraints:
        - Maximum wheel speed (RPM saturation)
        - Maximum torque output
    """

    def __init__(self,
                 I_wheel: float = 0.05,      # Wheel inertia [kg*m^2]
                 max_rpm: float = 6000.0,     # Maximum wheel speed [RPM]
                 max_torque: float = 0.2):    # Maximum torque [N*m]
        """
        Parameters
        ----------
        I_wheel : Moment of inertia of each wheel [kg*m^2]
        max_rpm : Maximum wheel speed [RPM]
        max_torque : Maximum torque output per wheel [N*m]
        """
        self.I_wheel = I_wheel
        self.max_rpm = max_rpm
        self.max_omega = max_rpm * 2.0 * np.pi / 60.0  # Convert to rad/s
        self.max_torque = max_torque

        # Wheel speeds [rad/s] for each axis
        self.omega_wheels = np.zeros(3)

    def apply_torque(self, torque_cmd: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply the commanded torque through the reaction wheels.

        The actual torque may be limited by wheel saturation.

        Parameters
        ----------
        torque_cmd : (3,) commanded torque for each axis [N*m]
        dt : time step [s]

        Returns
        -------
        actual_torque : (3,) actual torque applied to spacecraft [N*m]
        """
        actual_torque = np.zeros(3)

        for i in range(3):
            # Clip commanded torque to max
            t_cmd = np.clip(torque_cmd[i], -self.max_torque, self.max_torque)

            # Wheel spins in OPPOSITE direction to produce desired torque on SC
            # T_on_sc = -I_wheel * alpha_wheel  =>  alpha_wheel = -T_on_sc / I_wheel
            alpha_wheel = -t_cmd / self.I_wheel
            new_omega_wheel = self.omega_wheels[i] + alpha_wheel * dt

            # Check wheel speed saturation
            if abs(new_omega_wheel) > self.max_omega:
                new_omega_wheel = np.sign(new_omega_wheel) * self.max_omega
                # Recompute actual torque considering saturation
                alpha_wheel = (new_omega_wheel - self.omega_wheels[i]) / dt
                t_cmd = -self.I_wheel * alpha_wheel

            self.omega_wheels[i] = new_omega_wheel

            # Torque applied on spacecraft body (= what controller requested)
            actual_torque[i] = t_cmd

        return actual_torque

    def get_angular_momentum(self) -> np.ndarray:
        """Return the angular momentum stored in all wheels [N*m*s]."""
        return self.I_wheel * self.omega_wheels

    def get_wheel_speeds_rpm(self) -> np.ndarray:
        """Return wheel speeds in RPM."""
        return self.omega_wheels * 60.0 / (2.0 * np.pi)

    def is_saturated(self) -> np.ndarray:
        """Return boolean array indicating which wheels are saturated."""
        return np.abs(self.omega_wheels) >= 0.99 * self.max_omega
