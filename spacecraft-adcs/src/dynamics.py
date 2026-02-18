"""
Rigid-Body Rotational Dynamics
===============================
Implements Euler's rotational equations for rigid-body spacecraft
and quaternion kinematic propagation using RK4 integration.

Euler's Equations:
    M_ext = I * omega_dot + omega x (I * omega)
    => omega_dot = I^{-1} * (M_ext - omega x (I * omega))

Quaternion Kinematics:
    q_dot = 0.5 * Omega(omega) * q
"""

import numpy as np
from quaternion_utils import quat_normalize, omega_matrix


def euler_equation(omega: np.ndarray, I_body: np.ndarray,
                   I_body_inv: np.ndarray, torque: np.ndarray) -> np.ndarray:
    """
    Compute angular acceleration from Euler's rotational equations.

    omega_dot = I^{-1} * (torque - omega x (I * omega))

    Parameters
    ----------
    omega : (3,) angular velocity vector [rad/s]
    I_body : (3,3) inertia tensor [kg*m^2]
    I_body_inv : (3,3) inverse inertia tensor
    torque : (3,) external torque vector [N*m]

    Returns
    -------
    omega_dot : (3,) angular acceleration [rad/s^2]
    """
    h = I_body @ omega                         # Angular momentum
    gyroscopic = np.cross(omega, h)             # omega x (I * omega)
    omega_dot = I_body_inv @ (torque - gyroscopic)
    return omega_dot


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute quaternion time derivative from angular velocity.

    q_dot = 0.5 * Omega(omega) * q

    Parameters
    ----------
    q : (4,) unit quaternion [q0, q1, q2, q3]
    omega : (3,) angular velocity in body frame [rad/s]

    Returns
    -------
    q_dot : (4,) quaternion derivative
    """
    Omega = omega_matrix(omega)
    q_dot = 0.5 * Omega @ q
    return q_dot


class RigidBodyDynamics:
    """
    Rigid-body rotational dynamics simulator using RK4 integration.

    State = [q0, q1, q2, q3, wx, wy, wz]  (7-dimensional)
    """

    def __init__(self, I_body: np.ndarray):
        """
        Parameters
        ----------
        I_body : (3,3) spacecraft inertia tensor [kg*m^2]
        """
        self.I_body = np.array(I_body, dtype=np.float64)
        self.I_body_inv = np.linalg.inv(self.I_body)

    def state_derivative(self, state: np.ndarray, torque: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the full [q, omega] state vector.

        Parameters
        ----------
        state : (7,) [q0, q1, q2, q3, wx, wy, wz]
        torque : (3,) applied torque in body frame [N*m]

        Returns
        -------
        dstate : (7,) time derivative of state
        """
        q = state[0:4]
        omega = state[4:7]

        q_dot = quaternion_derivative(q, omega)
        omega_dot = euler_equation(omega, self.I_body, self.I_body_inv, torque)

        return np.concatenate([q_dot, omega_dot])

    def rk4_step(self, state: np.ndarray, torque: np.ndarray,
                 dt: float) -> np.ndarray:
        """
        Advance state by one RK4 step.

        Parameters
        ----------
        state : (7,) current state
        torque : (3,) applied torque (constant over step)
        dt : time step [s]

        Returns
        -------
        new_state : (7,) updated state (quaternion re-normalized)
        """
        k1 = self.state_derivative(state, torque)
        k2 = self.state_derivative(state + dt / 2.0 * k1, torque)
        k3 = self.state_derivative(state + dt / 2.0 * k2, torque)
        k4 = self.state_derivative(state + dt * k3, torque)

        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Re-normalize quaternion to maintain unit constraint
        new_state[0:4] = quat_normalize(new_state[0:4])

        return new_state
