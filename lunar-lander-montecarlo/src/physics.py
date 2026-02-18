"""
Lunar Lander 2-DOF Powered Descent Physics Engine
===================================================
Models vertical (altitude) and horizontal (downrange) motion
of a lunar lander during powered descent.

Dynamics:
    - Gravity: a_g = -g_moon (vertical)
    - Thrust: T directed along thrust vector angle
    - Mass depletion: m_dot = -T / (Isp * g0)

State vector: [altitude, downrange, v_vertical, v_horizontal, mass]
"""

import numpy as np


# ============================================================================
# Constants
# ============================================================================
G_MOON = 1.625            # Lunar surface gravity [m/s^2]
G0_EARTH = 9.80665        # Standard gravity for Isp [m/s^2]


class LanderPhysics:
    """
    2-DOF powered descent physics model for a lunar lander.

    State = [altitude, downrange, v_vertical, v_horizontal, mass]
    """

    def __init__(self,
                 dry_mass: float = 1000.0,          # Dry mass [kg]
                 fuel_mass: float = 2000.0,          # Initial fuel mass [kg]
                 max_thrust: float = 15000.0,        # Maximum thrust [N]
                 Isp: float = 311.0,                 # Specific impulse [s]
                 g_moon: float = G_MOON):
        """
        Parameters
        ----------
        dry_mass : Vehicle dry mass (no fuel) [kg]
        fuel_mass : Initial fuel mass [kg]
        max_thrust : Engine maximum thrust [N]
        Isp : Specific impulse [s]
        g_moon : Lunar surface gravity [m/s^2]
        """
        self.dry_mass = dry_mass
        self.fuel_mass_init = fuel_mass
        self.max_thrust = max_thrust
        self.Isp = Isp
        self.g_moon = g_moon

    def initial_state(self,
                      altitude: float = 15000.0,
                      downrange: float = 0.0,
                      v_vertical: float = -50.0,
                      v_horizontal: float = 200.0) -> np.ndarray:
        """Create the initial state vector."""
        total_mass = self.dry_mass + self.fuel_mass_init
        return np.array([altitude, downrange, v_vertical, v_horizontal,
                         total_mass])

    def state_derivative(self, state: np.ndarray,
                         thrust_mag: float,
                         thrust_angle: float) -> np.ndarray:
        """
        Compute the derivative of the state vector.

        Parameters
        ----------
        state : [alt, downrange, v_vert, v_horiz, mass]
        thrust_mag : Thrust magnitude [N]
        thrust_angle : Thrust angle from vertical [rad]
                       (0 = pure vertical, pi/2 = pure horizontal)

        Returns
        -------
        dstate : derivative of the state vector
        """
        alt, dr, v_v, v_h, mass = state

        # No fuel left? No thrust.
        if mass <= self.dry_mass:
            thrust_mag = 0.0
            mass = self.dry_mass

        # Clamp thrust
        thrust_mag = np.clip(thrust_mag, 0.0, self.max_thrust)

        # Thrust components
        T_vertical = thrust_mag * np.cos(thrust_angle)
        T_horizontal = -thrust_mag * np.sin(thrust_angle)  # Decelerate

        # Accelerations
        a_vert = -self.g_moon + T_vertical / mass
        a_horiz = T_horizontal / mass

        # Mass flow rate: m_dot = -T / (Isp * g0)
        m_dot = -thrust_mag / (self.Isp * G0_EARTH)

        return np.array([v_v, v_h, a_vert, a_horiz, m_dot])

    def rk4_step(self, state: np.ndarray, dt: float,
                 thrust_mag: float, thrust_angle: float) -> np.ndarray:
        """
        Advance state by one RK4 step.

        Parameters
        ----------
        state : current state vector
        dt : time step [s]
        thrust_mag : current thrust magnitude [N]
        thrust_angle : current thrust angle from vertical [rad]

        Returns
        -------
        new_state : updated state vector
        """
        k1 = self.state_derivative(state, thrust_mag, thrust_angle)
        k2 = self.state_derivative(state + dt/2 * k1, thrust_mag, thrust_angle)
        k3 = self.state_derivative(state + dt/2 * k2, thrust_mag, thrust_angle)
        k4 = self.state_derivative(state + dt * k3, thrust_mag, thrust_angle)

        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Clamp mass to dry mass
        if new_state[4] < self.dry_mass:
            new_state[4] = self.dry_mass

        return new_state

    def fuel_remaining(self, state: np.ndarray) -> float:
        """Return remaining fuel [kg]."""
        return max(0.0, state[4] - self.dry_mass)

    def has_landed(self, state: np.ndarray) -> bool:
        """Check if lander has reached the surface."""
        return state[0] <= 0.0
