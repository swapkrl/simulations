"""
Navigation Model — Sensor Noise Injection
===========================================
Corrupts the "truth" state with Gaussian sensor noise
to simulate realistic navigation errors.

Noise Sources:
    - Altimeter:   true_alt + N(0, sigma_alt^2)
    - IMU velocity: true_vel + N(0, sigma_vel^2)
    - Mass sensor:  true_mass + N(0, sigma_mass^2)
"""

import numpy as np


class NavigationModel:
    """
    Navigation model that corrupts truth state with sensor noise.
    """

    def __init__(self,
                 sigma_alt: float = 5.0,       # Altimeter noise 1σ [m]
                 sigma_vel: float = 0.3,        # Velocity noise 1σ [m/s]
                 sigma_mass: float = 2.0,       # Mass sensor noise 1σ [kg]
                 sigma_downrange: float = 10.0, # Downrange position noise [m]
                 rng: np.random.Generator = None):
        """
        Parameters
        ----------
        sigma_alt : 1σ altimeter noise [m]
        sigma_vel : 1σ velocity noise [m/s]
        sigma_mass : 1σ mass sensor noise [kg]
        sigma_downrange : 1σ downrange position noise [m]
        rng : Random number generator (for reproducibility)
        """
        self.sigma_alt = sigma_alt
        self.sigma_vel = sigma_vel
        self.sigma_mass = sigma_mass
        self.sigma_downrange = sigma_downrange
        self.rng = rng if rng is not None else np.random.default_rng()

    def corrupt_state(self, truth_state: np.ndarray) -> np.ndarray:
        """
        Produce a corrupted navigation state from the truth state.

        Parameters
        ----------
        truth_state : [alt, downrange, v_vert, v_horiz, mass]

        Returns
        -------
        nav_state : corrupted state with sensor noise applied
        """
        nav_state = truth_state.copy()

        # Altimeter noise
        nav_state[0] += self.rng.normal(0, self.sigma_alt)

        # Downrange position noise
        nav_state[1] += self.rng.normal(0, self.sigma_downrange)

        # Velocity noise (both axes)
        nav_state[2] += self.rng.normal(0, self.sigma_vel)
        nav_state[3] += self.rng.normal(0, self.sigma_vel)

        # Mass sensor noise
        nav_state[4] += self.rng.normal(0, self.sigma_mass)

        # Clamp altitude to positive
        nav_state[0] = max(nav_state[0], 0.1)

        return nav_state
