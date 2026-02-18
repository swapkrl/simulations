"""
Lunar Lander Guidance Algorithm
================================
Three-phase powered descent guidance:

Phase 1 - Pitchover Braking (alt > 2000m):
    High-tilt braking to eliminate horizontal velocity while
    opposing gravity. Uses full throttle tilted to balance
    vertical hover with horizontal deceleration.

Phase 2 - Vertical Braking (alt > 200m):
    Primarily vertical braking to control descent rate.
    Mop up any remaining horizontal velocity.

Phase 3 - Terminal Descent (alt <= 200m):
    Proportional velocity control for soft touchdown.
"""

import numpy as np


class DescentGuidance:
    """
    Three-phase powered descent guidance for lunar landing.
    """

    def __init__(self,
                 g_moon: float = 1.625,
                 v_target_touchdown: float = 1.0,
                 min_thrust_fraction: float = 0.10,
                 max_thrust_fraction: float = 1.0):
        self.g_moon = g_moon
        self.v_target = v_target_touchdown
        self.min_frac = min_thrust_fraction
        self.max_frac = max_thrust_fraction

    def compute_command(self, nav_state: np.ndarray,
                        max_thrust: float,
                        mass: float) -> tuple:
        """
        Compute thrust command.

        Parameters
        ----------
        nav_state : [alt, downrange, v_vert, v_horiz, mass]
        max_thrust : Maximum engine thrust [N]
        mass : Current vehicle mass [kg]

        Returns
        -------
        thrust_mag, thrust_angle
        """
        alt = max(nav_state[0], 0.1)
        v_vert = nav_state[2]
        v_horiz = nav_state[3]

        abs_v_vert = abs(v_vert)
        abs_v_horiz = abs(v_horiz)

        # ---------------------------------------------------------------
        # Phase 1: Pitchover braking (horizontal velocity > 5 m/s)
        # ---------------------------------------------------------------
        if abs_v_horiz > 1.0 and alt > 200.0:
            # Full thrust, angled to kill horizontal while hovering/braking
            # Compute tilt angle: more tilt when horizontal is dominant
            # The tilt angle is chosen so that:
            #   T*cos(theta) = m*g + m*a_vert_brake  (vertical)
            #   T*sin(theta) = m*a_horiz_brake        (horizontal)

            # Desired vertical: just enough to manage descent rate
            # Use constant-decel profile for vertical
            if alt > 500.0 and abs_v_vert > self.v_target:
                a_brake_vert = (abs_v_vert**2 - self.v_target**2) / (2.0 * alt)
            else:
                a_brake_vert = 0.0
            a_vert = a_brake_vert + self.g_moon

            # Desired horizontal: eliminate as fast as possible
            # Use constant decel to zero over remaining altitude
            a_horiz = 1.2 * abs_v_horiz**2 / (2.0 * max(alt, 1.0))
            # Minimum: ensure we finish braking in time
            a_horiz = max(a_horiz, abs_v_horiz / max(alt / max(abs_v_vert, 1.0), 1.0))

            # Compute angle and magnitude
            theta = np.arctan2(a_horiz, a_vert)
            theta = np.clip(theta, 0.0, np.deg2rad(60))  # Allow more tilt in this phase

            # Thrust magnitude from vertical requirement
            a_total = a_vert / max(np.cos(theta), 0.3)
            thrust_cmd = mass * a_total

        # ---------------------------------------------------------------
        # Phase 2: Vertical braking (alt > 200m, horiz < 5 m/s)
        # ---------------------------------------------------------------
        elif alt > 200.0:
            # Primarily vertical control
            if abs_v_vert > self.v_target:
                a_brake = (abs_v_vert**2 - self.v_target**2) / (2.0 * alt)
            else:
                a_brake = 0.0

            a_vert = a_brake + self.g_moon

            # Kill remaining horizontal
            if abs_v_horiz > 0.5:
                t_go = max(alt / max(abs_v_vert, 0.1), 1.0)
                a_horiz = abs_v_horiz / t_go
            else:
                a_horiz = 0.0

            theta = np.arctan2(a_horiz, a_vert)
            theta = np.clip(theta, 0.0, np.deg2rad(30))

            a_total = a_vert / max(np.cos(theta), 0.5)
            thrust_cmd = mass * a_total

        # ---------------------------------------------------------------
        # Phase 3: Terminal descent (alt <= 200m)
        # ---------------------------------------------------------------
        else:
            # Target velocity: linearly decrease to v_target at surface
            v_desired = self.v_target * (1.0 + alt / 30.0)
            v_error = abs_v_vert - v_desired
            Kp = 0.5
            a_vert = Kp * v_error + self.g_moon
            a_vert = max(a_vert, self.g_moon * 0.5)

            # Kill horizontal
            if abs_v_horiz > 0.1:
                t_remain = max(alt / max(abs_v_vert, 0.1), 0.5)
                a_horiz = abs_v_horiz / t_remain
                a_horiz = min(a_horiz, 2.0)
            else:
                a_horiz = 0.0

            theta = np.arctan2(a_horiz, a_vert)
            theta = np.clip(theta, 0.0, np.deg2rad(20))

            a_total = a_vert / max(np.cos(theta), 0.7)
            thrust_cmd = mass * a_total

        # ---------------------------------------------------------------
        # Clip thrust
        # ---------------------------------------------------------------
        thrust_cmd = np.clip(thrust_cmd,
                             self.min_frac * max_thrust,
                             self.max_frac * max_thrust)

        return float(thrust_cmd), float(theta)
