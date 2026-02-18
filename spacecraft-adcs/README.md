# Spacecraft ADCS Simulator

Rigid-body attitude dynamics simulator with a closed-loop **quaternion-based PD controller** and **3-axis reaction wheel array**. Stabilizes a tumbling spacecraft from an arbitrary initial angular velocity to a commanded target attitude.

## Mathematical Models

### Quaternion Kinematics (gimbal-lock free)

$$\dot{q} = \frac{1}{2} \Omega(\omega) \cdot q$$

### Euler's Rotational Equations

$$\dot{\omega} = I^{-1} \left( M_{ext} - \omega \times (I \cdot \omega) \right)$$

### PD Control Law

$$T_{cmd} = -K_p \cdot q_{err,vec} - K_d \cdot \omega$$

### Reaction Wheel Model

- Angular momentum: L = I_wheel × ω_wheel
- Saturates at configurable max RPM and max torque

## Quick Start

```bash
# Run simulation
cd spacecraft-adcs
python src/simulation.py

# Visualize results
pip install numpy matplotlib pandas
python scripts/visualize.py
```

## Output
- `output/adcs_telemetry.csv` — Time histories of quaternions, angular velocity, wheel speeds, torques
- `output/adcs_plots.png` — 4-panel convergence plot

## Configuration
Edit the `config` dict in `src/simulation.py`:
- `I_body`: 3×3 inertia tensor [kg⋅m²]
- `omega_init`: Initial angular velocity [rad/s]
- `Kp`, `Kd`: Controller gains
- `max_rpm`, `max_torque`: Wheel saturation limits

## Directory Structure
```
spacecraft-adcs/
├── README.md
├── .gitignore
├── src/
│   ├── quaternion_utils.py
│   ├── dynamics.py
│   ├── controller.py
│   └── simulation.py
├── scripts/
│   ├── visualize.py
│   └── requirements.txt
└── output/         (generated at runtime)
```
