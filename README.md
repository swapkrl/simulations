# Simulations

A collection of aerospace simulation projects covering orbital mechanics, attitude control, and powered descent.

## Projects

### [Lunar Lander Monte Carlo](./lunar-lander-montecarlo)
Powered-descent trajectory simulator with 10,000 randomized off-nominal conditions via Monte Carlo analysis. Statistically proves mission safety with >99.7% safe touchdown rate.

### [N-Body Orbital Propagator](./nbody-orbital-propagator)
High-fidelity numerical simulator that propagates spacecraft state vectors under multi-body gravitational influence (Earth, Moon, Sun) using RK4 integration. Includes a Hohmann transfer from LEO to GEO.

### [Spacecraft ADCS](./spacecraft-adcs)
Rigid-body attitude dynamics simulator with a closed-loop quaternion-based PD controller and 3-axis reaction wheel array. Stabilizes a tumbling spacecraft to a commanded target attitude.

## Tech Stack
- **Python** (NumPy, Matplotlib, Pandas) — Lunar Lander MC, Spacecraft ADCS
- **C++17** (CMake) — N-Body Orbital Propagator

## License
MIT 2026
