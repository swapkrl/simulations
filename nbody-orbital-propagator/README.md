# N-Body Orbital Propagator

High-fidelity numerical simulator that propagates spacecraft state vectors under the gravitational influence of multiple celestial bodies (Earth, Moon, Sun) using a **4th-Order Runge-Kutta (RK4)** integrator. Includes an impulsive **Hohmann transfer** from LEO (400 km) to GEO (35,793 km).

## Mathematical Models

### N-Body Gravitational Acceleration

$$\vec{a} = \sum_{i=1}^{N} \mu_i \frac{\vec{r}_i - \vec{r}}{|\vec{r}_i - \vec{r}|^3}$$

### RK4 Integration

The state vector **x** = [x, y, z, vx, vy, vz]ᵀ is updated at each time step using the classical 4th-order Runge-Kutta method (Euler integration is **not** used due to energy drift).

### Hohmann Transfer

Two impulsive burns are computed analytically:
- **Burn 1** (LEO departure): Δv₁ = v_transfer_periapsis − v_circular_LEO
- **Burn 2** (GEO insertion): Δv₂ = v_circular_GEO − v_transfer_apoapsis

## Build & Run

### Prerequisites
- C++17 compiler (g++, MSVC, or Clang)
- CMake ≥ 3.15
- Python 3.8+ with matplotlib, numpy, pandas

### Build (C++)
```bash
cmake -B build
cmake --build build --config Release
```

### Run Simulation
```bash
./build/nbody_propagator        # Linux/Mac
build\Release\nbody_propagator  # Windows
```

### Visualize Results
```bash
pip install -r scripts/requirements.txt
python scripts/visualize.py
```

## Output
- `output/trajectory.csv` — Time-stamped state vectors
- `output/trajectory_3d.png` — 3D orbital trajectory plot
- `output/altitude_speed.png` — Altitude and speed vs. time

## Directory Structure
```
nbody-orbital-propagator/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── src/
│   └── main.cpp
├── scripts/
│   ├── visualize.py
│   └── requirements.txt
└── output/          (generated at runtime)
```
