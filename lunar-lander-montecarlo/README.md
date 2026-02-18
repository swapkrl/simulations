# Lunar Lander Monte Carlo Simulation

Powered-descent trajectory simulator subjected to **10,000 randomized off-nominal conditions** via Monte Carlo analysis. Statistically proves mission safety by demonstrating >99.7% of landings achieve a touchdown velocity < 2 m/s.

## Architecture

```
Truth Model ──► Navigation Model ──► Guidance Model ──► Thrust Command
     │              (sensor noise)      (P-nav)              │
     └──────────── Physics Engine ◄──────────────────────────┘
```

## Mathematical Models

### Powered Descent Dynamics (2-DOF)
- Vertical + horizontal motion under lunar gravity (1.625 m/s²)
- Mass depletion: `ṁ = −T / (Isp · g₀)`
- RK4 integration

### Stochastic Dispersions
- **Altimeter noise**: N(0, σ²_alt)
- **Thrust variance**: Cmd × N(1.0, σ²_thrust)
- **Initial condition errors**: altitude, velocity biases
- **Engine performance**: Isp bias, thrust scaling

### Guidance
- Proportional navigation for vertical braking
- Proportional horizontal deceleration

## Quick Start

```bash
cd lunar-lander-montecarlo

# Install dependencies
pip install -r scripts/requirements.txt

# Run single nominal simulation
python src/simulation.py

# Run full Monte Carlo (10,000 runs)
python src/montecarlo.py -n 10000

# Generate statistical report
python scripts/report.py
```

## Configuration

Edit `config/dispersions.json` to adjust 1σ dispersion values:
- `alt_bias_sigma`: Initial altitude error [m]
- `thrust_scale_sigma`: Engine thrust scaling variance
- `sigma_alt_mean`: Altimeter noise level [m]
- See file for all parameters

## Output
- `output/monte_carlo_results.csv` — Per-run touchdown conditions
- `output/statistics_summary.json` — Statistical summary
- `output/monte_carlo_report.png` — 4-panel dashboard

## Directory Structure
```
lunar-lander-montecarlo/
├── README.md
├── .gitignore
├── config/
│   └── dispersions.json
├── src/
│   ├── physics.py
│   ├── guidance.py
│   ├── navigation.py
│   ├── simulation.py
│   └── montecarlo.py
├── scripts/
│   ├── report.py
│   └── requirements.txt
└── output/         (generated at runtime)
```
