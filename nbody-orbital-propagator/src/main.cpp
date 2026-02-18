/**
 * High-Fidelity N-Body Orbital Propagator
 * ========================================
 * Propagates spacecraft state vectors under the gravitational influence
 * of multiple celestial bodies using a 4th-Order Runge-Kutta integrator.
 * Includes impulsive Hohmann transfer maneuver (LEO -> GEO).
 *
 * Output: CSV telemetry file with timestamps and state vectors.
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <iomanip>
#include <sstream>
#include <algorithm>

// ============================================================================
// Constants
// ============================================================================
constexpr double G = 6.67430e-11;          // Gravitational constant [m^3 kg^-1 s^-2]
constexpr double PI = 3.14159265358979323846;

// Standard gravitational parameters [m^3/s^2]
constexpr double MU_EARTH = 3.986004418e14;
constexpr double MU_MOON  = 4.9048695e12;
constexpr double MU_SUN   = 1.32712440018e20;

// Orbital radii [m]
constexpr double R_EARTH = 6.371e6;             // Earth mean radius
constexpr double R_LEO   = R_EARTH + 400e3;     // LEO altitude ~400 km
constexpr double R_GEO   = 42164e3;             // GEO radius from Earth center

// Earth-Moon system
constexpr double EARTH_MOON_DIST = 384400e3;    // Average Earth-Moon distance [m]

// Time
constexpr double SECONDS_PER_DAY = 86400.0;

// ============================================================================
// 3D Vector Type
// ============================================================================
struct Vec3 {
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }

    double norm() const { return std::sqrt(x * x + y * y + z * z); }
    double norm_sq() const { return x * x + y * y + z * z; }
    Vec3 normalized() const { double n = norm(); return {x / n, y / n, z / n}; }
};

Vec3 operator*(double s, const Vec3& v) { return v * s; }

// ============================================================================
// State Vector (6-DOF Cartesian)
// ============================================================================
struct StateVector {
    Vec3 pos;   // Position [m]
    Vec3 vel;   // Velocity [m/s]

    StateVector operator+(const StateVector& o) const {
        return {pos + o.pos, vel + o.vel};
    }
    StateVector operator*(double s) const {
        return {pos * s, vel * s};
    }
};

// ============================================================================
// Celestial Body
// ============================================================================
class CelestialBody {
public:
    std::string name;
    double mu;          // Standard gravitational parameter [m^3/s^2]
    Vec3 position;      // Current position [m] (in inertial frame)
    Vec3 velocity;      // Current velocity [m/s]

    // For simple circular orbit propagation (Moon around Earth)
    double orbit_radius;
    double orbit_period;
    double orbit_phase0;  // Initial phase [rad]
    bool orbiting;        // Whether this body orbits (simplified)

    CelestialBody(const std::string& name_, double mu_, Vec3 pos_, Vec3 vel_,
                  double orb_r = 0, double orb_T = 0, double phase0 = 0, bool orb = false)
        : name(name_), mu(mu_), position(pos_), velocity(vel_),
          orbit_radius(orb_r), orbit_period(orb_T), orbit_phase0(phase0), orbiting(orb) {}

    // Update position for orbiting bodies (simplified circular orbit)
    void update_position(double t) {
        if (orbiting && orbit_period > 0) {
            double omega = 2.0 * PI / orbit_period;
            double phase = orbit_phase0 + omega * t;
            position.x = orbit_radius * std::cos(phase);
            position.y = orbit_radius * std::sin(phase);
            position.z = 0.0;
        }
    }
};

// ============================================================================
// Maneuver Definition
// ============================================================================
struct Maneuver {
    double time;        // Time of maneuver [s]
    Vec3 delta_v;       // Impulsive velocity change [m/s]
    bool executed;      // Has this maneuver been applied?
    std::string label;

    Maneuver(double t, Vec3 dv, const std::string& lbl)
        : time(t), delta_v(dv), executed(false), label(lbl) {}
};

// ============================================================================
// Spacecraft
// ============================================================================
class Spacecraft {
public:
    std::string name;
    StateVector state;
    std::vector<Maneuver> maneuvers;

    Spacecraft(const std::string& name_, const StateVector& initial_state)
        : name(name_), state(initial_state) {}

    void add_maneuver(const Maneuver& m) {
        maneuvers.push_back(m);
    }

    // Check and apply any maneuvers at the current time
    bool apply_maneuvers(double t, double dt) {
        bool applied = false;
        for (auto& m : maneuvers) {
            if (!m.executed && t >= m.time && t < m.time + dt) {
                state.vel += m.delta_v;
                m.executed = true;
                std::cout << "[MANEUVER] t=" << std::fixed << std::setprecision(1)
                          << t / 3600.0 << " hr | " << m.label
                          << " | dv=" << m.delta_v.norm() << " m/s\n";
                applied = true;
            }
        }
        return applied;
    }
};

// ============================================================================
// N-Body Gravitational Acceleration
// ============================================================================
// a = sum_i  mu_i * (r_i - r) / |r_i - r|^3
Vec3 compute_gravity(const Vec3& sc_pos, const std::vector<CelestialBody>& bodies) {
    Vec3 accel;
    for (const auto& body : bodies) {
        Vec3 r_diff = body.position - sc_pos;
        double dist = r_diff.norm();
        if (dist < 1.0) continue;  // Avoid singularity
        double dist_cubed = dist * dist * dist;
        accel += r_diff * (body.mu / dist_cubed);
    }
    return accel;
}

// ============================================================================
// RK4 Integrator
// ============================================================================
// State derivative: dx/dt = [vel, accel]
StateVector state_derivative(const StateVector& s, const std::vector<CelestialBody>& bodies) {
    Vec3 accel = compute_gravity(s.pos, bodies);
    return {s.vel, accel};
}

StateVector rk4_step(const StateVector& state, double dt,
                     const std::vector<CelestialBody>& bodies, double t,
                     std::vector<CelestialBody>& mutable_bodies) {
    // k1: derivative at current state
    StateVector k1 = state_derivative(state, bodies);

    // k2: derivative at midpoint using k1
    // Update body positions to t + dt/2 for k2
    for (auto& b : mutable_bodies) b.update_position(t + dt / 2.0);
    StateVector s2 = state + k1 * (dt / 2.0);
    StateVector k2 = state_derivative(s2, mutable_bodies);

    // k3: derivative at midpoint using k2
    StateVector s3 = state + k2 * (dt / 2.0);
    StateVector k3 = state_derivative(s3, mutable_bodies);

    // k4: derivative at end using k3
    for (auto& b : mutable_bodies) b.update_position(t + dt);
    StateVector s4 = state + k3 * dt;
    StateVector k4 = state_derivative(s4, mutable_bodies);

    // Weighted average
    StateVector delta;
    delta.pos = (k1.pos + 2.0 * k2.pos + 2.0 * k3.pos + k4.pos) * (dt / 6.0);
    delta.vel = (k1.vel + 2.0 * k2.vel + 2.0 * k3.vel + k4.vel) * (dt / 6.0);

    StateVector new_state;
    new_state.pos = state.pos + delta.pos;
    new_state.vel = state.vel + delta.vel;

    // Restore body positions to t + dt for next step
    for (auto& b : mutable_bodies) b.update_position(t + dt);

    return new_state;
}

// ============================================================================
// Hohmann Transfer Calculator
// ============================================================================
struct HohmannTransfer {
    double dv1;         // First burn [m/s]
    double dv2;         // Second burn [m/s]
    double transfer_time; // Half-period of transfer orbit [s]

    static HohmannTransfer compute(double mu, double r1, double r2) {
        HohmannTransfer h;
        double v_circ1 = std::sqrt(mu / r1);
        double v_circ2 = std::sqrt(mu / r2);

        double a_transfer = (r1 + r2) / 2.0;
        double v_peri = std::sqrt(mu * (2.0 / r1 - 1.0 / a_transfer));
        double v_apo  = std::sqrt(mu * (2.0 / r2 - 1.0 / a_transfer));

        h.dv1 = v_peri - v_circ1;
        h.dv2 = v_circ2 - v_apo;
        h.transfer_time = PI * std::sqrt(a_transfer * a_transfer * a_transfer / mu);

        return h;
    }
};

// ============================================================================
// CSV Telemetry Recorder
// ============================================================================
class TelemetryRecorder {
    std::ofstream file;
public:
    TelemetryRecorder(const std::string& filename) {
        file.open(filename);
        file << "time_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,altitude_km,speed_ms,maneuver\n";
    }

    void record(double t, const StateVector& s, bool maneuver = false) {
        double alt_km = (s.pos.norm() - R_EARTH) / 1000.0;
        double speed = s.vel.norm();
        file << std::fixed << std::setprecision(6)
             << t << ","
             << s.pos.x << "," << s.pos.y << "," << s.pos.z << ","
             << s.vel.x << "," << s.vel.y << "," << s.vel.z << ","
             << std::setprecision(3) << alt_km << ","
             << std::setprecision(3) << speed << ","
             << (maneuver ? "1" : "0") << "\n";
    }

    ~TelemetryRecorder() {
        if (file.is_open()) file.close();
    }
};

// ============================================================================
// Main Simulation
// ============================================================================
int main() {
    std::cout << "==============================================\n";
    std::cout << " N-Body Orbital Propagator v1.0\n";
    std::cout << " RK4 Integration | Hohmann Transfer LEO->GEO\n";
    std::cout << "==============================================\n\n";

    // ------------------------------------------------------------------
    // 1. Define celestial bodies
    // ------------------------------------------------------------------
    // Earth at origin
    CelestialBody earth("Earth", MU_EARTH, Vec3(0, 0, 0), Vec3(0, 0, 0));

    // Moon in circular orbit around Earth (simplified)
    double moon_period = 27.3217 * SECONDS_PER_DAY;  // Sidereal period
    CelestialBody moon("Moon", MU_MOON,
                       Vec3(EARTH_MOON_DIST, 0, 0), Vec3(0, 0, 0),
                       EARTH_MOON_DIST, moon_period, 0.0, true);

    // Sun (far away, provides perturbation)
    double au = 1.496e11; // 1 AU in meters
    CelestialBody sun("Sun", MU_SUN,
                      Vec3(au, 0, 0), Vec3(0, 0, 0),
                      au, 365.25 * SECONDS_PER_DAY, 0.0, true);

    std::vector<CelestialBody> bodies = {earth, moon, sun};

    // ------------------------------------------------------------------
    // 2. Compute Hohmann transfer parameters
    // ------------------------------------------------------------------
    auto hohmann = HohmannTransfer::compute(MU_EARTH, R_LEO, R_GEO);

    std::cout << "Hohmann Transfer LEO -> GEO:\n";
    std::cout << "  LEO radius:      " << R_LEO / 1e3 << " km\n";
    std::cout << "  GEO radius:      " << R_GEO / 1e3 << " km\n";
    std::cout << "  dv1 (inject):    " << std::fixed << std::setprecision(2)
              << hohmann.dv1 << " m/s\n";
    std::cout << "  dv2 (circularize): " << hohmann.dv2 << " m/s\n";
    std::cout << "  Transfer time:   " << hohmann.transfer_time / 3600.0
              << " hours\n";
    std::cout << "  Total dv:        " << hohmann.dv1 + hohmann.dv2 << " m/s\n\n";

    // ------------------------------------------------------------------
    // 3. Initialize spacecraft in LEO (circular orbit)
    // ------------------------------------------------------------------
    double v_leo = std::sqrt(MU_EARTH / R_LEO);
    StateVector initial_state;
    initial_state.pos = Vec3(R_LEO, 0, 0);
    initial_state.vel = Vec3(0, v_leo, 0);

    Spacecraft sc("TransferVehicle", initial_state);

    // Schedule maneuvers
    // Burn 1: After 2 orbits of coasting (~3 hours)
    double t_burn1 = 2.0 * 2.0 * PI * std::sqrt(R_LEO * R_LEO * R_LEO / MU_EARTH);
    Vec3 dv1_vec = Vec3(0, hohmann.dv1, 0);  // Prograde at (R_LEO, 0, 0)

    // We need to compute the direction at burn time properly
    // For now, the burn happens when the spacecraft is near its initial position
    // The dv is applied in the velocity direction at burn time
    sc.add_maneuver(Maneuver(t_burn1, dv1_vec, "Hohmann Burn 1 (LEO departure)"));

    // Burn 2: At apoapsis of transfer orbit
    double t_burn2 = t_burn1 + hohmann.transfer_time;
    Vec3 dv2_vec = Vec3(0, -hohmann.dv2, 0);  // Prograde direction at apoapsis
    sc.add_maneuver(Maneuver(t_burn2, dv2_vec, "Hohmann Burn 2 (GEO insertion)"));

    std::cout << "Maneuver Schedule:\n";
    std::cout << "  Burn 1 at t = " << t_burn1 / 3600.0 << " hr\n";
    std::cout << "  Burn 2 at t = " << t_burn2 / 3600.0 << " hr\n\n";

    // ------------------------------------------------------------------
    // 4. Simulation parameters
    // ------------------------------------------------------------------
    double dt = 10.0;                     // Time step [s]
    double t_total = 30.0 * SECONDS_PER_DAY;  // 30-day simulation
    int total_steps = static_cast<int>(t_total / dt);
    int record_interval = 10;             // Record every 10 steps

    std::cout << "Simulation Parameters:\n";
    std::cout << "  dt:           " << dt << " s\n";
    std::cout << "  Duration:     " << t_total / SECONDS_PER_DAY << " days\n";
    std::cout << "  Total steps:  " << total_steps << "\n\n";

    // ------------------------------------------------------------------
    // 5. Open telemetry recorder
    // ------------------------------------------------------------------
    TelemetryRecorder telemetry("output/trajectory.csv");

    // Record initial state
    telemetry.record(0.0, sc.state);

    // ------------------------------------------------------------------
    // 6. Main integration loop
    // ------------------------------------------------------------------
    std::cout << "Running simulation..." << std::flush;
    double t = 0.0;

    for (int step = 0; step < total_steps; ++step) {
        // Update celestial body positions
        for (auto& b : bodies) b.update_position(t);

        // Check for maneuvers (apply impulsive dv)
        // At burn time, we recompute dv direction along current velocity
        for (auto& m : sc.maneuvers) {
            if (!m.executed && t >= m.time && t < m.time + dt) {
                // Apply dv in the prograde direction
                Vec3 v_hat = sc.state.vel.normalized();
                double dv_mag = m.delta_v.norm();
                sc.state.vel += v_hat * dv_mag;
                m.executed = true;

                std::cout << "\n[MANEUVER] t=" << std::fixed << std::setprecision(1)
                          << t / 3600.0 << " hr | " << m.label
                          << " | dv=" << dv_mag << " m/s"
                          << " | alt=" << (sc.state.pos.norm() - R_EARTH) / 1e3 << " km";

                telemetry.record(t, sc.state, true);
            }
        }

        // RK4 integration step
        sc.state = rk4_step(sc.state, dt, bodies, t, bodies);

        t += dt;

        // Record telemetry periodically
        if (step % record_interval == 0) {
            telemetry.record(t, sc.state);
        }

        // Progress indicator
        if (step % (total_steps / 10) == 0) {
            std::cout << "." << std::flush;
        }
    }

    std::cout << "\n\nSimulation complete!\n";

    // ------------------------------------------------------------------
    // 7. Print final state
    // ------------------------------------------------------------------
    double final_alt = (sc.state.pos.norm() - R_EARTH) / 1e3;
    double final_speed = sc.state.vel.norm();
    double v_geo = std::sqrt(MU_EARTH / R_GEO);

    std::cout << "\nFinal State @ t = " << t / SECONDS_PER_DAY << " days:\n";
    std::cout << "  Position: [" << sc.state.pos.x / 1e3 << ", "
              << sc.state.pos.y / 1e3 << ", "
              << sc.state.pos.z / 1e3 << "] km\n";
    std::cout << "  Velocity: [" << sc.state.vel.x << ", "
              << sc.state.vel.y << ", "
              << sc.state.vel.z << "] m/s\n";
    std::cout << "  Altitude: " << final_alt << " km\n";
    std::cout << "  Speed:    " << final_speed << " m/s (GEO circular: "
              << v_geo << " m/s)\n";
    std::cout << "\nTelemetry written to output/trajectory.csv\n";

    return 0;
}
