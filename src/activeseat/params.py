"""
params.py — Central parameter dataclasses for the 2-DOF active seat suspension.

Every tuneable physical, controller, and simulation quantity lives here as a
typed, documented dataclass field with physically reasonable defaults in SI
units (kg, N, m, s, rad).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Physical / plant parameters
# ---------------------------------------------------------------------------

@dataclass
class SeatParams:
    """Physical parameters of the 2-DOF seat–driver system and rotary actuator.

    Coordinate convention (vertical, positive upward):
        z_0(t) — chassis / road excitation input
        z_s(t) — seat frame displacement
        z_d(t) — driver (human) displacement

    Spring–damper 1 connects chassis ↔ seat.
    Spring–damper 2 connects seat   ↔ driver.
    Actuator acts in parallel with spring–damper 1 (chassis ↔ seat).
    """

    # --- Masses -----------------------------------------------------------
    mass_seat: float = 15.0          # m_s  [kg]  seat frame + cushion base
    mass_driver: float = 75.0        # m_d  [kg]  driver body mass

    # --- Primary suspension (chassis ↔ seat) ------------------------------
    k1: float = 20_000.0             # [N/m]   spring stiffness
    c1: float = 1_500.0              # [N·s/m] viscous damping

    # --- Cushion (seat ↔ driver) ------------------------------------------
    k2: float = 50_000.0             # [N/m]   spring stiffness
    c2: float = 2_000.0              # [N·s/m] viscous damping

    # --- Rotary motor actuator --------------------------------------------
    gear_ratio: float = 628.0        # N  [rad/m]  ball-screw (10 mm lead: 2π/0.01)
    max_torque: float = 5.0          # τ_max [N·m]  peak motor torque
    motor_inertia: float = 2e-5      # J_m [kg·m²]  rotor inertia (small servo)
    viscous_friction: float = 0.01   # b_v [N·m·s/rad]  motor viscous friction
    coulomb_friction: float = 0.02   # τ_c [N·m]  Coulomb friction magnitude
    electrical_tau: float = 5e-3     # τ_e [s]  electrical time constant (1st-order lag)

    # --- Derived (computed post-init) -------------------------------------
    max_force: float = field(init=False)        # F_max = N · τ_max  [N]
    reflected_mass: float = field(init=False)    # J_m · N²  [kg]

    def __post_init__(self) -> None:
        self.max_force = self.gear_ratio * self.max_torque
        self.reflected_mass = self.motor_inertia * self.gear_ratio ** 2

    @property
    def effective_seat_mass(self) -> float:
        """Effective seat mass including reflected motor inertia."""
        return self.mass_seat + self.reflected_mass


# ---------------------------------------------------------------------------
# Controller parameters
# ---------------------------------------------------------------------------

@dataclass
class ControllerParams:
    """Tuning parameters for all supported controllers.

    Fields are grouped by controller type; unused fields are simply ignored
    by controllers that don't need them.
    """

    controller_type: str = "lqr"  # "passive" | "lqr" | "hinf" | "adaptive"

    # --- LQR weights ------------------------------------------------------
    #   Q = diag(Q_diag),  cost = ∫ (x'Qx + R u²) dt
    Q_diag: List[float] = field(default_factory=lambda: [5e5, 100.0, 1e6, 100.0])
    R: float = 1e-6

    # --- H∞ parameters ----------------------------------------------------
    hinf_gamma: float = 5.0          # performance bound γ
    hinf_perf_weight_accel: float = 500.0   # weight on driver acceleration
    hinf_perf_weight_force: float = 1e-4    # weight on control effort

    # --- Adaptive (MRAC) parameters ---------------------------------------
    adaptive_gamma_gain: float = 500.0      # adaptation rate Γ
    adaptive_ref_wn: float = 12.0           # reference model ωn [rad/s]
    adaptive_ref_zeta: float = 0.8          # reference model damping ratio


# ---------------------------------------------------------------------------
# Road / excitation configuration
# ---------------------------------------------------------------------------

@dataclass
class RoadConfig:
    """Road excitation profile specification."""

    road_type: str = "bump"          # "bump" | "sinusoidal" | "random"

    # Bump (half-sine) parameters
    bump_height: float = 0.05        # [m]
    bump_length: float = 0.5         # [m]
    vehicle_speed: float = 10.0      # [m/s]

    # Sinusoidal parameters
    sine_amplitude: float = 0.01     # [m]
    sine_frequency: float = 3.0      # [Hz]

    # Random (ISO 8608) parameters
    iso_class: str = "C"             # "A" through "E"
    iso_speed: float = 20.0          # vehicle speed [m/s]
    iso_seed: int = 42               # RNG seed for reproducibility


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """Numerical integration / simulation settings."""

    duration: float = 2.0            # total simulation time [s]
    dt: float = 1e-3                 # output time-step (for dense output) [s]
    solver: str = "Radau"            # scipy solve_ivp method
    rtol: float = 1e-8               # relative tolerance
    atol: float = 1e-10              # absolute tolerance
    max_step: float = 1e-3           # maximum solver step [s]

    # Frequency sweep (for transmissibility)
    freq_start: float = 0.5          # [Hz]
    freq_end: float = 20.0           # [Hz]
    freq_points: int = 80            # number of frequency points
    freq_steady_cycles: int = 10     # cycles to simulate per freq
