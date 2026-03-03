"""
plant.py — 2-DOF seat–driver dynamic model with rotary motor actuator.

State vector (4 mechanical + 1 electrical = 5 states):

    x = [x1, x2, x3, x4, tau_m]

where
    x1 = z_s - z_0       relative seat-to-chassis displacement  [m]
    x2 = dz_s            seat absolute velocity                 [m/s]
    x3 = z_d - z_s       relative driver-to-seat displacement   [m]
    x4 = dz_d            driver absolute velocity               [m/s]
    tau_m                 current motor torque (electrical lag)  [N·m]

Equations of motion
-------------------
    m_s_eff · ẍ_s = -k1·x1 - c1·(x2 - dz0) + k2·x3 + c2·(x4 - x2) + F_a
    m_d     · ẍ_d = -k2·x3 - c2·(x4 - x2)

Actuator force
--------------
    F_a = N · (tau_m - tau_friction)
    tau_friction = b_v · θ̇  +  τ_c · tanh(α · θ̇)        (smooth Coulomb)
    θ̇ = N · ẋ_s

Motor electrical dynamics (1st-order lag):
    τ̇_m = (τ_cmd - τ_m) / τ_e
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import signal as sig

from .params import SeatParams

# Coulomb friction smoothing parameter (larger → sharper)
_COULOMB_ALPHA: float = 100.0


# ---------------------------------------------------------------------------
# State-space matrices (linear, for controller design)
# ---------------------------------------------------------------------------

def build_state_space(p: SeatParams):
    """Build the continuous-time state-space representation of the *linear*
    2-DOF seat–driver plant (no friction, no saturation, no electrical lag).

    Returns
    -------
    A  : (4, 4) ndarray      — state matrix
    Bu : (4, 1) ndarray      — actuator input matrix  (input = F_a [N])
    Bw : (4, 1) ndarray      — disturbance input matrix (input = dz0 [m/s])
    C  : (1, 4) ndarray      — output matrix  (output = driver acceleration)
    D  : (1, 2) ndarray      — direct feed-through [u, w]
    ss : scipy.signal.StateSpace  — for frequency-domain analysis
    """
    ms = p.effective_seat_mass  # m_s + J_m·N²
    md = p.mass_driver
    k1, c1 = p.k1, p.c1
    k2, c2 = p.k2, p.c2

    # ---- A (4×4) ----
    #   ẋ1 = ż_s - ż_0 = x2 - ż_0
    #   ẋ2 = z̈_s = (1/ms)[-k1 x1 - c1(x2 - ż_0) + k2 x3 + c2(x4-x2) + Fa]
    #   ẋ3 = ż_d - ż_s = x4 - x2
    #   ẋ4 = z̈_d = (1/md)[-k2 x3 - c2(x4 - x2)]

    A = np.zeros((4, 4))
    # row 0: ẋ1 = x2  (- ż_0 goes into Bw)
    A[0, 1] = 1.0

    # row 1: ẋ2 = (-k1/ms)x1 + (-(c1+c2)/ms)x2 + (k2/ms)x3 + (c2/ms)x4
    A[1, 0] = -k1 / ms
    A[1, 1] = -(c1 + c2) / ms
    A[1, 2] = k2 / ms
    A[1, 3] = c2 / ms

    # row 2: ẋ3 = -x2 + x4
    A[2, 1] = -1.0
    A[2, 3] = 1.0

    # row 3: ẋ4 = (c2/md)x2 + (-k2/md)x3 + (-c2/md)x4
    A[3, 1] = c2 / md
    A[3, 2] = -k2 / md
    A[3, 3] = -c2 / md

    # ---- Bu (4×1)  — actuator force Fa enters seat acceleration ----
    Bu = np.array([[0.0], [1.0 / ms], [0.0], [0.0]])

    # ---- Bw (4×1)  — disturbance ż_0 enters rows 0 and 1 ----
    # ẋ1 += -ż_0   and   ẋ2 += (c1/ms)·ż_0
    Bw = np.array([[-1.0], [c1 / ms], [0.0], [0.0]])

    # ---- C — output = driver acceleration z̈_d ----
    # z̈_d = (c2/md)x2 + (-k2/md)x3 + (-c2/md)x4
    C = np.array([[0.0, c2 / md, -k2 / md, -c2 / md]])

    # ---- D (1×2) [Fa, ż_0] → z̈_d direct feed-through = 0 ----
    D = np.zeros((1, 2))

    # Build a scipy StateSpace for Bode / frequency analysis
    # Combined input [Fa, dz0]
    B_combined = np.hstack([Bu, Bw])
    ss_obj = sig.StateSpace(A, B_combined, C, D)

    return A, Bu, Bw, C, D, ss_obj


# ---------------------------------------------------------------------------
# Actuator physics (nonlinear)
# ---------------------------------------------------------------------------

def actuator_force(tau_current: float, x2_seat_vel: float, p: SeatParams) -> float:
    """Compute the linear actuator force from the current motor torque state.

    Accounts for viscous + smoothed Coulomb friction.

    Parameters
    ----------
    tau_current : float
        Current motor torque (after electrical lag) [N·m].
    x2_seat_vel : float
        Seat velocity ẋ_s [m/s].
    p : SeatParams

    Returns
    -------
    F_a : float  — linear force applied to the seat [N].
    """
    N = p.gear_ratio
    theta_dot = N * x2_seat_vel  # motor angular velocity [rad/s]

    # Friction torque (smooth Coulomb via tanh)
    tau_friction = (p.viscous_friction * theta_dot
                    + p.coulomb_friction * np.tanh(_COULOMB_ALPHA * theta_dot))

    # Net torque after friction
    tau_net = tau_current - tau_friction

    # Linear force
    return N * tau_net


def saturate_torque(tau_cmd: float, p: SeatParams) -> float:
    """Hard-clip the commanded torque to the motor's physical limits."""
    return float(np.clip(tau_cmd, -p.max_torque, p.max_torque))


# ---------------------------------------------------------------------------
# Full nonlinear ODE right-hand side
# ---------------------------------------------------------------------------

def dynamics_ode(
    t: float,
    state: np.ndarray,
    controller_compute: Callable,
    road_func: Callable,
    p: SeatParams,
    *,
    enable_electrical: bool = True,
) -> np.ndarray:
    """Right-hand side for ``solve_ivp``.

    Parameters
    ----------
    t : float
        Current time [s].
    state : ndarray, shape (5,)
        [x1, x2, x3, x4, tau_m].
    controller_compute : callable  (t, x_mech_4) → tau_cmd [N·m]
        Controller function returning desired motor torque.
    road_func : callable  (t) → (z0, dz0)
        Road excitation returning displacement and velocity.
    p : SeatParams
    enable_electrical : bool
        If False, bypass electrical lag (tau_m = tau_cmd instantly).

    Returns
    -------
    dstate : ndarray, shape (5,)
    """
    x1, x2, x3, x4, tau_m = state
    ms = p.effective_seat_mass
    md = p.mass_driver
    k1, c1 = p.k1, p.c1
    k2, c2 = p.k2, p.c2

    # Road input
    _z0, dz0 = road_func(t)

    # Controller output (desired motor torque)
    x_mech = np.array([x1, x2, x3, x4])
    tau_cmd = controller_compute(t, x_mech)
    tau_cmd = saturate_torque(tau_cmd, p)

    # Electrical dynamics
    if enable_electrical and p.electrical_tau > 0:
        dtau_m = (tau_cmd - tau_m) / p.electrical_tau
    else:
        tau_m = tau_cmd
        dtau_m = 0.0

    # Actuator force
    Fa = actuator_force(tau_m, x2, p)

    # Equations of motion
    # ẋ1 = x2 - dz0
    dx1 = x2 - dz0

    # ẋ2 = (1/ms)[-k1·x1 - c1·(x2 - dz0) + k2·x3 + c2·(x4 - x2) + Fa]
    dx2 = (-k1 * x1 - c1 * (x2 - dz0) + k2 * x3 + c2 * (x4 - x2) + Fa) / ms

    # ẋ3 = x4 - x2
    dx3 = x4 - x2

    # ẋ4 = (1/md)[-k2·x3 - c2·(x4 - x2)]
    dx4 = (-k2 * x3 - c2 * (x4 - x2)) / md

    return np.array([dx1, dx2, dx3, dx4, dtau_m])


# ---------------------------------------------------------------------------
# Extended ODE for controllers with internal states (H∞, adaptive)
# ---------------------------------------------------------------------------

def dynamics_ode_extended(
    t: float,
    state: np.ndarray,
    controller,
    road_func: Callable,
    p: SeatParams,
    n_ctrl_states: int,
    *,
    enable_electrical: bool = True,
) -> np.ndarray:
    """ODE RHS that supports controllers with internal dynamic states.

    The state vector layout is:
        [x1, x2, x3, x4, tau_m, ctrl_state_1, ..., ctrl_state_n]

    Parameters
    ----------
    controller : Controller instance with `compute_extended(t, x_mech, ctrl_states)`
                 that returns (tau_cmd, d_ctrl_states).
    n_ctrl_states : int
        Number of controller internal states.
    """
    x_plant = state[:5]
    ctrl_states = state[5:5 + n_ctrl_states] if n_ctrl_states > 0 else np.array([])

    x1, x2, x3, x4, tau_m = x_plant
    ms = p.effective_seat_mass
    md = p.mass_driver
    k1, c1 = p.k1, p.c1
    k2, c2 = p.k2, p.c2

    _z0, dz0 = road_func(t)
    x_mech = np.array([x1, x2, x3, x4])

    # Controller with internal states
    tau_cmd, d_ctrl = controller.compute_extended(t, x_mech, ctrl_states)
    tau_cmd = saturate_torque(tau_cmd, p)

    # Electrical dynamics
    if enable_electrical and p.electrical_tau > 0:
        dtau_m = (tau_cmd - tau_m) / p.electrical_tau
    else:
        tau_m = tau_cmd
        dtau_m = 0.0

    Fa = actuator_force(tau_m, x2, p)

    dx1 = x2 - dz0
    dx2 = (-k1 * x1 - c1 * (x2 - dz0) + k2 * x3 + c2 * (x4 - x2) + Fa) / ms
    dx3 = x4 - x2
    dx4 = (-k2 * x3 - c2 * (x4 - x2)) / md

    d_plant = np.array([dx1, dx2, dx3, dx4, dtau_m])
    return np.concatenate([d_plant, d_ctrl])
