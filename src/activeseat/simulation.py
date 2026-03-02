"""
simulation.py — Integration engine for the 2-DOF seat suspension.

Provides:
    run_simulation          — single (controller, road) simulation
    run_comparison          — passive vs one active controller
    run_all_controllers     — passive + LQR + H∞ + adaptive on same road
    run_parameter_sweep     — vary one parameter, collect metrics
    run_frequency_sweep     — transmissibility via repeated sinusoidal runs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .params import SeatParams, ControllerParams, RoadConfig, SimParams
from .plant import (
    build_state_space,
    dynamics_ode,
    dynamics_ode_extended,
)
from .controllers.base import Controller, PassiveController
from .controllers.lqr import LQRController
from .controllers.hinf import HinfController
from .controllers.adaptive import AdaptiveController
from .excitation import make_road_profile


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Container for a single simulation run's data."""

    controller_name: str
    road_type: str

    t: np.ndarray                # time [s]
    x1: np.ndarray               # seat–chassis relative disp [m]
    x2: np.ndarray               # seat velocity [m/s]
    x3: np.ndarray               # driver–seat relative disp [m]
    x4: np.ndarray               # driver velocity [m/s]
    tau_m: np.ndarray            # motor torque [N·m]

    z0: np.ndarray               # road displacement [m]
    dz0: np.ndarray              # road velocity [m/s]
    z_s: np.ndarray              # seat absolute displacement [m]
    z_d: np.ndarray              # driver absolute displacement [m]
    accel_driver: np.ndarray     # driver acceleration [m/s²]
    Fa: np.ndarray               # actuator linear force [N]
    power: np.ndarray            # instantaneous power [W]

    params: SeatParams = field(repr=False)


# ---------------------------------------------------------------------------
# Core simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    seat: SeatParams,
    ctrl: Controller,
    road_func: Callable,
    sim: SimParams,
    road_type_name: str = "",
) -> SimResult:
    """Integrate the nonlinear plant + controller ODE and post-process.

    Parameters
    ----------
    seat : SeatParams
    ctrl : Controller  (must have been ``design()``-ed already)
    road_func : callable  (t) → (z0, dz0)
    sim  : SimParams
    road_type_name : str  — label for the result

    Returns
    -------
    SimResult
    """
    n_ctrl = ctrl.n_states
    use_extended = n_ctrl > 0
    n_total = 5 + n_ctrl

    y0 = np.zeros(n_total)

    t_eval = np.arange(0, sim.duration + sim.dt, sim.dt)
    # Clip t_eval to t_span to avoid floating-point overshoot
    t_eval = t_eval[t_eval <= sim.duration]

    if use_extended:
        def rhs(t, y):
            return dynamics_ode_extended(
                t, y, ctrl, road_func, seat, n_ctrl, enable_electrical=True
            )
    else:
        def rhs(t, y):
            return dynamics_ode(
                t, y, ctrl.compute, road_func, seat, enable_electrical=True
            )

    sol = solve_ivp(
        rhs,
        t_span=(0, sim.duration),
        y0=y0,
        method=sim.solver,
        t_eval=t_eval,
        rtol=sim.rtol,
        atol=sim.atol,
        max_step=sim.max_step,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    t = sol.t
    x1 = sol.y[0]
    x2 = sol.y[1]
    x3 = sol.y[2]
    x4 = sol.y[3]
    tau_m_arr = sol.y[4]

    # Reconstruct road input at solution times
    z0_arr = np.zeros_like(t)
    dz0_arr = np.zeros_like(t)
    for i, ti in enumerate(t):
        z0_arr[i], dz0_arr[i] = road_func(ti)

    # Absolute displacements
    z_s = x1 + z0_arr
    z_d = x3 + z_s

    # Driver acceleration from EOM: z̈_d = (-k2 x3 - c2(x4-x2)) / m_d
    md = seat.mass_driver
    k2, c2 = seat.k2, seat.c2
    accel_driver = (-k2 * x3 - c2 * (x4 - x2)) / md

    # Actuator force (reconstructed with friction model)
    from .plant import actuator_force
    Fa = np.array([actuator_force(tau_m_arr[i], x2[i], seat) for i in range(len(t))])

    # Instantaneous power  P = Fa · ẋ_s  (ẋ_s = x2)
    power = Fa * x2

    return SimResult(
        controller_name=ctrl.name,
        road_type=road_type_name,
        t=t, x1=x1, x2=x2, x3=x3, x4=x4, tau_m=tau_m_arr,
        z0=z0_arr, dz0=dz0_arr, z_s=z_s, z_d=z_d,
        accel_driver=accel_driver, Fa=Fa, power=power,
        params=seat,
    )


# ---------------------------------------------------------------------------
# Convenience runners
# ---------------------------------------------------------------------------

def _design_controller(ctrl: Controller, seat: SeatParams) -> None:
    """Design a controller, passing extra matrices if needed (H∞)."""
    A, Bu, Bw, C, D, _ss = build_state_space(seat)
    if isinstance(ctrl, HinfController):
        ctrl.design(A, Bu, Bw, C)
    else:
        ctrl.design(A, Bu)


def run_comparison(
    seat: SeatParams,
    ctrl_params: ControllerParams,
    road_cfg: RoadConfig,
    sim: SimParams,
) -> Tuple[SimResult, SimResult]:
    """Run passive then active (specified by ctrl_params.controller_type).

    Returns (passive_result, active_result).
    """
    road_func, t_arr, z0_arr, dz0_arr = make_road_profile(road_cfg, sim)
    road_name = road_cfg.road_type

    # Passive
    passive = PassiveController(seat, ctrl_params)
    passive.design(np.zeros((4, 4)), np.zeros((4, 1)))
    res_passive = run_simulation(seat, passive, road_func, sim, road_name)

    # Active
    ctrl = _make_controller(ctrl_params.controller_type, seat, ctrl_params)
    _design_controller(ctrl, seat)
    res_active = run_simulation(seat, ctrl, road_func, sim, road_name)

    return res_passive, res_active


def run_all_controllers(
    seat: SeatParams,
    ctrl_params: ControllerParams,
    road_cfg: RoadConfig,
    sim: SimParams,
) -> Dict[str, SimResult]:
    """Run passive + all three active controllers on the same road profile.

    Returns dict mapping controller name → SimResult.
    """
    road_func, _, _, _ = make_road_profile(road_cfg, sim)
    road_name = road_cfg.road_type
    results = {}

    for ctype in ["passive", "lqr", "hinf", "adaptive"]:
        ctrl = _make_controller(ctype, seat, ctrl_params)
        _design_controller(ctrl, seat)
        ctrl.reset()
        res = run_simulation(seat, ctrl, road_func, sim, road_name)
        results[ctrl.name] = res

    return results


def _make_controller(ctype: str, seat: SeatParams, cp: ControllerParams) -> Controller:
    """Factory for controller instantiation by type string."""
    ctype = ctype.lower()
    if ctype == "passive":
        return PassiveController(seat, cp)
    elif ctype == "lqr":
        return LQRController(seat, cp)
    elif ctype == "hinf":
        return HinfController(seat, cp)
    elif ctype == "adaptive":
        return AdaptiveController(seat, cp)
    else:
        raise ValueError(f"Unknown controller type: {ctype}")


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def run_parameter_sweep(
    param_name: str,
    values: np.ndarray,
    base_seat: SeatParams,
    ctrl_params: ControllerParams,
    road_cfg: RoadConfig,
    sim: SimParams,
) -> List[Tuple[float, SimResult, SimResult]]:
    """Sweep one SeatParams field and return (value, passive, active) triples.

    Parameters
    ----------
    param_name : str
        Attribute name on SeatParams (e.g. 'mass_driver').
    values : array-like
        Values to sweep over.
    """
    from dataclasses import replace
    results = []
    for val in values:
        seat = replace(base_seat, **{param_name: float(val)})
        seat.__post_init__()  # recompute derived fields
        res_p, res_a = run_comparison(seat, ctrl_params, road_cfg, sim)
        results.append((float(val), res_p, res_a))
    return results


# ---------------------------------------------------------------------------
# Frequency sweep (transmissibility)
# ---------------------------------------------------------------------------

def run_frequency_sweep(
    seat: SeatParams,
    ctrl_params: ControllerParams,
    sim: SimParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute transmissibility by running sinusoidal sims at many frequencies.

    Returns
    -------
    freqs        : (N,) frequencies [Hz]
    T_passive    : (N,) passive transmissibility |z̈_d / z̈_0|
    T_active     : (N,) active transmissibility
    """
    from dataclasses import replace

    freqs = np.linspace(sim.freq_start, sim.freq_end, sim.freq_points)
    T_passive = np.zeros_like(freqs)
    T_active = np.zeros_like(freqs)

    for i, f in enumerate(freqs):
        # Duration = enough cycles to reach steady state
        duration = max(sim.freq_steady_cycles / f, 0.5)
        local_sim = replace(sim, duration=duration)

        road_cfg = RoadConfig(
            road_type="sinusoidal",
            sine_amplitude=0.01,
            sine_frequency=f,
        )
        road_func, _, _, _ = make_road_profile(road_cfg, local_sim)

        # Passive
        passive = PassiveController(seat, ctrl_params)
        passive.design(np.zeros((4, 4)), np.zeros((4, 1)))
        rp = run_simulation(seat, passive, road_func, local_sim, "sine")

        # Active
        ctrl = _make_controller(ctrl_params.controller_type, seat, ctrl_params)
        _design_controller(ctrl, seat)
        ra = run_simulation(seat, ctrl, road_func, local_sim, "sine")

        # Transmissibility = RMS(z̈_d) / RMS(z̈_0)
        # Use last 50% of data (steady state)
        n_half = len(rp.t) // 2

        # Road acceleration z̈_0 = -A·(2πf)²·sin(2πf·t)
        omega = 2 * np.pi * f
        accel_road_p = -road_cfg.sine_amplitude * omega**2 * np.sin(omega * rp.t[n_half:])
        accel_road_a = -road_cfg.sine_amplitude * omega**2 * np.sin(omega * ra.t[n_half:])

        rms_road_p = np.sqrt(np.mean(accel_road_p**2))
        rms_road_a = np.sqrt(np.mean(accel_road_a**2))

        rms_drv_p = np.sqrt(np.mean(rp.accel_driver[n_half:]**2))
        rms_drv_a = np.sqrt(np.mean(ra.accel_driver[n_half:]**2))

        T_passive[i] = rms_drv_p / max(rms_road_p, 1e-12)
        T_active[i] = rms_drv_a / max(rms_road_a, 1e-12)

    return freqs, T_passive, T_active
