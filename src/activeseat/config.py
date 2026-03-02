"""
config.py — YAML scenario file loading and saving.

A *scenario* is a complete description of a simulation case: physical
parameters, controller tuning, road profile, and solver settings.  This
module serialises / deserialises the four dataclasses in ``params.py`` to
and from human-readable YAML files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import yaml

from .params import SeatParams, ControllerParams, RoadConfig, SimParams


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_scenario(path: str | Path) -> Tuple[SeatParams, ControllerParams, RoadConfig, SimParams]:
    """Load a YAML scenario file and return the four parameter dataclasses.

    Parameters
    ----------
    path : str or Path
        Path to a ``.yaml`` file.

    Returns
    -------
    seat_params, ctrl_params, road_config, sim_params
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    seat = SeatParams(**data.get("seat", {}), **data.get("actuator", {}))
    ctrl = ControllerParams(**data.get("controller", {}))
    road = RoadConfig(**data.get("road", {}))
    sim  = SimParams(**data.get("simulation", {}))
    return seat, ctrl, road, sim


def save_scenario(
    path: str | Path,
    seat: SeatParams,
    ctrl: ControllerParams,
    road: RoadConfig,
    sim: SimParams,
) -> None:
    """Serialise current parameters to a YAML scenario file.

    Parameters
    ----------
    path : str or Path
        Destination ``.yaml`` file (created or overwritten).
    seat, ctrl, road, sim
        Parameter dataclass instances.
    """
    data = {
        "seat": {
            "mass_seat": seat.mass_seat,
            "mass_driver": seat.mass_driver,
            "k1": seat.k1,
            "c1": seat.c1,
            "k2": seat.k2,
            "c2": seat.c2,
        },
        "actuator": {
            "gear_ratio": seat.gear_ratio,
            "max_torque": seat.max_torque,
            "motor_inertia": seat.motor_inertia,
            "viscous_friction": seat.viscous_friction,
            "coulomb_friction": seat.coulomb_friction,
            "electrical_tau": seat.electrical_tau,
        },
        "controller": {
            "controller_type": ctrl.controller_type,
            "Q_diag": list(ctrl.Q_diag),
            "R": ctrl.R,
            "hinf_gamma": ctrl.hinf_gamma,
            "hinf_perf_weight_accel": ctrl.hinf_perf_weight_accel,
            "hinf_perf_weight_force": ctrl.hinf_perf_weight_force,
            "adaptive_gamma_gain": ctrl.adaptive_gamma_gain,
            "adaptive_ref_wn": ctrl.adaptive_ref_wn,
            "adaptive_ref_zeta": ctrl.adaptive_ref_zeta,
        },
        "road": {
            "road_type": road.road_type,
            "bump_height": road.bump_height,
            "bump_length": road.bump_length,
            "vehicle_speed": road.vehicle_speed,
            "sine_amplitude": road.sine_amplitude,
            "sine_frequency": road.sine_frequency,
            "iso_class": road.iso_class,
            "iso_speed": road.iso_speed,
            "iso_seed": road.iso_seed,
        },
        "simulation": {
            "duration": sim.duration,
            "dt": sim.dt,
            "solver": sim.solver,
            "rtol": sim.rtol,
            "atol": sim.atol,
            "max_step": sim.max_step,
            "freq_start": sim.freq_start,
            "freq_end": sim.freq_end,
            "freq_points": sim.freq_points,
            "freq_steady_cycles": sim.freq_steady_cycles,
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def list_scenarios(directory: str | Path) -> list[str]:
    """Return sorted list of ``.yaml`` filenames in *directory*."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.yaml"))
