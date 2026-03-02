"""
base.py — Abstract controller interface and passive (no-control) implementation.

Every controller implements:
    design(A, Bu, params)  — one-time offline synthesis
    compute(t, x_mech)     — real-time control law  → τ_cmd [N·m]
    reset()                — clear internal state for a fresh run

Controllers that carry internal dynamic states (H∞, adaptive) additionally
implement:
    n_states               — number of internal ODE states
    compute_extended(t, x_mech, ctrl_states) → (τ_cmd, d_ctrl_states)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ..params import SeatParams, ControllerParams


class Controller(ABC):
    """Abstract base class for seat-suspension controllers."""

    def __init__(self, seat_params: SeatParams, ctrl_params: ControllerParams):
        self.seat_params = seat_params
        self.ctrl_params = ctrl_params
        self._designed = False

    @abstractmethod
    def design(self, A: np.ndarray, Bu: np.ndarray) -> None:
        """Perform offline controller synthesis (Riccati, γ-iteration, …)."""
        ...

    @abstractmethod
    def compute(self, t: float, x_mech: np.ndarray) -> float:
        """Return commanded motor torque τ_cmd [N·m]."""
        ...

    def reset(self) -> None:
        """Reset any internal state for a new simulation run."""
        pass

    # --- Extended interface for controllers with dynamic states -----------

    @property
    def n_states(self) -> int:
        """Number of additional ODE states carried by this controller."""
        return 0

    def compute_extended(
        self, t: float, x_mech: np.ndarray, ctrl_states: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Return (τ_cmd, d_ctrl_states).  Default delegates to ``compute``."""
        return self.compute(t, x_mech), np.array([])

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Passive controller (F_a = 0 always)
# ---------------------------------------------------------------------------

class PassiveController(Controller):
    """No active control — actuator is off."""

    def design(self, A: np.ndarray, Bu: np.ndarray) -> None:
        self._designed = True

    def compute(self, t: float, x_mech: np.ndarray) -> float:
        return 0.0

    @property
    def name(self) -> str:
        return "Passive"
