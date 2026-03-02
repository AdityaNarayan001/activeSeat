"""
lqr.py — Linear-Quadratic Regulator for the 2-DOF seat suspension.

Solves the continuous algebraic Riccati equation (CARE):

    A'P + PA - PBR⁻¹B'P + Q = 0

and computes the state-feedback gain K = R⁻¹ B' P.

The control law in force space is  F_a = -K x,  which is converted to
motor torque  τ_cmd = F_a / N  (gear ratio).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from .base import Controller
from ..params import SeatParams, ControllerParams


class LQRController(Controller):
    """Full-state feedback LQR controller."""

    def __init__(self, seat_params: SeatParams, ctrl_params: ControllerParams):
        super().__init__(seat_params, ctrl_params)
        self.K_force: np.ndarray | None = None   # gain in force space (1×4)
        self.K_torque: np.ndarray | None = None   # gain in torque space (1×4)

    # ---- design ----------------------------------------------------------

    def design(self, A: np.ndarray, Bu: np.ndarray) -> None:
        """Solve CARE and compute LQR gain.

        ``Bu`` is assumed to be the (4×1) actuator-force input column.
        """
        Q = np.diag(self.ctrl_params.Q_diag)
        R = np.atleast_2d(self.ctrl_params.R)

        # Solve CARE
        P = solve_continuous_are(A, Bu, Q, R)

        # State-feedback gain (force domain)
        self.K_force = np.linalg.solve(R, Bu.T @ P)       # (1×4)

        # Convert to torque domain:  τ = F / N
        N = self.seat_params.gear_ratio
        self.K_torque = self.K_force / N

        self._designed = True

    # ---- compute ---------------------------------------------------------

    def compute(self, t: float, x_mech: np.ndarray) -> float:
        """Return τ_cmd = -(K_torque · x).  Saturation applied in plant."""
        if not self._designed:
            raise RuntimeError("LQRController.design() must be called first.")
        tau_cmd = -float(self.K_torque @ x_mech)
        return tau_cmd

    def reset(self) -> None:
        pass  # stateless controller

    @property
    def name(self) -> str:
        return "LQR"
