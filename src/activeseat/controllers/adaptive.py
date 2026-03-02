"""
adaptive.py — Model-Reference Adaptive Controller (MRAC) for the seat suspension.

Approach: direct MRAC with Lyapunov-based adaptation law.

Reference model
    ẍ_ref = -2 ζ ω_n ẋ_ref - ω_n² x_ref + ω_n² r(t)

where r(t) = 0 (regulation).  The reference model describes the desired
closed-loop dynamics of the relative driver–seat motion x3.

Adaptive law (normalised gradient):
    θ̇ = -Γ · φ(x) · e'Pb  /  (1 + φ'φ)

where  e = x3 - x3_ref,  φ is the regressor vector (plant states), and
P, b come from the Lyapunov equation A_m'P + PA_m = -Q_lyap.

The adaptive parameters θ are additional ODE states integrated alongside
the plant.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from .base import Controller
from ..params import SeatParams, ControllerParams


class AdaptiveController(Controller):
    """Model-Reference Adaptive Controller (MRAC).

    Internal ODE states: [x_ref, dx_ref, θ_1, θ_2, θ_3, θ_4]  (6 total).
        - x_ref, dx_ref : reference model states
        - θ_1..θ_4      : adaptive gain parameters (one per plant state)
    """

    _N_CTRL_STATES = 6  # 2 ref model + 4 adaptive params

    def __init__(self, seat_params: SeatParams, ctrl_params: ControllerParams):
        super().__init__(seat_params, ctrl_params)
        self._Gamma: float = 0.0
        self._wn: float = 0.0
        self._zeta: float = 0.0
        self._P_lyap: np.ndarray | None = None  # (2×2)
        self._b_ref: np.ndarray | None = None    # (2×1)

    @property
    def n_states(self) -> int:
        return self._N_CTRL_STATES

    @property
    def name(self) -> str:
        return "Adaptive"

    # ----- design ---------------------------------------------------------

    def design(self, A: np.ndarray, Bu: np.ndarray) -> None:
        """Set up the reference model and Lyapunov adaptation matrices."""
        cp = self.ctrl_params
        self._Gamma = cp.adaptive_gamma_gain
        self._wn = cp.adaptive_ref_wn
        self._zeta = cp.adaptive_ref_zeta

        # Reference model:  2nd-order  ẍ + 2ζωẋ + ω²x = 0
        #   State: [x_ref, dx_ref]
        #   A_m = [[0, 1], [-ω², -2ζω]]
        wn, zeta = self._wn, self._zeta
        A_m = np.array([[0.0, 1.0],
                        [-wn**2, -2.0 * zeta * wn]])
        b_m = np.array([[0.0], [wn**2]])  # input column (for reference input r=0)

        # Lyapunov equation: A_m' P + P A_m = -Q_lyap
        Q_lyap = np.eye(2) * 10.0
        self._P_lyap = solve_continuous_lyapunov(A_m.T, -Q_lyap)
        self._b_ref = b_m
        self._designed = True

    # ----- compute (simplified, static snapshot) --------------------------

    def compute(self, t: float, x_mech: np.ndarray) -> float:
        """Fallback static compute — uses zero adaptive params (no adaptation)."""
        return 0.0

    # ----- compute_extended (full dynamic adaptation) ---------------------

    def compute_extended(
        self, t: float, x_mech: np.ndarray, ctrl_states: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Evaluate MRAC and return (τ_cmd, ẋ_ctrl).

        ctrl_states layout: [x_ref, dx_ref, θ1, θ2, θ3, θ4]
        """
        if not self._designed:
            raise RuntimeError("AdaptiveController.design() must be called first.")

        x_ref = ctrl_states[0]
        dx_ref = ctrl_states[1]
        theta = ctrl_states[2:6]  # (4,) adaptive gains

        wn, zeta = self._wn, self._zeta
        N = self.seat_params.gear_ratio

        # --- Reference model dynamics (regulation: r=0) ---
        d_xref = dx_ref
        d_dxref = -wn**2 * x_ref - 2.0 * zeta * wn * dx_ref

        # --- Tracking error ---
        # Track relative driver-seat displacement x3 = x_mech[2]
        e = x_mech[2] - x_ref
        de = x_mech[3] - x_mech[1] - dx_ref  # ẋ3 - ẋ_ref  (ẋ3 = x4 - x2)
        e_vec = np.array([e, de])  # (2,)

        # --- Control law:  Fa = θ' · x_mech  →  τ = Fa / N ---
        Fa = float(theta @ x_mech)
        tau_cmd = Fa / N

        # --- Adaptation law (normalised Lyapunov-based) ---
        # θ̇ = -Γ · x_mech · (e_vec' P b) / (1 + x_mech' x_mech)
        Pb = self._P_lyap @ self._b_ref.flatten()  # (2,)
        sigma = float(e_vec @ Pb)   # scalar projection
        norm_factor = 1.0 + float(x_mech @ x_mech)
        d_theta = -self._Gamma * x_mech * sigma / norm_factor  # (4,)

        d_ctrl = np.array([d_xref, d_dxref, d_theta[0], d_theta[1],
                           d_theta[2], d_theta[3]])
        return tau_cmd, d_ctrl

    def reset(self) -> None:
        """Controller states are reset via initial conditions in simulation."""
        pass
