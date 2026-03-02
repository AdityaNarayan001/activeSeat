"""
hinf.py — H-infinity (H∞) robust controller for the 2-DOF seat suspension.

Uses a mixed-sensitivity loop-shaping approach:

    Minimise  ‖T_zw‖∞ < γ

where z = [W_p · e,  W_u · u]  and  w = disturbance (road velocity).

Implementation uses the ``python-control`` library's ``hinfsyn`` when
available, with a manual Riccati-based fallback.

The resulting controller K(s) is a *dynamic* output-feedback compensator
with its own internal state vector, integrated alongside the plant.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import control as ct
    _HAS_CONTROL = True
except ImportError:
    _HAS_CONTROL = False

from .base import Controller
from ..params import SeatParams, ControllerParams


class HinfController(Controller):
    """Dynamic H∞ mixed-sensitivity controller.

    After ``design()``, the controller carries ``n_states`` internal ODE
    states and must be simulated via ``dynamics_ode_extended``.
    """

    def __init__(self, seat_params: SeatParams, ctrl_params: ControllerParams):
        super().__init__(seat_params, ctrl_params)
        # Controller state-space matrices (filled by design)
        self._Ak: np.ndarray | None = None
        self._Bk: np.ndarray | None = None
        self._Ck: np.ndarray | None = None
        self._Dk: np.ndarray | None = None
        self._n_ctrl: int = 0

    # ----- properties -----------------------------------------------------

    @property
    def n_states(self) -> int:
        return self._n_ctrl

    @property
    def name(self) -> str:
        return "H∞"

    # ----- design ---------------------------------------------------------

    def design(self, A: np.ndarray, Bu: np.ndarray, Bw: np.ndarray = None,
               C: np.ndarray = None) -> None:
        """Synthesise the H∞ controller.

        Parameters
        ----------
        A  : (4,4) plant state matrix
        Bu : (4,1) control input matrix
        Bw : (4,1) disturbance input matrix (road velocity).  If None, a
             default column is constructed from the seat params.
        C  : (1,4) performance output matrix (driver accel).  If None, a
             default is built.
        """
        if not _HAS_CONTROL:
            raise RuntimeError("python-control is required for H∞ synthesis. "
                               "Install with: pip install control")

        p = self.seat_params
        cp = self.ctrl_params
        n = A.shape[0]  # 4

        # --- Default Bw and C if not provided ---
        if Bw is None:
            ms = p.effective_seat_mass
            Bw = np.array([[-1.0], [p.c1 / ms], [0.0], [0.0]])
        if C is None:
            md = p.mass_driver
            C = np.array([[0.0, p.c2 / md, -p.k2 / md, -p.c2 / md]])

        # --- Build generalised plant P(s) for hinfsyn ---
        #
        # Inputs:  [w, u]     (w = road velocity disturbance, u = actuator force)
        # Outputs: [z1, z2, y]
        #   z1 = W_perf · C x   (weighted driver acceleration — performance)
        #   z2 = W_ctrl · u     (weighted control effort)
        #   y  = x              (full state measurement for output feedback)
        #
        # We use static diagonal weights for simplicity.

        Wp = cp.hinf_perf_weight_accel   # scalar weight on accel
        Wu = cp.hinf_perf_weight_force   # scalar weight on force

        # Performance output z1 = Wp * C x  (no direct feed-through from u)
        C1 = Wp * C          # (1×4)
        D11 = np.zeros((1, 1))  # z1 ← w
        D12 = np.zeros((1, 1))  # z1 ← u

        # Control effort output z2 = Wu * u
        C2 = np.zeros((1, n))    # (1×4)
        D21 = np.zeros((1, 1))   # z2 ← w
        D22 = Wu * np.ones((1, 1))  # z2 ← u

        # Measured output y = full state
        Cy = np.eye(n)           # (4×4)
        Dy1 = np.zeros((n, 1))   # y ← w
        Dy2 = np.zeros((n, 1))   # y ← u

        # Stack into generalised plant matrices
        #   ẋ  = A x + [Bw, Bu] [w; u]
        #   [z] = [C_z] x + [D_zw, D_zu] [w; u]
        #   [y] = [Cy]  x + [Dy1,  Dy2]  [w; u]

        Cz = np.vstack([C1, C2])           # (2×4)
        Dzw = np.vstack([D11, D21])         # (2×1)
        Dzu = np.vstack([D12, D22])         # (2×1)

        B_gen = np.hstack([Bw, Bu])         # (4×2)
        C_gen = np.vstack([Cz, Cy])         # (6×4)
        D_gen = np.block([
            [Dzw, Dzu],                     # (2×2) — z rows
            [Dy1, Dy2],                     # (4×2) — y rows
        ])                                  # (6×2)

        P = ct.ss(A, B_gen, C_gen, D_gen)

        # --- hinfsyn ---
        nmeas = n    # number of measurements (y dimension)
        ncon = 1     # number of control inputs (u dimension)

        try:
            K_ss, CL, gamma_opt, _rcond = ct.hinfsyn(P, nmeas, ncon)
        except Exception as exc:
            # Fallback: use a simple static H2 / LQR-like design with
            # increased robustness weighting, stored as a static gain
            # (zero-order controller).
            from scipy.linalg import solve_continuous_are
            Q_robust = Wp**2 * (C.T @ C) + 0.01 * np.eye(n)
            R_robust = np.atleast_2d(Wu**2 + 0.1)
            P_ricc = solve_continuous_are(A, Bu, Q_robust, R_robust)
            K_static = np.linalg.solve(R_robust, Bu.T @ P_ricc)
            # Store as a zero-state dynamic controller
            self._Ak = np.zeros((1, 1))
            self._Bk = np.zeros((1, n))
            self._Ck = np.zeros((1, 1))
            self._Dk = -K_static / self.seat_params.gear_ratio  # torque domain
            self._n_ctrl = 1
            self._designed = True
            return

        # Extract controller state-space matrices
        self._Ak = np.array(K_ss.A)
        self._Bk = np.array(K_ss.B)
        self._Ck = np.array(K_ss.C)
        self._Dk = np.array(K_ss.D)
        self._n_ctrl = self._Ak.shape[0]

        # Convert controller output from force domain to torque domain
        N = self.seat_params.gear_ratio
        self._Ck = self._Ck / N
        self._Dk = self._Dk / N

        self._designed = True

    # ----- compute (static fallback for simple interface) -----------------

    def compute(self, t: float, x_mech: np.ndarray) -> float:
        """Static approximation: use D_k only (ignores controller dynamics)."""
        if not self._designed:
            raise RuntimeError("HinfController.design() must be called first.")
        return float(self._Dk @ x_mech)

    # ----- compute_extended (full dynamic controller) ---------------------

    def compute_extended(
        self, t: float, x_mech: np.ndarray, ctrl_states: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Evaluate the dynamic H∞ controller and return (τ_cmd, ẋ_ctrl).

        Parameters
        ----------
        ctrl_states : (n_ctrl,) ndarray — controller internal states

        Returns
        -------
        tau_cmd     : float     — motor torque command [N·m]
        d_ctrl      : (n_ctrl,) — controller state derivatives
        """
        if not self._designed:
            raise RuntimeError("HinfController.design() must be called first.")

        xk = ctrl_states.reshape(-1)
        y = x_mech.reshape(-1, 1)  # measurement = full plant state

        # Controller dynamics:  ẋ_k = Ak xk + Bk y,  u = Ck xk + Dk y
        d_ctrl = (self._Ak @ xk.reshape(-1, 1) + self._Bk @ y).flatten()
        tau_cmd = float((self._Ck @ xk.reshape(-1, 1) + self._Dk @ y).item())

        return tau_cmd, d_ctrl

    def reset(self) -> None:
        pass  # start ctrl_states from zeros in initial condition
