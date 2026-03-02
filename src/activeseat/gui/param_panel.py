"""
param_panel.py — Left-side parameter input panel for the GUI.

Groups: Masses, Suspension, Cushion, Actuator, Controller, Road, Simulation.
Each parameter is a labelled QDoubleSpinBox with SI-unit suffix and tooltip.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSpinBox, QComboBox, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy,
)

from ..params import SeatParams, ControllerParams, RoadConfig, SimParams


def _dspin(value, lo, hi, step, decimals=4, suffix=""):
    """Helper to create a configured QDoubleSpinBox."""
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(f"  {suffix}")
    sb.setMinimumWidth(130)
    return sb


def _ispin(value, lo, hi, step=1):
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setValue(value)
    sb.setMinimumWidth(130)
    return sb


class ParamPanel(QWidget):
    """Scrollable parameter-input panel occupying the left column of the GUI."""

    # Signals emitted when run buttons are clicked
    run_comparison_clicked = pyqtSignal()
    run_all_controllers_clicked = pyqtSignal()
    run_freq_sweep_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- Masses ---
        grp = QGroupBox("Masses")
        form = QFormLayout()
        self.mass_seat = _dspin(15.0, 1, 200, 1, 2, "kg")
        self.mass_driver = _dspin(75.0, 30, 200, 5, 1, "kg")
        form.addRow("Seat mass mₛ", self.mass_seat)
        form.addRow("Driver mass m_d", self.mass_driver)
        grp.setLayout(form)
        layout.addWidget(grp)

        # --- Suspension ---
        grp = QGroupBox("Suspension (Chassis ↔ Seat)")
        form = QFormLayout()
        self.k1 = _dspin(20000, 1000, 200000, 1000, 0, "N/m")
        self.c1 = _dspin(1500, 100, 20000, 100, 0, "N·s/m")
        form.addRow("Stiffness k₁", self.k1)
        form.addRow("Damping c₁", self.c1)
        grp.setLayout(form)
        layout.addWidget(grp)

        # --- Cushion ---
        grp = QGroupBox("Cushion (Seat ↔ Driver)")
        form = QFormLayout()
        self.k2 = _dspin(50000, 1000, 500000, 1000, 0, "N/m")
        self.c2 = _dspin(2000, 100, 50000, 100, 0, "N·s/m")
        form.addRow("Stiffness k₂", self.k2)
        form.addRow("Damping c₂", self.c2)
        grp.setLayout(form)
        layout.addWidget(grp)

        # --- Actuator ---
        grp = QGroupBox("Rotary Actuator")
        form = QFormLayout()
        self.gear_ratio = _dspin(628, 100, 50000, 50, 0, "rad/m")
        self.max_torque = _dspin(5.0, 0.1, 100.0, 0.5, 2, "N·m")
        self.motor_inertia = _dspin(0.00002, 0.000001, 0.01, 0.00001, 6, "kg·m²")
        self.viscous_friction = _dspin(0.0002, 0, 1.0, 0.0001, 6, "N·m·s/rad")
        self.coulomb_friction = _dspin(0.02, 0, 1.0, 0.01, 3, "N·m")
        self.electrical_tau = _dspin(0.005, 0, 0.1, 0.001, 4, "s")
        form.addRow("Gear ratio N", self.gear_ratio)
        form.addRow("Max torque τ_max", self.max_torque)
        form.addRow("Motor inertia J_m", self.motor_inertia)
        form.addRow("Viscous friction b_v", self.viscous_friction)
        form.addRow("Coulomb friction τ_c", self.coulomb_friction)
        form.addRow("Electrical τ_e", self.electrical_tau)
        grp.setLayout(form)
        layout.addWidget(grp)

        # --- Controller ---
        grp = QGroupBox("Controller")
        form = QFormLayout()
        self.ctrl_type = QComboBox()
        self.ctrl_type.addItems(["passive", "lqr", "hinf", "adaptive"])
        self.ctrl_type.setCurrentText("lqr")
        form.addRow("Type", self.ctrl_type)

        self.Q1 = _dspin(10, 0, 1e8, 10, 1)
        self.Q2 = _dspin(700, 0, 1e6, 50, 1)
        self.Q3 = _dspin(400000, 0, 1e8, 10000, 0)
        self.Q4 = _dspin(700, 0, 1e6, 50, 1)
        self.R_val = _dspin(0.00002, 1e-10, 100, 0.000005, 8)
        form.addRow("Q[1] (seat disp)", self.Q1)
        form.addRow("Q[2] (seat vel)", self.Q2)
        form.addRow("Q[3] (driver disp)", self.Q3)
        form.addRow("Q[4] (driver vel)", self.Q4)
        form.addRow("R (control)", self.R_val)

        self.hinf_gamma = _dspin(5.0, 0.1, 100, 0.5, 2)
        self.hinf_wp = _dspin(500.0, 0.1, 1e6, 10, 1)
        self.hinf_wu = _dspin(0.0001, 1e-8, 100, 0.0001, 6)
        form.addRow("H∞ γ", self.hinf_gamma)
        form.addRow("H∞ W_accel", self.hinf_wp)
        form.addRow("H∞ W_force", self.hinf_wu)

        self.adapt_gamma = _dspin(500.0, 0.1, 5000, 50, 1)
        self.adapt_wn = _dspin(12.0, 0.5, 50, 0.5, 1, "rad/s")
        self.adapt_zeta = _dspin(0.8, 0.01, 2.0, 0.05, 3)
        form.addRow("Adaptive Γ", self.adapt_gamma)
        form.addRow("Adaptive ωₙ", self.adapt_wn)
        form.addRow("Adaptive ζ", self.adapt_zeta)

        grp.setLayout(form)
        layout.addWidget(grp)

        # --- Road ---
        grp = QGroupBox("Road Excitation")
        form = QFormLayout()
        self.road_type = QComboBox()
        self.road_type.addItems(["bump", "sinusoidal", "random"])
        form.addRow("Type", self.road_type)

        self.bump_height = _dspin(0.05, 0.001, 1.0, 0.01, 3, "m")
        self.bump_length = _dspin(0.5, 0.05, 5.0, 0.1, 2, "m")
        self.vehicle_speed = _dspin(10.0, 0.5, 50, 1, 1, "m/s")
        form.addRow("Bump height", self.bump_height)
        form.addRow("Bump length", self.bump_length)
        form.addRow("Vehicle speed", self.vehicle_speed)

        self.sine_amp = _dspin(0.01, 0.001, 0.5, 0.005, 3, "m")
        self.sine_freq = _dspin(3.0, 0.1, 50, 0.5, 1, "Hz")
        form.addRow("Sine amplitude", self.sine_amp)
        form.addRow("Sine frequency", self.sine_freq)

        self.iso_class = QComboBox()
        self.iso_class.addItems(["A", "B", "C", "D", "E"])
        self.iso_class.setCurrentText("C")
        self.iso_speed = _dspin(20.0, 1, 60, 5, 1, "m/s")
        self.iso_seed = _ispin(42, 0, 999999)
        form.addRow("ISO class", self.iso_class)
        form.addRow("ISO speed", self.iso_speed)
        form.addRow("ISO seed", self.iso_seed)

        grp.setLayout(form)
        layout.addWidget(grp)

        # --- Simulation ---
        grp = QGroupBox("Simulation")
        form = QFormLayout()
        self.sim_duration = _dspin(2.0, 0.1, 60, 0.5, 2, "s")
        self.sim_dt = _dspin(0.001, 0.0001, 0.01, 0.0005, 4, "s")
        self.sim_solver = QComboBox()
        self.sim_solver.addItems(["Radau", "BDF", "RK45", "RK23", "DOP853"])
        self.sim_max_step = _dspin(0.001, 0.0001, 0.01, 0.0005, 4, "s")
        form.addRow("Duration", self.sim_duration)
        form.addRow("Output dt", self.sim_dt)
        form.addRow("Solver", self.sim_solver)
        form.addRow("Max step", self.sim_max_step)
        grp.setLayout(form)
        layout.addWidget(grp)

        layout.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # --- Run buttons (always visible, below scroll area) ---
        btn_box = QVBoxLayout()
        btn_box.setSpacing(6)
        btn_box.setContentsMargins(8, 6, 8, 8)

        self.btn_run_comparison = QPushButton("▶  Run Comparison")
        self.btn_run_comparison.setMinimumHeight(38)
        self.btn_run_comparison.setStyleSheet(
            "QPushButton { background-color: #2d8cf0; color: white; "
            "font-weight: bold; font-size: 13px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #1a6fd1; }"
            "QPushButton:pressed { background-color: #1260b8; }"
        )
        self.btn_run_comparison.clicked.connect(self.run_comparison_clicked)
        btn_box.addWidget(self.btn_run_comparison)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_run_all = QPushButton("All Controllers")
        self.btn_run_all.setMinimumHeight(32)
        self.btn_run_all.setStyleSheet(
            "QPushButton { background-color: #4a4a4a; color: white; "
            "font-size: 12px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #5a5a5a; }"
            "QPushButton:pressed { background-color: #3a3a3a; }"
        )
        self.btn_run_all.clicked.connect(self.run_all_controllers_clicked)
        btn_row.addWidget(self.btn_run_all)

        self.btn_run_freq = QPushButton("Freq Sweep")
        self.btn_run_freq.setMinimumHeight(32)
        self.btn_run_freq.setStyleSheet(
            "QPushButton { background-color: #4a4a4a; color: white; "
            "font-size: 12px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #5a5a5a; }"
            "QPushButton:pressed { background-color: #3a3a3a; }"
        )
        self.btn_run_freq.clicked.connect(self.run_freq_sweep_clicked)
        btn_row.addWidget(self.btn_run_freq)

        btn_box.addLayout(btn_row)
        outer.addLayout(btn_box)

    def set_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable all run buttons (e.g. while simulation is running)."""
        self.btn_run_comparison.setEnabled(enabled)
        self.btn_run_all.setEnabled(enabled)
        self.btn_run_freq.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Read / write dataclasses
    # ------------------------------------------------------------------

    def get_seat_params(self) -> SeatParams:
        return SeatParams(
            mass_seat=self.mass_seat.value(),
            mass_driver=self.mass_driver.value(),
            k1=self.k1.value(),
            c1=self.c1.value(),
            k2=self.k2.value(),
            c2=self.c2.value(),
            gear_ratio=self.gear_ratio.value(),
            max_torque=self.max_torque.value(),
            motor_inertia=self.motor_inertia.value(),
            viscous_friction=self.viscous_friction.value(),
            coulomb_friction=self.coulomb_friction.value(),
            electrical_tau=self.electrical_tau.value(),
        )

    def get_controller_params(self) -> ControllerParams:
        return ControllerParams(
            controller_type=self.ctrl_type.currentText(),
            Q_diag=[self.Q1.value(), self.Q2.value(),
                    self.Q3.value(), self.Q4.value()],
            R=self.R_val.value(),
            hinf_gamma=self.hinf_gamma.value(),
            hinf_perf_weight_accel=self.hinf_wp.value(),
            hinf_perf_weight_force=self.hinf_wu.value(),
            adaptive_gamma_gain=self.adapt_gamma.value(),
            adaptive_ref_wn=self.adapt_wn.value(),
            adaptive_ref_zeta=self.adapt_zeta.value(),
        )

    def get_road_config(self) -> RoadConfig:
        return RoadConfig(
            road_type=self.road_type.currentText(),
            bump_height=self.bump_height.value(),
            bump_length=self.bump_length.value(),
            vehicle_speed=self.vehicle_speed.value(),
            sine_amplitude=self.sine_amp.value(),
            sine_frequency=self.sine_freq.value(),
            iso_class=self.iso_class.currentText(),
            iso_speed=self.iso_speed.value(),
            iso_seed=self.iso_seed.value(),
        )

    def get_sim_params(self) -> SimParams:
        return SimParams(
            duration=self.sim_duration.value(),
            dt=self.sim_dt.value(),
            solver=self.sim_solver.currentText(),
            max_step=self.sim_max_step.value(),
        )

    def set_from_params(
        self,
        seat: SeatParams,
        ctrl: ControllerParams,
        road: RoadConfig,
        sim: SimParams,
    ) -> None:
        """Populate all spin boxes from dataclass values."""
        self.mass_seat.setValue(seat.mass_seat)
        self.mass_driver.setValue(seat.mass_driver)
        self.k1.setValue(seat.k1)
        self.c1.setValue(seat.c1)
        self.k2.setValue(seat.k2)
        self.c2.setValue(seat.c2)
        self.gear_ratio.setValue(seat.gear_ratio)
        self.max_torque.setValue(seat.max_torque)
        self.motor_inertia.setValue(seat.motor_inertia)
        self.viscous_friction.setValue(seat.viscous_friction)
        self.coulomb_friction.setValue(seat.coulomb_friction)
        self.electrical_tau.setValue(seat.electrical_tau)

        self.ctrl_type.setCurrentText(ctrl.controller_type)
        self.Q1.setValue(ctrl.Q_diag[0])
        self.Q2.setValue(ctrl.Q_diag[1])
        self.Q3.setValue(ctrl.Q_diag[2])
        self.Q4.setValue(ctrl.Q_diag[3])
        self.R_val.setValue(ctrl.R)
        self.hinf_gamma.setValue(ctrl.hinf_gamma)
        self.hinf_wp.setValue(ctrl.hinf_perf_weight_accel)
        self.hinf_wu.setValue(ctrl.hinf_perf_weight_force)
        self.adapt_gamma.setValue(ctrl.adaptive_gamma_gain)
        self.adapt_wn.setValue(ctrl.adaptive_ref_wn)
        self.adapt_zeta.setValue(ctrl.adaptive_ref_zeta)

        self.road_type.setCurrentText(road.road_type)
        self.bump_height.setValue(road.bump_height)
        self.bump_length.setValue(road.bump_length)
        self.vehicle_speed.setValue(road.vehicle_speed)
        self.sine_amp.setValue(road.sine_amplitude)
        self.sine_freq.setValue(road.sine_frequency)
        self.iso_class.setCurrentText(road.iso_class)
        self.iso_speed.setValue(road.iso_speed)
        self.iso_seed.setValue(road.iso_seed)

        self.sim_duration.setValue(sim.duration)
        self.sim_dt.setValue(sim.dt)
        self.sim_solver.setCurrentText(sim.solver)
        self.sim_max_step.setValue(sim.max_step)
