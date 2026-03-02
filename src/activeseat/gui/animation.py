"""
animation.py — 2-D schematic animation of the seat–driver system.

Visual elements (top to bottom):
    ┌───────────┐
    │  Driver    │  ← z_d(t)
    └─────┬─────┘
       ╔══╧══╗     spring-damper 2 (k2, c2)
       ╚══╤══╝
    ┌─────┴─────┐
    │   Seat    │  ← z_s(t)  + actuator arrow
    └─────┬─────┘
       ╔══╧══╗     spring-damper 1 (k1, c1)
       ╚══╤══╝
    ▓▓▓▓▓▓▓▓▓▓▓▓  ← chassis / ground  z_0(t)

Displacements are exaggerated by a configurable scale factor (default 50×)
so that mm-scale motions become visible.

The animation *plays back* pre-computed ``SimResult`` data; it does NOT
re-simulate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel,
)

from ..simulation import SimResult


class AnimationWidget(QWidget):
    """2-D seat–driver animation with playback controls."""

    SCALE = 50.0          # displacement exaggeration factor
    FPS = 30              # target frames per second
    BASE_Y_GROUND = 0.5   # base Y position of ground in axes coords
    BASE_Y_SEAT = 2.0
    BASE_Y_DRIVER = 3.5

    def __init__(self, parent=None):
        super().__init__(parent)

        self._result: Optional[SimResult] = None
        self._frame: int = 0
        self._playing: bool = False
        self._speed: float = 1.0
        self._step: int = 1  # frames to advance per tick

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / self.FPS))
        self._timer.timeout.connect(self._advance_frame)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Matplotlib canvas
        self.figure = Figure(figsize=(5, 7), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        # Readout labels
        self._lbl_info = QLabel("t = 0.000 s  |  z_d = 0.00 mm  |  a_d = 0.00 m/s²  |  Fa = 0.0 N")
        self._lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_info.setStyleSheet("font-family: 'Courier New', 'Menlo', monospace; font-size: 11px;")
        layout.addWidget(self._lbl_info)

        # Playback controls
        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_step = QPushButton("⏭ Step")
        self.btn_step.clicked.connect(self._single_step)
        self.btn_reset = QPushButton("⏮ Reset")
        self.btn_reset.clicked.connect(self._reset)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 50)
        self.speed_slider.setValue(10)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.valueChanged.connect(self._speed_changed)
        self._lbl_speed = QLabel("1.0×")

        ctrl_row.addWidget(self.btn_reset)
        ctrl_row.addWidget(self.btn_play)
        ctrl_row.addWidget(self.btn_step)
        ctrl_row.addStretch()
        ctrl_row.addWidget(QLabel("Speed:"))
        ctrl_row.addWidget(self.speed_slider)
        ctrl_row.addWidget(self._lbl_speed)
        layout.addLayout(ctrl_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_result(self, result: SimResult) -> None:
        """Load a simulation result for playback."""
        self._result = result
        self._frame = 0
        self._playing = False
        self.btn_play.setText("▶ Play")
        self._draw_frame()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_frame(self):
        if self._result is None:
            return

        res = self._result
        idx = min(self._frame, len(res.t) - 1)
        t_now = res.t[idx]
        z0 = res.z0[idx] * self.SCALE
        zs = res.z_s[idx] * self.SCALE
        zd = res.z_d[idx] * self.SCALE
        Fa = res.Fa[idx]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xlim(-2, 2)

        # Vertical positions
        y_ground = self.BASE_Y_GROUND + z0
        y_seat = self.BASE_Y_SEAT + zs
        y_driver = self.BASE_Y_DRIVER + zd

        # Auto y-limits
        y_lo = min(y_ground - 0.5, -0.5)
        y_hi = max(y_driver + 1.5, 5.0)
        ax.set_ylim(y_lo, y_hi)

        # Draw ground (hatched rectangle)
        ground = mpatches.FancyBboxPatch(
            (-1.5, y_ground - 0.25), 3.0, 0.25,
            boxstyle="round,pad=0.02", facecolor="#8B7355", edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(ground)
        ax.text(0, y_ground - 0.15, "Chassis", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

        # Draw spring-damper 1 (chassis → seat)
        self._draw_spring_damper(ax, 0, y_ground, y_seat, "k₁, c₁", "#4477AA")

        # Draw actuator arrow
        arrow_x = 0.8
        arrow_scale = Fa / max(abs(res.Fa).max(), 1) * 0.4
        ax.annotate("", xy=(arrow_x, y_seat - 0.05),
                     xytext=(arrow_x, y_ground + 0.05),
                     arrowprops=dict(arrowstyle="->", color="red",
                                     lw=1.5 + abs(arrow_scale) * 3))
        ax.text(arrow_x + 0.15, (y_ground + y_seat) / 2, f"Fₐ={Fa:.0f}N",
                fontsize=7, color="red", ha="left", va="center")

        # Draw seat platform
        seat = mpatches.FancyBboxPatch(
            (-1.0, y_seat - 0.15), 2.0, 0.3,
            boxstyle="round,pad=0.03", facecolor="#4477AA", edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(seat)
        ax.text(0, y_seat, "Seat", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

        # Draw spring-damper 2 (seat → driver)
        self._draw_spring_damper(ax, 0, y_seat + 0.15, y_driver - 0.15,
                                  "k₂, c₂", "#44AA77")

        # Draw driver mass
        driver = mpatches.FancyBboxPatch(
            (-0.8, y_driver - 0.15), 1.6, 0.5,
            boxstyle="round,pad=0.05", facecolor="#DD6644", edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(driver)
        ax.text(0, y_driver + 0.1, "Driver", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"t = {t_now:.3f} s", fontsize=11, fontweight="bold")

        self.canvas.draw_idle()

        # Update readout
        ad = res.accel_driver[idx]
        self._lbl_info.setText(
            f"t = {t_now:.3f} s  |  z_d = {res.z_d[idx]*1e3:.2f} mm  |  "
            f"a_d = {ad:.2f} m/s²  |  Fa = {Fa:.1f} N"
        )

    def _draw_spring_damper(self, ax, x_center, y_bot, y_top, label, color):
        """Draw a schematic spring (zigzag) between two Y positions."""
        n_zigs = 6
        height = y_top - y_bot
        if height < 0.1:
            height = 0.1  # minimum visible height

        # Zigzag for spring
        xs = x_center - 0.2
        ys = np.linspace(y_bot, y_top, n_zigs * 2 + 1)
        xz = np.zeros_like(ys)
        for i in range(len(ys)):
            if i == 0 or i == len(ys) - 1:
                xz[i] = xs
            elif i % 2 == 1:
                xz[i] = xs - 0.15
            else:
                xz[i] = xs + 0.15
        ax.plot(xz, ys, color=color, linewidth=1.2)

        # Damper (two parallel lines)
        xd = x_center + 0.2
        y_mid = (y_bot + y_top) / 2
        ax.plot([xd, xd], [y_bot, y_mid - height * 0.1], color=color, linewidth=1.5)
        ax.plot([xd - 0.08, xd + 0.08], [y_mid - height * 0.1] * 2,
                color=color, linewidth=2)
        ax.plot([xd - 0.08, xd - 0.08, xd + 0.08, xd + 0.08],
                [y_mid - height * 0.1, y_mid + height * 0.1,
                 y_mid + height * 0.1, y_mid - height * 0.1],
                color=color, linewidth=1.0)
        ax.plot([xd, xd], [y_mid + height * 0.1, y_top], color=color, linewidth=1.5)

        ax.text(x_center + 0.5, (y_bot + y_top) / 2, label,
                fontsize=7, color=color, ha="left", va="center")

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if self._playing:
            self._timer.stop()
            self._playing = False
            self.btn_play.setText("▶ Play")
        else:
            self._playing = True
            self.btn_play.setText("⏸ Pause")
            self._timer.start()

    def _single_step(self):
        if self._result is not None:
            self._frame = min(self._frame + self._step * 5, len(self._result.t) - 1)
            self._draw_frame()

    def _reset(self):
        self._timer.stop()
        self._playing = False
        self.btn_play.setText("▶ Play")
        self._frame = 0
        self._draw_frame()

    def _advance_frame(self):
        if self._result is None:
            return
        # Compute how many data frames to skip per animation frame
        # to achieve real-time at speed=1.0
        dt_data = self._result.t[1] - self._result.t[0] if len(self._result.t) > 1 else 0.001
        dt_anim = 1.0 / self.FPS
        frames_per_tick = max(1, int(self._speed * dt_anim / dt_data))

        self._frame += frames_per_tick
        if self._frame >= len(self._result.t):
            self._frame = 0  # loop
        self._draw_frame()

    def _speed_changed(self, val: int):
        self._speed = val / 10.0
        self._lbl_speed.setText(f"{self._speed:.1f}×")
