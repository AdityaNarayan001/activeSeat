"""
plot_tabs.py — Tabbed matplotlib canvas area for the centre of the GUI.

Each tab holds a ``FigureCanvasQTAgg`` with a navigation toolbar.
Tabs:
    0 — Time Histories
    1 — Actuator Response
    2 — Transmissibility
    3 — Power Analysis
    4 — Controller Comparison
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from ..simulation import SimResult
from .. import plotting as plt_mod


class _CanvasTab(QWidget):
    """A single tab holding one matplotlib figure + navigation toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def set_figure(self, fig: Figure):
        """Replace the contents of this tab with an externally created Figure.

        Swaps the canvas to the new figure, closes the old one to prevent
        memory leaks, and updates the toolbar reference.
        """
        import matplotlib.pyplot as plt

        old_fig = self.figure
        old_dpi = old_fig.get_dpi()

        # Resize new figure to match canvas dimensions
        w, h = self.canvas.get_width_height()
        fig.set_dpi(old_dpi)
        fig.set_size_inches(w / old_dpi, h / old_dpi)

        # Point canvas at the new figure
        self.figure = fig
        self.canvas.figure = fig
        fig.set_canvas(self.canvas)
        self.canvas.draw_idle()

        # Close the old figure to free memory and pyplot state
        plt.close(old_fig)


class PlotTabs(QWidget):
    """Tabbed widget containing all plot canvases."""

    TAB_TIME = 0
    TAB_ACTUATOR = 1
    TAB_TRANSMISSIBILITY = 2
    TAB_POWER = 3
    TAB_COMPARISON = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QTabWidget()
        self._canvases: list[_CanvasTab] = []

        names = [
            "Time Histories",
            "Actuator Response",
            "Transmissibility",
            "Power Analysis",
            "Controller Comparison",
        ]
        for name in names:
            ct = _CanvasTab()
            self._canvases.append(ct)
            self.tabs.addTab(ct, name)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    # ---- Update methods --------------------------------------------------

    def update_time_histories(self, passive: SimResult, active: SimResult):
        fig = plt_mod.plot_time_histories(passive, active)
        self._canvases[self.TAB_TIME].set_figure(fig)

    def update_actuator(self, active: SimResult):
        fig = plt_mod.plot_actuator_response(active)
        self._canvases[self.TAB_ACTUATOR].set_figure(fig)

    def update_transmissibility(self, freqs, T_p, T_a, name="Active"):
        fig = plt_mod.plot_transmissibility(freqs, T_p, T_a, name)
        self._canvases[self.TAB_TRANSMISSIBILITY].set_figure(fig)

    def update_power(self, active: SimResult):
        fig = plt_mod.plot_power_analysis(active)
        self._canvases[self.TAB_POWER].set_figure(fig)

    def update_controller_comparison(self, results: Dict[str, SimResult]):
        fig = plt_mod.plot_controller_comparison(results)
        self._canvases[self.TAB_COMPARISON].set_figure(fig)

    def clear_all(self):
        for c in self._canvases:
            c.clear()
