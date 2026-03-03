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
        memory leaks, and updates the toolbar reference.  Also connects
        interactive legend toggling (click legend entry to hide/show traces).
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

        # Connect interactive legend toggling
        self._connect_legend_toggle(fig)

    def _connect_legend_toggle(self, fig: Figure):
        """Wire up pick events so clicking a legend entry toggles traces.

        Toggling propagates across all subplots that share the same
        controller name, so one click hides/shows a controller everywhere.
        """
        # Disconnect any previous handler
        if hasattr(self, "_legend_cid") and self._legend_cid is not None:
            self.canvas.mpl_disconnect(self._legend_cid)
            self._legend_cid = None

        toggle = getattr(fig, "_legend_toggle", None)
        if not toggle:
            return

        h2n = toggle["h2n"]
        n2a = toggle["n2a"]
        n2l = toggle["n2l"]

        def _on_pick(event):
            name = h2n.get(event.artist)
            if not name:
                return
            artists = n2a.get(name, [])
            if not artists:
                return
            visible = not artists[0].get_visible()
            for a in artists:
                a.set_visible(visible)
            for lh in n2l.get(name, []):
                lh.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw_idle()

        self._legend_cid = self.canvas.mpl_connect("pick_event", _on_pick)


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

    def update_time_histories(self, results: Dict[str, SimResult]):
        fig = plt_mod.plot_time_histories(results)
        self._canvases[self.TAB_TIME].set_figure(fig)

    def update_actuator(self, results: Dict[str, SimResult]):
        fig = plt_mod.plot_actuator_response(results)
        self._canvases[self.TAB_ACTUATOR].set_figure(fig)

    def update_transmissibility(self, freq_data):
        """freq_data: Dict[str, (freqs, T)] mapping controller → arrays."""
        fig = plt_mod.plot_transmissibility(freq_data)
        self._canvases[self.TAB_TRANSMISSIBILITY].set_figure(fig)

    def update_power(self, results: Dict[str, SimResult]):
        fig = plt_mod.plot_power_analysis(results)
        self._canvases[self.TAB_POWER].set_figure(fig)

    def update_controller_comparison(self, results: Dict[str, SimResult]):
        fig = plt_mod.plot_controller_comparison(results)
        self._canvases[self.TAB_COMPARISON].set_figure(fig)

    def clear_all(self):
        for c in self._canvases:
            c.clear()
