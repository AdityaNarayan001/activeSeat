"""
main_window.py — Top-level PyQt6 window for the ActiveSeat application.

Layout:
    ┌─────────────┬──────────────────────┬──────────────┐
    │  Param      │  Tabs (plots) /      │  Metrics     │
    │  Panel      │  Animation           │  Dashboard   │
    │  (left)     │  (centre)            │  (right)     │
    └─────────────┴──────────────────────┴──────────────┘

Menu bar:  File (Load/Save scenario, Export plots)
           Run  (Comparison, All Controllers, Freq Sweep)
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QVBoxLayout, QTabWidget,
    QStatusBar, QFileDialog, QMessageBox, QApplication,
)

from .param_panel import ParamPanel
from .plot_tabs import PlotTabs
from .animation import AnimationWidget
from .metrics_panel import MetricsPanel
from .worker import SimWorker
from ..config import load_scenario, save_scenario


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self, scenario_path: str | None = None):
        super().__init__()
        self.setWindowTitle("ActiveSeat — 2-DOF Seat Suspension Simulator")
        self.resize(1600, 950)

        self._worker: SimWorker | None = None

        self._build_menus()
        self._build_central()
        self._build_statusbar()

        if scenario_path and os.path.isfile(scenario_path):
            self._load_scenario_file(scenario_path)

        # Connect param panel run button
        self.param_panel.run_clicked.connect(self._on_run)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menus(self):
        mb = self.menuBar()

        # ---- File ----
        file_menu = mb.addMenu("File")

        act_load = QAction("Load Scenario…", self)
        act_load.setShortcut("Ctrl+O")
        act_load.triggered.connect(self._on_load_scenario)
        file_menu.addAction(act_load)

        act_save = QAction("Save Scenario…", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._on_save_scenario)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ---- Run ----
        run_menu = mb.addMenu("Run")

        act_run = QAction("Run Simulation", self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(self._on_run)
        run_menu.addAction(act_run)

    # ------------------------------------------------------------------
    # Central widget
    # ------------------------------------------------------------------

    def _build_central(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left — parameter panel
        self.param_panel = ParamPanel()
        self.param_panel.setMinimumWidth(280)
        self.param_panel.setMaximumWidth(380)
        splitter.addWidget(self.param_panel)

        # Centre — animation + plot tabs
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)

        self.centre_tabs = QTabWidget()
        # Tab 0: Animation
        self.animation = AnimationWidget()
        self.centre_tabs.addTab(self.animation, "🎬 Animation")
        # Tab 1: Plots
        self.plot_tabs = PlotTabs()
        self.centre_tabs.addTab(self.plot_tabs, "📊 Plots")
        centre_layout.addWidget(self.centre_tabs)
        splitter.addWidget(centre)

        # Right — metrics dashboard
        self.metrics_panel = MetricsPanel()
        self.metrics_panel.setMinimumWidth(320)
        self.metrics_panel.setMaximumWidth(480)
        splitter.addWidget(self.metrics_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self.setCentralWidget(splitter)

    # ------------------------------------------------------------------
    # Status bar + progress
    # ------------------------------------------------------------------

    def _build_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — load a scenario or click Run.")

    # ------------------------------------------------------------------
    # Scenario load / save
    # ------------------------------------------------------------------

    def _on_load_scenario(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Scenario", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self._load_scenario_file(path)

    def _load_scenario_file(self, path: str):
        try:
            seat, ctrl, road, sim = load_scenario(path)
            self.param_panel.set_from_params(seat, ctrl, road, sim)
            self.status_bar.showMessage(f"Loaded: {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _on_save_scenario(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Scenario", "", "YAML Files (*.yaml);;All Files (*)"
        )
        if path:
            try:
                save_scenario(
                    path,
                    self.param_panel.get_seat_params(),
                    self.param_panel.get_controller_params(),
                    self.param_panel.get_road_config(),
                    self.param_panel.get_sim_params(),
                )
                self.status_bar.showMessage(f"Saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", str(e))

    # ------------------------------------------------------------------
    # Run simulations
    # ------------------------------------------------------------------

    def _on_run(self):
        """Start the full simulation pipeline."""
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "A simulation is already running.")
            return

        seat = self.param_panel.get_seat_params()
        ctrl = self.param_panel.get_controller_params()
        road = self.param_panel.get_road_config()
        sim = self.param_panel.get_sim_params()

        self._worker = SimWorker(SimWorker.TASK_FULL, seat, ctrl, road, sim, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.stage.connect(self._on_stage)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self.param_panel.set_run_enabled(False)
        self.param_panel.set_progress(0, "Starting…")
        self.status_bar.showMessage("Running simulation…")
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, val: float):
        self.param_panel.set_progress(val)

    def _on_stage(self, label: str):
        self.param_panel.set_progress(self.param_panel.progress_bar.value(), label)
        self.status_bar.showMessage(label)

    def _on_finished(self, data: dict):
        self.param_panel.set_run_enabled(True)
        task_type = data.get("type", "")

        if task_type == "full":
            results = data["results"]          # Dict[str, SimResult]
            freq_data = data["freq_data"]      # Dict[str, (freqs, T)]

            # Update all plot tabs with unified results
            self.plot_tabs.update_time_histories(results)
            self.plot_tabs.update_actuator(results)
            self.plot_tabs.update_power(results)
            self.plot_tabs.update_transmissibility(freq_data)
            self.plot_tabs.update_controller_comparison(results)

            # Update metrics panel (single unified table)
            self.metrics_panel.update_all_controllers(results)

            # Animation — use best active controller
            active_names = [n for n in results if n not in ("No Suspension", "Passive")]
            if active_names:
                from activeseat.metrics import compute_metrics
                best = min(active_names,
                           key=lambda n: compute_metrics(results[n])["rms_driver_accel"])
                self.animation.set_result(results[best])
            self.centre_tabs.setCurrentIndex(0)  # show animation

            road = next(iter(results.values())).road_type
            self.status_bar.showMessage(
                f"Done — All 5 controllers on {road}. Freq sweep complete."
            )

    def _on_error(self, tb: str):
        self.param_panel.set_run_enabled(True)
        self.status_bar.showMessage("Simulation error!")
        QMessageBox.critical(self, "Simulation Error", tb)
