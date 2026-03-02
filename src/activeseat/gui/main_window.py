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
    QStatusBar, QProgressBar, QFileDialog, QMessageBox, QApplication,
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

        # Connect param panel run buttons
        self.param_panel.run_comparison_clicked.connect(self._on_run_comparison)
        self.param_panel.run_all_controllers_clicked.connect(self._on_run_all_controllers)
        self.param_panel.run_freq_sweep_clicked.connect(self._on_run_freq_sweep)

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

        act_comp = QAction("Passive vs Active Comparison", self)
        act_comp.setShortcut("Ctrl+R")
        act_comp.triggered.connect(self._on_run_comparison)
        run_menu.addAction(act_comp)

        act_all = QAction("All Controllers Comparison", self)
        act_all.setShortcut("Ctrl+Shift+R")
        act_all.triggered.connect(self._on_run_all_controllers)
        run_menu.addAction(act_all)

        act_freq = QAction("Frequency Sweep (Transmissibility)", self)
        act_freq.setShortcut("Ctrl+T")
        act_freq.triggered.connect(self._on_run_freq_sweep)
        run_menu.addAction(act_freq)

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
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
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

    def _start_worker(self, task: str):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "A simulation is already running.")
            return

        seat = self.param_panel.get_seat_params()
        ctrl = self.param_panel.get_controller_params()
        road = self.param_panel.get_road_config()
        sim = self.param_panel.get_sim_params()

        self._worker = SimWorker(task, seat, ctrl, road, sim, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.param_panel.set_buttons_enabled(False)
        self.status_bar.showMessage(f"Running: {task} …")
        self._worker.start()

    def _on_run_comparison(self):
        self._start_worker(SimWorker.TASK_COMPARISON)

    def _on_run_all_controllers(self):
        self._start_worker(SimWorker.TASK_ALL_CTRL)

    def _on_run_freq_sweep(self):
        self._start_worker(SimWorker.TASK_FREQ_SWEEP)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, val: float):
        self.progress_bar.setValue(int(val))

    def _on_finished(self, data: dict):
        self.progress_bar.setVisible(False)
        self.param_panel.set_buttons_enabled(True)
        task_type = data.get("type", "")

        if task_type == "comparison":
            passive = data["passive"]
            active = data["active"]
            self.plot_tabs.update_time_histories(passive, active)
            self.plot_tabs.update_actuator(active)
            self.plot_tabs.update_power(active)
            self.metrics_panel.update_comparison(passive, active)
            self.animation.set_result(active)
            self.centre_tabs.setCurrentIndex(0)  # show animation
            self.status_bar.showMessage(
                f"Done — Passive vs {active.controller_name} on {active.road_type}."
            )

        elif task_type == "all_controllers":
            results = data["results"]
            self.plot_tabs.update_controller_comparison(results)
            self.metrics_panel.update_all_controllers(results)
            # Use first active controller for animation
            for name, res in results.items():
                if name != "Passive":
                    self.animation.set_result(res)
                    break
            self.centre_tabs.setCurrentIndex(1)  # show plots
            self.plot_tabs.tabs.setCurrentIndex(self.plot_tabs.TAB_COMPARISON)
            self.status_bar.showMessage("Done — All controllers compared.")

        elif task_type == "freq_sweep":
            freqs = data["freqs"]
            T_p = data["T_passive"]
            T_a = data["T_active"]
            ctrl_name = self.param_panel.get_controller_params().controller_type.upper()
            self.plot_tabs.update_transmissibility(freqs, T_p, T_a, ctrl_name)
            self.centre_tabs.setCurrentIndex(1)
            self.plot_tabs.tabs.setCurrentIndex(self.plot_tabs.TAB_TRANSMISSIBILITY)
            self.status_bar.showMessage("Done — Frequency sweep complete.")

    def _on_error(self, tb: str):
        self.progress_bar.setVisible(False)
        self.param_panel.set_buttons_enabled(True)
        self.status_bar.showMessage("Simulation error!")
        QMessageBox.critical(self, "Simulation Error", tb)
