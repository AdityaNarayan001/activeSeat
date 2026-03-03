"""
worker.py — Background QThread for running simulations without freezing the GUI.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
import traceback

from ..params import SeatParams, ControllerParams, RoadConfig, SimParams
from ..simulation import (
    SimResult,
    run_all_controllers,
    run_frequency_sweep,
)


class SimWorker(QThread):
    """Runs a simulation in a background thread.

    Signals
    -------
    finished(dict)   — emitted with results dict on success
    error(str)       — emitted with traceback string on failure
    progress(float)  — 0–100 progress estimate (coarse)
    """

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(float)
    stage = pyqtSignal(str)

    # Task types
    TASK_FULL = "full"

    def __init__(
        self,
        task: str,
        seat: SeatParams,
        ctrl: ControllerParams,
        road: RoadConfig,
        sim: SimParams,
        parent=None,
    ):
        super().__init__(parent)
        self.task = task
        self.seat = seat
        self.ctrl = ctrl
        self.road = road
        self.sim = sim

    def run(self) -> None:
        try:
            self.progress.emit(5.0)

            if self.task == self.TASK_FULL:
                # Phase 1: All controllers including No Suspension (0-55%)
                self.stage.emit("Step 1/2 \u2014 Running all controllers\u2026")
                self.progress.emit(2.0)

                def on_ctrl_progress(frac):
                    self.progress.emit(2.0 + frac * 53.0)

                results = run_all_controllers(
                    self.seat, self.ctrl, self.road, self.sim,
                    progress_callback=on_ctrl_progress,
                )
                self.progress.emit(55.0)

                # Phase 2: Frequency sweep (55-95%)
                self.stage.emit("Step 2/2 \u2014 Frequency sweep\u2026")

                def on_freq_progress(frac):
                    self.progress.emit(55.0 + frac * 40.0)

                freq_data = run_frequency_sweep(
                    self.seat, self.ctrl, self.sim,
                    progress_callback=on_freq_progress,
                )
                self.progress.emit(98.0)
                self.stage.emit("Finalising\u2026")

                self.finished.emit({
                    "type": "full",
                    "results": results,
                    "freq_data": freq_data,
                })

            else:
                self.error.emit(f"Unknown task: {self.task}")

            self.progress.emit(100.0)

        except Exception:
            self.error.emit(traceback.format_exc())
