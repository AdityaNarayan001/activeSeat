"""
worker.py — Background QThread for running simulations without freezing the GUI.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
import traceback

from ..params import SeatParams, ControllerParams, RoadConfig, SimParams
from ..simulation import (
    SimResult,
    run_comparison,
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
    TASK_COMPARISON = "comparison"
    TASK_ALL_CTRL = "all_controllers"
    TASK_FREQ_SWEEP = "freq_sweep"
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

            if self.task == self.TASK_COMPARISON:
                self.progress.emit(10.0)
                passive, active = run_comparison(
                    self.seat, self.ctrl, self.road, self.sim
                )
                self.progress.emit(90.0)
                self.finished.emit({
                    "type": "comparison",
                    "passive": passive,
                    "active": active,
                })

            elif self.task == self.TASK_ALL_CTRL:
                self.progress.emit(10.0)
                results = run_all_controllers(
                    self.seat, self.ctrl, self.road, self.sim
                )
                self.progress.emit(90.0)
                self.finished.emit({
                    "type": "all_controllers",
                    "results": results,
                })

            elif self.task == self.TASK_FREQ_SWEEP:
                self.progress.emit(10.0)
                freqs, T_p, T_a = run_frequency_sweep(
                    self.seat, self.ctrl, self.sim
                )
                self.progress.emit(90.0)
                self.finished.emit({
                    "type": "freq_sweep",
                    "freqs": freqs,
                    "T_passive": T_p,
                    "T_active": T_a,
                })

            elif self.task == self.TASK_FULL:
                # Phase 1: Passive vs Active comparison (0-20%)
                self.stage.emit("Step 1/3 \u2014 Passive vs Active comparison\u2026")
                self.progress.emit(2.0)
                passive, active = run_comparison(
                    self.seat, self.ctrl, self.road, self.sim
                )
                self.progress.emit(20.0)

                # Phase 2: All controllers (20-50%)
                self.stage.emit("Step 2/3 \u2014 All controllers comparison\u2026")
                def on_ctrl_progress(frac):
                    self.progress.emit(20.0 + frac * 30.0)
                results = run_all_controllers(
                    self.seat, self.ctrl, self.road, self.sim,
                    progress_callback=on_ctrl_progress,
                )
                self.progress.emit(50.0)

                # Phase 3: Frequency sweep (50-95%)
                self.stage.emit("Step 3/3 \u2014 Frequency sweep\u2026")
                def on_freq_progress(frac):
                    self.progress.emit(50.0 + frac * 45.0)
                freqs, T_p, T_a = run_frequency_sweep(
                    self.seat, self.ctrl, self.sim,
                    progress_callback=on_freq_progress,
                )
                self.progress.emit(98.0)
                self.stage.emit("Finalising\u2026")

                self.finished.emit({
                    "type": "full",
                    "passive": passive,
                    "active": active,
                    "all_results": results,
                    "freqs": freqs,
                    "T_passive": T_p,
                    "T_active": T_a,
                })

            else:
                self.error.emit(f"Unknown task: {self.task}")

            self.progress.emit(100.0)

        except Exception:
            self.error.emit(traceback.format_exc())
