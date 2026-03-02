"""
metrics_panel.py — Right-side metrics dashboard for the GUI.

Shows:
    • Summary "cards" for key metrics (coloured boxes)
    • Full comparison table (QTableWidget) with colour-coded % change
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QGroupBox, QGridLayout, QFrame, QSizePolicy,
)

from ..simulation import SimResult
from ..metrics import compute_metrics, compare_metrics


# Metric display names and units
_DISPLAY = {
    "rms_driver_accel":    ("RMS Driver Accel", "m/s²"),
    "peak_driver_accel":   ("Peak Driver Accel", "m/s²"),
    "rms_actuator_force":  ("RMS Force", "N"),
    "peak_actuator_force": ("Peak Force", "N"),
    "max_seat_travel":     ("Max Seat Travel", "mm"),
    "avg_power":           ("Avg Power", "W"),
    "peak_power":          ("Peak Power", "W"),
    "control_effort":      ("Control Effort", "N²·s"),
}

_SCALE = {
    "max_seat_travel": 1000.0,  # m → mm
}


class _MetricCard(QFrame):
    """A small summary card showing one metric value."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #f0f4f8; border-radius: 6px; padding: 6px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        self._title = QLabel(title)
        self._title.setFont(QFont("Arial", 9))
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value = QLabel("—")
        self._value.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._change = QLabel("")
        self._change.setFont(QFont("Arial", 9))
        self._change.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)
        layout.addWidget(self._value)
        layout.addWidget(self._change)

    def set_value(self, val: float, unit: str, pct: Optional[float] = None):
        if abs(val) >= 1000:
            self._value.setText(f"{val:.0f}")
        elif abs(val) >= 1:
            self._value.setText(f"{val:.2f}")
        else:
            self._value.setText(f"{val:.4f}")

        self._title.setText(f"{self._title.text().split(' [')[0]} [{unit}]")

        if pct is not None:
            arrow = "↓" if pct < 0 else "↑"
            color = "#2e7d32" if pct < 0 else "#c62828"
            self._change.setText(f"{pct:+.1f}% {arrow}")
            self._change.setStyleSheet(f"color: {color}; font-weight: bold;")
        else:
            self._change.setText("")


class MetricsPanel(QWidget):
    """Right-side dashboard showing metrics cards and comparison table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # Title
        title = QLabel("Performance Metrics")
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Summary cards (top 4 metrics)
        cards_group = QGroupBox("Key Indicators (Active)")
        cards_grid = QGridLayout()
        self._cards: dict[str, _MetricCard] = {}
        card_keys = ["rms_driver_accel", "peak_driver_accel", "max_seat_travel", "avg_power"]
        for i, key in enumerate(card_keys):
            label, unit = _DISPLAY.get(key, (key, ""))
            card = _MetricCard(f"{label} [{unit}]")
            self._cards[key] = card
            cards_grid.addWidget(card, i // 2, i % 2)
        cards_group.setLayout(cards_grid)
        layout.addWidget(cards_group)

        # Full comparison table
        tbl_group = QGroupBox("Passive vs Active Comparison")
        tbl_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Metric", "Passive", "Active", "Change"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        tbl_layout.addWidget(self.table)
        tbl_group.setLayout(tbl_layout)
        layout.addWidget(tbl_group)

        # All-controllers table
        multi_group = QGroupBox("All Controllers")
        multi_layout = QVBoxLayout()
        self.multi_table = QTableWidget()
        self.multi_table.verticalHeader().setVisible(False)
        self.multi_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.multi_table.setAlternatingRowColors(True)
        multi_layout.addWidget(self.multi_table)
        multi_group.setLayout(multi_layout)
        layout.addWidget(multi_group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Update from results
    # ------------------------------------------------------------------

    def update_comparison(self, passive: SimResult, active: SimResult):
        """Refresh cards and table from passive vs active results."""
        pm = compute_metrics(passive)
        am = compute_metrics(active)
        comp = compare_metrics(pm, am)

        # Update cards
        for key, card in self._cards.items():
            scale = _SCALE.get(key, 1.0)
            _, unit = _DISPLAY.get(key, (key, ""))
            card.set_value(am[key] * scale, unit, comp[key]["change_pct"])

        # Update table
        keys = list(pm.keys())
        self.table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            label, unit = _DISPLAY.get(key, (key, ""))
            scale = _SCALE.get(key, 1.0)
            pv = pm[key] * scale
            av = am[key] * scale
            pct = comp[key]["change_pct"]

            self.table.setItem(row, 0, QTableWidgetItem(f"{label} [{unit}]"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{pv:.4f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{av:.4f}"))

            pct_item = QTableWidgetItem(f"{pct:+.1f}%")
            if pct < -5:
                pct_item.setForeground(QColor("#2e7d32"))  # green
            elif pct > 5:
                pct_item.setForeground(QColor("#c62828"))  # red
            else:
                pct_item.setForeground(QColor("#f57f17"))  # yellow
            self.table.setItem(row, 3, pct_item)

        self.table.resizeColumnsToContents()

    def update_all_controllers(self, results: Dict[str, SimResult]):
        """Populate the all-controllers table."""
        names = list(results.keys())
        metrics = {n: compute_metrics(r) for n, r in results.items()}
        keys = list(next(iter(metrics.values())).keys())

        self.multi_table.setColumnCount(1 + len(names))
        self.multi_table.setHorizontalHeaderLabels(["Metric"] + names)
        self.multi_table.setRowCount(len(keys))

        for row, key in enumerate(keys):
            label, unit = _DISPLAY.get(key, (key, ""))
            scale = _SCALE.get(key, 1.0)
            self.multi_table.setItem(row, 0, QTableWidgetItem(f"{label} [{unit}]"))
            for col, name in enumerate(names, start=1):
                val = metrics[name][key] * scale
                self.multi_table.setItem(row, col, QTableWidgetItem(f"{val:.4f}"))

        self.multi_table.resizeColumnsToContents()
