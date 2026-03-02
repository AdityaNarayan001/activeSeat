"""
metrics.py — Performance metrics for the seat-suspension simulation.

All metrics are computed from a ``SimResult`` object using trapezoidal
integration (``np.trapz``) for accuracy.

Key metrics (ISO 2631 inspired):
    • RMS driver acceleration         — primary ride comfort indicator
    • Peak driver acceleration        — worst-case transient
    • RMS actuator force              — average control effort
    • Peak actuator force             — sizing requirement
    • Maximum seat travel (stroke)    — actuator stroke constraint
    • Average power consumption       — energy cost
    • Peak power                      — motor/driver sizing
    • Control effort (integral of F²) — total energy-like effort measure
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .simulation import SimResult


# ---------------------------------------------------------------------------
# Single-run metrics
# ---------------------------------------------------------------------------

def compute_metrics(res: SimResult) -> Dict[str, float]:
    """Compute all performance metrics from a simulation result.

    Returns
    -------
    dict with string keys and float values.
    """
    t = res.t
    T = t[-1] - t[0]
    if T <= 0:
        raise ValueError("Simulation duration must be positive.")

    ad = res.accel_driver
    Fa = res.Fa
    pw = res.power

    return {
        "rms_driver_accel":   float(np.sqrt(np.trapz(ad**2, t) / T)),
        "peak_driver_accel":  float(np.max(np.abs(ad))),
        "rms_actuator_force": float(np.sqrt(np.trapz(Fa**2, t) / T)),
        "peak_actuator_force": float(np.max(np.abs(Fa))),
        "max_seat_travel":    float(np.max(np.abs(res.x1))),
        "avg_power":          float(np.trapz(np.abs(pw), t) / T),
        "peak_power":         float(np.max(np.abs(pw))),
        "control_effort":     float(np.trapz(Fa**2, t)),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_metrics(
    passive: Dict[str, float],
    active: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Compare passive vs active metrics.

    Returns
    -------
    dict mapping metric_name → {"passive": val, "active": val, "change_pct": %}
    where change_pct = (active - passive) / |passive| × 100.
    Negative change_pct means improvement (reduction).
    """
    comparison = {}
    for key in passive:
        p = passive[key]
        a = active[key]
        denom = abs(p) if abs(p) > 1e-15 else 1e-15
        pct = (a - p) / denom * 100.0
        comparison[key] = {"passive": p, "active": a, "change_pct": pct}
    return comparison


def compare_all_controllers(
    results: Dict[str, SimResult],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for every controller in *results* dict.

    Returns
    -------
    dict mapping controller_name → metrics_dict
    """
    return {name: compute_metrics(res) for name, res in results.items()}


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "rms_driver_accel":    "RMS Driver Accel  [m/s²]",
    "peak_driver_accel":   "Peak Driver Accel [m/s²]",
    "rms_actuator_force":  "RMS Actuator Force  [N]",
    "peak_actuator_force": "Peak Actuator Force [N]",
    "max_seat_travel":     "Max Seat Travel     [m]",
    "avg_power":           "Avg Power           [W]",
    "peak_power":          "Peak Power          [W]",
    "control_effort":      "Control Effort    [N²·s]",
}


def format_comparison_table(
    comparison: Dict[str, Dict[str, float]],
) -> str:
    """Return a nicely formatted ASCII table string."""
    lines = []
    header = f"{'Metric':<30s}  {'Passive':>12s}  {'Active':>12s}  {'Change':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    for key, vals in comparison.items():
        label = _METRIC_LABELS.get(key, key)
        pct = vals["change_pct"]
        arrow = "↓" if pct < 0 else "↑" if pct > 0 else "–"
        lines.append(
            f"{label:<30s}  {vals['passive']:12.4f}  {vals['active']:12.4f}  "
            f"{pct:+7.1f}% {arrow}"
        )
    return "\n".join(lines)


def format_all_controllers_table(
    all_metrics: Dict[str, Dict[str, float]],
) -> str:
    """Return an ASCII table comparing all controllers."""
    names = list(all_metrics.keys())
    lines = []
    header = f"{'Metric':<30s}" + "".join(f"  {n:>12s}" for n in names)
    lines.append(header)
    lines.append("-" * len(header))
    metric_keys = list(next(iter(all_metrics.values())).keys())
    for key in metric_keys:
        label = _METRIC_LABELS.get(key, key)
        row = f"{label:<30s}"
        for name in names:
            row += f"  {all_metrics[name][key]:12.4f}"
        lines.append(row)
    return "\n".join(lines)
