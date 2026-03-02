"""
plotting.py — Publication-quality matplotlib figures for the seat-suspension
simulation.  All functions return ``Figure`` objects (not shown automatically)
so they can be embedded in the PyQt6 GUI or saved from the CLI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .simulation import SimResult
from .metrics import compute_metrics


# Consistent style
_COLORS = {
    "Passive":  "#1f77b4",
    "LQR":      "#d62728",
    "H∞":       "#2ca02c",
    "Adaptive": "#ff7f0e",
}

def _color(name: str) -> str:
    return _COLORS.get(name, "#333333")

def _style(name: str) -> str:
    return "--" if name == "Passive" else "-"


# ---------------------------------------------------------------------------
# 1.  Time-history comparison (passive vs active)
# ---------------------------------------------------------------------------

def plot_time_histories(
    passive: SimResult,
    active: SimResult,
) -> Figure:
    """4-subplot figure: road input, seat disp, driver disp, driver accel."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for res, ls, label, clr in [
        (passive, "--", passive.controller_name, _color(passive.controller_name)),
        (active,  "-",  active.controller_name,  _color(active.controller_name)),
    ]:
        t = res.t * 1e3  # ms for display
        axes[0].plot(t, res.z0 * 1e3, ls, color=clr, label=label, linewidth=1.2)
        axes[1].plot(t, res.z_s * 1e3, ls, color=clr, label=label, linewidth=1.2)
        axes[2].plot(t, res.z_d * 1e3, ls, color=clr, label=label, linewidth=1.2)
        axes[3].plot(t, res.accel_driver, ls, color=clr, label=label, linewidth=1.2)

    titles = ["Road Input z₀", "Seat Displacement zₛ",
              "Driver Displacement z_d", "Driver Acceleration z̈_d"]
    ylabels = ["[mm]", "[mm]", "[mm]", "[m/s²]"]
    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [ms]")
    fig.suptitle("Time-History Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# 2.  Actuator response
# ---------------------------------------------------------------------------

def plot_actuator_response(active: SimResult) -> Figure:
    """2-subplot: actuator force and motor torque over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = active.t * 1e3
    p = active.params

    ax1.plot(t, active.Fa, color=_color(active.controller_name), linewidth=1.0)
    ax1.axhline(p.max_force, color="gray", linestyle=":", linewidth=0.8,
                label=f"±F_max = {p.max_force:.0f} N")
    ax1.axhline(-p.max_force, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("Actuator Force [N]")
    ax1.set_title("Actuator Force F_a(t)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, active.tau_m, color=_color(active.controller_name), linewidth=1.0)
    ax2.axhline(p.max_torque, color="gray", linestyle=":", linewidth=0.8,
                label=f"±τ_max = {p.max_torque:.1f} N·m")
    ax2.axhline(-p.max_torque, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("Motor Torque [N·m]")
    ax2.set_xlabel("Time [ms]")
    ax2.set_title("Motor Torque τ_m(t)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Actuator Response — {active.controller_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# 3.  Transmissibility
# ---------------------------------------------------------------------------

def plot_transmissibility(
    freqs: np.ndarray,
    T_passive: np.ndarray,
    T_active: np.ndarray,
    active_name: str = "Active",
) -> Figure:
    """Bode-magnitude plot of transmissibility |z̈_d / z̈_0|."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(freqs, T_passive, "--", color=_color("Passive"), linewidth=1.5,
                label="Passive")
    ax.semilogy(freqs, T_active, "-", color=_color(active_name), linewidth=1.5,
                label=active_name)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Transmissibility |z̈_d / z̈_₀|")
    ax.set_title("Seat-to-Driver Transmissibility", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(freqs[0], freqs[-1])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4.  Power analysis
# ---------------------------------------------------------------------------

def plot_power_analysis(active: SimResult) -> Figure:
    """Instantaneous power and cumulative energy."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = active.t * 1e3

    ax1.plot(t, active.power, color=_color(active.controller_name), linewidth=0.8)
    ax1.set_ylabel("Instantaneous Power [W]")
    ax1.set_title("Power P(t) = Fₐ · żₛ", fontsize=10)
    ax1.grid(True, alpha=0.3)

    energy = np.cumsum(np.abs(active.power)) * (active.t[1] - active.t[0])
    ax2.plot(t, energy, color=_color(active.controller_name), linewidth=1.2)
    ax2.set_ylabel("Cumulative Energy [J]")
    ax2.set_xlabel("Time [ms]")
    ax2.set_title("Cumulative Energy Consumption", fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Power Analysis — {active.controller_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# 5.  Metrics comparison bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    passive_metrics: Dict[str, float],
    active_metrics: Dict[str, float],
    active_name: str = "Active",
) -> Figure:
    """Grouped bar chart of key metrics with percentage labels."""
    keys = list(passive_metrics.keys())
    labels = [k.replace("_", " ").title() for k in keys]
    p_vals = np.array([passive_metrics[k] for k in keys])
    a_vals = np.array([active_metrics[k] for k in keys])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w / 2, p_vals, w, label="Passive", color=_color("Passive"), alpha=0.85)
    ax.bar(x + w / 2, a_vals, w, label=active_name, color=_color(active_name), alpha=0.85)

    # Percentage labels
    for i, (pv, av) in enumerate(zip(p_vals, a_vals)):
        if abs(pv) > 1e-12:
            pct = (av - pv) / abs(pv) * 100
            color = "green" if pct < 0 else "red"
            ax.text(i + w / 2, av, f"{pct:+.0f}%", ha="center", va="bottom",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Performance Metrics Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6.  Controller comparison (all controllers, same road)
# ---------------------------------------------------------------------------

def plot_controller_comparison(
    results: Dict[str, SimResult],
) -> Figure:
    """Overlay driver acceleration time histories for all controllers,
    plus a bar chart of RMS driver accel."""
    names = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={"width_ratios": [2, 1]})

    for name, res in results.items():
        ax1.plot(res.t * 1e3, res.accel_driver, _style(name),
                 color=_color(name), label=name, linewidth=1.0)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Driver Acceleration [m/s²]")
    ax1.set_title("Driver Acceleration — All Controllers", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    rms_vals = []
    for name in names:
        m = compute_metrics(results[name])
        rms_vals.append(m["rms_driver_accel"])
    bars = ax2.bar(names, rms_vals, color=[_color(n) for n in names], alpha=0.85)
    ax2.set_ylabel("RMS Driver Accel [m/s²]")
    ax2.set_title("RMS Comparison", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, rms_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Controller Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# 7.  Parameter sweep
# ---------------------------------------------------------------------------

def plot_parameter_sweep(
    param_name: str,
    values: np.ndarray,
    passive_metric_vals: np.ndarray,
    active_metric_vals: np.ndarray,
    metric_name: str = "rms_driver_accel",
    active_name: str = "Active",
) -> Figure:
    """Line plot of one metric vs. one swept parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(values, passive_metric_vals, "o--", color=_color("Passive"),
            label="Passive", linewidth=1.5, markersize=5)
    ax.plot(values, active_metric_vals, "s-", color=_color(active_name),
            label=active_name, linewidth=1.5, markersize=5)
    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"Parameter Sweep: {param_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
