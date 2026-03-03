"""
plotting.py — Publication-quality matplotlib figures for the seat-suspension
simulation.  All functions accept ``Dict[str, SimResult]`` so every controller
(including the rigid "No Suspension" baseline) is overlaid on every plot.
Figures are returned without calling ``plt.show()`` so they can be embedded
in the PyQt6 GUI or saved from the CLI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D as _Line2D

from .simulation import SimResult
from .metrics import compute_metrics


# ── Consistent visual identity ──────────────────────────────────────────

_COLORS = {
    "No Suspension": "#888888",
    "Passive":       "#1f77b4",
    "LQR":           "#d62728",
    "H∞":            "#2ca02c",
    "Adaptive":      "#ff7f0e",
}

_LINESTYLES = {
    "No Suspension": ":",
    "Passive":       "--",
    "LQR":           "-",
    "H∞":            "-",
    "Adaptive":      "-",
}

# Canonical ordering for legends / bar charts
_ORDER = ["No Suspension", "Passive", "LQR", "H∞", "Adaptive"]


def _color(name: str) -> str:
    return _COLORS.get(name, "#333333")


def _ls(name: str) -> str:
    return _LINESTYLES.get(name, "-")


def _ordered(results: Dict[str, SimResult]) -> List[Tuple[str, SimResult]]:
    """Yield (name, result) pairs in canonical order, tolerating missing keys."""
    for n in _ORDER:
        if n in results:
            yield n, results[n]
    # Any unexpected names appended at the end
    for n in results:
        if n not in _ORDER:
            yield n, results[n]


# ── Interactive legend toggling ─────────────────────────────────────────

_TOGGLEABLE = set(_ORDER)


def _enable_legend_toggle(fig: Figure) -> None:
    """Prepare figure for interactive legend toggling.

    Clicking a controller's legend entry in any subplot toggles that
    controller's lines on **all** subplots of the figure.  Only
    ``Line2D``-based legend items whose labels match a known controller
    name are made pickable; limit-lines and bar-chart entries are ignored.
    """
    # Map controller name → original Line2D artists across all axes
    name_to_artists: Dict[str, list] = {}
    for ax in fig.axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            if label in _TOGGLEABLE and isinstance(h, _Line2D):
                name_to_artists.setdefault(label, []).append(h)

    # Map legend-handle → name and make pickable
    handle_to_name: Dict = {}
    name_to_leg: Dict[str, list] = {}
    for ax in fig.axes:
        leg = ax.get_legend()
        if not leg:
            continue
        leg_handles = getattr(leg, "legend_handles",
                              getattr(leg, "legendHandles", []))
        _, labels = ax.get_legend_handles_labels()
        for lh, label in zip(leg_handles, labels):
            if label in _TOGGLEABLE and isinstance(lh, _Line2D):
                lh.set_picker(8)
                handle_to_name[lh] = label
                name_to_leg.setdefault(label, []).append(lh)

    if handle_to_name:
        fig._legend_toggle = {
            "h2n": handle_to_name,
            "n2a": name_to_artists,
            "n2l": name_to_leg,
        }


# ---------------------------------------------------------------------------
# 1.  Time-history comparison (all controllers)
# ---------------------------------------------------------------------------

def plot_time_histories(results: Dict[str, SimResult]) -> Figure:
    """4-subplot figure: road input, seat disp, driver disp, driver accel."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             layout="constrained")

    for name, res in _ordered(results):
        t = res.t * 1e3  # ms
        clr, ls = _color(name), _ls(name)
        axes[0].plot(t, res.z0 * 1e3, ls, color=clr, label=name, linewidth=1.2)
        axes[1].plot(t, res.z_s * 1e3, ls, color=clr, label=name, linewidth=1.2)
        axes[2].plot(t, res.z_d * 1e3, ls, color=clr, label=name, linewidth=1.2)
        axes[3].plot(t, res.accel_driver, ls, color=clr, label=name, linewidth=1.2)

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
    _enable_legend_toggle(fig)
    return fig


# ---------------------------------------------------------------------------
# 2.  Actuator response (all controllers overlaid)
# ---------------------------------------------------------------------------

def plot_actuator_response(results: Dict[str, SimResult]) -> Figure:
    """2-subplot: actuator force and motor torque for every controller."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   layout="constrained")

    # Use any active result to get force/torque limits
    any_res = next(iter(results.values()))
    p = any_res.params

    for name, res in _ordered(results):
        t = res.t * 1e3
        clr, ls = _color(name), _ls(name)
        ax1.plot(t, res.Fa, ls, color=clr, label=name, linewidth=1.0)
        ax2.plot(t, res.tau_m, ls, color=clr, label=name, linewidth=1.0)

    ax1.axhline(p.max_force, color="gray", linestyle=":", linewidth=0.8,
                label=f"±F_max = {p.max_force:.0f} N")
    ax1.axhline(-p.max_force, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("Actuator Force [N]")
    ax1.set_title("Actuator Force F_a(t)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(p.max_torque, color="gray", linestyle=":", linewidth=0.8,
                label=f"±τ_max = {p.max_torque:.1f} N·m")
    ax2.axhline(-p.max_torque, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("Motor Torque [N·m]")
    ax2.set_xlabel("Time [ms]")
    ax2.set_title("Motor Torque τ_m(t)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Actuator Response — All Controllers",
                 fontsize=13, fontweight="bold")
    _enable_legend_toggle(fig)
    return fig


# ---------------------------------------------------------------------------
# 3.  Transmissibility (all controllers)
# ---------------------------------------------------------------------------

def plot_transmissibility(
    freq_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Figure:
    """Bode-magnitude plot of transmissibility |z̈_d / z̈_0| for all controllers.

    Parameters
    ----------
    freq_data : dict mapping controller name → (freqs, T) arrays.
    """
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for name, (freqs, T) in _ordered_freq(freq_data):
        ax.semilogy(freqs, T, _ls(name), color=_color(name), linewidth=1.5,
                    label=name)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Transmissibility |z̈_d / z̈_₀|")
    ax.set_title("Seat-to-Driver Transmissibility", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    freqs_all = next(iter(freq_data.values()))[0]
    ax.set_xlim(freqs_all[0], freqs_all[-1])
    _enable_legend_toggle(fig)
    return fig


def _ordered_freq(freq_data):
    """Yield (name, (freqs, T)) in canonical order."""
    for n in _ORDER:
        if n in freq_data:
            yield n, freq_data[n]
    for n in freq_data:
        if n not in _ORDER:
            yield n, freq_data[n]


# ---------------------------------------------------------------------------
# 4.  Power analysis (all controllers overlaid)
# ---------------------------------------------------------------------------

def plot_power_analysis(results: Dict[str, SimResult]) -> Figure:
    """Instantaneous power and cumulative energy for every controller."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   layout="constrained")

    for name, res in _ordered(results):
        t = res.t * 1e3
        clr, ls = _color(name), _ls(name)
        ax1.plot(t, res.power, ls, color=clr, label=name, linewidth=0.8)

        energy = np.cumsum(np.abs(res.power)) * (res.t[1] - res.t[0])
        ax2.plot(t, energy, ls, color=clr, label=name, linewidth=1.2)

    ax1.set_ylabel("Instantaneous Power [W]")
    ax1.set_title("Power P(t) = Fₐ · żₛ", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Cumulative Energy [J]")
    ax2.set_xlabel("Time [ms]")
    ax2.set_title("Cumulative Energy Consumption", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Power Analysis — All Controllers",
                 fontsize=13, fontweight="bold")
    _enable_legend_toggle(fig)
    return fig


# ---------------------------------------------------------------------------
# 5.  Metrics comparison bar chart (all controllers)
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    all_metrics: Dict[str, Dict[str, float]],
) -> Figure:
    """Grouped bar chart of key metrics for all controllers."""
    # Use canonical order (only keep available names)
    names = [n for n in _ORDER if n in all_metrics]
    if not names:
        names = list(all_metrics.keys())

    keys = list(next(iter(all_metrics.values())).keys())
    labels = [k.replace("_", " ").title() for k in keys]
    n_groups = len(keys)
    n_bars = len(names)

    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    x = np.arange(n_groups)
    w = 0.8 / n_bars  # bar width

    for j, name in enumerate(names):
        vals = np.array([all_metrics[name][k] for k in keys])
        offset = (j - (n_bars - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=name,
                      color=_color(name), alpha=0.85)
        # Percentage change relative to "No Suspension" baseline
        if "No Suspension" in all_metrics and name != "No Suspension":
            base = all_metrics["No Suspension"]
            for i, k in enumerate(keys):
                bv = base[k]
                if abs(bv) > 1e-12:
                    pct = (all_metrics[name][k] - bv) / abs(bv) * 100
                    clr = "green" if pct < 0 else "red"
                    ax.text(x[i] + offset, vals[i],
                            f"{pct:+.0f}%", ha="center", va="bottom",
                            fontsize=6, color=clr, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Performance Metrics Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _enable_legend_toggle(fig)
    return fig


# ---------------------------------------------------------------------------
# 6.  Controller comparison (driver accel time + RMS bar)
# ---------------------------------------------------------------------------

def plot_controller_comparison(
    results: Dict[str, SimResult],
) -> Figure:
    """Overlay driver acceleration time histories for all controllers,
    plus a bar chart of RMS driver accel."""
    names = [n for n in _ORDER if n in results]
    if not names:
        names = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                   gridspec_kw={"width_ratios": [2, 1]},
                                   layout="constrained")

    for name in names:
        res = results[name]
        ax1.plot(res.t * 1e3, res.accel_driver, _ls(name),
                 color=_color(name), label=name, linewidth=1.0)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Driver Acceleration [m/s²]")
    ax1.set_title("Driver Acceleration — All Controllers", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    rms_vals = [compute_metrics(results[n])["rms_driver_accel"] for n in names]
    bars = ax2.bar(names, rms_vals, color=[_color(n) for n in names], alpha=0.85)
    ax2.set_ylabel("RMS Driver Accel [m/s²]")
    ax2.set_title("RMS Comparison", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, rms_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.tick_params(axis="x", rotation=30)

    fig.suptitle("Controller Comparison", fontsize=13, fontweight="bold")
    _enable_legend_toggle(fig)
    return fig


# ---------------------------------------------------------------------------
# 7.  Parameter sweep
# ---------------------------------------------------------------------------

def plot_parameter_sweep(
    param_name: str,
    values: np.ndarray,
    sweep_metrics: Dict[str, np.ndarray],
    metric_name: str = "rms_driver_accel",
) -> Figure:
    """Line plot of one metric vs. one swept parameter for all controllers."""
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for name in _ORDER:
        if name not in sweep_metrics:
            continue
        marker = "o" if name in ("No Suspension", "Passive") else "s"
        ax.plot(values, sweep_metrics[name], f"{marker}{_ls(name)}",
                color=_color(name), label=name, linewidth=1.5, markersize=5)
    # Any extras
    for name in sweep_metrics:
        if name not in _ORDER:
            ax.plot(values, sweep_metrics[name], "s-", color=_color(name),
                    label=name, linewidth=1.5, markersize=5)
    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"Parameter Sweep: {param_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _enable_legend_toggle(fig)
    return fig
