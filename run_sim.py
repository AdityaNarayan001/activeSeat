#!/usr/bin/env python3
"""
run_sim.py — CLI entry point for headless batch simulation.

Demonstrates the full pipeline: passive vs active comparison, all-controller
comparison, metrics tables, and plot generation.

Usage:
    python run_sim.py                           # default parameters
    python run_sim.py --scenario scenarios/default.yaml
    python run_sim.py --save-plots output/      # save figures to directory
"""

import sys
import os
import argparse

# Ensure the src/ directory is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless use
import matplotlib.pyplot as plt

from activeseat.params import SeatParams, ControllerParams, RoadConfig, SimParams
from activeseat.config import load_scenario
from activeseat.simulation import (
    run_all_controllers,
    run_frequency_sweep,
    run_parameter_sweep,
)
from activeseat.metrics import (
    compute_metrics,
    compare_all_controllers,
    format_all_controllers_table,
)
from activeseat import plotting as plotter


def main():
    parser = argparse.ArgumentParser(description="ActiveSeat CLI Simulator")
    parser.add_argument("--scenario", "-s", type=str, default=None)
    parser.add_argument("--save-plots", type=str, default="output",
                        help="Directory to save plot PNGs (default: output/).")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # 1. Load or create parameters
    # -------------------------------------------------------------------
    if args.scenario:
        seat, ctrl, road, sim = load_scenario(args.scenario)
        print(f"Loaded scenario: {args.scenario}")
    else:
        seat = SeatParams()
        ctrl = ControllerParams(controller_type="lqr")
        road = RoadConfig(road_type="bump")
        sim = SimParams(duration=2.0)
        print("Using default parameters.")

    print(f"\n{'='*70}")
    print(f"  Driver mass: {seat.mass_driver} kg  |  Seat mass: {seat.mass_seat} kg")
    print(f"  k1={seat.k1:.0f} N/m  c1={seat.c1:.0f} N·s/m")
    print(f"  k2={seat.k2:.0f} N/m  c2={seat.c2:.0f} N·s/m")
    print(f"  Gear ratio: {seat.gear_ratio}  |  Max torque: {seat.max_torque} N·m")
    print(f"  Max actuator force: {seat.max_force:.0f} N")
    print(f"  Reflected motor mass: {seat.reflected_mass:.1f} kg")
    print(f"{'='*70}\n")

    save_dir = args.save_plots
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _save(fig, name):
        if save_dir:
            path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")

    # -------------------------------------------------------------------
    # 2. All controllers comparison (including No Suspension baseline)
    # -------------------------------------------------------------------
    print(f"─── All Controllers on {road.road_type} road ───")
    all_results = run_all_controllers(seat, ctrl, road, sim)
    all_metrics = compare_all_controllers(all_results)
    print(format_all_controllers_table(all_metrics))

    fig1 = plotter.plot_time_histories(all_results)
    _save(fig1, "time_histories")
    fig2 = plotter.plot_actuator_response(all_results)
    _save(fig2, "actuator_response")
    fig3 = plotter.plot_power_analysis(all_results)
    _save(fig3, "power_analysis")
    fig4 = plotter.plot_metrics_comparison(all_metrics)
    _save(fig4, "metrics_comparison")
    fig5 = plotter.plot_controller_comparison(all_results)
    _save(fig5, "controller_comparison")

    # -------------------------------------------------------------------
    # 3. Frequency sweep (transmissibility — all controllers)
    # -------------------------------------------------------------------
    print(f"\n─── Frequency Sweep ({sim.freq_start}–{sim.freq_end} Hz) ───")
    freq_data = run_frequency_sweep(seat, ctrl, sim)
    fig6 = plotter.plot_transmissibility(freq_data)
    _save(fig6, "transmissibility")
    for name, (freqs, T) in freq_data.items():
        print(f"  {name:16s} peak T = {T.max():.3f} at {freqs[T.argmax()]:.1f} Hz")

    # -------------------------------------------------------------------
    # 4. Parameter sweep: driver mass 50–120 kg
    # -------------------------------------------------------------------
    print("\n─── Parameter Sweep: Driver Mass ───")
    masses = np.linspace(50, 120, 8)
    sweep = run_parameter_sweep("mass_driver", masses, seat, ctrl, road, sim)
    p_rms = np.array([compute_metrics(r[1])["rms_driver_accel"] for r in sweep])
    a_rms = np.array([compute_metrics(r[2])["rms_driver_accel"] for r in sweep])
    sweep_metrics = {"Passive": p_rms, ctrl.controller_type.upper(): a_rms}
    fig7 = plotter.plot_parameter_sweep("mass_driver", masses, sweep_metrics,
                                        "rms_driver_accel")
    _save(fig7, "sweep_driver_mass")
    for val, prms, arms in zip(masses, p_rms, a_rms):
        print(f"  {val:.0f} kg  →  passive {prms:.3f}  active {arms:.3f}  "
              f"({(arms-prms)/prms*100:+.1f}%)")

    # -------------------------------------------------------------------
    # 5. Engineering interpretation
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  ENGINEERING SUMMARY")
    print(f"{'='*70}")
    best_ctrl = min(all_metrics, key=lambda n: all_metrics[n]["rms_driver_accel"])
    worst_ctrl = max(all_metrics, key=lambda n: all_metrics[n]["rms_driver_accel"])
    print(f"  Best  RMS driver accel: {best_ctrl} "
          f"({all_metrics[best_ctrl]['rms_driver_accel']:.4f} m/s²)")
    print(f"  Worst RMS driver accel: {worst_ctrl} "
          f"({all_metrics[worst_ctrl]['rms_driver_accel']:.4f} m/s²)")

    # Check if saturation was hit (use selected active controller)
    active_name = ctrl.controller_type.upper() if ctrl.controller_type.lower() != "hinf" else "H∞"
    if active_name in all_metrics:
        am = all_metrics[active_name]
    else:
        # Fallback to first non-baseline
        am = next(v for k, v in all_metrics.items() if k not in ("No Suspension", "Passive"))
    peak_force = am["peak_actuator_force"]
    if peak_force >= seat.max_force * 0.95:
        print(f"  ⚠ Actuator near saturation: peak force {peak_force:.0f} N "
              f"vs limit {seat.max_force:.0f} N")
    else:
        print(f"  ✓ Actuator within limits: peak {peak_force:.0f} / {seat.max_force:.0f} N "
              f"({peak_force/seat.max_force*100:.0f}%)")

    print(f"  Average power consumption: {am['avg_power']:.2f} W")
    print(f"  Max seat travel: {am['max_seat_travel']*1e3:.2f} mm")
    print(f"{'='*70}\n")

    print(f"\nAll plots saved to: {os.path.abspath(save_dir)}/")


if __name__ == "__main__":
    main()
