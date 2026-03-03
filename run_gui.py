#!/usr/bin/env python3
"""
run_gui.py — Launch the ActiveSeat desktop GUI application.

Usage:
    python run_gui.py                       # default parameters
    python run_gui.py --scenario path.yaml  # pre-load a scenario
"""

import sys
import os
import argparse
import warnings

# Silence harmless matplotlib layout-engine conflict warning
warnings.filterwarnings(
    "ignore",
    message=".*layout engine that is incompatible with subplots_adjust.*",
    category=UserWarning,
)

# Ensure the src/ directory is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Set matplotlib backend BEFORE any mpl imports (prevents macOS Tk conflicts)
import matplotlib
matplotlib.use("Agg")

from PyQt6.QtWidgets import QApplication
from activeseat.gui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description="ActiveSeat GUI Simulator")
    parser.add_argument(
        "--scenario", "-s", type=str, default=None,
        help="Path to a YAML scenario file to pre-load.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("ActiveSeat")
    app.setStyle("Fusion")

    win = MainWindow(scenario_path=args.scenario)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
