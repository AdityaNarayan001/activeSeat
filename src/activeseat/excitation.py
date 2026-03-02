"""
excitation.py — Road profile generators for the seat-suspension simulation.

Each public function returns:
    road_func : callable  (t: float) → (z0, dz0)
        Interpolation function giving road displacement and velocity at any
        time *t*.  Suitable for passing directly to the ODE integrator.
    t_arr     : ndarray  — pre-computed time array  [s]
    z0_arr    : ndarray  — pre-computed displacement array [m]
    dz0_arr   : ndarray  — pre-computed velocity array [m/s]

Supported profiles:
    1. Half-sine bump
    2. Sinusoidal (single-frequency)
    3. Random road (ISO 8608)
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy import interpolate

from .params import RoadConfig, SimParams


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
RoadResult = Tuple[Callable, np.ndarray, np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# 1.  Half-sine bump
# ---------------------------------------------------------------------------

def bump_profile(
    cfg: RoadConfig,
    sim: SimParams,
) -> RoadResult:
    """Half-sine bump: z0(t) = h·sin(π v t / L)  for  0 ≤ t ≤ L/v.

    Parameters are read from *cfg*:  bump_height, bump_length, vehicle_speed.
    """
    h = cfg.bump_height
    L = cfg.bump_length
    v = cfg.vehicle_speed
    T = sim.duration
    dt = sim.dt

    t_bump = L / v  # duration the wheel is on the bump [s]
    t_arr = np.arange(0, T + dt, dt)
    z0_arr = np.zeros_like(t_arr)
    dz0_arr = np.zeros_like(t_arr)

    mask = t_arr <= t_bump
    omega = np.pi * v / L
    z0_arr[mask] = h * np.sin(omega * t_arr[mask])
    dz0_arr[mask] = h * omega * np.cos(omega * t_arr[mask])

    # Build interpolation function (linear — fast & sufficient at small dt)
    _z0_interp = interpolate.interp1d(t_arr, z0_arr, kind="linear",
                                       bounds_error=False, fill_value=0.0)
    _dz0_interp = interpolate.interp1d(t_arr, dz0_arr, kind="linear",
                                        bounds_error=False, fill_value=0.0)

    def road_func(t: float) -> Tuple[float, float]:
        return float(_z0_interp(t)), float(_dz0_interp(t))

    return road_func, t_arr, z0_arr, dz0_arr


# ---------------------------------------------------------------------------
# 2.  Sinusoidal excitation
# ---------------------------------------------------------------------------

def sinusoidal_profile(
    cfg: RoadConfig,
    sim: SimParams,
) -> RoadResult:
    """Single-frequency sinusoidal: z0(t) = A·sin(2πf·t)."""
    A = cfg.sine_amplitude
    f = cfg.sine_frequency
    T = sim.duration
    dt = sim.dt

    t_arr = np.arange(0, T + dt, dt)
    omega = 2.0 * np.pi * f
    z0_arr = A * np.sin(omega * t_arr)
    dz0_arr = A * omega * np.cos(omega * t_arr)

    def road_func(t: float) -> Tuple[float, float]:
        return (A * np.sin(omega * t),
                A * omega * np.cos(omega * t))

    return road_func, t_arr, z0_arr, dz0_arr


# ---------------------------------------------------------------------------
# 3.  Random road — ISO 8608
# ---------------------------------------------------------------------------

# ISO 8608 reference PSD values at n0=0.1 cycles/m  [m³/cycle]
_ISO_8608_GD: dict[str, float] = {
    "A": 1e-6,
    "B": 4e-6,
    "C": 16e-6,
    "D": 64e-6,
    "E": 256e-6,
}


def random_iso8608_profile(
    cfg: RoadConfig,
    sim: SimParams,
) -> RoadResult:
    """Generate a random road profile conforming to ISO 8608.

    The spatial PSD is  G_d(n) = G_d(n0)·(n/n0)^(-2)  with n0=0.1 cycles/m.
    Temporal PSD is obtained via  Φ(f) = G_d(f/v) / v .

    Implementation:
        1. Generate white noise in the frequency domain.
        2. Shape its spectrum to match the ISO 8608 PSD.
        3. IFFT back to time domain.
        4. Compute velocity via central differences.
    """
    v = cfg.iso_speed          # vehicle speed [m/s]
    T = sim.duration
    dt = sim.dt
    iso_class = cfg.iso_class.upper()

    Gd_n0 = _ISO_8608_GD.get(iso_class, _ISO_8608_GD["C"])
    n0 = 0.1  # reference spatial frequency [cycles/m]

    rng = np.random.default_rng(cfg.iso_seed)

    t_arr = np.arange(0, T + dt, dt)
    N = len(t_arr)
    fs = 1.0 / dt               # sampling frequency [Hz]
    freqs = np.fft.rfftfreq(N, d=dt)  # positive frequencies [Hz]

    # Temporal PSD:  Φ(f) = Gd(n0) · (f/(v·n0))^(-2) / v
    # Avoid division by zero at f=0
    Phi = np.zeros_like(freqs)
    nonzero = freqs > 0
    n = freqs[nonzero] / v       # spatial frequency [cycles/m]
    Phi[nonzero] = Gd_n0 * (n / n0) ** (-2) / v

    # Amplitude spectrum from PSD:  |X(f)| = sqrt(Phi(f) · df)
    df = fs / N
    amplitude = np.sqrt(Phi * df)

    # Random phase
    phase = rng.uniform(0, 2 * np.pi, size=len(freqs))
    phase[0] = 0  # DC component is real

    # Complex spectrum → IFFT
    spectrum = amplitude * np.exp(1j * phase)
    z0_arr = np.fft.irfft(spectrum, n=N)

    # Remove mean (ensure zero-mean road)
    z0_arr -= z0_arr.mean()

    # Velocity via central differences (2nd-order accurate)
    dz0_arr = np.gradient(z0_arr, dt)

    # Build interpolation function
    _z0_interp = interpolate.interp1d(t_arr[:len(z0_arr)], z0_arr, kind="linear",
                                       bounds_error=False, fill_value=0.0)
    _dz0_interp = interpolate.interp1d(t_arr[:len(dz0_arr)], dz0_arr, kind="linear",
                                        bounds_error=False, fill_value=0.0)

    def road_func(t: float) -> Tuple[float, float]:
        return float(_z0_interp(t)), float(_dz0_interp(t))

    # Trim arrays to same length
    n_pts = min(len(t_arr), len(z0_arr))
    return road_func, t_arr[:n_pts], z0_arr[:n_pts], dz0_arr[:n_pts]


# ---------------------------------------------------------------------------
# Factory — dispatch by road_type string
# ---------------------------------------------------------------------------

def make_road_profile(cfg: RoadConfig, sim: SimParams) -> RoadResult:
    """Create the road profile specified by ``cfg.road_type``."""
    dispatch = {
        "bump": bump_profile,
        "sinusoidal": sinusoidal_profile,
        "random": random_iso8608_profile,
    }
    key = cfg.road_type.lower()
    if key not in dispatch:
        raise ValueError(f"Unknown road type '{cfg.road_type}'. "
                         f"Choose from: {list(dispatch.keys())}")
    return dispatch[key](cfg, sim)
