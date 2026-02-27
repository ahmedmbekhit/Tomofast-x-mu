from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# --- import path helper ----------------------------------------------------

def pytest_configure():
    """Allow running tests without requiring `pip install -e .` first."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


# --- synthetic fixtures ----------------------------------------------------


@pytest.fixture
def synthetic_mat(tmp_path):
    """Create a tiny synthetic QDM-style .mat file.

    Variables:
      - Bz: Tesla
      - step: meters
      - h: meters (default fixture uses meters; tests can set h_units)
    """
    import scipy.io

    ny, nx = 6, 8
    step_m = 4.4e-6
    h_m = 5e-6

    # deterministic-ish field
    Bz_T = (np.arange(ny * nx, dtype=float).reshape(ny, nx) * 1e-12)

    mat_path = tmp_path / "synthetic.mat"
    scipy.io.savemat(
        str(mat_path),
        {
            "Bz": Bz_T,
            "step": np.array([[step_m]]),
            "h": np.array([[h_m]]),
        },
    )

    return {
        "path": mat_path,
        "ny": ny,
        "nx": nx,
        "step_um": step_m * 1e6,
        "Bz_nT": Bz_T * 1e9,
        "h_um": h_m * 1e6,
    }


@pytest.fixture
def synthetic_csv(tmp_path):
    """Create a synthetic CSV in the same format produced by load_qdm_mat().

    The geometry is chosen to match expectations in unit tests:
      - x is 0..19 (20 points)
      - y is 0..19 (20 points)
      - clipping x_start=5,x_end=15 yields nx=10
      - clipping y_start=2,y_end=10 yields ny=8

    Also includes a central positive anomaly so blob detection has something to find.
    """
    nx, ny = 20, 20
    spacing = 1.0
    x = np.arange(nx) * spacing
    y = np.arange(ny) * spacing

    z = 5.0

    # Build a 2D field with a central anomaly.
    bz2 = np.zeros((ny, nx), dtype=float)
    bz2[8:12, 9:13] = 100.0  # nT anomaly patch

    import pandas as pd

    df = pd.DataFrame(
        {
            "x (um)": np.tile(x, ny),
            "y (um)": np.repeat(y, nx),
            "z (um)": np.repeat(z, nx * ny),
            "bz (nT)": bz2.ravel(order="C"),
        }
    ).sort_values(by=["y (um)", "x (um)"])

    p = tmp_path / "synthetic.csv"
    df.to_csv(p, index=False)

    return {"path": p, "nx": nx, "ny": ny, "z": z}


@pytest.fixture
def synthetic_obs(tmp_path, synthetic_csv):
    """Create a Mag_Inversion.obs using format_tomofast_obs."""
    from tomofast_x_mu.format import format_tomofast_obs

    out = tmp_path / "Mag_Inversion.obs"
    format_tomofast_obs(str(synthetic_csv["path"]), str(out), flip_bz=False)
    return {"path": out, "nx": synthetic_csv["nx"], "ny": synthetic_csv["ny"]}
