"""I/O utilities for loading QDM microscopy data and Tomofast mesh/field files."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import xarray as xr

# Unit conversion constants
METER_TO_MICROMETER = 1e6
TESLA_TO_NANOTESLA = 1e9


def load_qdm_mat(path, save_csv=True, csv_path=None, *, h_units: str = "m"):
    """Load QDM microscopy data from a MATLAB .mat file and return as xarray Dataset.

    Parameters
    ----------
    path : str or Path
        Path to the .mat file.
    save_csv : bool, optional
        If True, save the output as a CSV file (default True).
    csv_path : str or Path, optional
        Path to save the CSV file.  If None, saves as ``<mat_file_stem>.csv``
        in the same folder.
    h_units : {'m','um','auto'}, optional
        Units of the scalar `h` (sensor-sample distance) stored in the .mat file.

        - 'm': interpret `h` as meters and convert to µm.
        - 'um': interpret `h` as already in µm.
        - 'auto': infer based on typical QDM conventions (often: step in meters, h in µm).

        Default is 'm' for backwards compatibility.

    Returns
    -------
    xarray.Dataset
        Dataset with bz(y, x) vertical magnetic field [nT], coords in [um].
    """
    path = Path(path)
    contents = scipy.io.loadmat(path)

    # Pixel spacing (m -> um)
    spacing = contents["step"].ravel()[0] * METER_TO_MICROMETER

    # Vertical magnetic field (T -> nT)
    bz = contents["Bz"] * TESLA_TO_NANOTESLA

    # Sensor-sample distance (h -> um)
    # NOTE: scipy.io.loadmat typically returns a (1, 1) ndarray for scalar values.
    # Converting an ndarray directly to float is deprecated in NumPy.
    h_raw = np.asarray(contents["h"]).squeeze().item()

    h_units_norm = str(h_units).strip().lower()

    if h_units_norm in {"auto", "infer"}:
        # Heuristic: for QDM standoff, typical values are O(1–100) µm.
        # If step is in meters (<< 1) and h is O(1+) it is almost certainly already µm.
        # Conversely, if h is tiny (e.g. 5e-6) it is likely meters.
        step_m = float(contents["step"].ravel()[0])
        h_val = float(h_raw)
        if step_m < 1e-3 and h_val >= 1e-3:
            h_units_norm = "um"
        else:
            h_units_norm = "m"

    if h_units_norm in {"m", "meter", "metre", "meters", "metres"}:
        sensor_sample_distance = float(h_raw) * METER_TO_MICROMETER
    elif h_units_norm in {"um", "µm", "micrometer", "micrometre", "micrometers", "micrometres"}:
        sensor_sample_distance = float(h_raw)
    else:
        raise ValueError(f"Unsupported h_units={h_units!r} (expected 'm', 'um', or 'auto')")

    # Coordinates
    x = np.arange(bz.shape[1]) * spacing
    y = np.arange(bz.shape[0])[::-1] * spacing  # flip to cartesian
    z = np.full_like(bz, sensor_sample_distance)

    data = xr.Dataset(
        data_vars={"bz": (("y", "x"), bz)},
        coords={
            "x": ("x", x, {"units": "um"}),
            "y": ("y", y, {"units": "um"}),
            "z": (("y", "x"), z, {"long_name": "sensor sample distance", "units": "um"}),
        },
        attrs={"file_name": str(path)},
    )
    data.bz.attrs = {"long_name": "vertical magnetic field", "units": "nT"}

    if save_csv:
        if csv_path is None:
            csv_path = path.with_suffix(".csv")
        df = pd.DataFrame(
            {
                "x (um)": np.tile(x, len(y)),
                "y (um)": np.repeat(y, len(x)),
                "z (um)": np.repeat(sensor_sample_distance, len(x) * len(y)),
                "bz (nT)": bz.ravel(order="C"),
            }
        )
        df = df.sort_values(by=["y (um)", "x (um)"], ascending=[True, True])
        df.to_csv(csv_path, index=False)

    return data


def load_field_table(path, names=("x", "y", "z", "value")):
    """Read whitespace-delimited field data into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the text file.
    names : tuple of str
        Column names.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(path, sep=r"\s+", names=list(names), engine="python")


def read_mesh_bounds(mesh_path):
    """Read a Tomofast mesh file (skip 1 header line).

    Returns a dict with x_centers, y_centers, z_centers, raw mesh array,
    nx, ny, nz inferred from unique centers.

    Parameters
    ----------
    mesh_path : str or Path
        Path to the meshgrid text file.

    Returns
    -------
    dict or None
        Keys: x_centers, y_centers, z_centers, mesh, nx, ny, nz.
        Returns None if file has fewer than 2 lines.
    """
    mesh = np.genfromtxt(mesh_path, skip_header=1)
    if mesh.ndim < 2 or mesh.shape[0] == 0:
        return None

    x_centers = (mesh[:, 0] + mesh[:, 1]) / 2
    y_centers = (mesh[:, 2] + mesh[:, 3]) / 2
    z_centers = (mesh[:, 4] + mesh[:, 5]) / 2

    nx = len(np.unique(x_centers))
    ny = len(np.unique(y_centers))

    # Validate that mesh dimensions are consistent
    expected_cells = nx * ny
    if mesh.shape[0] % expected_cells != 0:
        raise ValueError(
            f"Mesh has {mesh.shape[0]} cells but nx={nx} * ny={ny} = {expected_cells}. "
            "Cell count is not evenly divisible."
        )

    nz = mesh.shape[0] // expected_cells

    return {
        "x_centers": x_centers,
        "y_centers": y_centers,
        "z_centers": z_centers,
        "mesh": mesh,
        "nx": nx,
        "ny": ny,
        "nz": nz,
    }
