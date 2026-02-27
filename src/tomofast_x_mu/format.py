"""Formatting utilities for Tomofast observation and meshgrid files."""

import os

import numpy as np


def format_tomofast_obs(
    csv_path,
    out_path,
    flip_bz=True,
    x_start=None,
    x_end=None,
    y_start=None,
    y_end=None,
):
    """Read CSV (x,y,z,bz columns), optionally clip, flip Bz sign, write Tomofast obs file.

    Parameters
    ----------
    csv_path : str or Path
        Path to input CSV with columns x, y, z, bz (comma-delimited, 1 header row).
        Data must be sorted with y varying slowly and x varying fast (row-major order).
    out_path : str or Path
        Path for the output Tomofast obs file.
    flip_bz : bool
        If True, flip Bz sign for Tomofast convention (Z positive down).
    x_start, x_end, y_start, y_end : int or None
        Optional cell-index clipping bounds.  Defaults cover full range.

    Returns
    -------
    dict
        Keys: x_coords, y_coords, bz (1-D arrays), nx, ny.
    """
    data_z = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    x_coords = data_z[:, 0]
    y_coords = data_z[:, 1]
    z_coords = 0 * data_z[:, 2]  # Make Z=0

    if flip_bz:
        bz = -1 * data_z[:, 3]
    else:
        bz = data_z[:, 3]

    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    nx = len(unique_x)
    ny = len(unique_y)

    # Validate data ordering before reshaping
    # Expected: y varies slowly (outer loop), x varies fast (inner loop)
    x_coords_reshaped = x_coords.reshape(ny, nx)
    if not np.allclose(x_coords_reshaped[0, :], unique_x):
        raise ValueError(
            "CSV data is not in expected row-major order (y slow, x fast). "
            "First row x-coordinates do not match unique x values."
        )

    x_coords = x_coords_reshaped
    y_coords = y_coords.reshape(ny, nx)
    z_coords = z_coords.reshape(ny, nx)
    bz = bz.reshape(ny, nx)

    # Clipping
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = nx
    if y_start is None:
        y_start = 0
    if y_end is None:
        y_end = ny

    x_coords = x_coords[y_start:y_end, x_start:x_end]
    y_coords = y_coords[y_start:y_end, x_start:x_end]
    z_coords = z_coords[y_start:y_end, x_start:x_end]
    bz = bz[y_start:y_end, x_start:x_end]

    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = z_coords.flatten()
    bz = bz.flatten()

    data_combined = np.column_stack((x_coords, y_coords, z_coords, bz))
    Ndataz = len(bz)

    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    np.savetxt(
        out_path,
        data_combined,
        delimiter=" ",
        fmt="%f %f %f %f",
        header=str(Ndataz),
        comments="",
    )

    # Recompute clipped grid dims
    nx_out = len(np.unique(x_coords))
    ny_out = len(np.unique(y_coords))

    return {
        "x_coords": x_coords,
        "y_coords": y_coords,
        "bz": bz,
        "nx": nx_out,
        "ny": ny_out,
    }


def write_obs(out_path, header, arr):
    """Write obs file updating Ndata count in header.

    Parameters
    ----------
    out_path : str or Path
        Output file path.
    header : str
        Header string; if it starts with an integer, that integer is replaced
        with the number of rows in *arr*.
    arr : numpy.ndarray
        Data array to write (one row per observation).
    """
    parts = header.split()
    try:
        int(parts[0])
        parts[0] = str(arr.shape[0])
        header = " ".join(parts)
    except (ValueError, IndexError):
        pass

    with open(out_path, "w") as f:
        f.write(header + "\n")
        np.savetxt(f, arr, fmt="%.6f")


def write_meshgrid_tomofast(filename, line_data):
    """Write Tomofast meshgrid file (3-component model format).

    Parameters
    ----------
    filename : str or Path
        Output file path.
    line_data : numpy.ndarray
        Array with columns: X1 X2 Y1 Y2 Z1 Z2 V1 V2 V3 I J K.
    """
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"{line_data.shape[0]}\n")
        np.savetxt(f, line_data, fmt="%f %f %f %f %f %f %f %f %f %d %d %d")
