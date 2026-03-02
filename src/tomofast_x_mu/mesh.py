"""Mesh construction utilities for Tomofast inversions."""

import os

import numpy as np

from tomofast_x_mu.format import write_meshgrid_tomofast


def make_edges(core_min, d_core, n_core, npad, r):
    """Build edges with uniform core + geometric padding.

    Parameters
    ----------
    core_min : float
        Starting coordinate of the core region.
    d_core : float
        Core cell width.
    n_core : int
        Number of core cells.
    npad : int
        Number of padding cells on each side.
    r : float
        Geometric growth factor for padding.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        (edges, widths) arrays.
    """
    core_widths = np.full(n_core, d_core, dtype=float)

    left_widths = d_core * (r ** np.arange(npad - 1, -1, -1))
    right_widths = d_core * (r ** np.arange(0, npad))

    widths = np.concatenate([left_widths, core_widths, right_widths])

    start_edge = core_min - left_widths.sum()
    edges = start_edge + np.concatenate(([0.0], np.cumsum(widths)))
    return edges, widths


def _build_mesh_for_window(
    x_min,
    x_max,
    y_min,
    y_max,
    out_file,
    dx_core=4.4,
    dy_core=4.4,
    npad_x=20,
    npad_y=20,
    rx=1.15,
    ry=1.15,
    z_min=0.0,
    nz=120,
    dz0=4.4,
    rz=1.0,
    expand_half_cell=True,
):
    """Build and write a non-uniform mesh for one blob window.

    Returns a dict with mesh metadata.
    """
    if expand_half_cell:
        x_core_min = x_min - 0.5 * dx_core
        x_core_max = x_max + 0.5 * dx_core
        y_core_min = y_min - 0.5 * dy_core
        y_core_max = y_max + 0.5 * dy_core
    else:
        x_core_min, x_core_max = x_min, x_max
        y_core_min, y_core_max = y_min, y_max

    nx_core = int(np.round((x_core_max - x_core_min) / dx_core))
    ny_core = int(np.round((y_core_max - y_core_min) / dy_core))

    if nx_core < 1 or ny_core < 1:
        raise ValueError(
            f"Window too small for given dx/dy. "
            f"Got nx_core={nx_core}, ny_core={ny_core}."
        )

    x_core_max = x_core_min + nx_core * dx_core
    y_core_max = y_core_min + ny_core * dy_core

    x_edges, x_widths = make_edges(x_core_min, dx_core, nx_core, npad_x, rx)
    y_edges, y_widths = make_edges(y_core_min, dy_core, ny_core, npad_y, ry)

    nx_cells = nx_core + 2 * npad_x
    ny_cells = ny_core + 2 * npad_y
    nz_cells = nz

    dz_list = dz0 * (rz ** np.arange(nz_cells))
    z_edges = z_min + np.concatenate(([0.0], np.cumsum(dz_list)))

    X1, Y1, Z1 = np.meshgrid(x_edges[:-1], y_edges[:-1], z_edges[:-1], indexing="ij")
    X2, Y2, Z2 = np.meshgrid(x_edges[1:], y_edges[1:], z_edges[1:], indexing="ij")

    I, J, K = np.meshgrid(
        np.arange(1, nx_cells + 1),
        np.arange(1, ny_cells + 1),
        np.arange(1, nz_cells + 1),
        indexing="ij",
    )

    x1 = X1.flatten(order="F")
    x2 = X2.flatten(order="F")
    y1 = Y1.flatten(order="F")
    y2 = Y2.flatten(order="F")
    z1 = Z1.flatten(order="F")
    z2 = Z2.flatten(order="F")
    ii = I.flatten(order="F")
    jj = J.flatten(order="F")
    kk = K.flatten(order="F")

    N = x1.size
    zeros = np.zeros(N, dtype=float)

    line_data = np.column_stack([x1, x2, y1, y2, z1, z2, zeros, zeros, zeros, ii, jj, kk])

    write_meshgrid_tomofast(out_file, line_data)

    info = {
        "nx_core": nx_core,
        "ny_core": ny_core,
        "nz": nz_cells,
        "nx": nx_cells,
        "ny": ny_cells,
        "x_min": x_edges[0],
        "x_max": x_edges[-1],
        "y_min": y_edges[0],
        "y_max": y_edges[-1],
        "z_min": z_edges[0],
        "z_max": z_edges[-1],
        "dz_top": dz_list[0],
        "dz_bottom": dz_list[-1],
        "dx_min": x_widths.min(),
        "dx_max": x_widths.max(),
        "dy_min": y_widths.min(),
        "dy_max": y_widths.max(),
    }
    return info


def build_mesh_per_blob(
    windows_path,
    out_dir,
    summary_path,
    dx_core=4.4,
    dy_core=4.4,
    npad_x=20,
    npad_y=20,
    rx=1.15,
    ry=1.15,
    z_min=0.0,
    nz=120,
    dz0=4.4,
    rz=1.0,
    expand_half_cell=True,
):
    """Read blob_windows.txt and build a mesh per blob.

    Parameters
    ----------
    windows_path : str
        Path to blob_windows.txt (columns: blob_id x_min x_max y_min y_max).
    out_dir : str
        Output directory for meshgrid files.
    summary_path : str
        Path to write updated blob_window_summary.txt.
    dx_core, dy_core : float
        Core cell sizes in X/Y (um).
    npad_x, npad_y : int
        Padding cells on each side.
    rx, ry : float
        Padding growth factors.
    z_min : float
        Top of model.
    nz : int
        Number of Z cells.
    dz0 : float
        Top layer thickness.
    rz : float
        Z growth factor.
    expand_half_cell : bool
        Expand each window by half a core cell.

    Returns
    -------
    list of dict
        One info dict per blob.
    """
    os.makedirs(out_dir, exist_ok=True)

    windows = np.loadtxt(windows_path, comments="#", ndmin=2)

    with open(summary_path, "w") as fs:
        fs.write("# blob_id nx ny nz ndata\n")

    results = []

    for row in windows:
        blob_id = int(row[0])
        x_min_w, x_max_w, y_min_w, y_max_w = row[1], row[2], row[3], row[4]

        out_file = os.path.join(out_dir, f"Meshgrid_Blob_{blob_id:03d}.txt")
        info = _build_mesh_for_window(
            x_min_w,
            x_max_w,
            y_min_w,
            y_max_w,
            out_file,
            dx_core=dx_core,
            dy_core=dy_core,
            npad_x=npad_x,
            npad_y=npad_y,
            rx=rx,
            ry=ry,
            z_min=z_min,
            nz=nz,
            dz0=dz0,
            rz=rz,
            expand_half_cell=expand_half_cell,
        )

        # ndata: use 6th column if present, else fallback
        if row.size >= 6:
            ndata = int(row[5])
        else:
            ndata = int(info["nx_core"] * info["ny_core"])

        with open(summary_path, "a") as fs:
            fs.write(
                f"{blob_id:d} "
                f"{info['nx']:d} {info['ny']:d} {info['nz']:d} "
                f"{ndata:d}\n"
            )

        info["blob_id"] = blob_id
        info["ndata"] = ndata
        info["out_file"] = out_file
        results.append(info)

    return results


def create_single_cell_meshes(
    blob_root,
    thr=0.9,
    rx=(-5, 5, 1),
    ry=(-5, 5, 1),
    rz=(-5, 5, 1),
    outer_root=None,
):
    """Create single-cell meshes around max grain centers for source detection.

    Parameters
    ----------
    blob_root : str
        Root folder containing Blob_XXX subfolders.
    thr : float
        Threshold value used in grain_max_meshrow filename.
    rx, ry, rz : tuple of (int, int, int)
        (start, stop, step) for search offsets in each axis.
    outer_root : str or None
        Output root directory.  If None, raises ValueError.

    Returns
    -------
    list of dict
        One dict per blob with keys: blob, grain_id, count, folder.
    """
    if outer_root is None:
        raise ValueError("outer_root must be specified")

    os.makedirs(outer_root, exist_ok=True)

    in_name = f"grain_max_meshrow_{thr:.1f}.txt"

    blob_dirs = sorted(
        d
        for d in os.listdir(blob_root)
        if d.startswith("Blob_") and os.path.isdir(os.path.join(blob_root, d))
    )

    results = []

    for blob in blob_dirs:
        input_path = os.path.join(blob_root, blob, blob, "Maxs_meshs", in_name)
        if not os.path.exists(input_path):
            continue

        blob_out = os.path.join(outer_root, blob)
        os.makedirs(blob_out, exist_ok=True)

        # Read grain rows
        grains = []
        with open(input_path, "r") as f:
            first = f.readline().strip()
            try:
                _ = int(first)
            except ValueError:
                f.seek(0)

            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split()
                if len(cols) < 7:
                    continue

                grain_id = int(float(cols[0]))
                xmin = float(cols[1])
                xmax = float(cols[2])
                ymin = float(cols[3])
                ymax = float(cols[4])
                zmin = float(cols[5])
                zmax = float(cols[6])
                grains.append((grain_id, xmin, xmax, ymin, ymax, zmin, zmax))

        for grain_id, xmin, xmax, ymin, ymax, zmin, zmax in grains:
            dx = xmax - xmin
            dy = ymax - ymin
            dz = zmax - zmin

            cx0 = 0.5 * (xmin + xmax)
            cy0 = 0.5 * (ymin + ymax)
            cz0 = 0.5 * (zmin + zmax)

            grain_folder = os.path.join(blob_out, f"grain_{grain_id}")
            os.makedirs(grain_folder, exist_ok=True)

            count = 0
            for ix in range(rx[0], rx[1], rx[2]):
                for iy in range(ry[0], ry[1], ry[2]):
                    for iz in range(rz[0], rz[1], rz[2]):
                        cx = cx0 + ix * dx
                        cy = cy0 + iy * dy
                        cz = cz0 + iz * dz

                        xmin_new = cx - dx / 2
                        xmax_new = cx + dx / 2
                        ymin_new = cy - dy / 2
                        ymax_new = cy + dy / 2
                        zmin_new = cz - dz / 2
                        zmax_new = cz + dz / 2

                        count += 1
                        out_name = f"meshgrid_{count:03d}.txt"
                        out_path = os.path.join(grain_folder, out_name)

                        with open(out_path, "w") as fout:
                            fout.write("1\n")
                            fout.write(
                                f"{xmin_new:.6f} {xmax_new:.6f} "
                                f"{ymin_new:.6f} {ymax_new:.6f} "
                                f"{zmin_new:.6f} {zmax_new:.6f} "
                                f"0.000000 0.000000 0.000000 1 1 1\n"
                            )

            results.append(
                {
                    "blob": blob,
                    "grain_id": grain_id,
                    "count": count,
                    "folder": grain_folder,
                }
            )

    return results
