"""Anomaly detection and grain extraction from Tomofast inversion results."""

import os

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, generate_binary_structure, label

from tomofast_x_mu.format import write_obs


def detect_anomalies(
    obs_path,
    threshold_factor=2.75,
    pad_x=40,
    pad_y=10,
    min_blob_cells=150,
    max_blob_cells=100000,
    nz_fixed=120,
    out_obs_dir=None,
    windows_path=None,
    summary_path=None,
):
    """Detect anomalous blobs in observation data.

    Reads obs, reshapes to grid, thresholds, labels blobs via scipy.ndimage,
    clips rectangles per blob.  Optionally writes blob obs files,
    blob_windows.txt, and blob_window_summary.txt.

    Parameters
    ----------
    obs_path : str or Path
        Path to the Tomofast observation file (header = Ndata, then X Y Z Bz rows).
    threshold_factor : float
        Multiplier of standard deviation for anomaly threshold.
    pad_x, pad_y : int
        Padding cells to extend detected blob rectangles.
    min_blob_cells, max_blob_cells : int
        Size filters for blobs.
    nz_fixed : int
        Fixed nz for Tomofast mesh (same for all blobs).
    out_obs_dir : str or None
        Directory for per-blob obs files.  If None, no obs files are written.
    windows_path : str or None
        Path to write blob_windows.txt.  If None, not written.
    summary_path : str or None
        Path to write blob_window_summary.txt.  If None, not written.

    Returns
    -------
    list of dict
        One dict per kept blob with keys: blob_id, x_min, x_max, y_min, y_max,
        nx_win, ny_win, nz_win, ndata, obs_out.
    """
    with open(obs_path, "r") as f:
        header_line = f.readline().strip()

    data = np.loadtxt(obs_path, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    bz = data[:, 3]

    nx = len(np.unique(x))
    ny = len(np.unique(y))

    # Validate data ordering before reshaping
    x_grid = x.reshape(ny, nx)
    if not np.all(np.diff(x_grid[0, :]) > 0):
        raise ValueError(
            "Observation data is not in expected row-major order (y slow, x fast). "
            "First row x-coordinates are not monotonically increasing."
        )

    y_grid = y.reshape(ny, nx)
    bz_grid = bz.reshape(ny, nx)

    # Detect blobs
    thr = threshold_factor * np.std(bz_grid)
    mask = np.abs(bz_grid) > thr

    structure = np.ones((3, 3))
    mask_dilated = binary_dilation(mask, structure=structure, iterations=1)

    labels_arr, n_blobs = ndi.label(mask_dilated)
    # Count cells in dilated mask (not original mask) for consistent filtering
    sizes = ndi.sum(mask_dilated, labels_arr, index=np.arange(1, n_blobs + 1))
    slices = ndi.find_objects(labels_arr)

    # Prepare output dirs/files
    if out_obs_dir is not None:
        os.makedirs(out_obs_dir, exist_ok=True)

    if windows_path is not None:
        with open(windows_path, "w") as fw:
            fw.write("# blob_id x_min x_max y_min y_max\n")

    if summary_path is not None:
        with open(summary_path, "w") as fs:
            fs.write("# blob_id nx ny nz ndata\n")

    results = []

    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        if sizes[i - 1] < min_blob_cells:
            continue
        if sizes[i - 1] > max_blob_cells:
            continue

        ys, xs = sl
        r0, r1 = ys.start, ys.stop - 1
        c0, c1 = xs.start, xs.stop - 1

        r0e = max(0, r0 - pad_y)
        r1e = min(ny - 1, r1 + pad_y)
        c0e = max(0, c0 - pad_x)
        c1e = min(nx - 1, c1 + pad_x)

        rect_mask = np.zeros((ny, nx), dtype=bool)
        rect_mask[r0e : r1e + 1, c0e : c1e + 1] = True

        blob_data = data[rect_mask.ravel(), :]

        x_min = x_grid[r0e, c0e]
        x_max = x_grid[r1e, c1e]
        y_min = y_grid[r0e, c0e]
        y_max = y_grid[r1e, c1e]

        nx_win = c1e - c0e + 1
        ny_win = r1e - r0e + 1
        nz_win = nz_fixed
        ndata = blob_data.shape[0]

        obs_out = None
        if out_obs_dir is not None:
            obs_out = os.path.join(out_obs_dir, f"Blob_{i:03d}_RECT.obs")
            write_obs(obs_out, header_line, blob_data)

        if windows_path is not None:
            with open(windows_path, "a") as fw:
                fw.write(f"{i} {x_min:.6f} {x_max:.6f} {y_min:.6f} {y_max:.6f}\n")

        if summary_path is not None:
            with open(summary_path, "a") as fs:
                fs.write(f"{i} {nx_win} {ny_win} {nz_win} {ndata}\n")

        results.append(
            {
                "blob_id": i,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "nx_win": nx_win,
                "ny_win": ny_win,
                "nz_win": nz_win,
                "ndata": ndata,
                "obs_out": obs_out,
            }
        )

    return results


def extract_grains(
    blob_root,
    mesh_root,
    threshold_list=None,
    structure=None,
):
    """Loop blobs, load model + mesh, threshold + 3D labeling, save grain CSVs.

    Parameters
    ----------
    blob_root : str
        Root folder containing Blob_XXX subfolders (Tomofast output structure).
    mesh_root : str
        Folder containing Meshgrid_Blob_XXX.txt files.
    threshold_list : array-like or None
        Fractional thresholds (0-1) of max magnetization amplitude.
        Defaults to np.arange(0.1, 1.0, 0.1).
    structure : numpy.ndarray or None
        Structuring element for 3D connectivity labeling.
        Defaults to 26-connectivity.

    Returns
    -------
    dict
        Mapping ``(blob_name, threshold)`` -> path of saved grain CSV.
    """
    if threshold_list is None:
        threshold_list = np.arange(0.1, 1.0, 0.1)
    if structure is None:
        structure = generate_binary_structure(3, 3)

    blob_dirs = sorted(
        d
        for d in os.listdir(blob_root)
        if d.startswith("Blob_") and os.path.isdir(os.path.join(blob_root, d))
    )

    results = {}

    for blob in blob_dirs:
        model_file = os.path.join(blob_root, blob, blob, "model", "mag_final_model_full.txt")
        blob_num = blob.split("_")[1]
        mesh_file = os.path.join(mesh_root, f"Meshgrid_Blob_{blob_num}.txt")

        if not os.path.exists(model_file) or not os.path.exists(mesh_file):
            continue

        data = np.genfromtxt(model_file, skip_header=1)
        Mx = data[:, 0]
        My = data[:, 1]
        Mz = data[:, 2]
        mag = np.sqrt(Mx**2 + My**2 + Mz**2)

        mesh = np.genfromtxt(mesh_file, skip_header=1)
        x_centers = (mesh[:, 0] + mesh[:, 1]) / 2
        y_centers = (mesh[:, 2] + mesh[:, 3]) / 2
        z_centers = (mesh[:, 4] + mesh[:, 5]) / 2

        nx = len(np.unique(x_centers))
        ny = len(np.unique(y_centers))

        # Validate mesh dimensions before reshaping
        expected_cells = nx * ny
        if mesh.shape[0] % expected_cells != 0:
            continue

        nz = mesh.shape[0] // expected_cells

        if nx * ny * nz != mesh.shape[0]:
            continue

        for threshold_per in threshold_list:
            threshold = threshold_per * mag.max()
            mask = mag > threshold
            # Reshape with order='C' (row-major, Z varies slowest)
            mask_3d = mask.reshape((nz, ny, nx), order='C')

            labeled_grains_3d, num_grains = label(mask_3d, structure=structure)

            df = pd.DataFrame(
                {
                    "X": x_centers.ravel(),
                    "Y": y_centers.ravel(),
                    "Z": -1 * z_centers.ravel(),
                    "GrainID": labeled_grains_3d.ravel(),
                    "Mx": Mx,
                    "My": My,
                    "Mz": Mz,
                }
            )
            df = df[df["GrainID"] != 0]

            out_folder = os.path.join(blob_root, blob, blob, "Grains_detection")
            os.makedirs(out_folder, exist_ok=True)

            csv_file = os.path.join(out_folder, f"grain_labels_{threshold_per:.1f}.csv")
            df.to_csv(csv_file, index=False)

            results[(blob, float(threshold_per))] = csv_file

    return results


def extract_max_cells(
    blob_root,
    mesh_root,
    threshold_list=None,
    target_grain_ids=None,
    tol=1e-6,
):
    """Find max-amplitude mesh row per grain per threshold.

    Parameters
    ----------
    blob_root : str
        Root folder containing Blob_XXX subfolders.
    mesh_root : str
        Folder containing Meshgrid_Blob_XXX.txt files.
    threshold_list : array-like or None
        Fractional thresholds.  Defaults to np.arange(0.1, 1.0, 0.1).
    target_grain_ids : list of int or None
        If set, only process these grain IDs.
    tol : float
        Tolerance for coordinate matching.

    Returns
    -------
    dict
        Mapping ``(blob_name, threshold)`` -> path of saved txt file.
    """
    if threshold_list is None:
        threshold_list = np.arange(0.1, 1.0, 0.1)

    blob_dirs = sorted(
        d
        for d in os.listdir(blob_root)
        if d.startswith("Blob_") and os.path.isdir(os.path.join(blob_root, d))
    )

    output_paths = {}

    for blob in blob_dirs:
        blob_num = blob.split("_")[1]
        grains_folder = os.path.join(blob_root, blob, blob, "Grains_detection")
        mesh_path = os.path.join(mesh_root, f"Meshgrid_Blob_{blob_num}.txt")

        if not os.path.isdir(grains_folder) or not os.path.exists(mesh_path):
            continue

        mesh = np.genfromtxt(mesh_path, skip_header=1)
        x_c = (mesh[:, 0] + mesh[:, 1]) / 2
        y_c = (mesh[:, 2] + mesh[:, 3]) / 2
        z_c = (mesh[:, 4] + mesh[:, 5]) / 2

        meshs_folder = os.path.join(blob_root, blob, blob, "Maxs_meshs")
        os.makedirs(meshs_folder, exist_ok=True)

        for threshold_per in threshold_list:
            csv_path = os.path.join(
                grains_folder, f"grain_labels_largest_{threshold_per:.1f}.csv"
            )
            if not os.path.isfile(csv_path):
                # Fall back to non-filtered file
                csv_path = os.path.join(
                    grains_folder, f"grain_labels_{threshold_per:.1f}.csv"
                )
            if not os.path.isfile(csv_path):
                continue

            df = pd.read_csv(csv_path)

            if target_grain_ids is not None:
                df = df[df["GrainID"].isin(target_grain_ids)]

            if df.empty:
                continue

            rows_out = []

            for grain_id, group in df.groupby("GrainID"):
                mags = np.sqrt(
                    group["Mx"].values ** 2
                    + group["My"].values ** 2
                    + group["Mz"].values ** 2
                )

                i_max = np.argmax(mags)
                row = group.iloc[i_max]
                X0, Y0, Z0 = row["X"], row["Y"], row["Z"]

                # Z0 is stored as -z_centers in grain CSV, so match with z_c directly
                idx_mesh_arr = np.where(
                    (np.abs(x_c - X0) < tol)
                    & (np.abs(y_c - Y0) < tol)
                    & (np.abs(z_c + Z0) < tol)  # Z0 = -z_c, so z_c = -Z0
                )[0]

                if len(idx_mesh_arr) == 0:
                    continue

                idx_mesh = idx_mesh_arr[0]
                rows_out.append(np.hstack(([grain_id], mesh[idx_mesh, :])))

            if len(rows_out) > 0:
                rows_out = np.array(rows_out)
                out_txt = os.path.join(
                    meshs_folder, f"grain_max_meshrow_{threshold_per:.1f}.txt"
                )

                with open(out_txt, "w") as f:
                    f.write(f"{rows_out.shape[0]}\n")
                    for row in rows_out:
                        f.write(
                            f"{int(row[0])} "
                            + " ".join([f"{v:.6f}" for v in row[1:]])
                            + "\n"
                        )

                output_paths[(blob, float(threshold_per))] = out_txt

    return output_paths
