"""Analysis utilities for grain statistics and vector direction calculations."""

import os
import re

import numpy as np
import pandas as pd


def _compute_dec_inc(Mx, My, Mz, tol=1e-15):
    """Compute declination and inclination from magnetization components.

    Uses the notebook's sign convention: Mz is flipped, dec = 90 - phi.

    Parameters
    ----------
    Mx, My, Mz : float
        Magnetization components.
    tol : float
        Tolerance for zero magnitude check.

    Returns
    -------
    dec, inc : float
        Declination and inclination in degrees.
    """
    m = np.sqrt(Mx**2 + My**2 + Mz**2)
    if m < tol:
        return np.nan, np.nan

    # Clamp arccos argument to [-1, 1] to avoid numerical issues
    cos_arg = np.clip(Mz / m, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_arg))
    phi = np.degrees(np.arctan2(My, Mx))

    inc = theta - 90
    dec = 90 - phi

    # Normalize to standard ranges: inc in [-90, 90], dec in [0, 360)
    # Guard against infinite loops from non-finite values
    if not (np.isfinite(inc) and np.isfinite(dec)):
        return np.nan, np.nan

    change = True
    max_iterations = 10  # Safety limit
    iterations = 0
    while change and iterations < max_iterations:
        change = False
        iterations += 1
        if inc > 90:
            inc = 180 - inc
            dec = dec + 180
            change = True
        if inc < -90:
            inc = -180 - inc
            dec = dec + 180
            change = True
        if dec < 0:
            dec = dec + 360
            change = True
        if dec >= 360:
            dec = dec - 360
            change = True

    return dec, inc


def grain_statistics(blob_root, dx=4.4, dy=4.4, dz=4.4, threshold_list=None):
    """Compute per-grain mean M, dec/inc, dipole moment, weighted position.

    Parameters
    ----------
    blob_root : str
        Root folder containing Blob_XXX subfolders.
    dx, dy, dz : float
        Cell sizes (um) for dipole moment calculation.
    threshold_list : array-like or None
        Thresholds to process.  Defaults to np.arange(0.1, 1.0, 0.1).

    Returns
    -------
    pandas.DataFrame
        One row per (blob, threshold, grain) with columns: Blob, Threshold,
        GrainID, Num_Voxels, Mx_mean, My_mean, Mz_mean, Mean_Magnitude,
        Mean_Moment_Amplitude, Resultant_R, Angular_STD_deg, Dec, Inc,
        Dipole_X, Dipole_Y, Dipole_Z.
    """
    if threshold_list is None:
        threshold_list = np.arange(0.1, 1.0, 0.1)

    V = dx * dy * dz

    blob_dirs = sorted(
        d
        for d in os.listdir(blob_root)
        if d.startswith("Blob_") and os.path.isdir(os.path.join(blob_root, d))
    )

    all_results = []

    for blob in blob_dirs:
        grains_folder = os.path.join(blob_root, blob, blob, "Grains_detection")
        if not os.path.isdir(grains_folder):
            continue

        for threshold_per in threshold_list:
            csv_path = os.path.join(grains_folder, f"grain_labels_{threshold_per:.1f}.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            for grain_id, group in df.groupby("GrainID"):
                vectors = group[["Mx", "My", "Mz"]].to_numpy()
                n_vectors = len(vectors)

                mean_Mx, mean_My, mean_Mz = vectors.mean(axis=0)
                mean_vec = np.array([mean_Mx, mean_My, mean_Mz])

                mags = np.linalg.norm(vectors, axis=1)
                mean_magnitude = mags.mean()

                # Compute Fisher R statistic using unit vectors
                unit_vectors = vectors / mags[:, np.newaxis]
                mean_unit_vec = unit_vectors.mean(axis=0)
                R = np.linalg.norm(mean_unit_vec)

                if (not np.isnan(R)) and (R > 0):
                    sigma_rad = np.sqrt(-2 * np.log(R))
                    sigma_deg = np.degrees(sigma_rad)
                else:
                    sigma_deg = np.nan

                # dec/inc from mean vector using consistent formula
                dec, inc = _compute_dec_inc(mean_Mx, mean_My, mean_Mz)

                moment_amp = mean_magnitude * V

                # moment-weighted dipole position
                Mi = mags
                Xi = group["X"].to_numpy()
                Yi = group["Y"].to_numpy()
                Zi = group["Z"].to_numpy()

                weight = Mi * V
                w_sum = weight.sum()

                if w_sum > 0:
                    r_dip_x = (weight * Xi).sum() / w_sum
                    r_dip_y = (weight * Yi).sum() / w_sum
                    r_dip_z = (weight * Zi).sum() / w_sum
                else:
                    r_dip_x = r_dip_y = r_dip_z = np.nan

                all_results.append(
                    {
                        "Blob": blob,
                        "Threshold": float(threshold_per),
                        "GrainID": int(grain_id),
                        "Num_Voxels": int(n_vectors),
                        "Mx_mean": mean_Mx,
                        "My_mean": mean_My,
                        "Mz_mean": mean_Mz,
                        "Mean_Magnitude": mean_magnitude,
                        "Mean_Moment_Amplitude": moment_amp,
                        "Resultant_R": R,
                        "Angular_STD_deg": sigma_deg,
                        "Dec": dec,
                        "Inc": inc,
                        "Dipole_X": r_dip_x,
                        "Dipole_Y": r_dip_y,
                        "Dipole_Z": r_dip_z,
                    }
                )

    return pd.DataFrame(all_results)


def vector_analysis(root_folder, grain_folders=None, dx=4.4e-6, dy=4.4e-6, dz=4.4e-6):
    """Load final model and compute per-grain dec/inc/moment.

    Parameters
    ----------
    root_folder : str
        Root folder of Tomofast outputs.
    grain_folders : list of str or None
        Subfolder names under root_folder.  Defaults to ["combined_grains"].
    dx, dy, dz : float
        Cell sizes in metres for moment calculation.

    Returns
    -------
    pandas.DataFrame
        Columns: folder, grain_idx, Mx, My, Mz_raw, Mz_flipped, dec, inc,
        M_amp, moment.
    """
    if grain_folders is None:
        grain_folders = ["combined_grains"]

    V = dx * dy * dz
    results = []

    for folder in grain_folders:
        model_path = os.path.join(root_folder, folder, "model", "mag_final_model_full.txt")
        if not os.path.exists(model_path):
            continue

        with open(model_path, "r") as f:
            lines = f.readlines()

        Ngrains = int(lines[0].strip())
        data = np.loadtxt(lines[1:])
        data = np.atleast_2d(data)

        for idx, row in enumerate(data, start=1):
            Mx, My, Mz_raw = row[0], row[1], row[2]
            Mz = -Mz_raw  # sign flip
            dec, inc = _compute_dec_inc(Mx, My, Mz)

            M_amp = np.sqrt(Mx**2 + My**2 + Mz**2)
            moment = M_amp * V

            results.append(
                {
                    "folder": folder,
                    "grain_idx": idx,
                    "Mx": Mx,
                    "My": My,
                    "Mz_raw": Mz_raw,
                    "Mz_flipped": Mz,
                    "dec": dec,
                    "inc": inc,
                    "M_amp": M_amp,
                    "moment": moment,
                }
            )

    return pd.DataFrame(results)


def combine_best_locations(summary_path, out_path):
    """Pick lowest data-cost location per blob and write combined mesh.

    Parameters
    ----------
    summary_path : str
        Path to all_final_data_costs_ALL_BLOBS.txt.
    out_path : str
        Path to write combined_best_blobs_mesh.txt.

    Returns
    -------
    pandas.DataFrame
        Best row per blob with assigned i, j, k indices.
    """
    colnames = [
        "blob",
        "grain",
        "meshfile",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
        "final_data_cost",
    ]
    df = pd.read_csv(summary_path, sep=r"\s+", names=colnames, skiprows=1, engine="python")

    best = df.loc[df.groupby("blob")["final_data_cost"].idxmin()].reset_index(drop=True)
    n_blobs = len(best)

    best["i"] = best.index + 1
    best["j"] = 1
    best["k"] = 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"{n_blobs}\n")
        for _, row in best.iterrows():
            xmin, xmax = row["xmin"], row["xmax"]
            ymin, ymax = row["ymin"], row["ymax"]
            zmin, zmax = row["zmin"], row["zmax"]
            i = int(row["i"])
            f.write(f"{xmin} {xmax} {ymin} {ymax} {zmin} {zmax} 0 0 0 {i} 1 1\n")

    return best


def extract_blob_data_costs(blob_output_dir):
    """Parse final data cost from Tomofast log files for each blob.

    Searches for ``Blob_XXX/QDM_log_Blob_XXX.txt`` files and extracts the last
    occurrence of a "data cost (new)" value from each log (the final iteration).

    Parameters
    ----------
    blob_output_dir : str
        Directory containing Blob_XXX subdirectories.

    Returns
    -------
    pandas.DataFrame
        Columns: ``blob`` (str), ``final_data_cost`` (float).
        Empty DataFrame if no logs found.
    """
    pattern = re.compile(r"[-+]?\d+\.\d+E[-+]?\d+", re.IGNORECASE)
    rows = []

    if not os.path.isdir(blob_output_dir):
        return pd.DataFrame(columns=["blob", "final_data_cost"])

    for entry in sorted(os.listdir(blob_output_dir)):
        blob_dir = os.path.join(blob_output_dir, entry)
        if not (os.path.isdir(blob_dir) and entry.startswith("Blob_")):
            continue

        log_path = os.path.join(blob_dir, f"QDM_log_{entry}.txt")
        if not os.path.exists(log_path):
            continue

        last_cost = None
        with open(log_path, "r") as f:
            for line in f:
                if "data cost (new)" in line:
                    match = pattern.search(line)
                    if match:
                        try:
                            last_cost = float(match.group())
                        except ValueError:
                            pass

        if last_cost is not None:
            rows.append({"blob": entry, "final_data_cost": last_cost})

    return pd.DataFrame(rows, columns=["blob", "final_data_cost"])


def compute_field_data_cost(obs_path, calc_path):
    """Compute normalised L2 misfit between observed and predicted fields.

    Reads two Tomofast ``.obs`` files (space-delimited, first line is a count
    that is skipped) and computes:

    .. math::
        \\frac{\\|bz_{\\text{pred}} - bz_{\\text{obs}}\\|_2}{\\|bz_{\\text{obs}}\\|_2}

    Parameters
    ----------
    obs_path : str
        Path to the observed field ``.obs`` file.
    calc_path : str
        Path to the predicted/calculated field ``.obs`` file.

    Returns
    -------
    float
        Relative L2 data misfit.
    """
    col_names = ["x", "y", "z", "bz"]

    obs_df = pd.read_csv(obs_path, sep=r"\s+", names=col_names, skiprows=1, engine="python")
    calc_df = pd.read_csv(calc_path, sep=r"\s+", names=col_names, skiprows=1, engine="python")

    bz_obs = obs_df["bz"].to_numpy(dtype=float)
    bz_calc = calc_df["bz"].to_numpy(dtype=float)

    norm_obs = float(np.linalg.norm(bz_obs))
    if norm_obs == 0.0:
        return 0.0

    return float(np.linalg.norm(bz_calc - bz_obs) / norm_obs)
