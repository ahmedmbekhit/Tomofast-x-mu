"""Orchestration for running Tomofast-x inversions via WSL or native shell."""

import glob
import math
import os
import re
import subprocess
import sys
import time


def to_wsl_path(win_path):
    """Convert a Windows path to a WSL /mnt/ path.

    Parameters
    ----------
    win_path : str
        Windows-style path (e.g. ``C:\\Users\\...`` or ``C:/Users/...``).

    Returns
    -------
    str
        WSL-style path (e.g. ``/mnt/c/Users/...``).

    Raises
    ------
    ValueError
        If the path is not a valid Windows absolute path with a drive letter.
    """
    win_path = win_path.replace("\\", "/")

    # Validate Windows path format: must start with drive letter and colon
    if len(win_path) < 3 or win_path[1] != ':' or not win_path[0].isalpha():
        raise ValueError(
            f"Invalid Windows path format: {win_path!r}. "
            "Expected format: 'C:/path' or 'C:\\path'"
        )

    drive = win_path[0].lower()
    rest = win_path[3:]  # Skip "C:/"
    return f"/mnt/{drive}/{rest}"


def _print_progress(done, total, blob, grain, mesh, cost):
    """Single-line overwriting progress update for source detection loop."""
    cost_str = f"{cost:.6e}" if cost is not None and not math.isnan(cost) else "NaN"
    sys.stdout.write(f"\rProgress: {done}/{total} | {blob} {grain} {mesh} | cost={cost_str}   ")
    sys.stdout.flush()


def _run_live(cmd, wsl_exe=None, bash_time=False):
    """Run a shell command, capturing output for error reporting."""
    if bash_time:
        cmd = f"{{ time {cmd}; }} 2>&1"
    if wsl_exe is not None:
        args = [wsl_exe, "bash", "-lc", cmd]
    else:
        args = ["bash", "-lc", cmd]

    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = p.stdout.read()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={p.returncode}): {cmd}\n"
            f"Output:\n{output}"
        )


def _run_quiet(cmd, wsl_exe=None):
    """Run a shell command quietly, capturing stdout and parsing data cost.

    Returns (returncode, stdout_text, last_data_cost).
    """
    if wsl_exe is not None:
        args = [wsl_exe, "bash", "-lc", cmd]
    else:
        args = ["bash", "-lc", cmd]

    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    last_data_cost = None
    stdout_lines = []
    for line in p.stdout:
        stdout_lines.append(line)
        if "data cost (new)" in line:
            try:
                last_data_cost = float(line.split("=")[-1].strip())
            except Exception:
                pass

    p.wait()
    return p.returncode, "".join(stdout_lines), last_data_cost


def run_tomofast_blobs(base_dir, tomofast_home, nproc=7, wsl_exe=None):
    """Run Tomofast for each blob: copy files, run mpirun, copy back.

    Parameters
    ----------
    base_dir : str
        Project base directory (Windows path if using WSL).
    tomofast_home : str
        Tomofast-X installation directory (WSL or native path).
    nproc : int
        Number of MPI processes.
    wsl_exe : str or None
        Path to wsl.exe (e.g. ``r"C:\\Windows\\System32\\wsl.exe"``).
        If None, runs natively without WSL.

    Returns
    -------
    list of str
        Paths to output directories per blob.
    """
    stage_start = time.time()

    # Validate tomofast_home
    if not tomofast_home or tomofast_home.strip() == "":
        raise ValueError("tomofast_home must be a non-empty path")

    obs_dir = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Blob_Clipped_obs_RECT")
    mesh_dir = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Blob_Meshgrids")
    par_dir = os.path.join(base_dir, "0-Parfiles")
    out_win_root = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")
    os.makedirs(out_win_root, exist_ok=True)

    use_wsl = wsl_exe is not None
    if use_wsl:
        obs_dir_wsl = to_wsl_path(obs_dir)
        mesh_dir_wsl = to_wsl_path(mesh_dir)
        par_dir_wsl = to_wsl_path(par_dir)

    obs_files = sorted(glob.glob(os.path.join(obs_dir, "Blob_*_RECT.obs")))
    if not obs_files:
        raise FileNotFoundError(f"No blob obs found in: {obs_dir}")

    blob_ids = []
    for f in obs_files:
        name = os.path.basename(f)
        blob_ids.append(int(name.split("_")[1]))

    output_dirs = []

    for bid in blob_ids:
        obs_win = os.path.join(obs_dir, f"Blob_{bid:03d}_RECT.obs")
        mesh_win = os.path.join(mesh_dir, f"Meshgrid_Blob_{bid:03d}.txt")
        par_win = os.path.join(par_dir, f"Parfile_QDM_Inversion_Blob_{bid:03d}.txt")

        for p in [obs_win, mesh_win, par_win]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        if use_wsl:
            obs_wsl = f"{obs_dir_wsl}/Blob_{bid:03d}_RECT.obs"
            mesh_wsl = f"{mesh_dir_wsl}/Meshgrid_Blob_{bid:03d}.txt"
            par_wsl = f"{par_dir_wsl}/Parfile_QDM_Inversion_Blob_{bid:03d}.txt"

            _run_live(f"cp '{par_wsl}' {tomofast_home}/parfiles/", wsl_exe)
            _run_live(f"cp '{obs_wsl}' {tomofast_home}/data/QDM/", wsl_exe)
            _run_live(f"cp '{mesh_wsl}' {tomofast_home}/data/QDM/", wsl_exe)
            _run_live(f"rm -rf {tomofast_home}/output/QDM/*", wsl_exe)

            par_name = f"Parfile_QDM_Inversion_Blob_{bid:03d}.txt"
            _run_live(
                f"cd {tomofast_home} && "
                f"mpirun -np {nproc} ./tomofastx -p ./parfiles/{par_name}",
                wsl_exe,
                bash_time=True,
            )

            out_blob_win = os.path.join(out_win_root, f"Blob_{bid:03d}")
            os.makedirs(out_blob_win, exist_ok=True)
            out_blob_wsl = to_wsl_path(out_blob_win)

            _run_live(
                f"rsync -av --exclude='SENSIT/' "
                f"{tomofast_home}/output/QDM/ '{out_blob_wsl}/'",
                wsl_exe,
            )
        else:
            # Native (non-WSL) execution
            _run_live(f"cp '{par_win}' {tomofast_home}/parfiles/")
            _run_live(f"cp '{obs_win}' {tomofast_home}/data/QDM/")
            _run_live(f"cp '{mesh_win}' {tomofast_home}/data/QDM/")
            _run_live(f"rm -rf {tomofast_home}/output/QDM/*")

            par_name = f"Parfile_QDM_Inversion_Blob_{bid:03d}.txt"
            _run_live(
                f"cd {tomofast_home} && "
                f"mpirun -np {nproc} ./tomofastx -p ./parfiles/{par_name}",
                bash_time=True,
            )

            out_blob_win = os.path.join(out_win_root, f"Blob_{bid:03d}")
            os.makedirs(out_blob_win, exist_ok=True)
            _run_live(
                f"rsync -av --exclude='SENSIT/' "
                f"{tomofast_home}/output/QDM/ '{out_blob_win}/'"
            )

        output_dirs.append(out_blob_win)

    total = time.time() - stage_start
    print(f"\nTool 1 total: {total / 60:.2f} min")
    return output_dirs


def run_source_detection(base_dir, tomofast_home, nproc=7, wsl_exe=None):
    """Run source detection inversions: per blob -> per grain -> per meshgrid.

    Parameters
    ----------
    base_dir : str
        Project base directory.
    tomofast_home : str
        Tomofast-X installation directory.
    nproc : int
        Number of MPI processes.
    wsl_exe : str or None
        Path to wsl.exe.  If None, runs natively.

    Returns
    -------
    list of tuple
        Summary rows: (blob_name, grain_name, mesh_name, xmin, xmax, ymin, ymax,
        zmin, zmax, final_data_cost).
    """
    stage_start = time.time()
    use_wsl = wsl_exe is not None

    mesh_root = os.path.join(base_dir, "4-Source_Detection", "Single_cell_meshs")
    obs_dir = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Blob_Clipped_obs_RECT")
    par_dir = os.path.join(base_dir, "0-Parfiles")

    blob_folders = sorted(
        d for d in glob.glob(os.path.join(mesh_root, "Blob_*")) if os.path.isdir(d)
    )
    if not blob_folders:
        raise FileNotFoundError("No Blob_* folders found in: " + mesh_root)

    total_meshes = sum(
        len(glob.glob(os.path.join(gf, "meshgrid_*.txt")))
        for bf in blob_folders
        for gf in glob.glob(os.path.join(bf, "grain_*"))
        if os.path.isdir(gf)
    )
    done_count = 0

    summary_rows = []

    for blob_folder in blob_folders:
        blob_name = os.path.basename(blob_folder)
        blob_id = int(blob_name.split("_")[1])

        obs_win = os.path.join(obs_dir, f"Blob_{blob_id:03d}_RECT.obs")
        par_win = os.path.join(par_dir, f"Parfile_QDM_Inversion_Blob_{blob_id:03d}.txt")

        if not os.path.exists(obs_win) or not os.path.exists(par_win):
            continue

        if use_wsl:
            _run_quiet(f"cp '{to_wsl_path(par_win)}' {tomofast_home}/parfiles/", wsl_exe)
            _run_quiet(
                f"cp '{to_wsl_path(obs_win)}' {tomofast_home}/data/QDM/Mag_Inversion.obs",
                wsl_exe,
            )
        else:
            _run_quiet(f"cp '{par_win}' {tomofast_home}/parfiles/")
            _run_quiet(f"cp '{obs_win}' {tomofast_home}/data/QDM/Mag_Inversion.obs")

        par_name = f"Parfile_QDM_Inversion_Blob_{blob_id:03d}.txt"

        grain_folders = sorted(
            d
            for d in glob.glob(os.path.join(blob_folder, "grain_*"))
            if os.path.isdir(d)
        )

        for grain_folder in grain_folders:
            grain_name = os.path.basename(grain_folder)
            if use_wsl:
                grain_folder_wsl = to_wsl_path(grain_folder)

            mesh_files = sorted(glob.glob(os.path.join(grain_folder, "meshgrid_*.txt")))

            for mesh_path in mesh_files:
                mesh_name = os.path.basename(mesh_path)

                # Read mesh bounds
                bounds = _read_mesh_bounds_simple(mesh_path)
                if bounds is None:
                    summary_rows.append(
                        (blob_name, grain_name, mesh_name)
                        + (float("nan"),) * 7
                    )
                    continue

                xmin, xmax, ymin, ymax, zmin, zmax = bounds

                if use_wsl:
                    src = f"{grain_folder_wsl}/{mesh_name}"
                    _run_quiet(f"cp '{src}' {tomofast_home}/data/QDM/meshgrid.txt", wsl_exe)
                    cmd = f"cd {tomofast_home} && ./tomofastx -p ./parfiles/{par_name}"
                    rc, _, last_cost = _run_quiet(cmd, wsl_exe)
                else:
                    _run_quiet(f"cp '{mesh_path}' {tomofast_home}/data/QDM/meshgrid.txt")
                    cmd = f"cd {tomofast_home} && ./tomofastx -p ./parfiles/{par_name}"
                    rc, _, last_cost = _run_quiet(cmd)

                final_cost = last_cost if (rc == 0 and last_cost is not None) else float("nan")
                done_count += 1
                _print_progress(done_count, total_meshes, blob_name, grain_name, mesh_name, final_cost)
                summary_rows.append(
                    (blob_name, grain_name, mesh_name, xmin, xmax, ymin, ymax, zmin, zmax, final_cost)
                )

    print()
    total = time.time() - stage_start
    print(f"Tool 2 total: {total / 60:.2f} min")
    return summary_rows


def run_dipole_fitting(base_dir, tomofast_home, nproc=7, wsl_exe=None):
    """Run combined dipole fitting inversion.

    Parameters
    ----------
    base_dir : str
        Project base directory.
    tomofast_home : str
        Tomofast-X installation directory.
    nproc : int
        Number of MPI processes.
    wsl_exe : str or None
        Path to wsl.exe.

    Returns
    -------
    str
        Path to the output directory.
    """
    stage_start = time.time()
    use_wsl = wsl_exe is not None

    if use_wsl:
        base_dir_wsl = to_wsl_path(base_dir)

        _run_live(
            f"cp '{base_dir_wsl}/0-Parfiles/Parfile_QDM_Dipole.txt' {tomofast_home}/parfiles/",
            wsl_exe,
        )
        _run_live(
            f"cp '{base_dir_wsl}/4-3D_Inversion_Filtered/Mag_Inversion.obs' {tomofast_home}/data/QDM/",
            wsl_exe,
        )

        combined_mesh_win = os.path.join(base_dir, "4-Source_Detection", "combined_best_blobs_mesh.txt")
        combined_mesh_wsl = to_wsl_path(combined_mesh_win)
        _run_live(f"cp '{combined_mesh_wsl}' '{tomofast_home}/data/QDM/meshgrid.txt'", wsl_exe)

        _run_live(
            f"cd {tomofast_home} && "
            f"mpirun -np {nproc} ./tomofastx -p ./parfiles/Parfile_QDM_Dipole.txt",
            wsl_exe,
            bash_time=True,
        )

        output_win = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output", "combined_grains")
        os.makedirs(output_win, exist_ok=True)
        output_wsl = to_wsl_path(output_win)

        for subdir in ["data", "model", "Paraview", "QDM_log.txt", "Parfile_copy.txt", "costs.txt"]:
            _run_live(f"cp -r {tomofast_home}/output/QDM/{subdir} '{output_wsl}/'", wsl_exe)
    else:
        _run_live(f"cp '{base_dir}/0-Parfiles/Parfile_QDM_Dipole.txt' {tomofast_home}/parfiles/")
        _run_live(
            f"cp '{base_dir}/4-3D_Inversion_Filtered/Mag_Inversion.obs' {tomofast_home}/data/QDM/"
        )

        combined_mesh = os.path.join(base_dir, "4-Source_Detection", "combined_best_blobs_mesh.txt")
        _run_live(f"cp '{combined_mesh}' '{tomofast_home}/data/QDM/meshgrid.txt'")

        _run_live(
            f"cd {tomofast_home} && "
            f"mpirun -np {nproc} ./tomofastx -p ./parfiles/Parfile_QDM_Dipole.txt",
            bash_time=True,
        )

        output_win = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output", "combined_grains")
        os.makedirs(output_win, exist_ok=True)

        for subdir in ["data", "model", "Paraview", "QDM_log.txt", "Parfile_copy.txt", "costs.txt"]:
            _run_live(f"cp -r {tomofast_home}/output/QDM/{subdir} '{output_win}/'")

    total = time.time() - stage_start
    print(f"\nTool 3 total: {total / 60:.2f} min")
    return output_win


def _read_mesh_bounds_simple(mesh_path):
    """Read mesh bounds from a single-cell mesh file (2 lines: count + data)."""
    with open(mesh_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    s = lines[1].split()
    return float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])
