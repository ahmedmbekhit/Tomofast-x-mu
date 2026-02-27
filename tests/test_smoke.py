"""End-to-end smoke test: synthetic .mat -> load -> format -> detect -> mesh -> parfile chain."""

import numpy as np
import pytest
import scipy.io

from tomofast_x_mu.io import load_qdm_mat
from tomofast_x_mu.format import format_tomofast_obs
from tomofast_x_mu.detect import detect_anomalies
from tomofast_x_mu.mesh import build_mesh_per_blob
from tomofast_x_mu.parfile import create_inversion_parfiles


def test_full_pipeline(tmp_path):
    """Run the full pipeline on synthetic data: mat -> csv -> obs -> detect -> mesh -> parfile."""
    # Step 1: Create synthetic .mat with a strong anomaly
    ny, nx = 60, 100
    step = 4.4e-6
    h = 10e-6

    rng = np.random.default_rng(123)
    Bz = rng.normal(0, 1e-10, (ny, nx))

    # Inject a clear anomaly blob
    cx, cy = 50, 30
    for iy in range(ny):
        for ix in range(nx):
            dist = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
            if dist < 15:
                Bz[iy, ix] += 5e-7 * np.exp(-dist / 5)

    mat_path = tmp_path / "smoke_test.mat"
    scipy.io.savemat(str(mat_path), {"Bz": Bz, "step": np.array([[step]]), "h": np.array([[h]])})

    # Step 2: Load .mat
    ds = load_qdm_mat(mat_path, save_csv=True, csv_path=tmp_path / "smoke.csv")
    assert ds.bz.shape == (ny, nx)

    # Step 3: Format to Tomofast obs
    obs_path = tmp_path / "obs" / "smoke.obs"
    result = format_tomofast_obs(str(tmp_path / "smoke.csv"), str(obs_path))
    assert obs_path.exists()
    assert result["nx"] == nx
    assert result["ny"] == ny

    # Step 4: Detect anomalies
    win_path = tmp_path / "blob_windows.txt"
    sum_path = tmp_path / "summary.txt"
    obs_out_dir = tmp_path / "blob_obs"

    blobs = detect_anomalies(
        str(obs_path),
        threshold_factor=2.5,
        pad_x=5,
        pad_y=5,
        min_blob_cells=5,
        max_blob_cells=100000,
        nz_fixed=10,
        out_obs_dir=str(obs_out_dir),
        windows_path=str(win_path),
        summary_path=str(sum_path),
    )
    assert len(blobs) >= 1, "Pipeline should detect at least one anomaly blob"

    # Step 5: Build meshes
    mesh_out_dir = tmp_path / "meshgrids"
    mesh_summary = tmp_path / "mesh_summary.txt"

    mesh_results = build_mesh_per_blob(
        str(win_path),
        str(mesh_out_dir),
        str(mesh_summary),
        npad_x=3,
        npad_y=3,
        nz=5,
        dz0=4.4,
    )
    assert len(mesh_results) >= 1

    # Step 6: Create parfiles
    par_dir = tmp_path / "parfiles"
    par_paths = create_inversion_parfiles(str(mesh_summary), str(par_dir))
    assert len(par_paths) >= 1

    # Verify parfile content references the blob
    with open(par_paths[0]) as f:
        text = f.read()
    assert "Blob_" in text
    assert "modelGrid.size" in text
