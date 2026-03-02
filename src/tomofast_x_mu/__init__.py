"""tomofast_x_mu -- QDM 3D inversion pipeline wrapping Tomofast-x."""

from tomofast_x_mu.io import load_qdm_mat, load_field_table, read_mesh_bounds
from tomofast_x_mu.format import format_tomofast_obs, write_obs, write_meshgrid_tomofast
from tomofast_x_mu.detect import detect_anomalies, extract_grains, extract_max_cells
from tomofast_x_mu.mesh import make_edges, build_mesh_per_blob, create_single_cell_meshes
from tomofast_x_mu.parfile import (
    create_inversion_parfiles,
    create_source_parfiles,
    create_fitting_parfile,
)
from tomofast_x_mu.runner import (
    run_tomofast_blobs,
    run_source_detection,
    run_dipole_fitting,
    to_wsl_path,
)
from tomofast_x_mu.analysis import (
    grain_statistics,
    vector_analysis,
    combine_best_locations,
    extract_blob_data_costs,
    compute_field_data_cost,
)
from tomofast_x_mu.plotting import plot_bz_led, plot_misfits, plot_stereonet, plot_results_on_bz

__all__ = [
    # io
    "load_qdm_mat",
    "load_field_table",
    "read_mesh_bounds",
    # format
    "format_tomofast_obs",
    "write_obs",
    "write_meshgrid_tomofast",
    # detect
    "detect_anomalies",
    "extract_grains",
    "extract_max_cells",
    # mesh
    "make_edges",
    "build_mesh_per_blob",
    "create_single_cell_meshes",
    # parfile
    "create_inversion_parfiles",
    "create_source_parfiles",
    "create_fitting_parfile",
    # runner
    "run_tomofast_blobs",
    "run_source_detection",
    "run_dipole_fitting",
    "to_wsl_path",
    # analysis
    "grain_statistics",
    "vector_analysis",
    "combine_best_locations",
    "extract_blob_data_costs",
    "compute_field_data_cost",
    # plotting
    "plot_bz_led",
    "plot_misfits",
    "plot_stereonet",
    "plot_results_on_bz",
]
