# =============================================================================
# TEC FILE -> PLOT (TEC + TOMOFAST) + TOMOFAST TXT + VTK + FORWARD MODEL FILES
# =============================================================================
# Paraview workflow:
#   M = Mx*iHat + My*jHat + Mz*kHat   (Calculator1)
#   V = curl(M)                         (GradientOfUnstructuredDataSet, vorticity)
#   H = M.V                             (Calculator2, helicity)
#
# Tomofast forward files per grain (auto-sized to grain):
#   Meshgrid_<stem>.txt       nx*ny*nz rows, ALL cells, x-fastest order
#   Magnetization_<stem>.txt  nx*ny*nz rows, Mx My Mz (0 outside grain)
#   Forward_<stem>.obs        nx*ny rows, sensor XY centres
#   Parfile_<stem>.txt        ready to run
# =============================================================================

import re
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path

import sys
import datetime

# ============================================================
#  LOGGING -- duplicate everything printed to console into a log file too
# ============================================================
class Tee:
    """Writes to both the original stream (console) and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()   # flush immediately so the log stays current if the run crashes

    def flush(self):
        for s in self.streams:
            s.flush()

log_filename = f"tec_converter_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
log_file = open(log_filename, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)   # also capture errors/tracebacks, not just prints

print(f"Logging this run to: {log_filename}\n")


# ============================================================
#  USER CONFIG
# ============================================================
base_dir = Path(os.getcwd())
tec_dir  = base_dir / "Tec_files_bellon"
out_dir  = tec_dir  / "converted"
out_dir.mkdir(exist_ok=True)

PLOT_TEC      = False    # interactive PyVista plot of original tetrahedral mesh
PLOT_TOMOFAST = False    # interactive PyVista plot of converted prism mesh
PLOT_SLICES   = False    # matplotlib XY/XZ/YZ slice plots
SAVE_VTK_TEC  = True    # save .vtu for original tec (tetrahedral)
SAVE_VTK_TOMO = True    # save .vtu for tomofast prisms (hexahedral)
SAVE_TOMOFAST_FILES = True  # save four Tomofast forward model files

Ms = 482476.91085597320   # saturation magnetisation A/m (magnetite)

GRID_MODE    = "n_divisions"  # options: "n_divisions" or "cell_size"
N_DIVISIONS_XY  = 100    # cells per axis, X and Y (used only if GRID_MODE == "n_divisions")
N_DIVISIONS_Z   = 50    # cells along Z (used only if GRID_MODE == "n_divisions")
CELL_SIZE_XY_UM = 0.00435 #matched to the median tetrahedron volume  # voxel edge length, X and Y, um (used only if GRID_MODE == "cell_size")
CELL_SIZE_Z_UM  = 0.00435 #matched to the median tetrahedron volume  # voxel edge length, Z, um (used only if GRID_MODE == "cell_size")

#PAD_FRAC     = 0.1       # padding around grain bounding box

#SENSOR_HEIGHT = 0.01   # sensor height above top of model (same units as grain)
SENSOR_HEIGHT = float(os.environ.get("TEC_SENSOR_HEIGHT", 0.01))   # sensor height above top of model (same units as grain)
WINDOW_FACTOR = SENSOR_HEIGHT*10 if SENSOR_HEIGHT <= 1.5 else SENSOR_HEIGHT * 5       # 0.01 # sensor window = height ** this 

MESH_PAD_UM = SENSOR_HEIGHT/10   # EXTRA padding added to the mesh domain, beyond the data window (user-defined)
MESH_Z_PAD_UM = (WINDOW_FACTOR+MESH_PAD_UM)/2      # extra padding in Z direction only, to ensure the mesh extends well below the grain for tomofast's half-space assumption

#COMPRESSION_RATE = float(os.environ.get("TEC_COMPRESSION_RATE", 0.02))   # forward.matrixCompression.rate, used in forward + inversion parfiles only


#MESH_PAD_UM = 2*CELL_SIZE_XY_UM    # EXTRA padding added to the mesh domain, beyond the data window (user-defined)
#MESH_Z_PAD_UM = 2*CELL_SIZE_Z_UM      # extra padding in Z direction only, to ensure the mesh extends well below the grain for tomofast's half-space assumption


#INCLINATION  = 90.0       # vertical field = no directional bias on remanenceS
#DECLINATION  = 0.0
#INTENSITY_NT = 53829.0




# ============================================================
#  Optimal compression rate prediction (Bruce et al. 2026, Eq. 10) and minimum model size for compression
# ============================================================
COMPRESSION_TARGET_ERROR = float(os.environ.get("TEC_COMPRESSION_ERROR", 0.01))  # c_e, target compression error (1% recommended by Bruce et al.)
COMPRESSION_MIN_CELLS    = 400000 #   # below this disable compression 

def calc_ideal_compression_rate(nx, ny, nz, c_e=COMPRESSION_TARGET_ERROR, wavelet="haar"):
    """
    Predicts the optimal wavelet compression rate (forward.matrixCompression.rate)
    from model size and target compression error, using Equation 10 of:

        Bruce, Ogarko, Giraud & Jessell (2026), "Wavelet Compression for Gravity
        Inversion: Optimising and Predicting Memory Efficiency in 3D Geophysical
        Modelling" (preprint, SSRN 6262158).

            c_r = a * c_e^k * N_model^b                              (Eq. 10)

    Constants (Table 2, fitted via SciPy curve_fit):
        Haar: a=35.07, k=-0.872, b=-0.884   (R^2=0.95, RMSE=0.04)
        D4:   a=42.53, k=-0.614, b=-0.852   (R^2=0.89, RMSE=0.001)

    Mirrors Tomofast-x-q's update_ideal_compression_ratio() (Tomofast_x.py),
    which applies a 2x safety margin on top of the raw equation output and
    clamps the result to a max of 1.0.
    """
    n_model = nx * ny * nz

    if n_model <= COMPRESSION_MIN_CELLS:
        # Below the smallest model size Bruce et al. actually tested (0.4M cells) --
        # Equation 10 is extrapolating outside its fitted range, so skip compression.
        return 1.0

    if wavelet.lower() == "d4":
        a, k, b = 42.53, -0.614, -0.852
    else:  # default: haar
        a, k, b = 35.07, -0.872, -0.884

    ideal_cr = a * (c_e ** k) * (n_model ** b)
    rate = min(ideal_cr * 10.0, 1.0)   # 10x safety margin, capped at 1.0 (no compression)
    return rate

# ============================================================
#  READ TEC FILE
# ============================================================
def read_tec(tec_file):
    with open(tec_file, "r") as f:
        text = f.read()
    N = int(re.search(r"N\s*=\s*(\d+)", text).group(1))
    E = int(re.search(r"E\s*=\s*(\d+)", text).group(1))
    lines = text.splitlines()
    data_start = None
    for idx, line in enumerate(lines):
        if "VARLOCATION" in line:
            data_start = idx + 1
            break
    if data_start is None:
        raise ValueError(f"Could not find VARLOCATION in {tec_file.name}")
    numbers = []
    for line in lines[data_start:]:
        numbers.extend(line.split())
    numbers = np.array(numbers, dtype=float)
    n_nodal_vars     = 6
    total_var_values = n_nodal_vars * N + E
    values           = numbers[:total_var_values]
    connectivity     = numbers[total_var_values:].astype(int)
    X  = values[0*N:1*N];  Y  = values[1*N:2*N];  Z  = values[2*N:3*N].max() - values[2*N:3*N] #anchor to 0 ,flip z becuase tomofast is z potivite dpwn (depth) but MERRILL is z positive up (height)
    Mx = values[3*N:4*N];  My = values[4*N:5*N];  Mz = -values[5*N:6*N] 
    SD = values[6*N:6*N+E]
    tetra = connectivity.reshape(E, 4) - 1
    return dict(N=N, E=E, X=X, Y=Y, Z=Z, Mx=Mx, My=My, Mz=Mz, SD=SD, tetra=tetra)


# ============================================================
#  BUILD MESH FROM GRAIN - auto-sized per grain
# ============================================================
def build_grain_mesh(d):
    cx = (d['X'].min() + d['X'].max()) / 2.0
    cy = (d['Y'].min() + d['Y'].max()) / 2.0
    zlo, zhi = d['Z'].min(), d['Z'].max()

    # XY: centred on grain, window = SENSOR_HEIGHT * WINDOW_FACTOR each side
    #window = SENSOR_HEIGHT * WINDOW_FACTOR
    #window = WINDOW_FACTOR
    window = WINDOW_FACTOR + max(d['X'].max()-d['X'].min(), d['Y'].max()-d['Y'].min()) / 2.0
    #window = SENSOR_HEIGHT ** WINDOW_FACTOR
    x0 = cx - window;  x1 = cx + window
    y0 = cy - window;  y1 = cy + window

    # Z: grain bottom to grain top + sensor height
    z0 = zlo
    z1 = zhi #+ SENSOR_HEIGHT

    # extra mesh padding, applied AFTER the data window for tomofast boundary condition as the inversion smoothes the model
    x0 -= MESH_PAD_UM;  x1 += MESH_PAD_UM
    y0 -= MESH_PAD_UM;  y1 += MESH_PAD_UM
    z1 += MESH_Z_PAD_UM          # only pad deeper (downward) stays positive
    z0 = max(z0, 0.0)          # never let z0 go negative - Tomofast requires positive depth

    # all three axes use N_DIVISIONS or cell size 
    #nx = ny = nz = N_DIVISIONS
    xr = x1 - x0
    yr = y1 - y0
    zr = z1 - z0

    if GRID_MODE == "n_divisions":
        cube_size_xy = min(xr, yr) / N_DIVISIONS_XY
        cube_size_z  = zr / N_DIVISIONS_Z
    elif GRID_MODE == "cell_size":
        cube_size_xy = CELL_SIZE_XY_UM
        cube_size_z  = CELL_SIZE_Z_UM
    else:
        raise ValueError(f"Unknown GRID_MODE: {GRID_MODE}")

    nx = max(1, int(np.ceil(xr / cube_size_xy)))
    ny = max(1, int(np.ceil(yr / cube_size_xy)))
    nz = max(1, int(np.ceil(zr / cube_size_z)))

    x1 = x0 + nx * cube_size_xy
    y1 = y0 + ny * cube_size_xy
    z1 = z0 + nz * cube_size_z

    # auto-compute divisions from body tetrahedra count
    #n_body = int(np.sum(d['SD'] == 101.0))
    #n_div  = int(np.ceil(n_body ** (1/3)))
    #nx = ny = nz = n_div
    #print(f"  Auto: {n_div} divisions ({n_body} body tetra -> {n_div}³={n_div**3} prisms)")

    x_edges = np.linspace(x0, x1, nx + 1)
    y_edges = np.linspace(y0, y1, ny + 1)
    z_edges = np.linspace(z0, z1, nz + 1)

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / nz
    print(f"  Window: {window:.4f} um each side | centre: ({cx:.4f}, {cy:.4f})")
    print(f"  Mesh: {nx}x{ny}x{nz} = {nx*ny*nz} cells | "
          f"dx={dx:.4f} dy={dy:.4f} dz={dz:.4f}")
    return x_edges, y_edges, z_edges, nx, ny, nz

def plot_mesh_map(d, x_edges, y_edges, z_edges, title, save_path):
    cx = (d['X'].min() + d['X'].max()) / 2.0
    cy = (d['Y'].min() + d['Y'].max()) / 2.0
    half_extent = max(d['X'].max()-d['X'].min(), d['Y'].max()-d['Y'].min()) / 2.0
    zlo, zhi = d['Z'].min(), d['Z'].max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- XY panel ---
    for xe in x_edges:
        ax1.axvline(xe, color='lightgray', lw=0.3)
    for ye in y_edges:
        ax1.axhline(ye, color='lightgray', lw=0.3)

    ax1.add_patch(plt.Rectangle((cx-half_extent, cy-half_extent), 2*half_extent, 2*half_extent,
                                 facecolor='orange', alpha=0.3, edgecolor='none', label='core (grain footprint)'))
    ax1.scatter(d['X'], d['Y'], s=1, color='red', label='grain nodes')

    ax1.set_xlim(x_edges[0], x_edges[-1])
    ax1.set_ylim(y_edges[0], y_edges[-1])
    ax1.set_aspect('equal')
    ax1.set_title(f"{title} -- XY mesh")
    ax1.legend(loc='upper right', fontsize=8)

    # --- XZ panel ---
    for xe in x_edges:
        ax2.axvline(xe, color='lightgray', lw=0.3)
    for ze in z_edges:
        ax2.axhline(ze, color='lightgray', lw=0.3)

    ax2.add_patch(plt.Rectangle((cx-half_extent, zlo), 2*half_extent, zhi-zlo,
                                 facecolor='orange', alpha=0.3, edgecolor='none', label='core (grain footprint)'))
    ax2.scatter(d['X'], d['Z'], s=1, color='red', label='grain nodes')

    ax2.set_xlim(x_edges[0], x_edges[-1])
    ax2.set_ylim(z_edges[0], z_edges[-1])
    ax2.set_aspect('equal')
    ax2.set_title(f"{title} -- XZ mesh")
    ax2.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Mesh map plot   : {save_path.name}")
# ============================================================
#  MAP TETRA -> VOXELS - returns dict (i,j,k) -> (mx,my,mz unit)
# ============================================================
#def map_tetra_to_voxels(d, x_edges, y_edges, z_edges, nx, ny, nz):
 #   """
 #   Average and renormalise magnetisation for each occupied voxel.
 #   Returns mag_dict keyed by 1-based (i,j,k).
 #   """
  #  tetra = d['tetra']
 #   X, Y, Z = d['X'], d['Y'], d['Z']
  #  vox_mx = {};  vox_my = {};  vox_mz = {};  vox_cnt = {}
 #   SD_BODY = 101.0   # MERRILL body index - only use grain body tetrahedra
 #   for e in range(d['E']):
 #       # Skip tetrahedra not belonging to grain body
 #       if d['SD'][e] != SD_BODY:
   #         continue
  #      nids = tetra[e]
   #     cx = X[nids].mean();  cy = Y[nids].mean();  cz = Z[nids].mean()
  #      mx_e = d['Mx'][nids].mean() 
  #      my_e = d['My'][nids].mean() 
  #      mz_e = d['Mz'][nids].mean() 

   #     # sample centroid + 4 corners (was centroid only)
   #     #points = [(cx, cy, cz)] + [(X[n], Y[n], Z[n]) for n in nids]

   #     corners = [(X[n], Y[n], Z[n]) for n in nids]
   #     edge_pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
   #     midpoints = [((corners[a][0]+corners[b][0])/2,
    #                  (corners[a][1]+corners[b][1])/2,
   #                   (corners[a][2]+corners[b][2])/2) for a,b in edge_pairs]
    #    points = [(cx, cy, cz)] + corners + midpoints

    #    for (px, py, pz) in points:
    #        i = int(np.clip(np.searchsorted(x_edges, px, side='right'), 1, nx))
    #        j = int(np.clip(np.searchsorted(y_edges, py, side='right'), 1, ny))
    #        k = int(np.clip(np.searchsorted(z_edges, pz, side='right'), 1, nz))
    #        key = (i, j, k)
    #        cx_vox = (x_edges[i-1] + x_edges[i]) / 2
    #        cy_vox = (y_edges[j-1] + y_edges[j]) / 2
    #        cz_vox = (z_edges[k-1] + z_edges[k]) / 2
     #       dist = max(np.sqrt((px-cx_vox)**2 + (py-cy_vox)**2 + (pz-cz_vox)**2), 1e-10)
     #       w = 1.0 / dist
     #       if key not in vox_mx:
     #           vox_mx[key]=0.0; vox_my[key]=0.0; vox_mz[key]=0.0; vox_cnt[key]=0.0
      #      vox_mx[key]+=mx_e*w; vox_my[key]+=my_e*w; vox_mz[key]+=mz_e*w; vox_cnt[key]+=w

    #mag_dict = {}
    #for key, cnt in vox_cnt.items():
     #   mx = vox_mx[key]/cnt;  my = vox_my[key]/cnt;  mz = vox_mz[key]/cnt
     #   mag_dict[key] = (mx, my, mz)

    #print(f"  Occupied: {len(mag_dict)} / {nx*ny*nz} voxels")
    #return mag_dict
# ============================================================
#  MAP TETRA -> VOXELS (pyvista) - returns dict (i,j,k) -> (mx,my,mz unit)
# ============================================================

def map_tetra_to_voxels(d, x_edges, y_edges, z_edges, nx, ny, nz, mvh=None):
    """
    Resample tetrahedral Mx,My,Mz (and optionally H, H_normalized, V from mvh)
    onto the regular voxel grid using PyVista's sample() (proper geometric
    interpolation, no point-sampling holes). Returns (mag_dict, mvh_dict),
    both keyed by 1-based (i,j,k). mvh_dict is empty if mvh is None.
    """
    SD_BODY = 101.0   # MERRILL body index - only use grain body tetrahedra

    # ---- build a PyVista tetrahedral grid containing only the grain body ----
    tetra = d['tetra']
    X, Y, Z = d['X'], d['Y'], d['Z']
    body_mask = d['SD'] == SD_BODY
    body_tetra = tetra[body_mask]

    points = np.column_stack([X, Y, Z])
    E_body = len(body_tetra)
    cells = np.hstack([np.full((E_body, 1), 4), body_tetra]).astype(np.int64).flatten()
    celltypes = np.full(E_body, 10)  # VTK_TETRA
    tec_body_grid = pv.UnstructuredGrid(cells, celltypes, points)
    tec_body_grid.point_data["Mx"] = d['Mx']
    tec_body_grid.point_data["My"] = d['My']
    tec_body_grid.point_data["Mz"] = d['Mz']
    if mvh is not None:
        tec_body_grid.point_data["H"] = mvh['H']
        tec_body_grid.point_data["H_normalized"] = mvh['H_normalized']
        tec_body_grid.point_data["Vx"] = mvh['V'][:, 0]
        tec_body_grid.point_data["Vy"] = mvh['V'][:, 1]
        tec_body_grid.point_data["Vz"] = mvh['V'][:, 2]

    # ---- build the target voxel grid (cell centres at i,j,k) ----
    cx_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    cy_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    cz_centres = (z_edges[:-1] + z_edges[1:]) / 2.0

    target_grid = pv.ImageData()
    target_grid.dimensions = (nx + 1, ny + 1, nz + 1)
    target_grid.origin = (x_edges[0], y_edges[0], z_edges[0])
    target_grid.spacing = (
        (x_edges[-1] - x_edges[0]) / nx,
        (y_edges[-1] - y_edges[0]) / ny,
        (z_edges[-1] - z_edges[0]) / nz,
    )

    # ---- resample: control how points outside the tetra mesh are handled ----
    # tolerance: search radius for nearest cell (0 = exact containment only)
    # snap_to_closest_point=False: points outside the mesh get 0 / NaN, not nearest extrapolation
    resampled = target_grid.sample(
        tec_body_grid,
        tolerance=0,
        snap_to_closest_point=False,
        mark_blank=True,           # marks outside-mesh points so we can zero them explicitly
    )

    Mx_v = np.nan_to_num(resampled.point_data["Mx"], nan=0.0)
    My_v = np.nan_to_num(resampled.point_data["My"], nan=0.0)
    Mz_v = np.nan_to_num(resampled.point_data["Mz"], nan=0.0)

    # "vtkValidPointMask" is 1 where the sample point fell inside the source mesh, 0 otherwise
    valid = resampled.point_data.get("vtkValidPointMask", np.ones_like(Mx_v))

    # ---- convert point-grid samples (corners of UniformGrid) into cell-centre dict ----
    # UniformGrid.sample() returns values at grid POINTS (nx+1 x ny+1 x nz+1).
    # We need values at CELL CENTRES (nx x ny x nz) to match mag_dict's (i,j,k) convention,
    # so resample a second time directly at the cell-centre coordinates.
    cc_points = np.array([[xc, yc, zc]
                          for zc in cz_centres
                          for yc in cy_centres
                          for xc in cx_centres])
    cc_poly = pv.PolyData(cc_points)
    cc_sampled = cc_poly.sample(
        tec_body_grid,
        tolerance=0,
        snap_to_closest_point=False,
        mark_blank=True,
    )
    Mx_cc = np.nan_to_num(cc_sampled.point_data["Mx"], nan=0.0)
    My_cc = np.nan_to_num(cc_sampled.point_data["My"], nan=0.0)
    Mz_cc = np.nan_to_num(cc_sampled.point_data["Mz"], nan=0.0)
    valid_cc = cc_sampled.point_data.get("vtkValidPointMask", np.ones_like(Mx_cc))

    have_mvh = mvh is not None
    if have_mvh:
        H_cc  = np.nan_to_num(cc_sampled.point_data["H"], nan=0.0)
        Hn_cc = np.nan_to_num(cc_sampled.point_data["H_normalized"], nan=0.0)
        Vx_cc = np.nan_to_num(cc_sampled.point_data["Vx"], nan=0.0)
        Vy_cc = np.nan_to_num(cc_sampled.point_data["Vy"], nan=0.0)
        Vz_cc = np.nan_to_num(cc_sampled.point_data["Vz"], nan=0.0)

    mag_dict = {}
    mvh_dict = {}
    idx = 0
    for k in range(1, nz + 1):
        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                if valid_cc[idx] > 0 and (abs(Mx_cc[idx]) + abs(My_cc[idx]) + abs(Mz_cc[idx])) > 0:
                    mag_dict[(i, j, k)] = (Mx_cc[idx], My_cc[idx], Mz_cc[idx])
                    if have_mvh:
                        mvh_dict[(i, j, k)] = (H_cc[idx], Hn_cc[idx], Vx_cc[idx], Vy_cc[idx], Vz_cc[idx])
                idx += 1

    if len(mag_dict) == 0:
        # Grain fell entirely between sample points (cell size > grain size).
        # Force-assign it to the single cell containing the grain's centroid,
        # using a VOLUME-WEIGHTED average direction (matches classify_domain_state
        # exactly) -- a simple node-mean gets skewed for non-uniform domain states
        # like Vortex, where magnetization direction varies across the grain.
        node_ids = np.unique(body_tetra.flatten())
        gx = X[node_ids].mean()
        gy = Y[node_ids].mean()
        gz = Z[node_ids].mean()

        v0 = points[body_tetra[:,0]]; v1 = points[body_tetra[:,1]]
        v2 = points[body_tetra[:,2]]; v3 = points[body_tetra[:,3]]
        a = v1-v0; b = v2-v0; c = v3-v0
        tetra_volumes = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0
        Mx_t = d['Mx'][body_tetra].mean(axis=1)
        My_t = d['My'][body_tetra].mean(axis=1)
        Mz_t = d['Mz'][body_tetra].mean(axis=1)

        V_grain = tetra_volumes.sum()
        gMx = np.sum(Mx_t * tetra_volumes) / V_grain
        gMy = np.sum(My_t * tetra_volumes) / V_grain
        gMz = np.sum(Mz_t * tetra_volumes) / V_grain

        i = min(max(np.searchsorted(x_edges, gx), 1), nx)
        j = min(max(np.searchsorted(y_edges, gy), 1), ny)
        k = min(max(np.searchsorted(z_edges, gz), 1), nz)

        dx_cell = x_edges[1] - x_edges[0]
        dy_cell = y_edges[1] - y_edges[0]
        dz_cell = z_edges[1] - z_edges[0]
        V_cell = dx_cell * dy_cell * dz_cell

        scale = V_grain / V_cell
        mag_dict[(i, j, k)] = (gMx*scale, gMy*scale, gMz*scale)
        print(f"  [fallback] Grain smaller than one cell -- force-assigned to cell ({i},{j},{k}), "
              f"V_grain/V_cell={scale:.2e} (volume-weighted, conserves true moment)")
        # Note: H/V not computed for the single-cell fallback case -- mvh_dict stays empty.

    print(f"  Occupied: {len(mag_dict)} / {nx*ny*nz} voxels  (PyVista sample(), tolerance=0.0)")
    return mag_dict, mvh_dict



def verify_voxelization(d, mag_dict, x_edges, y_edges, z_edges, Ms,
                         moment_tol_pct=5.0, angle_tol_deg=2.0):
    """
    Compares the TRUE magnetic moment/direction (computed directly from the
    raw tetrahedral grain) against the VOXELIZED moment/direction (computed
    from mag_dict after gridding), using the same convention as
    classify_domain_state(). Flags if voxelization lost moment or rotated
    the magnetization direction.
    """
    # ---- TRUE moment, from raw tetrahedra ----
    nodes = np.column_stack([d['X'], d['Y'], d['Z']])
    body_mask = d['SD'] == 101.0
    tetra = d['tetra'][body_mask]
    v0 = nodes[tetra[:,0]]; v1 = nodes[tetra[:,1]]
    v2 = nodes[tetra[:,2]]; v3 = nodes[tetra[:,3]]
    a = v1-v0; b = v2-v0; c = v3-v0
    volumes = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0
    Mx_t = d['Mx'][tetra].mean(axis=1)
    My_t = d['My'][tetra].mean(axis=1)
    Mz_t = d['Mz'][tetra].mean(axis=1)
    true_mx = Ms * np.sum(Mx_t * volumes)
    true_my = Ms * np.sum(My_t * volumes)
    true_mz = Ms * np.sum(Mz_t * volumes)
    true_mag = np.sqrt(true_mx**2 + true_my**2 + true_mz**2)
    true_inc = np.degrees(np.arcsin(np.clip(true_mz/(true_mag+1e-30), -1, 1)))
    true_dec = np.degrees(np.arctan2(true_my, true_mx)) % 360

    # ---- VOXELIZED moment, from mag_dict ----
    dx_cell = x_edges[1] - x_edges[0]
    dy_cell = y_edges[1] - y_edges[0]
    dz_cell = z_edges[1] - z_edges[0]
    V_cell = dx_cell * dy_cell * dz_cell
    vox_mx = Ms * sum(m[0]*V_cell for m in mag_dict.values())
    vox_my = Ms * sum(m[1]*V_cell for m in mag_dict.values())
    vox_mz = Ms * sum(m[2]*V_cell for m in mag_dict.values())
    vox_mag = np.sqrt(vox_mx**2 + vox_my**2 + vox_mz**2)
    vox_inc = np.degrees(np.arcsin(np.clip(vox_mz/(vox_mag+1e-30), -1, 1)))
    vox_dec = np.degrees(np.arctan2(vox_my, vox_mx)) % 360

    moment_err_pct = 100*abs(vox_mag - true_mag)/(true_mag + 1e-30)
    inc_err = abs(vox_inc - true_inc)
    dec_err = min(abs(vox_dec - true_dec), 360 - abs(vox_dec - true_dec))  # wraparound-safe

    status = "PASS" if (moment_err_pct < moment_tol_pct and inc_err < angle_tol_deg
                         and dec_err < angle_tol_deg) else "WARN"

    print(f"  [{status}] Voxelization check: moment true={true_mag:.4e} vox={vox_mag:.4e} "
          f"(err={moment_err_pct:.2f}%) | Inc true={true_inc:.1f} vox={vox_inc:.1f} (err={inc_err:.2f}°) "
          f"| Dec true={true_dec:.1f} vox={vox_dec:.1f} (err={dec_err:.2f}°)")

    return status == "PASS"

# ============================================================
#  CONVERT TEC -> TOMOFAST ROWS (for Paraview prism VTK)
#  Uses per-grain mesh edges
# ============================================================
def tec_to_tomofast(d, x_edges, y_edges, z_edges, nx_cells, ny_cells, nz_cells):
    """
    Each tetra -> bounding-box prism row for Paraview visualisation.
    Uses the per-grain mesh edges.
    """
    X, Y, Z = d['X'], d['Y'], d['Z']
    tetra   = d['tetra']
    E       = d['E']
    rows = []
    for e in range(E):
        node_ids = tetra[e]
        xs = X[node_ids]; ys = Y[node_ids]; zs = Z[node_ids]
        xmin_e, xmax_e = xs.min(), xs.max()
        ymin_e, ymax_e = ys.min(), ys.max()
        zmin_e, zmax_e = zs.min(), zs.max()
        cx = xs.mean(); cy = ys.mean(); cz = zs.mean()
        i = int(np.clip(np.searchsorted(x_edges, cx, side='right'), 1, nx_cells))
        j = int(np.clip(np.searchsorted(y_edges, cy, side='right'), 1, ny_cells))
        k = int(np.clip(np.searchsorted(z_edges, cz, side='right'), 1, nz_cells))
        rows.append([xmin_e, xmax_e, ymin_e, ymax_e, zmin_e, zmax_e, i, j, k])
    return np.array(rows)


# ============================================================
#  COMPUTE M, V, H - matching manual Paraview workflow
# ============================================================
def compute_MVH(d):
    nodes = np.column_stack([d['X'], d['Y'], d['Z']])
    tetra = d['tetra']
    E = d['E'];  N = d['N']
    M = np.column_stack([d['Mx'], d['My'], d['Mz']])
    curl_node  = np.zeros((N, 3))
    node_count = np.zeros(N)
    for e in range(E):
        n0, n1, n2, n3 = tetra[e]
        T = np.column_stack([nodes[n1]-nodes[n0],
                             nodes[n2]-nodes[n0],
                             nodes[n3]-nodes[n0]])
        det = np.linalg.det(T)
        if abs(det) < 1e-30:
            continue
        T_inv  = np.linalg.inv(T)
        M0=M[n0]; M1=M[n1]; M2=M[n2]; M3=M[n3]
        dM     = np.column_stack([M1-M0, M2-M0, M3-M0])
        grad_M = T_inv.T @ dM
        curl_e = np.array([
            grad_M[2,1] - grad_M[1,2],
            grad_M[0,2] - grad_M[2,0],
            grad_M[1,0] - grad_M[0,1],
        ])
        for nid in [n0, n1, n2, n3]:
            curl_node[nid] += curl_e
            node_count[nid] += 1
    nonzero = node_count > 0
    curl_node[nonzero] /= node_count[nonzero, np.newaxis]
    V = curl_node
    H = np.einsum('ij,ij->i', M, V)


    # normalized helicity: cos(angle between M and V), bounded [-1, 1],
    # independent of |M| and |V| magnitudes
    M_norm = np.linalg.norm(M, axis=1)
    V_norm = np.linalg.norm(V, axis=1)
    denom = M_norm * V_norm
    H_normalized = np.divide(H, denom, out=np.zeros_like(H), where=denom > 1e-30)

    print(f"  H: min={H.min():.3f}  max={H.max():.3f}  mean={H.mean():.3f}")
    print(f"  H_normalized: min={H_normalized.min():.3f}  max={H_normalized.max():.3f}  mean={H_normalized.mean():.3f}")
    return dict(M=M, V=V, H=H, H_normalized=H_normalized)


# ============================================================
#  CLASSIFY DOMAIN STATE
# ============================================================
def classify_domain_state(d):
    nodes = np.column_stack([d['X'], d['Y'], d['Z']])
    #tetra = d['tetra']
    body_mask = d['SD'] == 101.0
    tetra = d['tetra'][body_mask]
    v0=nodes[tetra[:,0]]; v1=nodes[tetra[:,1]]
    v2=nodes[tetra[:,2]]; v3=nodes[tetra[:,3]]
    a=v1-v0; b=v2-v0; c=v3-v0
    volumes = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0
    Mx_t = d['Mx'][tetra].mean(axis=1)
    My_t = d['My'][tetra].mean(axis=1)
    Mz_t = d['Mz'][tetra].mean(axis=1)
    V_total = volumes.sum()
    mu_x = Ms*np.sum(Mx_t*volumes)
    mu_y = Ms*np.sum(My_t*volumes)
    mu_z = Ms*np.sum(Mz_t*volumes)
    mu_mag = np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
    Q   = mu_mag / (Ms * V_total + 1e-30)
    inc = np.degrees(np.arcsin(np.clip(mu_z/(mu_mag+1e-30), -1, 1)))
    #dec = np.degrees(np.arctan2(mu_y, mu_x))
    dec = np.degrees(np.arctan2(mu_y, mu_x)) % 360
    if Q > 0.90:   state = "Single Domain (SD)"
    elif Q > 0.75: state = "Flower / SD-like"
    elif Q > 0.30: state = "PSD"
    elif Q > 0.05: state = "Vortex"
    else:          state = "Multi-Domain (MD)"
    print(f"  Domain: {state}  Q={Q:.3f}  Inc={inc:.1f}°  Dec={dec:.1f}°")
    return state, Q, inc, dec


def compare_averaging_schemes(d, body_tetra, points):
    """One-off diagnostic: plain mean vs inverse-distance-weighted per-tetra
    magnetization, and the resulting net moment difference. Does not modify
    any pipeline output -- print-only."""
    Mx, My, Mz = d['Mx'], d['My'], d['Mz']

    v0 = points[body_tetra[:,0]]; v1 = points[body_tetra[:,1]]
    v2 = points[body_tetra[:,2]]; v3 = points[body_tetra[:,3]]
    volumes = np.abs(np.einsum('ij,ij->i', v1-v0, np.cross(v2-v0, v3-v0))) / 6.0

    # --- plain mean ---
    Mx_plain = Mx[body_tetra].mean(axis=1)
    My_plain = My[body_tetra].mean(axis=1)
    Mz_plain = Mz[body_tetra].mean(axis=1)

    # --- inverse-distance weighted (paper's method) ---
    centroid = (v0 + v1 + v2 + v3) / 4.0
    verts = np.stack([v0, v1, v2, v3], axis=1)
    dist = np.linalg.norm(verts - centroid[:, None, :], axis=2)
    dist = np.maximum(dist, 1e-12)
    w = 1.0 / dist
    w /= w.sum(axis=1, keepdims=True)

    Mnodal = np.stack([Mx[body_tetra], My[body_tetra], Mz[body_tetra]], axis=2)
    Mweighted = np.einsum('ec,ecd->ed', w, Mnodal)
    Mx_wt, My_wt, Mz_wt = Mweighted[:,0], Mweighted[:,1], Mweighted[:,2]

    m_plain = Ms * np.array([np.sum(Mx_plain*volumes), np.sum(My_plain*volumes), np.sum(Mz_plain*volumes)])
    m_wt    = Ms * np.array([np.sum(Mx_wt*volumes),    np.sum(My_wt*volumes),    np.sum(Mz_wt*volumes)])

    ang_diff = np.degrees(np.arccos(np.clip(np.dot(m_plain, m_wt)/(np.linalg.norm(m_plain)*np.linalg.norm(m_wt)), -1, 1)))
    amp_diff = 100*abs(np.linalg.norm(m_wt) - np.linalg.norm(m_plain)) / np.linalg.norm(m_plain)

    print(f"  [averaging check] plain vs weighted: angular diff={ang_diff:.4f}°, amplitude diff={amp_diff:.4f}%")


# ============================================================
#  BUILD TEC PYVISTA GRID
# ============================================================
def build_tec_grid(d, mvh):
    points    = np.column_stack([d['X'], d['Y'], -d['Z']]) #flip z becuase tomofast is z potivite down but MERRILL is z positive up
    tetra     = d['tetra'];  E = d['E']
    cells     = np.hstack([np.full((E,1),4), tetra]).astype(np.int64).flatten()
    celltypes = np.full(E, 10)
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.point_data["Mx"] = d['Mx']
    grid.point_data["My"] = d['My']
    grid.point_data["Mz"] = -d['Mz']#flip z becuase tomofast is z potivite 'down' but MERRILL is z positive up
    grid.point_data["M"]  = mvh['M']
    grid.point_data["M"]  = np.column_stack([d['Mx'], d['My'], -d['Mz']]) #flip z becuase tomofast is z potivite down but MERRILL is z positive up
    grid.point_data["V"]  = mvh['V']
    grid.point_data["H"]  = mvh['H']
    grid.point_data["H_normalized"] = mvh['H_normalized']
    grid.cell_data["SD"]  = d['SD']
    grid.cell_data["M_magnitude"] = np.sqrt(
        d['Mx'][tetra].mean(axis=1)**2 +
        d['My'][tetra].mean(axis=1)**2 +
        d['Mz'][tetra].mean(axis=1)**2)
    return grid

# ============================================================
#  BUILD TOMOFAST PYVISTA GRID
# ============================================================
def build_tomo_grid(rows, d, mvh):
    E = len(rows)
    points=[]; cells=[]; celltypes=[]
    ijk_i=[]; ijk_j=[]; ijk_k=[]
    tetra   = d['tetra']
    Mx_cell = d['Mx'][tetra].mean(axis=1)
    My_cell = d['My'][tetra].mean(axis=1)
    Mz_cell = d['Mz'][tetra].mean(axis=1)
    Mmag    = np.sqrt(Mx_cell**2 + My_cell**2 + Mz_cell**2)
    Vx_cell = mvh['V'][:,0][tetra].mean(axis=1) if mvh['V'].shape[0]==d['N'] else np.zeros(E)
    Vy_cell = mvh['V'][:,1][tetra].mean(axis=1) if mvh['V'].shape[0]==d['N'] else np.zeros(E)
    Vz_cell = mvh['V'][:,2][tetra].mean(axis=1) if mvh['V'].shape[0]==d['N'] else np.zeros(E)
    H_cell  = mvh['H'][tetra].mean(axis=1)       if mvh['H'].shape[0]==d['N'] else np.zeros(E)
    Hn_cell = mvh['H_normalized'][tetra].mean(axis=1) if mvh['H_normalized'].shape[0]==d['N'] else np.zeros(E)

    for idx, row in enumerate(rows):
        xmin,xmax,ymin,ymax,zmin,zmax = row[0],row[1],row[2],row[3],row[4],row[5]
        i,j,k = int(row[6]),int(row[7]),int(row[8])
        start = len(points)
        pts = [[xmin,ymin,-zmin],[xmax,ymin,-zmin],[xmax,ymax,-zmin],[xmin,ymax,-zmin], #flip z becuase tomofast is z potivite down but MERRILL is z positive up
               [xmin,ymin,-zmax],[xmax,ymin,-zmax],[xmax,ymax,-zmax],[xmin,ymax,-zmax]]
        points.extend(pts)
        cells.append([8]+list(range(start, start+8)))
        celltypes.append(12)
        ijk_i.append(i); ijk_j.append(j); ijk_k.append(k)
    points=np.array(points); cells=np.array(cells).flatten(); celltypes=np.array(celltypes)
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.cell_data["i_x"]=np.array(ijk_i); grid.cell_data["j_y"]=np.array(ijk_j)
    grid.cell_data["k_z"]=np.array(ijk_k); grid.cell_data["SD"]=d['SD']
    grid.cell_data["Mx"]=Mx_cell; grid.cell_data["My"]=My_cell; grid.cell_data["Mz"]=-Mz_cell #flip z becuase tomofast is z potivite down but MERRILL is z positive up
    grid.cell_data["M_magnitude"]=Mmag
    grid.cell_data["M"]=np.column_stack([Mx_cell,My_cell,-Mz_cell])#flip z becuase tomofast is z potivite down but MERRILL is z positive up
    grid.cell_data["V"]=np.column_stack([Vx_cell,Vy_cell,Vz_cell])
    grid.cell_data["H"]=H_cell
    grid.cell_data["H_normalized"]=Hn_cell
    return grid

def build_regular_grid_vtk(x_edges, y_edges, z_edges, mag_dict, nx, ny, nz):
    """
    Build VTK from the actual Tomofast regular voxel grid - ALL cells, including empty (zero) ones.
    This matches exactly what Meshgrid_*.txt and Magnetization_*.txt contain.
    """
    points = []; cells = []; celltypes = []
    mx_arr = []; my_arr = []; mz_arr = []; mmag_arr = []
    for k in range(1, nz+1):
        for j in range(1, ny+1):
            for i in range(1, nx+1):
                key = (i, j, k)
                mx, my, mz = mag_dict.get(key, (0.0, 0.0, 0.0))  # 0 if not in grain
                start = len(points)
                pts = [
                    [x_edges[i-1], y_edges[j-1], -z_edges[k-1]],#flip z becuase tomofast is z potivite down but MERRILL is z positive up
                    [x_edges[i],   y_edges[j-1], -z_edges[k-1]],
                    [x_edges[i],   y_edges[j],   -z_edges[k-1]],
                    [x_edges[i-1], y_edges[j],   -z_edges[k-1]],
                    [x_edges[i-1], y_edges[j-1], -z_edges[k]  ],
                    [x_edges[i],   y_edges[j-1], -z_edges[k]  ],
                    [x_edges[i],   y_edges[j],   -z_edges[k]  ],
                    [x_edges[i-1], y_edges[j],   -z_edges[k]  ],
                ]
                points.extend(pts)
                cells.append([8] + list(range(start, start+8)))
                celltypes.append(12)
                mx_arr.append(mx); my_arr.append(my); mz_arr.append(-mz)#flip z becuase tomofast is z potivite down but MERRILL is z positive up
                mmag_arr.append(np.sqrt(mx**2+my**2+mz**2))
    points = np.array(points)
    cells  = np.array(cells).flatten()
    celltypes = np.array(celltypes)
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.cell_data["Mx"] = np.array(mx_arr)
    grid.cell_data["My"] = np.array(my_arr)
    grid.cell_data["Mz"] = np.array(mz_arr)
    grid.cell_data["M_magnitude"] = np.array(mmag_arr)
    print("x_edges diffs:", np.diff(x_edges).min(), np.diff(x_edges).max())
    print("y_edges diffs:", np.diff(y_edges).min(), np.diff(y_edges).max())
    print("z_edges diffs:", np.diff(z_edges).min(), np.diff(z_edges).max())
    return grid


def build_voxel_mesh_occupied_vtk(x_edges, y_edges, z_edges, mag_dict, mvh_dict=None):
    """
    Build VTK showing only OCCUPIED voxels, using the true cubic
    Tomofast cell size (dx=dy=dz). If mvh_dict is provided, also writes
    H, H_normalized, and V as cell data (falls back to 0 for cells not
    present in mvh_dict, e.g. the single-cell fallback case).
    """
    if mvh_dict is None:
        mvh_dict = {}

    points = []; cells = []; celltypes = []
    mx_arr = []; my_arr = []; mz_arr = []
    H_arr = []; Hn_arr = []; Vx_arr = []; Vy_arr = []; Vz_arr = []
    for (i, j, k), (mx, my, mz) in mag_dict.items():
        start = len(points)
        pts = [
            [x_edges[i-1], y_edges[j-1], -z_edges[k-1]],
            [x_edges[i],   y_edges[j-1], -z_edges[k-1]],
            [x_edges[i],   y_edges[j],   -z_edges[k-1]],
            [x_edges[i-1], y_edges[j],   -z_edges[k-1]],
            [x_edges[i-1], y_edges[j-1], -z_edges[k]  ],
            [x_edges[i],   y_edges[j-1], -z_edges[k]  ],
            [x_edges[i],   y_edges[j],   -z_edges[k]  ],
            [x_edges[i-1], y_edges[j],   -z_edges[k]  ],
        ]
        points.extend(pts)
        cells.append([8] + list(range(start, start+8)))
        celltypes.append(12)
        mx_arr.append(mx); my_arr.append(my); mz_arr.append(-mz)
        Hval, Hnval, vx, vy, vz = mvh_dict.get((i, j, k), (0.0, 0.0, 0.0, 0.0, 0.0))
        H_arr.append(Hval); Hn_arr.append(Hnval)
        Vx_arr.append(vx); Vy_arr.append(vy); Vz_arr.append(-vz)  # -vz to match the Z-flip used elsewhere

    points = np.array(points)
    cells  = np.array(cells, dtype=np.int64).flatten()
    celltypes = np.array(celltypes)
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    grid.cell_data["Mx"] = np.array(mx_arr)
    grid.cell_data["My"] = np.array(my_arr)
    grid.cell_data["Mz"] = np.array(mz_arr)
    grid.cell_data["M"] = np.column_stack([mx_arr, my_arr, mz_arr])
    grid.cell_data["M_magnitude"] = np.sqrt(np.array(mx_arr)**2 + np.array(my_arr)**2 + np.array(mz_arr)**2)
    grid.cell_data["H"] = np.array(H_arr)
    grid.cell_data["H_normalized"] = np.array(Hn_arr)
    grid.cell_data["V"] = np.column_stack([Vx_arr, Vy_arr, Vz_arr])
    # ------------------------------------------------------------
    # write M and M_magnitude as POINT data, since
    # ------------------------------------------------------------
    #point_grid = grid.cell_data_to_point_data()
    #grid.point_data["M"] = point_grid.point_data["M"]
    #grid.point_data["M_magnitude"] = point_grid.point_data["M_magnitude"]

    return grid
# ============================================================
#  MATPLOTLIB SLICE PLOTS
# ============================================================
def plot_slices(d, rows, title, save_path):
    X,Y,Z = d['X'],d['Y'],d['Z']
    Mmag_nodes = np.sqrt(d['Mx']**2 + d['My']**2 + d['Mz']**2)
    px=(rows[:,0]+rows[:,1])/2; py=(rows[:,2]+rows[:,3])/2; pz=(rows[:,4]+rows[:,5])/2
    tetra=d['tetra']
    Mmag_p = np.sqrt(d['Mx'][tetra].mean(axis=1)**2 +
                     d['My'][tetra].mean(axis=1)**2 +
                     d['Mz'][tetra].mean(axis=1)**2)

    # shared colour scale across both rows
    vmin = min(Mmag_nodes.min(), Mmag_p.min())
    vmax = max(Mmag_nodes.max(), Mmag_p.max())
    cmap = 'viridis'

    fig,axes = plt.subplots(2,3,figsize=(15,8)); fig.suptitle(title,fontsize=12)
    views=[("XY",X,Y,"X","Y"),("XZ",X,Z,"X","Z"),("YZ",Y,Z,"Y","Z")]
    for col,(label,a,b,xl,yl) in enumerate(views):
        sc=axes[0,col].scatter(a,b,c=Mmag_nodes,cmap=cmap,s=2,vmin=vmin,vmax=vmax)
        axes[0,col].set_xlabel(xl); axes[0,col].set_ylabel(yl)
        axes[0,col].set_title(f"TEC - {label}")
        plt.colorbar(sc,ax=axes[0,col],label="|M|")
    prism_coords=[(px,py,"X","Y"),(px,pz,"X","Z"),(py,pz,"Y","Z")]
    for col,(a,b,xl,yl) in enumerate(prism_coords):
        sc=axes[1,col].scatter(a,b,c=Mmag_p,cmap=cmap,s=4,marker='s',vmin=vmin,vmax=vmax)
        axes[1,col].set_xlabel(xl); axes[1,col].set_ylabel(yl)
        axes[1,col].set_title(f"TOMOFAST - {views[col][0]}")
        plt.colorbar(sc,ax=axes[1,col],label="|M|")
    plt.tight_layout(); plt.savefig(save_path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Slices saved       : {save_path.name}")
# ============================================================
#  SIDE-BY-SIDE 3D PLOT
# ============================================================
def plot_both_3d(tec_grid, tomo_grid, title):
    # compute shared colour scale from M_magnitude on both grids
    mag_tec  = tec_grid.cell_data["M_magnitude"]
    mag_tomo = tomo_grid.cell_data["M_magnitude"]
    vmin = min(mag_tec.min(), mag_tomo.min())
    vmax = max(mag_tec.max(), mag_tomo.max())
    cmap = "viridis"

    plotter = pv.Plotter(shape=(1,2), title=title)
    plotter.subplot(0,0)
    plotter.add_text("TEC (tetrahedral)", font_size=10)
    plotter.add_mesh(tec_grid, scalars="M_magnitude", show_edges=True,
                     opacity=0.7, cmap=cmap, clim=[vmin, vmax])
    plotter.add_axes(); plotter.show_grid()
    plotter.subplot(0,1)
    plotter.add_text("TOMOFAST (prisms)", font_size=10)
    plotter.add_mesh(tomo_grid, scalars="M_magnitude", show_edges=True,
                     opacity=0.7, cmap=cmap, clim=[vmin, vmax])
    plotter.add_axes(); plotter.show_grid()
    plotter.link_views(); plotter.show()
# ============================================================
#  TOMOFAST FILE 1: MESHGRID
#  ALL nx*ny*nz cells, x-fastest order
#  Format: N \n xmin xmax ymin ymax zmin zmax i j k
# ============================================================
def save_tomofast_meshgrid(x_edges, y_edges, z_edges, nx, ny, nz, out_path):
    N = nx * ny * nz
    with open(out_path, 'w') as f:
        f.write(f"{N}\n")
        for k in range(1, nz+1):
            for j in range(1, ny+1):
                for i in range(1, nx+1):
                    f.write(f"{x_edges[i-1]} {x_edges[i]} "
                            f"{y_edges[j-1]} {y_edges[j]} "
                            f"{z_edges[k-1]} {z_edges[k]} "
                            f"{i} {j} {k}\n")
    print(f"  [1] Meshgrid        : {out_path.name}  ({N} = {nx}x{ny}x{nz})")


# ============================================================
#  TOMOFAST FILE 2: MAGNETISATION
#  ALL nx*ny*nz cells, same x-fastest order
#  Format: N \n Mx My Mz  (0 0 0 outside grain)
# ============================================================
def save_tomofast_magnetisation(x_edges, y_edges, z_edges, nx, ny, nz,
                                 mag_dict, out_path):
    N = nx * ny * nz
    with open(out_path, 'w') as f:
        f.write(f"{N}\n")
        for k in range(1, nz+1):
            for j in range(1, ny+1):
                for i in range(1, nx+1):
                    key = (i, j, k)
                    if key in mag_dict:
                        mx, my, mz = mag_dict[key]
                        mx *= Ms;  my *= Ms;  mz *= Ms
                    else:
                        mx = my = mz = 0.0
                    f.write(f"{mx:.16e} {my:.16e} {mz:.16e}\n")
    print(f"  [2] Magnetisation   : {out_path.name}  ({N} cells, "
          f"{len(mag_dict)} non-zero)")



# ============================================================
#  TOMOFAST FILE 3: OBS
#  nx*ny sensors at XY cell centres, x-fastest order
#  Format: N \n X Y Z Bz or bz by bz
# ============================================================
#def save_tomofast_obs(x_edges, y_edges, z_edges, nx, ny, out_path):
def save_tomofast_obs(x_edges, y_edges, nx, ny, sensor_z, out_path):
    #sensor_z = z_edges[0] - SENSOR_Z_OFFSET
    #sensor_z = h
    #sensor_z = d['Z'].max() + SENSOR_HEIGHT
    #sensor_z = np.max(z_edges) + SENSOR_Z_OFFSET
    N = nx * ny
    with open(out_path, 'w') as f:
        f.write(f"{N}\n")
        for j in range(1, ny+1):
            for i in range(1, nx+1):
                xc = (x_edges[i-1] + x_edges[i]) / 2.0
                yc = (y_edges[j-1] + y_edges[j]) / 2.0
                #f.write(f"{xc:.6f} {yc:.6f} {sensor_z:.6f} 0.000000\n") #x y z bz
                f.write(f"{xc:.6f} {yc:.6f} {sensor_z:.6f} 0.000000 0.000000 0.000000\n") # x y z bx by bz
    print(f"  [3] Obs             : {out_path.name}  ({N} sensors, Z={sensor_z:.4f})")
    return N


def save_tomofast_obs_bz(x_edges, y_edges, nx, ny, sensor_z, out_path):
    N = nx * ny
    with open(out_path, 'w') as f:
        f.write(f"{N}\n")
        for j in range(1, ny+1):
            for i in range(1, nx+1):
                xc = (x_edges[i-1] + x_edges[i]) / 2.0
                yc = (y_edges[j-1] + y_edges[j]) / 2.0
                f.write(f"{xc:.6f} {yc:.6f} {sensor_z:.6f} 0.000000\n")  # x y z bz (placeholder)
    print(f"  [3b] Obs (Bz only) : {out_path.name}  ({N} sensors, Z={sensor_z:.4f})")
    return N
# ============================================================
#  TOMOFAST FILE 4: PARFILE
# ============================================================
def save_tomofast_parfile(stem, nx, ny, nz, n_sensors,
                          mesh_path, mag_path, obs_path,
                          #sensit_dir, #output_dir, 
                          out_path):
    compression_rate = calc_ideal_compression_rate(nx, ny, nz)   # Eq. 10, Bruce et al. 2025
    content = f"""==================================================================================
GLOBAL
==================================================================================
global.outputFolderPath         = ./output/TEC_{stem}
global.description              = Forward model from MERRILL grain: {stem}
#global.zAxisDirection = -1

==================================================================================
MODEL GRID parameters
==================================================================================
modelGrid.size                        = {nx} {ny} {nz}
modelGrid.magn.file                   = data/TEC/Meshgrid_{stem}.txt
modelGrid.magn.nModelComponents       = 3

==================================================================================
DATA parameters
==================================================================================
forward.data.magn.nData                = {n_sensors}
forward.data.magn.dataGridFile         = data/TEC/Forward_{stem}.obs
forward.data.magn.nDataComponents      = 3

==================================================================================
MAGNETIC FIELD constants
==================================================================================
forward.magneticField.inclination      = 90
forward.magneticField.declination      = 0
forward.magneticField.intensity_nT     = 0
forward.magneticField.XaxisDeclination = 0.d0

==================================================================================
DEPTH WEIGHTING
==================================================================================
forward.depthWeighting.type            = 1 
forward.depthWeighting.magn.Z0         = {SENSOR_HEIGHT}

==================================================================================
SENSITIVITY KERNEL
==================================================================================
sensit.readFromFiles                   = 0
sensit.folderPath                      = ./output/TEC_{stem}/SENSIT/

==================================================================================
MATRIX COMPRESSION
==================================================================================
forward.matrixCompression.type         = 1
forward.matrixCompression.rate         = {compression_rate:.5f}

==================================================================================
PRIOR MODEL
==================================================================================
inversion.priorModel.type              = 2
inversion.priorModel.magn.file         = data/TEC/Magnetization_{stem}.txt

#inversion.priorModel.type                  = 1
#inversion.priorModel.magn.value            = 0.000000

==================================================================================
INVERSION parameters
==================================================================================
inversion.nMajorIterations             = 0
inversion.nMinorIterations             = 100

==================================================================================
MODEL DAMPING (m - m_prior)
==================================================================================
inversion.modelDamping.magn.weight     = 0.0

==================================================================================
JOINT INVERSION parameters
==================================================================================
inversion.joint.grav.problemWeight     = 0.00000000
inversion.joint.magn.problemWeight     = 1.00000000
"""
    with open(out_path, 'w') as f:
        f.write(content)
    print(f"  [4] Parfile         : {out_path.name}  (compression rate={compression_rate:.5f})")




def save_tomofast_parfile_inversion(stem, nx, ny, nz, n_sensors,
                                        mesh_path, mag_path, obs_bz_path,
                                        #sensit_dir, #output_dir, 
                                        out_path):
    compression_rate = calc_ideal_compression_rate(nx, ny, nz)   # Eq. 10, Bruce et al. 2025
    content = f"""==================================================================================
GLOBAL
==================================================================================
global.outputFolderPath         = ./output/TEC_{stem}_inversion
global.description              = inversion from MERRILL grain: {stem}

==================================================================================
MODEL GRID parameters
==================================================================================
modelGrid.size                        = {nx} {ny} {nz}
modelGrid.magn.file                   = data/TEC/Meshgrid_{stem}.txt
modelGrid.magn.nModelComponents       = 3

==================================================================================
DATA parameters
==================================================================================
forward.data.magn.nData                = {n_sensors}
forward.data.magn.dataGridFile         = data/TEC/Forward_{stem}.obs
#forward.data.magn.dataValuesFile       = data/TEC/Forward_{stem}.obs
forward.data.magn.nDataComponents      = 3

==================================================================================
MAGNETIC FIELD constants
==================================================================================
forward.magneticField.inclination      = 90
forward.magneticField.declination      = 0
forward.magneticField.intensity_nT     = 0
forward.magneticField.XaxisDeclination = 0.d0

==================================================================================
DEPTH WEIGHTING
==================================================================================
forward.depthWeighting.type            = 1
forward.depthWeighting.magn.Z0         = {SENSOR_HEIGHT}
==================================================================================
SENSITIVITY KERNEL
==================================================================================
sensit.readFromFiles                   = 0
sensit.folderPath                      = ./output/TEC_{stem}_inversion/SENSIT/

==================================================================================
MATRIX COMPRESSION
==================================================================================
forward.matrixCompression.type         = 1
forward.matrixCompression.rate         = {compression_rate:.5f}

==================================================================================
PRIOR MODEL
==================================================================================
inversion.priorModel.type              = 1
inversion.priorModel.magn.value        = 0.000000

==================================================================================
INVERSION parameters
==================================================================================
inversion.nMajorIterations             = 3
inversion.nMinorIterations             = 200

==================================================================================
MODEL DAMPING (m - m_prior)
==================================================================================
inversion.modelDamping.magn.weight     = 0.0

==================================================================================
JOINT INVERSION parameters
==================================================================================
inversion.joint.grav.problemWeight     = 0.00000000
inversion.joint.magn.problemWeight     = 1.00000000
"""
    with open(out_path, 'w') as f:
        f.write(content)
    print(f"  [4b] Parfile inversion  : {out_path.name}  (compression rate={compression_rate:.5f})")





def save_tomofast_parfile_fitting(stem, n_sensors, out_path):
    content = f"""==================================================================================
GLOBAL
==================================================================================
global.outputFolderPath         = ./output/TEC_{stem}_fitting
global.description              = inversion from MERRILL grain: {stem}

==================================================================================
MODEL GRID parameters
==================================================================================
modelGrid.size                        = 1 1 1
modelGrid.magn.file                   = data/TEC/Meshgrid_{stem}.txt
modelGrid.magn.nModelComponents       = 3

==================================================================================
DATA parameters
==================================================================================
forward.data.magn.nData                = {n_sensors}
forward.data.magn.dataGridFile         = data/TEC/Forward_{stem}.obs
#forward.data.magn.dataValuesFile       = data/TEC/Forward_{stem}.obs
forward.data.magn.nDataComponents      = 3

==================================================================================
MAGNETIC FIELD constants
==================================================================================
forward.magneticField.inclination      = 90
forward.magneticField.declination      = 0
forward.magneticField.intensity_nT     = 0
forward.magneticField.XaxisDeclination = 0.d0

==================================================================================
DEPTH WEIGHTING
==================================================================================
forward.depthWeighting.type            = 1
forward.depthWeighting.magn.Z0         = {SENSOR_HEIGHT}

==================================================================================
SENSITIVITY KERNEL
==================================================================================
sensit.readFromFiles                   = 0
sensit.folderPath                      = ./output/TEC_{stem}_inversion/SENSIT/

==================================================================================
MATRIX COMPRESSION
==================================================================================
forward.matrixCompression.type         = 0
forward.matrixCompression.rate         = 0.02

==================================================================================
PRIOR MODEL
==================================================================================
inversion.priorModel.type              = 1
inversion.priorModel.magn.value        = 0.000000

==================================================================================
INVERSION parameters
==================================================================================
inversion.nMajorIterations             = 3
inversion.nMinorIterations             = 100

==================================================================================
MODEL DAMPING (m - m_prior)
==================================================================================
inversion.modelDamping.magn.weight     = 0.0

==================================================================================
JOINT INVERSION parameters
==================================================================================
inversion.joint.grav.problemWeight     = 0.00000000
inversion.joint.magn.problemWeight     = 1.00000000
"""
    with open(out_path, 'w') as f:
        f.write(content)
    print(f"  [4c] Parfile fitting    : {out_path.name}")
# ============================================================
#  MAIN LOOP
# ============================================================
tec_files = sorted(tec_dir.glob("*.tec"))
print(f"Found {len(tec_files)} .tec file(s) in {tec_dir}\n")

#for tec_file in tec_files:
    #stem = tec_file.stem
for tec_file in tec_files:
    stem = tec_file.stem
    stem = f"{stem}_h{SENSOR_HEIGHT:.3f}"   # e.g. body_118_24_mult_h0.010 -- distinct body per height
    #stem = f"{stem}_h{SENSOR_HEIGHT:.3f}_c{COMPRESSION_RATE:.3f}"   # e.g. body_118_24_mult_h0.010_c0.020 -- distinct body per height AND compression rate
    print(f"{'='*60}")
    print(f"Processing: {tec_file.name}")

    # Read
    d = read_tec(tec_file)
    print(f"  Nodes: {d['N']}  Elements: {d['E']}")
    print(f"  X: [{d['X'].min():.4f}, {d['X'].max():.4f}]")
    print(f"  Y: [{d['Y'].min():.4f}, {d['Y'].max():.4f}]")
    print(f"  Z: [{d['Z'].min():.4f}, {d['Z'].max():.4f}]")

    # Domain state
    classify_domain_state(d)
    
     # Diagnostic: does averaging scheme (plain mean vs inverse-distance) matter?
    body_tetra_check = d['tetra'][d['SD'] == 101.0]
    points_check = np.column_stack([d['X'], d['Y'], d['Z']])
    compare_averaging_schemes(d, body_tetra_check, points_check)

    # Build per-grain mesh
    x_edges, y_edges, z_edges, nx, ny, nz = build_grain_mesh(d)
    plot_mesh_map(d, x_edges, y_edges, z_edges, title=stem, save_path=out_dir / f"{stem}_mesh_map.png")

    # Compute M, V, H for Paraview (moved up so it's available for voxelization)
    mvh = compute_MVH(d)

    # Map tetra to voxels (now also resamples H/H_normalized/V for the voxel VTU)
    mag_dict, mvh_dict = map_tetra_to_voxels(d, x_edges, y_edges, z_edges, nx, ny, nz, mvh=mvh)
    verify_voxelization(d, mag_dict, x_edges, y_edges, z_edges, Ms)


    # Build full regular grid VTK (all cells, including empty) - matches actual Tomofast input
    #tomo_grid_full = build_regular_grid_vtk(x_edges, y_edges, z_edges, mag_dict, nx, ny, nz)


    # Convert to prism rows (uses per-grain mesh)
    rows = tec_to_tomofast(d, x_edges, y_edges, z_edges, nx, ny, nz)

    # Build Paraview grids
    tec_grid  = build_tec_grid(d, mvh)
    #tomo_grid = build_tomo_grid(rows, d, mvh)

    # NEW: voxel mesh, occupied cells only, true cubic cell size
    tomo_grid_voxel = build_voxel_mesh_occupied_vtk(x_edges, y_edges, z_edges, mag_dict, mvh_dict)

    # Plots
    if PLOT_TEC or PLOT_TOMOFAST:
        plot_both_3d(tec_grid, tomo_grid, title=stem)
    if PLOT_SLICES:
        plot_slices(d, rows, title=stem,
                    save_path=out_dir / f"{stem}_slices.png")

    # VTK files
    if SAVE_VTK_TEC:
        p = out_dir / f"{stem}_tec.vtu"
        tec_grid.save(str(p))
        print(f"  TEC vtu            : {p.name}")
    if SAVE_VTK_TOMO:
        #p = out_dir / f"{stem}_tomofast.vtu"
        #tomo_grid.save(str(p))
        #print(f"  TOMOFAST vtu       : {p.name}")

        #p2 = out_dir / f"{stem}_tomofast_full.vtu"
        #tomo_grid_full.save(str(p2))
        #print(f"  TOMOFAST full vtu  : {p2.name}")

        p3 = out_dir / f"{stem}_tomofast_voxel.vtu"
        tomo_grid_voxel.save(str(p3))
        print(f"  TOMOFAST voxel vtu : {p3.name}")

    # Four Tomofast forward files - per grain, auto-sized
    if SAVE_TOMOFAST_FILES:
        grain_dir = out_dir / "tomofast_forward" / stem
        grain_dir.mkdir(parents=True, exist_ok=True)
        #sensit_dir = grain_dir / "SENSIT";  sensit_dir.mkdir(exist_ok=True)
        #output_dir = grain_dir / "output";  output_dir.mkdir(exist_ok=True)

        mesh_path = grain_dir / f"Meshgrid_{stem}.txt"
        mag_path  = grain_dir / f"Magnetization_{stem}.txt"
        obs_path  = grain_dir / f"Forward_{stem}.obs"
        par_path  = grain_dir / f"Parfile_{stem}.txt"
        obs_bz_path = grain_dir / f"Forward_{stem}_bz.obs"
        par_bz_path = grain_dir / f"Parfile_{stem}_inversion.txt"
        par_fit_path = grain_dir / f"Parfile_{stem}_fitting.txt"

        save_tomofast_meshgrid(x_edges, y_edges, z_edges, nx, ny, nz, mesh_path)
        save_tomofast_magnetisation(x_edges, y_edges, z_edges, nx, ny, nz,
                                     mag_dict, mag_path)
        #n_sensors = save_tomofast_obs(x_edges, y_edges, z_edges, nx, ny, obs_path)
        #sensor_z = float(d['Z'].max()) + SENSOR_HEIGHT
        sensor_z = float(d['Z'].min()) - SENSOR_HEIGHT #flip for tomofast convention (Z downwards, sensors below sample)
        #sensor_z = float(z_edges[-1]) + SENSOR_HEIGHT
        n_sensors = save_tomofast_obs(x_edges, y_edges, nx, ny, sensor_z, obs_path)
        n_sensors_bz = save_tomofast_obs_bz(x_edges, y_edges, nx, ny, sensor_z, obs_bz_path)



        save_tomofast_parfile(
            stem=stem, nx=nx, ny=ny, nz=nz, n_sensors=n_sensors,
            mesh_path=str(mesh_path.resolve()),
            mag_path=str(mag_path.resolve()),
            obs_path=str(obs_path.resolve()),
            #sensit_dir=str(sensit_dir.resolve()) + "/",
            #output_dir=str(output_dir.resolve()),
            out_path=par_path,
        )

        save_tomofast_parfile_inversion(
            stem=stem, nx=nx, ny=ny, nz=nz, n_sensors=n_sensors,
            mesh_path=str(mesh_path.resolve()),
            mag_path=str(mag_path.resolve()),
            obs_bz_path=str(obs_bz_path.resolve()),
            #sensit_dir=str(sensit_dir.resolve()) + "/",
            #output_dir=str(output_dir.resolve()),
            out_path=par_bz_path,
        )

        save_tomofast_parfile_fitting(
            stem=stem, n_sensors=n_sensors,
            out_path=par_fit_path,
        )


        n_total = nx * ny * nz
        print(f"\n  File sizes (must all match):")
        print(f"    Mesh rows     = {n_total}  ({nx}x{ny}x{nz})")
        print(f"    Mag rows      = {n_total}  ")
        print(f"    Sensor rows   = {n_sensors}  ({nx}x{ny})")
        print(f"    Grain cells   = {len(mag_dict)}  non-zero")
        print(f"    Empty cells   = {n_total - len(mag_dict)}  zeros")

    print()

print("All done.")
print()