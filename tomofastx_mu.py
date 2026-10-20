"""
Tomofast-x-µ functions.

This file holds the code that used to sit inside the notebook cells.
Every block starts with a ### header ### that is also the first line of the
matching notebook cell, so the two can be compared side by side.

Printing: PRINT_LIMIT and PLOT_LIMIT below control how much is shown for each
step (useful when there are thousands of dipoles). Files are always written.
"""

# ============================================================
### Import Libraries and Settings ###
# ============================================================

# paths, files, system
import os
import re
import sys
import glob
import math
import time
import shutil
import threading
import subprocess
import concurrent.futures
from pathlib import Path

# arrays, tables, datasets
import numpy as np
import pandas as pd
import xarray as xr

# matlab files
import scipy.io
from scipy.io import loadmat

# connectivity
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, label, generate_binary_structure

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# paleomag
import pmagpy.ipmag as ipmag


# ------------------------------------------------------------
# printing control
# ------------------------------------------------------------
PRINT_LIMIT = 30    # max lines printed per step (None = print everything)
PLOT_LIMIT = 10     # max figures shown per step (None = show all), all are still saved


class Out:
    """Prints the first PRINT_LIMIT lines of a step, counts the rest."""

    def __init__(self):
        self.n_lines = 0
        self.hidden_lines = 0
        self.n_plots = 0
        self.hidden_plots = 0
        self.lock = threading.Lock()

    def say(self, *args, always=False, **kw):
        with self.lock:
            if always:
                print(*args, **kw)
                return
            self.n_lines += 1
            if PRINT_LIMIT is None or self.n_lines <= PRINT_LIMIT:
                print(*args, **kw)
            else:
                self.hidden_lines += 1

    def table(self, df, index=True):
        if PRINT_LIMIT is not None and len(df) > PRINT_LIMIT:
            print(df.head(PRINT_LIMIT).to_string(index=index))
            self.hidden_lines += len(df) - PRINT_LIMIT
        else:
            print(df.to_string(index=index))

    def show(self):
        if PLOT_LIMIT is None or self.n_plots < PLOT_LIMIT:
            self.n_plots += 1
            plt.show()
        else:
            self.hidden_plots += 1
            plt.close()

    def end(self):
        if self.hidden_lines or self.hidden_plots:
            print(f"\n[{self.hidden_lines} line(s) and {self.hidden_plots} figure(s) not shown; "
                  f"raise PRINT_LIMIT / PLOT_LIMIT to see them, files were saved anyway]")


# ------------------------------------------------------------
# settings used by almost every step
# ------------------------------------------------------------
def next_run_id(tomofast_home_wsl, wsl, prefix="QDM"):
    """pick the next free "QDM"/"QDM1"/"QDM2"/... folder name"""
    try:
        existing = set(os.listdir(tomofast_home_wsl))  # native Linux: path is local
    except (FileNotFoundError, NotADirectoryError):
        p = subprocess.Popen([wsl, "bash", "-lc", f"ls -1 {tomofast_home_wsl} 2>/dev/null"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = p.communicate()
        existing = set(out.split())

    if prefix not in existing:
        return prefix
    n = 1
    while f"{prefix}{n}" in existing:
        n += 1
    return f"{prefix}{n}"


class Settings:
    def __init__(self, base_dir, d, tomofast_home, tomofast_home_wsl, wsl, nproc, use_wsl=False):
        self.base_dir = base_dir
        self.d = d                                  # cell size = data spacing (um)
        self.tomofast_home = tomofast_home          # Linux
        self.tomofast_home_wsl = tomofast_home_wsl  # WSL
        self.wsl = wsl
        self.nproc = nproc                          # mpirun -np
        self.use_wsl = use_wsl
        print("Current working directory:", base_dir)
        self.run_id = next_run_id(tomofast_home_wsl, wsl)
        print("This run's Tomofast folder name:", self.run_id)

    @property
    def home(self):
        # where Tomofast-x lives, as seen from the machine that runs it
        return self.tomofast_home_wsl if self.use_wsl else self.tomofast_home


# ------------------------------------------------------------
# small helpers shared by several steps
# ------------------------------------------------------------
def to_wsl_path(win_path):
    win_path = win_path.replace("\\", "/")
    drive = win_path[0].lower()
    rest = win_path[2:]
    return f"/mnt/{drive}/{rest}"


def host_path(cfg, path):
    # path as the Tomofast machine sees it (WSL needs /mnt/c/... , Linux is unchanged)
    return to_wsl_path(path) if cfg.use_wsl else path


def run_hidden(cfg, cmd):
    """Run a command (WSL or Linux), do not print its output, stop if it fails."""
    if cfg.use_wsl:
        p = subprocess.Popen([cfg.wsl, "bash", "-lc", cmd],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    else:
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _ = p.stdout.read()   # consume the output, do not print it
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def run_quiet(cfg, cmd):
    """Run a command, return (return code, last 'data cost' number, full output)."""
    if cfg.use_wsl:
        p = subprocess.Popen([cfg.wsl, "bash", "-lc", cmd],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    else:
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    cost = None
    for line in p.stdout:
        out_lines.append(line)
        if "data cost" in line.lower():
            try:
                cost = float(line.split("=")[-1].strip())
            except:
                pass

    p.wait()
    return p.returncode, cost, "".join(out_lines)


def run_live(cfg, cmd, out):
    """Run a command and show its output as it comes (limited by PRINT_LIMIT)."""
    if cfg.use_wsl:
        process = subprocess.Popen([cfg.wsl, "bash", "-lc", cmd],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
    else:
        process = subprocess.Popen(cmd, shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
    for line in process.stdout:
        out.say(line, end="")
        sys.stdout.flush()
    process.wait()
    out.say(f"\n[Return code: {process.returncode}]\n")


def list_blobs(root):
    return sorted(
        d for d in os.listdir(root)
        if d.startswith("Blob_") and os.path.isdir(os.path.join(root, d))
    )


def blob_num(b):
    # Blob_001, Blob_002, ... -> 1, 2, ...
    m = re.search(r"(\d+)$", b)
    return int(m.group(1)) if m else 10**9


def load_field_table(path):
    # first line may be a header (e.g. Ndata count): if it has a different number
    # of fields than the second line, skip it
    with open(path) as f:
        first_line = f.readline().split()
        second_line = f.readline().split()

    if len(first_line) != len(second_line):
        skiprows = 1
        ncols = len(second_line)
    else:
        skiprows = 0
        ncols = len(first_line)

    if ncols >= 6:
        names = ("x", "y", "z", "bx", "by", "bz")
    else:
        names = ("x", "y", "z", "value")

    return pd.read_csv(path, sep=r"\s+", names=list(names), skiprows=skiprows)


def clean_df(df):
    value_cols = [c for c in ("bx", "by", "bz", "value") if c in df.columns]
    for c in ["x", "y"] + value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y"] + value_cols)
    df = df[np.all([np.isfinite(df[c]) for c in ["x", "y"] + value_cols], axis=0)]
    return df


def df_to_grid(df, value_col="value"):
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    if xs.size == 0 or ys.size == 0:
        return None, None, None
    grid = df.pivot_table(index="y", columns="x", values=value_col)
    grid = grid.reindex(index=ys, columns=xs)
    return grid.values, xs, ys


# groupby+mean instead of df_to_grid's pivot_table: blob windows can touch
# at shared edge points, and pivot_table errors on duplicate (x,y) entries
def stitch_to_grid(df, value_col):
    g = df.groupby(["y", "x"])[value_col].mean().unstack()
    return g.to_numpy(), g.columns.to_numpy(), g.index.to_numpy()


def fit_quality(obs_vals, calc_vals):
    """data cost, R2 and RMS between observed and calculated values"""
    residuals = calc_vals - obs_vals
    data_cost = np.linalg.norm(residuals) / np.linalg.norm(obs_vals)
    rms = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((obs_vals - obs_vals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return data_cost, r2, rms


def dec_inc_from_vector(Mx, My, Mz):
    """declination / inclination of a vector (x north, y east, z up)"""
    m = np.sqrt(Mx**2 + My**2 + Mz**2)

    if m == 0:
        return np.nan, np.nan, np.nan, np.nan

    theta = np.degrees(np.arccos(Mz / m))
    phi = np.degrees(np.arctan2(My, Mx))

    inc = theta - 90
    dec = 90 - phi

    change = True
    while change:
        change = False

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

    return dec, inc, theta, phi


def read_model_vectors(model_file):
    # first line = number of cells, then Mx My Mz per cell
    with open(model_file, "r") as f:
        lines = f.readlines()

    n_cells = int(lines[0].strip())
    data = np.loadtxt(lines[1:])
    data = np.atleast_2d(data)    # keep it 2D even for a single cell
    return n_cells, data


def get_moment_column(columns):
    candidates = ["moment(A·m²)", "moment(A·m2)", "moment", "Moment", "moment_amp"]
    return next((c for c in candidates if c in columns), None)


def read_bz_obs(base_dir, flip=True):
    """x, y, Bz of Mag_Inversion.obs (flip=True goes back to the QDM convention)"""
    mag_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Mag_Inversion.obs")
    mag = np.loadtxt(mag_path, skiprows=1)
    x = mag[:, 0]
    y = mag[:, 1]

    # 4 cols = x,y,z,bz | 6 cols = x,y,z,bx,by,bz
    ncols = mag.shape[1]
    bz_col = 5 if ncols >= 6 else 3
    bz = -1 * mag[:, bz_col] if flip else mag[:, bz_col]
    return x, y, bz


# ============================================================
### Create Folders ###
# ============================================================
def create_folders(cfg):
    base_dir = cfg.base_dir
    out = Out()

    folders = [
        os.path.join(base_dir, "4-3D_Inversion_Filtered"),
        os.path.join(base_dir, "0-Parfiles"),
        os.path.join(base_dir, "4-Source_Detection"),
        os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output"),
    ]

    for folder_path in folders:
        if os.path.exists(folder_path):
            out.say("Folder already exists:", folder_path)
        else:
            os.mkdir(folder_path)
            out.say("Folder created:", folder_path)

    out.end()


# ============================================================
### Quick Visualization: Show Bz vs LED ###
# ============================================================
def show_bz_vs_led(cfg):
    out = Out()

    # load mat file
    mat_file = glob.glob(os.path.join(cfg.base_dir, "*.mat"))[0]
    print("Using:", mat_file)

    mat = loadmat(mat_file)
    print("Variables:", [k for k in mat.keys() if not k.startswith("__")])

    Bz = np.squeeze(mat["Bz"]) * 1e9
    led = np.squeeze(mat["newLED"])

    # pixel size in um
    pix = float(np.squeeze(mat["step"])) * 1e6
    print("Pixel size (µm):", pix)

    ny, nx = Bz.shape
    extent = [0, nx * pix, 0, ny * pix]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Bz
    v = np.percentile(np.abs(Bz), 99)
    im = axs[0].imshow(Bz, cmap="seismic", vmin=-v, vmax=v, origin="upper", extent=extent)
    axs[0].set_title("Bz (nT)")
    axs[0].set_xlabel("X (µm)")
    axs[0].set_ylabel("Y (µm)")
    plt.colorbar(im, ax=axs[0], label="Bz (nT)", fraction=0.03)

    # LED
    axs[1].imshow(led, cmap="gray", origin="upper", extent=extent)
    axs[1].set_title("LED")
    axs[1].set_xlabel("X (µm)")
    axs[1].set_ylabel("Y (µm)")

    plt.tight_layout()
    out.show()
    out.end()


# ============================================================
### Quick Visualization: Show Bz overlaid by LED ###
# ============================================================
def show_bz_led_overlay(cfg):
    out = Out()
    base_dir = cfg.base_dir

    mat_file = glob.glob(os.path.join(base_dir, "*.mat"))[0]
    mat = loadmat(mat_file)

    Bz = np.squeeze(mat["Bz"]) * 1e9
    led = np.squeeze(mat["newLED"])

    # pixel size
    pix = cfg.d  # µm
    ny, nx = Bz.shape

    x = np.linspace(0, nx * pix, nx)
    y = np.linspace(0, ny * pix, ny)

    X, Y = np.meshgrid(x, y)

    # clip outliers
    bz_min = np.percentile(Bz, 2)
    bz_max = np.percentile(Bz, 98)

    plt.style.use("seaborn-v0_8-white")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Bz base
    im = ax.imshow(Bz, cmap="seismic", vmin=bz_min, vmax=bz_max, origin="upper",
                   extent=[x.min(), x.max(), y.min(), y.max()])

    cbar = plt.colorbar(im, ax=ax, shrink=0.66, pad=0.02)
    cbar.set_label("Bz (nT)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # LED overlay
    ax.imshow(led, cmap="gray", alpha=0.7, origin="upper",
              extent=[x.min(), x.max(), y.min(), y.max()],
              vmin=np.percentile(led, 5), vmax=np.percentile(led, 95))

    # scale bar
    bar_len = 50  # µm
    x0 = x.min() + 0.05 * (x.max() - x.min())
    y0 = y.min() + 0.05 * (y.max() - y.min())

    ax.plot([x0, x0 + bar_len], [y0, y0], color="k", linewidth=3, solid_capstyle="butt")
    ax.text(x0 + bar_len / 2, y0 + 0.02 * (y.max() - y.min()), f"{int(bar_len)} µm",
            ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("X (µm)", fontsize=12)
    ax.set_ylabel("Y (µm)", fontsize=12)

    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

    plt.tight_layout()

    out_png = os.path.join(base_dir, "Bz_LED_overlay.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    out.show()
    out.end()


# ============================================================
### Data Preparation: 1. Read Bx, By, Bz mat file ###
# ============================================================
METER_TO_MICROMETER = 1e6
TESLA_TO_NANOTESLA = 1e9

# which field components to look for in the .mat file
FIELD_KEYS = ["Bx", "By", "Bz"]


def load_qdm_mat(path, save_csv=True, csv_path=None):
    """
    Load QDM microscopy data from a MATLAB .mat file and return it as an
    xarray Dataset (bx, by, bz in nT; x, y, z in µm). Only the components
    present in the file are used. The csv is saved next to the .mat file.
    """
    path = Path(path)
    contents = scipy.io.loadmat(path)

    # pixel spacing (m -> µm)
    spacing = contents['step'].ravel()[0] * METER_TO_MICROMETER

    # which components are actually in the file
    present_keys = [k for k in FIELD_KEYS if k in contents]
    if not present_keys:
        raise KeyError(
            f"None of {FIELD_KEYS} found in {path.name}. "
            f"Available keys: {list(contents.keys())}"
        )

    # T -> nT
    fields = {k: contents[k] * TESLA_TO_NANOTESLA for k in present_keys}

    # shape of the first field gives the coordinates
    ref_field = fields[present_keys[0]]

    # sensor-sample distance (µm)
    sensor_sample_distance = float(contents['h'].ravel()[0])

    x = np.arange(ref_field.shape[1]) * spacing
    y = np.arange(ref_field.shape[0])[::-1] * spacing  # flip from image coords (origin up) to cartesian (origin low)
    z = np.full_like(ref_field, sensor_sample_distance)

    data_vars = {
        k.lower(): (("y", "x"), v) for k, v in fields.items()
    }

    data = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": ("x", x, {"units": "µm"}),
            "y": ("y", y, {"units": "µm"}),
            "z": (("y", "x"), z, {"long_name": "sensor sample distance", "units": "µm"})
        },
        attrs={"file_name": str(path)}
    )

    long_names = {"bx": "in-plane magnetic field (x)",
                  "by": "in-plane magnetic field (y)",
                  "bz": "vertical magnetic field"}
    for k in data_vars:
        data[k].attrs = {"long_name": long_names.get(k, k), "units": "nT"}

    # save to csv
    if save_csv:
        if csv_path is None:
            csv_path = path.with_suffix(".csv")  # same name as the .mat, .csv extension

        df_dict = {
            "x (µm)": np.tile(x, len(y)),      # x varies fastest
            "y (µm)": np.repeat(y, len(x)),    # y varies slowest
            "z (µm)": np.repeat(sensor_sample_distance, len(x) * len(y)),
        }
        for k, v in fields.items():
            df_dict[f"{k.lower()} (nT)"] = v.ravel(order="C")

        df = pd.DataFrame(df_dict)
        df = df.sort_values(by=["y (µm)", "x (µm)"], ascending=[True, True])  # sort Y after flipping
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved CSV file at: {csv_path}")

    return data


# ============================================================
### Data Preparation: 1. Read Bx, By, Bz mat file (plots) ###
# ============================================================
def plot_components(data):
    for comp in ["bx", "by", "bz"]:
        if comp in data:
            data[comp].plot.pcolormesh(cmap="seismic")  # add vmin/vmax as needed
            plt.title(comp.upper())
            plt.show()


# ============================================================
### Data Preparation: 2. Format data into Tomofast-x format ###
# ============================================================
def plot_data(x_coords, y_coords, values, title, label='Bz [nT]', out_dir=None, out=None):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    vmax = np.nanmax(np.abs(values))
    vmin = -vmax
    scatter = plt.scatter(x_coords, y_coords, c=values, cmap='seismic', s=3, vmin=vmin, vmax=vmax)
    plt.colorbar(scatter, label=label)
    plt.xlabel('x [µm]')
    plt.ylabel('y [µm]')
    filename = os.path.join(out_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    out.say(f"Plot saved to {filename}")
    out.show()


def prepare_tomofast_data(cfg, csv_name="Bz.csv", x_start=0, x_end=None, y_start=0, y_end=None):
    """
    Reads the csv (Bz only, or Bx/By/Bz), optionally clips it by cell index
    (x_end / y_end = None means the full scan) and writes Mag_Inversion.obs.
    """
    base_dir = cfg.base_dir
    out = Out()
    out_dir = os.path.join(base_dir, "4-3D_Inversion_Filtered")

    filename_z = os.path.join(base_dir, csv_name)
    data_z = np.loadtxt(filename_z, delimiter=",", skiprows=1)
    print(data_z.shape)

    x_coords = data_z[:, 0]
    y_coords = data_z[:, 1]
    z_coords = -np.abs(data_z[:, 2])  # force z (height) negative regardless of original sign

    # 4 cols = x,y,z,bz | 6 cols = x,y,z,bx,by,bz
    ncols = data_z.shape[1]
    if ncols >= 6:
        bx = data_z[:, 3]
        by = data_z[:, 4]
        bz = -1 * data_z[:, 5]  # flip Bz for tomofast (Z positive down, QDM Bz positive up)
    else:
        bx = None
        by = None
        bz = -1 * data_z[:, 3]  # flip Bz for tomofast (Z positive down, QDM Bz positive up)

    # plot the image of the data
    plot_data(x_coords, y_coords, bz, 'bz', label='Bz [nT]', out_dir=out_dir, out=out)
    if bx is not None:
        plot_data(x_coords, y_coords, bx, 'bx', label='Bx [nT]', out_dir=out_dir, out=out)
        plot_data(x_coords, y_coords, by, 'by', label='By [nT]', out_dir=out_dir, out=out)

    # unique coordinates give the grid dimensions
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    nx = len(unique_x)  # number of points in X direction
    ny = len(unique_y)  # number of points in Y direction
    print("Grid dimensions:", nx, ny)

    x_coords = x_coords.reshape(ny, nx)
    y_coords = y_coords.reshape(ny, nx)
    z_coords = z_coords.reshape(ny, nx)
    bz = bz.reshape(ny, nx)
    if bx is not None:
        bx = bx.reshape(ny, nx)
        by = by.reshape(ny, nx)

    # clipping, by cell index (None = to the end)
    if x_end is None:
        x_end = nx
    if y_end is None:
        y_end = ny

    x_coords = x_coords[y_start:y_end, x_start:x_end]
    y_coords = y_coords[y_start:y_end, x_start:x_end]
    z_coords = z_coords[y_start:y_end, x_start:x_end]
    bz = bz[y_start:y_end, x_start:x_end]
    if bx is not None:
        bx = bx[y_start:y_end, x_start:x_end]
        by = by[y_start:y_end, x_start:x_end]
    print(bz.shape)

    # back to 1D arrays
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = z_coords.flatten()
    bz = bz.flatten()
    if bx is not None:
        bx = bx.flatten()
        by = by.flatten()

    plot_data(x_coords, y_coords, bz, 'Clipped bz', label='Bz [nT]', out_dir=out_dir, out=out)
    if bx is not None:
        plot_data(x_coords, y_coords, bx, 'Clipped bx', label='Bx [nT]', out_dir=out_dir, out=out)
        plot_data(x_coords, y_coords, by, 'Clipped by', label='By [nT]', out_dir=out_dir, out=out)

    print('Xmin, Xmax =', np.min(x_coords), np.max(x_coords))
    print('Ymin, Ymax =', np.min(y_coords), np.max(y_coords))

    # write tomofast data grid
    if bx is not None:
        data_combined = np.column_stack((x_coords, y_coords, z_coords, bx, by, bz))
        fmt = "%f %f %f %f %f %f"
    else:
        data_combined = np.column_stack((x_coords, y_coords, z_coords, bz))
        fmt = "%f %f %f %f"

    Ndataz = len(bz)
    print('Ndata =', Ndataz)
    np.savetxt(os.path.join(base_dir, "4-3D_Inversion_Filtered", "Mag_Inversion.obs"),
               data_combined, delimiter=' ', fmt=fmt, header=str(Ndataz), comments='')

    out.end()


# ============================================================
### Data Preparation: 3. Anomalies Detection and Windowing ###
# ============================================================
def write_obs(out_path, header, arr):
    # write an obs file with the updated Ndata in the header
    parts = header.split()
    try:
        int(parts[0])           # if header starts with Ndata
        parts[0] = str(arr.shape[0])
        header = " ".join(parts)
    except:
        pass

    with open(out_path, "w") as f:
        f.write(header + "\n")
        np.savetxt(f, arr, fmt="%.6f")


def detect_anomalies(cfg,
                     threshold_factors=None,    # None = sweep from 10 down to 0.5 in steps of 0.5
                     lock_margin_factor=0.4,    # once a blob is picked, lock out this fraction of its size
                     min_lock_margin=0,
                     pad_x=5,                   # extend rectangle in X (cols)
                     pad_y=5,                   # extend rectangle in Y (rows)
                     skip_blobs=(),             # ignore noisy blobs (empty = keep everything)
                     choose_blobs=(),           # if not empty, ONLY keep these blob IDs
                     min_blob_cells=1,          # ignore tiny blobs
                     max_blob_cells=100000,     # ignore huge blobs
                     nz_fixed=10,               # nz for Tomofast (same for all blobs)
                     merge_close=True,          # True = nearby anomalies join into one blob, False = keep them separate
                     plot_vlim=20000):          # colour limit of the overview plot (nT)
    base_dir = cfg.base_dir
    d = cfg.d
    out = Out()

    if threshold_factors is None:
        threshold_factors = list(np.arange(10, 0.0000, -0.5))   # sweep instead of one fixed value

    # a single number like choose_blobs=1 or (1) is accepted too
    if isinstance(skip_blobs, (int, np.integer)):
        skip_blobs = (int(skip_blobs),)
    if isinstance(choose_blobs, (int, np.integer)):
        choose_blobs = (int(choose_blobs),)

    # cell size + z center at half model depth (for Steepest search)
    dx_cell = d
    dy_cell = d
    dz_cell = d
    z_center = (nz_fixed // 4 + 0.5) * dz_cell

    # store seed rows to write after loop
    seed_rows = []  # (blob_id, xmin, xmax, ymin, ymax, zmin, zmax)

    # paths
    obs_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Mag_Inversion.obs")
    out_obs_dir = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Blob_Clipped_obs_RECT")

    # clear old outputs
    if os.path.exists(out_obs_dir):
        for fname in os.listdir(out_obs_dir):
            fpath = os.path.join(out_obs_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
    else:
        os.makedirs(out_obs_dir)

    # blob windows (xmin xmax ymin ymax)
    windows_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "blob_windows.txt")
    # summary for parfiles (nx ny nz ndata)
    summary_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "blob_window_summary.txt")
    # seed single-cell mesh rows
    seed_meshrow_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "blob_seed_meshrow.txt")

    # write headers (overwrite each run)
    with open(windows_path, "w") as fw:
        fw.write("# blob_id x_min x_max y_min y_max\n")

    with open(summary_path, "w") as fs:
        fs.write("# blob_id nx ny nz ndata\n")

    # 1) read obs
    with open(obs_path, "r") as f:
        header_line = f.readline().strip()

    data = np.loadtxt(obs_path, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]

    # 4 cols = x,y,z,bz | 6 cols = x,y,z,bx,by,bz
    ncols = data.shape[1]
    bz_col = 5 if ncols >= 6 else 3
    bz = data[:, bz_col]

    # 2) grid reshape
    nx = len(np.unique(x))
    ny = len(np.unique(y))

    # assumes grid-ordered obs
    x_grid = x.reshape(ny, nx)
    y_grid = y.reshape(ny, nx)
    bz_grid = bz.reshape(ny, nx)

    print("Grid:", nx, ny)
    print("Threshold levels to sweep:", len(threshold_factors))

    # 4) overview plot
    plt.figure(figsize=(8, 6), dpi=300)

    im = plt.imshow(
        bz_grid,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        cmap="seismic",
        aspect="equal", vmax=plot_vlim, vmin=-plot_vlim)

    cbar = plt.colorbar(im, pad=0.01, shrink=0.61)
    cbar.set_label("Bz (nT)", fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel("X (µm)", fontsize=14)
    plt.ylabel("Y (µm)", fontsize=14)

    ax = plt.gca()
    ax.tick_params(axis="both", length=4, labelsize=14)
    ax.margins(0)

    plt.tight_layout()

    # 4+5) threshold sweep + detect + loop blobs: save obs + windows + summary
    thr_base_std = np.std(bz_grid)
    picked_mask = np.zeros((ny, nx), dtype=bool)   # locks out already-picked regions
    kept = 0
    n_levels = len(threshold_factors)

    for pass_id, threshold_factor in enumerate(threshold_factors, start=1):

        if PRINT_LIMIT != 0:
            print(f"\rlevel {pass_id}/{n_levels} (factor={threshold_factor:.5f}) -- kept {kept} blob(s) so far          ",
                  end="", flush=True)

        thr = threshold_factor * thr_base_std
        mask = np.abs(bz_grid) > thr
        mask &= ~picked_mask   # never re-detect what a stronger threshold already picked

        if merge_close:
            # allows nearby high-amplitude regions to connect if they are spatially close.
            structure = np.ones((3, 3))  # allows diagonal + small gaps
            mask_dilated = binary_dilation(mask, structure=structure, iterations=1)
            labels, n_blobs = ndi.label(mask_dilated)
        else:
            # not merging:
            labels, n_blobs = ndi.label(mask)

        sizes = ndi.sum(mask, labels, index=np.arange(1, n_blobs + 1))
        slices = ndi.find_objects(labels)

        for i, sl in enumerate(slices, start=1):
            if sl is None:
                continue

            # skip tiny blobs
            if sizes[i-1] < min_blob_cells:
                continue

            # skip huge blobs
            if sizes[i-1] > max_blob_cells:
                continue

            # skip noisy blobs (by stable global id, not per-pass label i)
            if (kept + 1) in skip_blobs:
                kept += 1
                continue

            # if choose_blobs is set, only keep blobs explicitly listed there
            if choose_blobs and (kept + 1) not in choose_blobs:
                kept += 1
                continue

            ys, xs = sl
            r0, r1 = ys.start, ys.stop - 1
            c0, c1 = xs.start, xs.stop - 1

            # extend bbox
            r0e = max(0, r0 - pad_y)
            r1e = min(ny - 1, r1 + pad_y)
            c0e = max(0, c0 - pad_x)
            c1e = min(nx - 1, c1 + pad_x)

            # rectangular mask
            rect_mask = np.zeros((ny, nx), dtype=bool)
            rect_mask[r0e:r1e+1, c0e:c1e+1] = True

            # clip rows (rectangle)
            blob_data = data[rect_mask.ravel(), :]

            # rectangle coords (µm)
            x_min = x_grid[r0e, c0e]
            x_max = x_grid[r1e, c1e]
            y_min = y_grid[r0e, c0e]
            y_max = y_grid[r1e, c1e]

            # window size in cells (for parfile modelGrid.size)
            nx_win = (c1e - c0e + 1)  # X = columns
            ny_win = (r1e - r0e + 1)  # Y = rows
            nz_win = nz_fixed

            # number of data points in clipped obs (for parfile nData)
            ndata = blob_data.shape[0]

            kept += 1   # global id across all threshold levels

            # (A) save blob obs
            obs_out = os.path.join(out_obs_dir, f"Blob_{kept:03d}_RECT.obs")
            write_obs(obs_out, header_line, blob_data)

            # (B) save window file line
            with open(windows_path, "a") as fw:
                fw.write(f"{kept} {x_min:.6f} {x_max:.6f} {y_min:.6f} {y_max:.6f}\n")

            # (C) save summary file line
            with open(summary_path, "a") as fs:
                fs.write(f"{kept} {nx_win} {ny_win} {nz_win} {ndata}\n")

            # (D) save seed single-cell mesh row
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)

            # snap to nearest grid
            cx = round(cx / dx_cell) * dx_cell
            cy = round(cy / dy_cell) * dy_cell

            xmin_seed = cx - 0.5 * dx_cell
            xmax_seed = cx + 0.5 * dx_cell
            ymin_seed = cy - 0.5 * dy_cell
            ymax_seed = cy + 0.5 * dy_cell

            zmin_seed = z_center - 0.5 * dz_cell
            zmax_seed = z_center + 0.5 * dz_cell

            seed_rows.append((kept, xmin_seed, xmax_seed, ymin_seed, ymax_seed, zmin_seed, zmax_seed))

            # (E) lock out this blob's region so a lower threshold level
            # doesn't re-detect it (or its fringe) as a "new" blob
            lock_margin_x = max(min_lock_margin, int(round(lock_margin_factor * nx_win)))
            lock_margin_y = max(min_lock_margin, int(round(lock_margin_factor * ny_win)))
            lock_r0 = max(0, r0e - lock_margin_y); lock_r1 = min(ny - 1, r1e + lock_margin_y)
            lock_c0 = max(0, c0e - lock_margin_x); lock_c1 = min(nx - 1, c1e + lock_margin_x)
            picked_mask[lock_r0:lock_r1+1, lock_c0:lock_c1+1] = True

            # draw rectangle on overview
            ax.add_patch(
                patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min, y_max - y_min,
                    edgecolor="black", facecolor="none", linewidth=1.5
                )
            )

            # label with anomaly number
            ax.text(
                x_min, y_max, str(kept),
                color="black", fontsize=8,
                va="bottom", ha="left"
            )

    print()  # newline after the in-place progress line

    plt.title(f"Detected anomalies -- {kept} blob(s)", fontsize=14)
    plt.tight_layout()
    out_png = os.path.join(base_dir, "4-3D_Inversion_Filtered", "BZ_Blobs.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out.show()

    # write seed meshrow file, first line = count
    with open(seed_meshrow_path, "w") as f:
        f.write(f"{len(seed_rows)}\n")
        for r in seed_rows:
            blob_id, xmin_s, xmax_s, ymin_s, ymax_s, zmin_s, zmax_s = r

            blob_name = f"Blob_{blob_id:03d}"
            grain_id = 1

            i_idx = 1
            j_idx = 1
            k_idx = 1

            f.write(
                f"{blob_name} {grain_id} "
                f"{xmin_s:.6f} {xmax_s:.6f} "
                f"{ymin_s:.6f} {ymax_s:.6f} "
                f"{zmin_s:.6f} {zmax_s:.6f} "
                f"0.000000 0.000000 0.000000 {i_idx} {j_idx} {k_idx}\n"
            )

    print("\n====================================")
    print("DONE")
    print("Kept blobs:", kept)
    print("Obs folder:", out_obs_dir)
    print("Windows file:", windows_path)
    print("Summary file:", summary_path)
    print("Seed meshrow file:", seed_meshrow_path)
    print("====================================")

    print(min(bz), max(bz))
    out.end()


# ============================================================
### Tool 1: 1. Create Meshs ###
# ============================================================
def write_meshgrid_tomofast(filename, line_data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"{line_data.shape[0]}\n")
        np.savetxt(f, line_data, fmt="%f %f %f %f %f %f %d %d %d")


def make_edges(core_min, d_core, n_core, npad, r):
    """
    Edges with uniform core widths (d_core, n_core cells) and geometric
    padding widths growing outward from the core.
    Returns edges (n_core + 2*npad + 1) and widths (n_core + 2*npad).
    """
    core_widths = np.full(n_core, d_core, dtype=float)

    # padding widths: closest pad cell = d_core, then grows outward
    left_widths = d_core * (r ** np.arange(npad - 1, -1, -1))  # big -> ... -> d_core (toward core)
    right_widths = d_core * (r ** np.arange(0, npad))           # d_core -> ... -> big (outward)

    widths = np.concatenate([left_widths, core_widths, right_widths])

    start_edge = core_min - left_widths.sum()
    edges = start_edge + np.concatenate(([0.0], np.cumsum(widths)))
    return edges, widths


def build_mesh_for_window(x_min, x_max, y_min, y_max, out_file,
                          dx_core, dy_core, npad_x, npad_y, rx, ry,
                          z_min, nz, dz0, rz, expand_half_cell):
    # 1) core extents
    if expand_half_cell:
        x_core_min = x_min - 0.5 * dx_core
        x_core_max = x_max + 0.5 * dx_core
        y_core_min = y_min - 0.5 * dy_core
        y_core_max = y_max + 0.5 * dy_core
    else:
        x_core_min, x_core_max = x_min, x_max
        y_core_min, y_core_max = y_min, y_max

    # core cell counts
    nx_core = int(np.round((x_core_max - x_core_min) / dx_core))
    ny_core = int(np.round((y_core_max - y_core_min) / dy_core))

    if nx_core < 1 or ny_core < 1:
        raise ValueError(
            f"Window too small for given dx/dy. "
            f"Got nx_core={nx_core}, ny_core={ny_core}. Check units/extents."
        )

    # snap core max so it is exactly n_core * d_core from core_min
    x_core_max = x_core_min + nx_core * dx_core
    y_core_max = y_core_min + ny_core * dy_core

    # 2) non-uniform X/Y edges
    x_edges, x_widths = make_edges(x_core_min, dx_core, nx_core, npad_x, rx)
    y_edges, y_widths = make_edges(y_core_min, dy_core, ny_core, npad_y, ry)

    # totals including pads
    nx_cells = nx_core + 2 * npad_x
    ny_cells = ny_core + 2 * npad_y
    nz_cells = nz

    # 3) non-uniform Z edges (geometric growth)
    dz_list = dz0 * (rz ** np.arange(nz_cells))                 # len = nz_cells
    z_edges = z_min + np.concatenate(([0.0], np.cumsum(dz_list)))

    # 4) cell bounds grids (Tomofast ordering): indexing="ij" and flatten(order="F")
    X1, Y1, Z1 = np.meshgrid(x_edges[:-1], y_edges[:-1], z_edges[:-1], indexing="ij")
    X2, Y2, Z2 = np.meshgrid(x_edges[1:], y_edges[1:], z_edges[1:], indexing="ij")

    # 1-based indices for Tomofast
    I, J, K = np.meshgrid(
        np.arange(1, nx_cells + 1),
        np.arange(1, ny_cells + 1),
        np.arange(1, nz_cells + 1),
        indexing="ij"
    )

    # flatten (Fortran order)
    x1 = X1.flatten(order="F"); x2 = X2.flatten(order="F")
    y1 = Y1.flatten(order="F"); y2 = Y2.flatten(order="F")
    z1 = Z1.flatten(order="F"); z2 = Z2.flatten(order="F")
    ii = I.flatten(order="F"); jj = J.flatten(order="F"); kk = K.flatten(order="F")

    # X1 X2 Y1 Y2 Z1 Z2 I J K
    line_data = np.column_stack([x1, x2, y1, y2, z1, z2, ii, jj, kk])

    write_meshgrid_tomofast(out_file, line_data)

    info = {
        "nx_core": nx_core, "ny_core": ny_core, "nz": nz_cells,
        "nx": nx_cells, "ny": ny_cells,
        "x_min": x_edges[0], "x_max": x_edges[-1],
        "y_min": y_edges[0], "y_max": y_edges[-1],
        "z_min": z_edges[0], "z_max": z_edges[-1],
        "dz_top": dz_list[0], "dz_bottom": dz_list[-1],
        "dx_min": x_widths.min(), "dx_max": x_widths.max(),
        "dy_min": y_widths.min(), "dy_max": y_widths.max()
    }
    return info


def create_meshgrids(cfg,
                     npad_x=1, npad_y=1,        # pad cells on each side
                     rx=1.05, ry=1.05,          # pad growth factor (outwards)
                     z_min=0.0,                 # top of model (positive down)
                     nz=30,                     # number of z cells
                     dz0=None,                  # top layer thickness (None = cell size d)
                     rz=1,                      # z growth factor per layer
                     expand_half_cell=True,     # expand each window by half a core cell
                     use_ndata_from_file=True):
    base_dir = cfg.base_dir
    out = Out()

    if dz0 is None:
        dz0 = cfg.d

    # core (data) cell sizes in X/Y (µm)
    dx_core = cfg.d
    dy_core = cfg.d

    # input blob windows: blob_id x_min x_max y_min y_max
    windows_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "blob_windows.txt")

    out_dir = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Blob_Meshgrids")
    os.makedirs(out_dir, exist_ok=True)

    # summary for parfiles (nx ny nz ndata)
    summary_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "blob_window_summary.txt")

    windows = np.loadtxt(windows_path, comments="#", ndmin=2)

    print("Windows loaded:", windows.shape[0])
    print("Output folder:", out_dir)
    print(f"Pads: npad_x={npad_x}, npad_y={npad_y}, rx={rx}, ry={ry}")
    print(f"Z: nz={nz}, dz0={dz0}, rz={rz}\n")

    # write summary header once
    with open(summary_path, "w") as fs:
        fs.write("# blob_id nx ny nz ndata\n")

    for row in windows:
        blob_id = int(row[0])
        x_min, x_max, y_min, y_max = row[1], row[2], row[3], row[4]

        out_file = os.path.join(out_dir, f"Meshgrid_Blob_{blob_id:03d}.txt")
        info = build_mesh_for_window(x_min, x_max, y_min, y_max, out_file,
                                     dx_core, dy_core, npad_x, npad_y, rx, ry,
                                     z_min, nz, dz0, rz, expand_half_cell)

        # ndata: if blob_windows.txt has a 6th column (ndata) use it,
        # otherwise the number of core cells in XY
        if use_ndata_from_file and row.size >= 6:
            ndata = int(row[5])
        else:
            ndata = int(info["nx_core"] * info["ny_core"])

        # summary line: padded final nx ny nz
        with open(summary_path, "a") as fs:
            fs.write(
                f"{blob_id:d} "
                f"{info['nx']:d} {info['ny']:d} {info['nz']:d} "
                f"{ndata:d}\n"
            )

        out.say(
            f"Blob {blob_id:03d}: "
            f"core(nx,ny)=({info['nx_core']},{info['ny_core']})  "
            f"total(nx,ny,nz)=({info['nx']},{info['ny']},{info['nz']})  "
            f"ndata={ndata}"
        )
        out.say(
            f"  X[{info['x_min']:.3f}, {info['x_max']:.3f}]  "
            f"Y[{info['y_min']:.3f}, {info['y_max']:.3f}]  "
            f"Z[{info['z_min']:.3f}, {info['z_max']:.3f}]"
        )
        out.say(
            f"  Δx(min,max)=({info['dx_min']:.3f},{info['dx_max']:.3f})  "
            f"Δy(min,max)=({info['dy_min']:.3f},{info['dy_max']:.3f})  "
            f"Δz(top,bottom)=({info['dz_top']:.3f},{info['dz_bottom']:.3f})"
        )
        out.say("  saved:", out_file, "\n")

    print("Summary saved:", summary_path)
    print("Done.")
    out.end()


# ============================================================
### Tool 1: 2. Create Parfiles ###
# ============================================================
# target compression error (1%, per Bruce et al. 2026)
COMPRESSION_TARGET_ERROR = 0.01
COMPRESSION_MIN_CELLS = 400000


def calc_ideal_compression_rate(nx, ny, nz, c_e=COMPRESSION_TARGET_ERROR, wavelet="haar"):
    n_model = nx * ny * nz
    if n_model <= COMPRESSION_MIN_CELLS:
        return 1.0
    if wavelet.lower() == "d4":
        a, k, b = 42.53, -0.614, -0.852
    else:
        a, k, b = 35.07, -0.872, -0.884
    ideal_cr = a * (c_e ** k) * (n_model ** b)
    return min(ideal_cr * 4.0, 1.0)  # 2x safety margin, capped at 1.0 (no compression)


# Parfile template for the full 3D inversion of each blob (only the placeholders change)
PAR_TEMPLATE_3D = """==================================================================================
GLOBAL
==================================================================================
global.outputFolderPath         = ./{run_id}/output/Blob_{blob_id:03d}
global.description              = QDM inversion – Blob {blob_id}

==================================================================================
MODEL GRID parameters
==================================================================================
# nx ny nz
modelGrid.size                        = {nx} {ny} {nz}
modelGrid.magn.file                   = {run_id}/data/Meshgrid_Blob_{blob_id:03d}.txt
modelGrid.magn.nModelComponents       = 3

==================================================================================
DATA parameters
==================================================================================
forward.data.magn.nData                = {ndata}
forward.data.magn.dataGridFile         = {run_id}/data/Blob_{blob_id:03d}_RECT.obs
forward.data.magn.nDataComponents      = {ncomp}

==================================================================================
MAGNETIC FIELD constants
==================================================================================
forward.magneticField.inclination              = 90.0
forward.magneticField.declination              = 0.0
forward.magneticField.XaxisDeclination         = 0.d0

==================================================================================
DEPTH WEIGHTING
==================================================================================
# 1-depth weight, 2-distance weight.
forward.depthWeighting.type                    = 1
forward.depthWeighting.magn.Z0             = {z0} #adjust z0 paramter to data level (5)

==================================================================================
SENSITIVITY KERNEL
==================================================================================
sensit.readFromFiles                  = 0
sensit.folderPath                     = ./{run_id}/output/SENSIT/

==================================================================================
MATRIX COMPRESSION
==================================================================================
forward.matrixCompression.type                        = 1
forward.matrixCompression.rate                        = {compression_rate:.5f}

==================================================================================
PRIOR MODEL
==================================================================================
inversion.priorModel.type                  = 1
inversion.priorModel.magn.value            = 0.000000

==================================================================================
INVERSION parameters
==================================================================================
inversion.nMajorIterations              = 3
inversion.nMinorIterations              = 100
inversion.targetMisfit = 100 #RMS_nT target misfit (measurement error) to avoid overfitting

==================================================================================
MODEL DAMPING (m - m_prior)
==================================================================================
inversion.modelDamping.magn.weight      = 0.0

==================================================================================
JOINT INVERSION parameters
==================================================================================
inversion.joint.grav.problemWeight      = 0.00000000
inversion.joint.magn.problemWeight      = 1.00000000
"""


def create_inversion_parfiles(cfg, n_data_components=3, z0=10.0):
    """
    n_data_components: 3 if using Bx, By, Bz, or 1 if using Bz only
    z0: depth weighting Z0, the instrument height (matching the Z column in the data)
    """
    base_dir = cfg.base_dir
    out = Out()

    # input: window summary (blob_id nx ny nz ndata)
    summary_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "blob_window_summary.txt")

    # output: parfiles
    par_out_dir = os.path.join(base_dir, "0-Parfiles")
    os.makedirs(par_out_dir, exist_ok=True)

    summary = np.loadtxt(summary_path, comments="#", ndmin=2)

    print("Creating Parfiles...")

    for row in summary:
        blob_id = int(row[0])
        nx = int(row[1])
        ny = int(row[2])
        nz = int(row[3])
        ndata = int(row[4])

        compression_rate = calc_ideal_compression_rate(nx, ny, nz)

        par_text = PAR_TEMPLATE_3D.format(
            run_id=cfg.run_id,
            blob_id=blob_id,
            nx=nx,
            ny=ny,
            nz=nz,
            ndata=ndata,
            compression_rate=compression_rate,
            ncomp=n_data_components,
            z0=z0
        )

        out_path = os.path.join(par_out_dir, f"Parfile_QDM_Inversion_Blob_{blob_id:03d}.txt")

        with open(out_path, "w") as f:
            f.write(par_text)

        out.say(f"  Blob {blob_id:03d}: nx={nx}, ny={ny}, nz={nz}, nData={ndata}, compression_rate={compression_rate:.5f}")
        out.say("    ->", out_path)

    print("\nDone. Parfiles written to:", par_out_dir)
    out.end()


# ============================================================
### Tool 1: 3. Run Tomofast-x 3D full vector inversion ###
# ============================================================
def blob_id_from_name(path):
    # Blob_001_RECT.obs -> 1
    name = os.path.basename(path)
    return int(name.split("_")[1])


def run_3d_inversions(cfg):
    """
    Runs Tomofast-x for each blob (WSL or Linux, set by cfg.use_wsl) and copies
    the outputs back. Blobs that already have a log file are skipped, so an
    interrupted run can be started again.
    """
    base_dir = cfg.base_dir
    home = cfg.home
    run_id = cfg.run_id
    out = Out()

    script_start_time = time.time()

    OBS_DIR = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Blob_Clipped_obs_RECT")
    MESH_DIR = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Blob_Meshgrids")
    PAR_DIR = os.path.join(base_dir, "0-Parfiles")

    # where to copy the outputs back
    OUT_WIN_ROOT = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")
    os.makedirs(OUT_WIN_ROOT, exist_ok=True)

    print("Base dir:", base_dir)
    print("Obs dir :", OBS_DIR)
    print("Mesh dir:", MESH_DIR)
    print("Par dir :", PAR_DIR)
    print("Out dir :", OUT_WIN_ROOT)

    # find blob IDs from obs files
    obs_files = sorted(glob.glob(os.path.join(OBS_DIR, "Blob_*_RECT.obs")))
    if not obs_files:
        raise FileNotFoundError(f"No blob obs found in: {OBS_DIR}")

    blob_ids = [blob_id_from_name(f) for f in obs_files]
    out.say("Found blobs:", blob_ids)

    # paths as seen from the Tomofast machine
    OBS_DIR_WSL = host_path(cfg, OBS_DIR)
    MESH_DIR_WSL = host_path(cfg, MESH_DIR)
    PAR_DIR_WSL = host_path(cfg, PAR_DIR)

    run_hidden(cfg, f"mkdir -p '{home}/{run_id}/data' '{home}/{run_id}/output' '{home}/{run_id}/parfile'")

    # loop over blobs
    n_blobs = len(blob_ids)
    for idx, bid in enumerate(blob_ids, start=1):
        # checkpoint: skip if this blob's output log already exists from a previous run
        out_blob_win = os.path.join(OUT_WIN_ROOT, f"Blob_{bid:03d}")
        log_name = f"QDM_log_Blob_{bid:03d}.txt"
        if os.path.exists(os.path.join(out_blob_win, log_name)):
            out.say(f"Skipping Blob {bid:03d} (already completed, found {log_name})")
            continue

        if PRINT_LIMIT != 0:
            print(f"Blob {bid:03d} ({idx}/{n_blobs})".ljust(80), end="\r", flush=True)

        # filenames for this blob
        obs_win = os.path.join(OBS_DIR, f"Blob_{bid:03d}_RECT.obs")
        mesh_win = os.path.join(MESH_DIR, f"Meshgrid_Blob_{bid:03d}.txt")
        par_win = os.path.join(PAR_DIR, f"Parfile_QDM_Inversion_Blob_{bid:03d}.txt")

        # check they exist
        if not os.path.exists(obs_win):
            raise FileNotFoundError(obs_win)
        if not os.path.exists(mesh_win):
            raise FileNotFoundError(mesh_win)
        if not os.path.exists(par_win):
            raise FileNotFoundError(par_win)

        # filenames on the Tomofast machine (source)
        obs_wsl = f"{OBS_DIR_WSL}/Blob_{bid:03d}_RECT.obs"
        mesh_wsl = f"{MESH_DIR_WSL}/Meshgrid_Blob_{bid:03d}.txt"
        par_wsl = f"{PAR_DIR_WSL}/Parfile_QDM_Inversion_Blob_{bid:03d}.txt"

        run_hidden(cfg, f"cp '{par_wsl}'  {home}/{run_id}/parfile/")
        run_hidden(cfg, f"cp '{obs_wsl}'  {home}/{run_id}/data/")
        run_hidden(cfg, f"cp '{mesh_wsl}' {home}/{run_id}/data/")

        # clean previous output to avoid mixing
        run_hidden(cfg, f"rm -rf {home}/{run_id}/output/*")

        # run Tomofast for this blob
        par_name = f"Parfile_QDM_Inversion_Blob_{bid:03d}.txt"
        if cfg.use_wsl:
            run_hidden(
                cfg,
                f"cd {home} && "
                f"{{ time mpirun -np {cfg.nproc} ./tomofastx -p ./{run_id}/parfile/{par_name}; }} "  # time logging
                f"2>&1 | tee ./{run_id}/output/{log_name}"
            )
        else:
            run_hidden(
                cfg,
                f"cd {home} && "
                f"mpirun -np {cfg.nproc} ./tomofastx -p ./{run_id}/parfile/{par_name} | tee ./{run_id}/output/{log_name}"
            )

        # copy outputs back into a separate folder per blob
        os.makedirs(out_blob_win, exist_ok=True)
        out_blob_wsl = host_path(cfg, out_blob_win)

        # everything from this run's output except SENSIT
        run_hidden(
            cfg,
            f"rsync -av --exclude='SENSIT/' "
            f"{home}/{run_id}/output/ '{out_blob_wsl}/'"
        )

    print("\nALL BLOBS DONE.")

    total_time = time.time() - script_start_time
    print(f"\nTotal runtime: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    out.end()


# ============================================================
### Tool 1: 4. Models Misfits ###
# ============================================================
def plot_blob_misfits(cfg):
    """observed, calculated and misfit for every blob and component, plus data cost / R2 / RMS"""
    out = Out()
    root_out = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")

    blob_dirs = list_blobs(root_out)
    out.say("Found blobs:", blob_dirs, always=True)

    metrics_results = []

    for blob in blob_dirs:
        out.say(f"\nProcessing {blob}")

        data_dir = os.path.join(root_out, blob, blob, "data")

        obs_file = os.path.join(data_dir, "mag_observed.txt")
        calc_file = os.path.join(data_dir, "mag_final.txt")
        misfit_file = os.path.join(data_dir, "mag_misfit.txt")

        # check files exist (skip blob if any missing)
        missing = False
        for f in [obs_file, calc_file, misfit_file]:
            if not os.path.exists(f):
                out.say("   Missing:", f)
                missing = True
        if missing:
            continue

        # load + clean
        obs_df = clean_df(load_field_table(obs_file))
        calc_df = clean_df(load_field_table(calc_file))
        misfit_df = clean_df(load_field_table(misfit_file))

        if obs_df.empty or calc_df.empty or misfit_df.empty:
            out.say(f"   Empty/invalid data after cleaning for {blob}, skipping.")
            continue

        # components present: bx/by/bz (vector) or just "value" (Bz only)
        components = [c for c in ("bx", "by", "bz") if c in obs_df.columns]
        if not components:
            components = ["value"]

        for comp in components:
            obs_grid, xs, ys = df_to_grid(obs_df, comp)
            calc_grid, xs2, ys2 = df_to_grid(calc_df, comp)
            misfit_grid, xs3, ys3 = df_to_grid(misfit_df, comp)

            if obs_grid is None or calc_grid is None or misfit_grid is None:
                out.say(f"   No valid grid for {blob} [{comp}], skipping.")
                continue

            # observed grid coords for the extent
            extent = [xs.min(), xs.max(), ys.min(), ys.max()]
            if not np.all(np.isfinite(extent)):
                out.say(f"   Bad extent (NaN/Inf) for {blob} [{comp}], skipping.")
                continue

            # 3-panel figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

            v = np.percentile(np.abs(obs_grid[np.isfinite(obs_grid)]), 99)

            for ax, grid, cbar_label in zip(
                axes,
                [obs_grid, calc_grid, misfit_grid],
                ["Observed", "Calculated", "Misfit (Obs − Calc)"]
            ):
                im = ax.imshow(grid, origin="lower", cmap="seismic", vmin=-v, vmax=v,
                               extent=extent, aspect="equal")

                ax.set_xlabel("X (µm)")
                ax.set_ylabel("Y (µm)")
                ax.margins(0)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.08)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label(f"{cbar_label}  {comp.upper()} (nT)")

            plt.tight_layout(pad=0.2)
            # save per blob, per component
            out_png = os.path.join(root_out, blob, f"{blob}_{comp}_misfit.png")
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            out.show()
            out.say(f"   Saved: {out_png}")

            # data cost, R2, RMS
            obs_vals = obs_df[comp].to_numpy()
            calc_vals = calc_df[comp].to_numpy()
            data_cost, r2, rms = fit_quality(obs_vals, calc_vals)

            out.say(f"   [{comp}] Data cost = {data_cost:.6f}")
            out.say(f"   [{comp}] R2        = {r2:.6f}")
            out.say(f"   [{comp}] RMS       = {rms:.6f}")

            metrics_results.append({
                "Blob": blob,
                "component": comp,
                "n_data": len(obs_vals),
                "data_cost": data_cost,
                "r2": r2,
                "rms": rms,
            })

    print("\nAll blob misfit plots done.")

    # save metrics summary across all blobs
    metrics_df = pd.DataFrame(metrics_results)
    metrics_out = os.path.join(root_out, "blob_misfit_metrics.csv")
    metrics_df.to_csv(metrics_out, index=False)
    print("\nSaved metrics summary:", metrics_out)
    out.table(metrics_df)
    out.end()


# ============================================================
### Tool 1: 4. Models Misfits (all blobs stitched) ###
# ============================================================
def plot_combined_misfit(cfg):
    """stitch all blobs into one combined observed / calculated / misfit map"""
    out = Out()
    root_out = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")
    blob_dirs = list_blobs(root_out)

    obs_all, calc_all, misfit_all = [], [], []

    for blob in blob_dirs:
        data_dir = os.path.join(root_out, blob, blob, "data")
        obs_file = os.path.join(data_dir, "mag_observed.txt")
        calc_file = os.path.join(data_dir, "mag_final.txt")
        misfit_file = os.path.join(data_dir, "mag_misfit.txt")
        if not all(os.path.exists(f) for f in (obs_file, calc_file, misfit_file)):
            continue
        obs_all.append(clean_df(load_field_table(obs_file)))
        calc_all.append(clean_df(load_field_table(calc_file)))
        misfit_all.append(clean_df(load_field_table(misfit_file)))

    obs_df = pd.concat(obs_all, ignore_index=True)
    calc_df = pd.concat(calc_all, ignore_index=True)
    misfit_df = pd.concat(misfit_all, ignore_index=True)

    components = [c for c in ("bx", "by", "bz") if c in obs_df.columns] or ["value"]

    for comp in components:
        comp_label = comp.upper() if comp != "value" else "Bz"

        obs_grid, xs, ys = stitch_to_grid(obs_df, comp)
        calc_grid, _, _ = stitch_to_grid(calc_df, comp)
        misfit_grid, _, _ = stitch_to_grid(misfit_df, comp)
        extent = [xs.min(), xs.max(), ys.min(), ys.max()]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
        vmax = np.nanmax(np.abs(np.concatenate([obs_grid.ravel(), calc_grid.ravel(), misfit_grid.ravel()])))
        vmin = -vmax

        for ax, grid, title in zip(axes, [obs_grid, calc_grid, misfit_grid],
                                   ["Observed", "Calculated", "Misfit (Obs - Calc)"]):
            im = ax.imshow(-grid, origin="lower", cmap="seismic", extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
            ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)"); ax.margins(0)
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.8)
            cbar.set_label(f"{title}  {comp_label} (nT)")

        plt.tight_layout(pad=0.2)
        fig_path = os.path.join(root_out, f"combined_all_blobs_{comp}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        out.show()
        print("Saved:", fig_path)

        data_cost, r2, rms = fit_quality(obs_df[comp].to_numpy(), calc_df[comp].to_numpy())

        print(f"[{comp_label}] Data cost = {data_cost:.6f}")
        print(f"[{comp_label}] R2 = {r2:.6f}")
        print(f"[{comp_label}] RMS = {rms:.6f}")

    out.end()


# ============================================================
### Tool 1: 4. Final Data Costs ###
# ============================================================
def print_final_data_costs(cfg):
    """last 'data cost' line of every blob log"""
    out = Out()
    root_dir = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")

    results = []

    for blob in sorted(os.listdir(root_dir)):
        if not blob.startswith("Blob_"):
            continue

        log_file = os.path.join(root_dir, blob, f"QDM_log_{blob}.txt")

        if not os.path.exists(log_file):
            continue

        last_cost = None

        with open(log_file, "r") as f:
            for line in f:
                if "data cost (new)" in line:
                    # extract the number
                    match = re.search(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", line)
                    if match:
                        last_cost = float(match.group())

        if last_cost is not None:
            results.append({
                "Blob": blob,
                "Final_Data_Cost": last_cost
            })

    df = pd.DataFrame(results).sort_values("Blob").reset_index(drop=True)
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("\n=== Final Data Cost (Last Iteration) ===")
    out.table(df)
    out.end()


# ============================================================
### Tool 1: 5. Extracting Magnetic Volumes from the Inverted Model ###
# ============================================================
def load_blob_model(blob_root, mesh_root, blob, out):
    """magnetization (Mx, My, Mz), mesh centres and cell counts of one blob (None if something is missing)"""
    model_file = os.path.join(blob_root, blob, blob, "model", "mag_final_model_full.txt")
    mesh_file = os.path.join(mesh_root, f"Meshgrid_Blob_{blob.split('_')[1]}.txt")  # uses 001 etc.

    if not os.path.exists(model_file):
        out.say("   Missing model:", model_file)
        return None
    if not os.path.exists(mesh_file):
        out.say("   Missing mesh :", mesh_file)
        return None

    # magnetization model: Mx My Mz ...
    data = np.genfromtxt(model_file, skip_header=1)
    Mx = data[:, 0]
    My = data[:, 1]
    Mz = data[:, 2]
    mag = np.sqrt(Mx**2 + My**2 + Mz**2)

    # mesh (cell bounds) -> centers
    mesh = np.genfromtxt(mesh_file, skip_header=1)

    x_centers = (mesh[:, 0] + mesh[:, 1]) / 2
    y_centers = (mesh[:, 2] + mesh[:, 3]) / 2
    z_centers = (mesh[:, 4] + mesh[:, 5]) / 2

    # nx, ny, nz from the mesh
    n_cells = mesh.shape[0]
    nx = len(np.unique(x_centers))
    ny = len(np.unique(y_centers))
    nz = int(n_cells / (nx * ny))

    if nx * ny * nz != n_cells:
        out.say(f"   Cannot infer nx,ny,nz correctly for {blob}.")
        out.say(f"     nx*ny*nz={nx*ny*nz} but n_cells={n_cells}")
        return None

    out.say(f"  Mesh dims inferred: nx={nx}, ny={ny}, nz={nz}")

    return dict(Mx=Mx, My=My, Mz=Mz, mag=mag, mesh=mesh,
                x_centers=x_centers, y_centers=y_centers, z_centers=z_centers,
                nx=nx, ny=ny, nz=nz)


def extract_grains(cfg, thresholds=None, keep_largest=False):
    """
    Grains from connected cells above each amplitude threshold (fraction of the
    maximum). keep_largest=True also saves the largest grain of every threshold,
    useful for filtering compression artifacts.
    """
    out = Out()

    BLOB_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")
    MESH_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Blob_Meshgrids")

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)  # 0.1..0.9

    # 26-connectivity in 3D
    structure = generate_binary_structure(3, 3)

    blob_dirs = list_blobs(BLOB_ROOT)
    out.say("Found blobs:", blob_dirs, always=True)

    for blob in blob_dirs:
        out.say("\n===================================================")
        out.say("Processing", blob)
        out.say("===================================================")

        m = load_blob_model(BLOB_ROOT, MESH_ROOT, blob, out)
        if m is None:
            continue

        out_folder = os.path.join(BLOB_ROOT, blob, blob, "Grains_detection")
        os.makedirs(out_folder, exist_ok=True)

        # largest grain info per threshold (one row per threshold)
        blob_largest_summary = []

        for threshold_per in thresholds:
            thr_str = f"{threshold_per:.1f}"
            out.say(f"   -> threshold = {thr_str}")

            threshold = threshold_per * m["mag"].max()
            mask = m["mag"] > threshold

            # reshape into (nz, ny, nx)
            mask_3d = mask.reshape((m["nz"], m["ny"], m["nx"]))

            labeled_grains_3d, num_grains = label(mask_3d, structure=structure)
            out.say(f"      grains detected: {num_grains}")

            df = pd.DataFrame({
                "X": m["x_centers"].ravel(),
                "Y": m["y_centers"].ravel(),
                "Z": -1 * m["z_centers"].ravel(),     # keep the sign convention
                "GrainID": labeled_grains_3d.ravel(),
                "Mx": m["Mx"],
                "My": m["My"],
                "Mz": m["Mz"]
            })

            # remove background
            df = df[df["GrainID"] != 0]

            csv_file = os.path.join(out_folder, f"grain_labels_{thr_str}.csv")
            df.to_csv(csv_file, index=False)
            out.say(f"      saved: {csv_file}")

            if not keep_largest:
                continue

            ### Tool 1: 5. Extracting Magnetic Volumes (largest grains only) ###
            # no grains: skip largest outputs
            if df.empty:
                out.say("      (no voxels above threshold) skip largest")
                continue

            # largest grain by voxel count
            counts = df["GrainID"].value_counts()
            largest_id = int(counts.idxmax())
            largest_nvox = int(counts.max())

            df_largest = df[df["GrainID"] == largest_id].copy()

            largest_csv = os.path.join(out_folder, f"grain_labels_largest_{thr_str}.csv")
            df_largest.to_csv(largest_csv, index=False)
            out.say(f"       saved largest: {largest_csv} (GrainID={largest_id}, Num_Voxels={largest_nvox})")

            blob_largest_summary.append({
                "Blob": blob,
                "Threshold": float(threshold_per),
                "Threshold_abs": float(threshold),   # threshold in mag units
                "Largest_GrainID": largest_id,
                "Num_Voxels": largest_nvox
            })

        if keep_largest:
            # per-blob summary over all thresholds
            if blob_largest_summary:
                summary_df = pd.DataFrame(blob_largest_summary).sort_values("Threshold")
                summary_path = os.path.join(out_folder, "grain_labels_largest_ALL.csv")
                summary_df.to_csv(summary_path, index=False)
                out.say(f"\n   saved blob largest summary: {summary_path}")
            else:
                out.say("\n  (no largest summary rows written for this blob)")

    print("\nDONE.")
    out.end()


# ============================================================
### Tool 1: 5. 3D grains plots ###
# ============================================================
# voxel rendering gets slow above roughly this many filled cells
VOXEL_WARN_LIMIT = 100_000


def plot_3D_grains_voxels(labeled_grains_3d, dzyx, filename, title, out):
    """Detected grains as coloured 3D voxel cubes, one solid colour per GrainID."""
    # labeled_grains_3d comes in as (nz, ny, nx); transpose to (nx, ny, nz)
    model = labeled_grains_3d.T
    filled = model != 0
    L, W, H = model.shape

    if not np.any(filled):
        out.say("      (no voxels above threshold) skip plot")
        return

    if filled.sum() > VOXEL_WARN_LIMIT:
        out.say(f"      WARNING: {filled.sum()} filled voxels -- ax.voxels() "
                f"can be slow/heavy above ~{VOXEL_WARN_LIMIT}. Rendering anyway.")

    grain_ids = np.unique(model[filled])
    cmap = plt.colormaps["tab20"] if len(grain_ids) <= 20 else plt.colormaps["turbo"]
    id_to_color = {gid: cmap(i / max(len(grain_ids) - 1, 1)) for i, gid in enumerate(grain_ids)}

    facecolors = np.empty(model.shape, dtype=object)
    edgecolors = np.empty(model.shape, dtype=object)
    for i, j, k in zip(*np.where(filled)):
        rgba = id_to_color[model[i, j, k]]
        facecolors[i, j, k] = colors.rgb2hex(rgba)
        edgecolors[i, j, k] = "#00000055"  # thin, semi-transparent voxel edges

    # transparent bounding-box corners so every plot shares the same axis extent
    corners = [(0, 0, 0), (L - 1, 0, 0), (0, W - 1, 0), (0, 0, H - 1),
               (L - 1, W - 1, 0), (L - 1, 0, H - 1), (0, W - 1, H - 1), (L - 1, W - 1, H - 1)]
    for i, j, k in corners:
        filled[i, j, k] = True
        facecolors[i, j, k] = "#00000000"
        edgecolors[i, j, k] = "#00000000"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(45, -45)

    x, y, z = np.indices(np.array(filled.shape) + 1)
    x = x * dzyx[2]; y = y * dzyx[1]; z = z * dzyx[0]

    ax.voxels(x, y, z, filled, facecolors=facecolors, edgecolors=edgecolors, shade=False)
    plt.axis("scaled")
    ax.invert_zaxis()
    ax.set_xlabel("X", labelpad=1)
    ax.set_ylabel("Y", labelpad=1)
    ax.set_zlabel("Z", labelpad=-4)
    ax.set_title(title)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grains_3d(cfg, thresholds=None):
    """3d grains plots (voxel-style coloured by GrainID), saved in each blob folder"""
    out = Out()

    BLOB_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")
    MESH_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Blob_Meshgrids")

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)  # 0.1 .. 0.9
    structure = generate_binary_structure(3, 3)  # 26-connectivity

    blob_dirs = list_blobs(BLOB_ROOT)
    out.say("Found blobs:", blob_dirs, always=True)

    for blob in blob_dirs:
        out.say("\n===================================================")
        out.say("Processing", blob)
        out.say("===================================================")

        # Blob_001/Blob_001/model/mag_final_model_full.txt
        model_file = os.path.join(BLOB_ROOT, blob, blob, "model", "mag_final_model_full.txt")

        # Meshgrid_Blob_001.txt etc
        blob_num_str = blob.split("_")[1]
        mesh_file = os.path.join(MESH_ROOT, f"Meshgrid_Blob_{blob_num_str}.txt")

        if not os.path.exists(model_file):
            out.say("   Missing model:", model_file)
            continue
        if not os.path.exists(mesh_file):
            out.say("   Missing mesh :", mesh_file)
            continue

        # model (Mx, My, Mz)
        model = np.genfromtxt(model_file, skip_header=1)
        Mx = model[:, 0]
        My = model[:, 1]
        Mz = model[:, 2]
        mag = np.sqrt(Mx**2 + My**2 + Mz**2)

        # mesh -> centers + cell spacing
        mesh = np.genfromtxt(mesh_file, skip_header=1)
        x_centers = (mesh[:, 0] + mesh[:, 1]) / 2
        y_centers = (mesh[:, 2] + mesh[:, 3]) / 2
        z_centers = (mesh[:, 4] + mesh[:, 5]) / 2

        # cell size (assumed uniform per blob); needed for correctly-scaled voxels
        dx_cell = np.median(mesh[:, 1] - mesh[:, 0])
        dy_cell = np.median(mesh[:, 3] - mesh[:, 2])
        dz_cell = np.median(mesh[:, 5] - mesh[:, 4])
        dzyx = (dz_cell, dy_cell, dx_cell)

        # nx, ny, nz per blob
        n_cells = mesh.shape[0]
        nx = len(np.unique(x_centers))
        ny = len(np.unique(y_centers))
        nz = int(n_cells / (nx * ny))

        if nx * ny * nz != n_cells:
            out.say(f"   Bad dims: nx*ny*nz={nx*ny*nz} but cells={n_cells}")
            continue

        out.say(f"  Mesh dims: nx={nx}, ny={ny}, nz={nz}")

        out_folder = os.path.join(BLOB_ROOT, blob, blob, "Grains_detection_3D")
        os.makedirs(out_folder, exist_ok=True)

        for threshold_per in thresholds:
            out.say(f"   -> threshold = {threshold_per:.1f}")

            threshold = threshold_per * mag.max()
            mask = mag > threshold
            mask_3d = mask.reshape((nz, ny, nx))

            labeled_grains_3d, num_grains = label(mask_3d, structure=structure)
            out.say(f"      grains detected: {num_grains}")

            if num_grains == 0:
                out.say("      (no points above threshold) skip plot")
                continue

            out_fig = os.path.join(out_folder, f"{blob}_grains_3D_{threshold_per:.1f}.png")
            title = f"{blob} — 3D Grains (thr={threshold_per:.1f}, n={num_grains})"
            plot_3D_grains_voxels(labeled_grains_3d, dzyx, out_fig, title, out)

            out.say(f"      saved: {out_fig}")

    print("\nDONE.")
    out.end()


# ============================================================
### Tool 1: 6. 3D Calculations ###
# ============================================================
def grain_statistics(cfg, thresholds=None, keep_largest=False):
    """
    Per-grain mean direction and net moment from the grain_labels files.
    keep_largest=True also saves the largest grain of each threshold
    (grain_dec_inc_largest_*.csv and grain_dec_inc_largest_ALL.csv).
    """
    out = Out()

    BLOB_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")

    # cell size (m), used for the moment
    dx = dy = dz = cfg.d * 1e-6
    V = dx * dy * dz

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.1)

    blob_dirs = list_blobs(BLOB_ROOT)
    out.say("Found blobs:", blob_dirs, always=True)

    for blob in blob_dirs:
        out.say("\n===================================================")
        out.say("Processing", blob)
        out.say("===================================================")

        # Blob_001/Blob_001/Grains_detection
        grains_folder = os.path.join(BLOB_ROOT, blob, blob, "Grains_detection")

        if not os.path.isdir(grains_folder):
            out.say("   Missing Grains_detection folder:", grains_folder)
            out.say("     (Run the grain_labels script first for this blob.)")
            continue

        out_folder = grains_folder

        # largest grain per threshold for this blob
        blob_largest_rows = []

        for threshold_per in thresholds:
            thr_str = f"{threshold_per:.1f}"
            out.say(f"\n  === threshold {thr_str} ===")

            csv_path = os.path.join(grains_folder, f"grain_labels_{thr_str}.csv")

            if not os.path.exists(csv_path):
                out.say("    Missing:", csv_path)
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                out.say("   (empty file) skip")
                continue

            results = []

            for grain_id, group in df.groupby("GrainID"):
                vectors = group[["Mx", "My", "Mz"]].to_numpy()
                n_vectors = len(vectors)

                mean_Mx, mean_My, mean_Mz_raw = vectors.mean(axis=0)
                mean_Mz = -mean_Mz_raw  # sign flip, matches tomofast convention
                mean_vec = np.array([mean_Mx, mean_My, mean_Mz], dtype=float)

                mags = np.linalg.norm(vectors, axis=1)
                mean_magnitude = float(mags.mean()) if len(mags) else np.nan

                mean_vec_length = float(np.linalg.norm(mean_vec))
                if mean_magnitude and mean_magnitude > 0:
                    R = mean_vec_length / mean_magnitude
                else:
                    R = np.nan

                # angular standard deviation
                if (not np.isnan(R)) and (R > 0):
                    sigma_rad = np.sqrt(-2.0 * np.log(R))
                    sigma_deg = float(np.degrees(sigma_rad))
                else:
                    sigma_deg = np.nan

                # dec/inc
                m = mean_vec_length
                if m == 0 or np.isnan(m):
                    dec, inc = np.nan, np.nan
                else:
                    theta = float(np.degrees(np.arccos(mean_Mz / m)))
                    phi = float(np.degrees(np.arctan2(mean_My, mean_Mx)))
                    inc = theta - 90.0
                    dec = float(np.mod(90.0 - phi, 360.0))

                # total grain moment = net (vector sum) of the voxel magnetizations x fixed cell volume V
                moment_amp = np.linalg.norm(vectors.sum(axis=0)) * V if len(mags) else np.nan

                # moment-weighted dipole position
                Mi = mags
                Xi = group["X"].to_numpy()
                Yi = group["Y"].to_numpy()
                Zi = group["Z"].to_numpy()

                weight = Mi * V
                w_sum = float(weight.sum())

                if w_sum > 0:
                    r_dip_x = float((weight * Xi).sum() / w_sum)
                    r_dip_y = float((weight * Yi).sum() / w_sum)
                    r_dip_z = float((weight * Zi).sum() / w_sum)
                else:
                    r_dip_x = r_dip_y = r_dip_z = np.nan

                results.append({
                    "Blob": blob,
                    "Threshold": float(threshold_per),
                    "GrainID": int(grain_id),
                    "Num_Voxels": int(n_vectors),
                    "Mx_mean": mean_Mx,
                    "My_mean": mean_My,
                    "Mz_mean": mean_Mz,
                    "Mean_Magnitude": mean_magnitude,
                    "Net_Moment_Amplitude": moment_amp,
                    "Resultant_R": R,
                    "Angular_STD_deg": sigma_deg,
                    "Dec": dec,
                    "Inc": inc,
                    "Dipole_X": r_dip_x,
                    "Dipole_Y": r_dip_y,
                    "Dipole_Z": r_dip_z,
                })

            # save the full per-grain results
            out_name = f"grain_dec_inc_{thr_str}.csv"
            out_path = os.path.join(out_folder, out_name)

            out_df = pd.DataFrame(results)
            out_df.to_csv(out_path, index=False)
            out.say("    Saved:", out_path)

            if not keep_largest:
                continue

            ### Tool 1: 6. 3D Calculations (largest grains only) ###
            # pick and save only the largest grain (by Num_Voxels)
            if out_df.empty:
                out.say("   (no grains computed) skip largest")
                continue

            largest_row = out_df.loc[out_df["Num_Voxels"].idxmax()].copy()

            out_largest_path = os.path.join(out_folder, f"grain_dec_inc_largest_{thr_str}.csv")

            pd.DataFrame([largest_row]).to_csv(out_largest_path, index=False)
            out.say("    Saved largest:", out_largest_path,
                    f"(GrainID={int(largest_row['GrainID'])}, Num_Voxels={int(largest_row['Num_Voxels'])})")

            blob_largest_rows.append(largest_row)

        if keep_largest and blob_largest_rows:
            # "largest per threshold" summary for this blob
            blob_largest_df = pd.DataFrame(blob_largest_rows).sort_values("Threshold")
            blob_largest_all_path = os.path.join(out_folder, "grain_dec_inc_largest_ALL.csv")
            blob_largest_df.to_csv(blob_largest_all_path, index=False)
            out.say("\n    Saved blob largest summary:", blob_largest_all_path)

    print("\nDONE.")
    out.end()


# shared by the two Dec, Inc, Moment plots below
def _plot_dec_inc_moment(plot_df, BLOB_ROOT, out_name, out):
    # one panel each for Dec, Inc, Moment vs threshold, one line per blob
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    blobs = sorted(plot_df["Blob"].unique())
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, blob in enumerate(blobs):
        sub = plot_df[plot_df["Blob"] == blob].sort_values("Threshold")
        color = cmap(i % 10)
        marker = markers[i % len(markers)]

        axes[0].plot(sub["Threshold"], sub["Dec"], marker=marker, color=color, label=blob)
        axes[1].plot(sub["Threshold"], sub["Inc"], marker=marker, color=color, label=blob)
        axes[2].plot(sub["Threshold"], sub["Net_Moment_Amplitude"], marker=marker, color=color, label=blob)

    axes[0].set_ylabel("Declination (°)", fontsize=12)
    axes[1].set_ylabel("Inclination (°)", fontsize=12)
    axes[2].set_ylabel("Net Moment Amplitude (A·m²)", fontsize=12)

    for ax in axes:
        ax.set_xlabel("Threshold", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(BLOB_ROOT, out_name)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    out.show()
    print("Saved:", out_png)


# ============================================================
### Tool 1: 6. Plot Dec, Inc, Moment vs Threshold (largest grains) ###
# ============================================================
def plot_dec_inc_moment_largest(cfg):
    """Dec, Inc and moment vs threshold, one line per blob (reads grain_dec_inc_largest_ALL.csv)"""
    out = Out()
    BLOB_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")

    blob_dirs = list_blobs(BLOB_ROOT)

    all_rows = []

    for blob in blob_dirs:
        grains_folder = os.path.join(BLOB_ROOT, blob, blob, "Grains_detection")
        csv_path = os.path.join(grains_folder, "grain_dec_inc_largest_ALL.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No grain_dec_inc_largest_ALL.csv files found - run the largest-grain script first.")

    plot_df = pd.concat(all_rows, ignore_index=True)

    _plot_dec_inc_moment(plot_df, BLOB_ROOT, "dec_inc_moment_vs_threshold_largest.png", out)
    out.end()


# ============================================================
### Tool 1: 6. Plot Dec, Inc, Moment vs Threshold ###
# ============================================================
def plot_dec_inc_moment(cfg, thresholds=None):
    """Dec, Inc and moment vs threshold, one line per blob (reads grain_dec_inc_{threshold}.csv)"""
    out = Out()
    BLOB_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.1)

    blob_dirs = list_blobs(BLOB_ROOT)

    all_rows = []

    for blob in blob_dirs:
        grains_folder = os.path.join(BLOB_ROOT, blob, blob, "Grains_detection")
        if not os.path.isdir(grains_folder):
            continue

        for threshold_per in thresholds:
            csv_path = os.path.join(grains_folder, f"grain_dec_inc_{threshold_per:.1f}.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # keep only the largest grain at this threshold, for a clean per-blob trend
            largest_row = df.loc[df["Num_Voxels"].idxmax()]
            all_rows.append(largest_row)

    if not all_rows:
        raise RuntimeError("No data found - check BLOB_ROOT / thresholds / file naming.")

    plot_df = pd.DataFrame(all_rows)

    _plot_dec_inc_moment(plot_df, BLOB_ROOT, "dec_inc_moment_vs_threshold.png", out)
    out.end()


# ============================================================
### Tool 1: 7. Maximum cell Extraction ###
# ============================================================
def extract_max_cells(cfg, thresholds=None, target_grain_ids=None, tol=1e-6):
    """3D location (mesh row) of the maximum cell of each largest grain, per threshold"""
    out = Out()

    BLOB_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")
    MESH_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Blob_Meshgrids")

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    blob_dirs = list_blobs(BLOB_ROOT)
    out.say("Found blobs:", blob_dirs, always=True)

    for blob in blob_dirs:
        out.say("\n===================================================")
        out.say("Processing", blob)
        out.say("===================================================")

        blob_num_str = blob.split("_")[1]

        # blob grain labels folder
        grains_folder = os.path.join(BLOB_ROOT, blob, blob, "Grains_detection")
        if not os.path.isdir(grains_folder):
            out.say("   Missing Grains_detection:", grains_folder)
            continue

        # blob mesh file
        mesh_path = os.path.join(MESH_ROOT, f"Meshgrid_Blob_{blob_num_str}.txt")
        if not os.path.exists(mesh_path):
            out.say("   Missing mesh file:", mesh_path)
            continue

        # output folder inside blob
        meshs_folder = os.path.join(BLOB_ROOT, blob, blob, "Maxs_meshs")
        os.makedirs(meshs_folder, exist_ok=True)

        # load blob mesh once
        mesh = np.genfromtxt(mesh_path, skip_header=1)

        # centers
        x_c = (mesh[:, 0] + mesh[:, 1]) / 2
        y_c = (mesh[:, 2] + mesh[:, 3]) / 2
        z_c = (mesh[:, 4] + mesh[:, 5]) / 2  # positive down in Tomofast

        for threshold_per in thresholds:
            out.say(f"\n  === Threshold {threshold_per:.1f} ===")

            # largest grain per blob (filtered); for all grains use grain_labels_{thr}.csv
            csv_path = os.path.join(grains_folder, f"grain_labels_largest_{threshold_per:.1f}.csv")
            if not os.path.isfile(csv_path):
                out.say("   CSV not found:", csv_path)
                continue

            df = pd.read_csv(csv_path)

            if target_grain_ids is not None:
                df = df[df["GrainID"].isin(target_grain_ids)]

            if df.empty:
                out.say("   No rows in this CSV.")
                continue

            rows_out = []

            # for each grain: pick the voxel with max |M|
            for grain_id, group in df.groupby("GrainID"):
                mags = np.sqrt(
                    group["Mx"].values**2 +
                    group["My"].values**2 +
                    group["Mz"].values**2
                )

                i_max = np.argmax(mags)
                row = group.iloc[i_max]

                X0, Y0, Z0 = row["X"], row["Y"], row["Z"]

                # match to mesh centers (df stores Z = -z_center)
                idx_mesh_arr = np.where(
                    (np.abs(x_c - X0) < tol) &
                    (np.abs(y_c - Y0) < tol) &
                    (np.abs(-z_c - Z0) < tol)
                )[0]

                if len(idx_mesh_arr) == 0:
                    out.say(f"   Grain {grain_id}: no mesh match (float mismatch).")
                    continue

                idx_mesh = idx_mesh_arr[0]

                # GrainID + mesh row
                rows_out.append(np.hstack(([grain_id], mesh[idx_mesh, :])))

            # save output
            if len(rows_out) > 0:
                rows_out = np.array(rows_out)

                out_txt = os.path.join(meshs_folder, f"grain_max_meshrow_{threshold_per:.1f}.txt")

                with open(out_txt, "w") as f:
                    f.write(f"{rows_out.shape[0]}\n")
                    for row in rows_out:
                        # GrainID as int, rest as floats
                        f.write(f"{int(row[0])} " + " ".join([f"{v:.6f}" for v in row[1:]]) + "\n")

                out.say("    Saved:", out_txt)
            else:
                out.say("   No mesh rows found for this threshold.")

    print("\nDONE.")
    out.end()


# ============================================================
### Tool 1: 7. Combine all max cells ###
# ============================================================
def combine_max_cells(cfg, thr=0.9):
    """all the max cells in one file, for the source location search (thr = the best amplitude threshold)"""
    out = Out()
    BLOB_ROOT = os.path.join(cfg.base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")

    fname = f"grain_max_meshrow_{thr:.1f}.txt"

    all_rows = []

    blob_dirs = list_blobs(BLOB_ROOT)
    out.say("Found blobs:", blob_dirs, always=True)

    for blob in blob_dirs:
        path_txt = os.path.join(BLOB_ROOT, blob, blob, "Maxs_meshs", fname)

        if not os.path.exists(path_txt):
            out.say("Missing:", path_txt)
            continue

        # first line is count -> skip it
        arr = np.loadtxt(path_txt, skiprows=1)

        # if only one row, make it 2D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # add blob column
        for row in arr:
            all_rows.append([blob] + row.tolist())

        out.say(f"Read {blob}: {arr.shape[0]} rows")

    # save combined csv
    if len(all_rows) == 0:
        print("\nNo files found / no rows read.")
        df = None
    else:
        # columns: Blob + GrainID + mesh columns
        cols = ["Blob", "GrainID",
                "X1", "X2", "Y1", "Y2", "Z1", "Z2",
                "I", "J", "K"]

        df = pd.DataFrame(all_rows, columns=cols)

        out_csv = os.path.join(BLOB_ROOT, f"ALL_BLOBS_grain_max_meshrow_{thr:.1f}.csv")
        df.to_csv(out_csv, index=False)

        print("\nSaved combined csv:", out_csv)

        # combined txt (Tomofast-style: first line = total)
        out_txt = os.path.join(BLOB_ROOT, f"ALL_BLOBS_grain_max_meshrow_{thr:.1f}.txt")

        with open(out_txt, "w") as f:
            f.write(f"{len(df)}\n")
            for _, r in df.iterrows():
                # blob name first, then the numbers
                f.write(
                    f"{r['Blob']} {int(r['GrainID'])} "
                    f"{r['X1']:.6f} {r['X2']:.6f} "
                    f"{r['Y1']:.6f} {r['Y2']:.6f} "
                    f"{r['Z1']:.6f} {r['Z2']:.6f} "
                    f"{int(r['I'])} {int(r['J'])} {int(r['K'])}\n"
                )

        print("Saved combined txt:", out_txt)

    print("\nAll grains completed; max cells:")
    if df is not None:
        out.table(df)
    out.end()
    return df


# ============================================================
### Tool 1: 7. Max cells plot ###
# ============================================================
def plot_max_cells(cfg, thr=0.9):
    """max cells on the Bz map"""
    out = Out()
    base_dir = cfg.base_dir

    # 1. Bz data (QDM convention)
    x, y, bz = read_bz_obs(base_dir)

    # 2. max cells (combined file)
    max_cells_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs",
                                  f"ALL_BLOBS_grain_max_meshrow_{thr:.1f}.csv")
    df = pd.read_csv(max_cells_path)

    # cell centres
    mx = 0.5 * (df["X1"].values + df["X2"].values)
    my = 0.5 * (df["Y1"].values + df["Y2"].values)

    # 3. colour scaling (clip outliers)
    bz_min = np.percentile(bz, 2)
    bz_max = np.percentile(bz, 98)

    # 4. plot
    plt.style.use("seaborn-v0_8-white")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    sc = ax.scatter(x, y, c=bz, s=10, cmap="seismic", vmin=bz_min, vmax=bz_max, rasterized=True)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Bz (nT)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # 5. overlay max cells
    ax.scatter(mx, my, s=55, c="lime", edgecolors="black", linewidths=0.6, marker="s",
               label=f"Max cells (THR={thr:.1f})", zorder=3)

    # 6. labels and styling
    ax.set_xlabel("X coordinate (µm)", fontsize=12)
    ax.set_ylabel("Y coordinate (µm)", fontsize=12)
    ax.set_title("Bz Map with Tomofast Max Cells", fontsize=14)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    # 7. save figure
    out_png = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs",
                           f"BZ_max_cells_THR_{thr:.1f}.png")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    out.show()

    print("Saved figure:", out_png)
    out.end()


# ============================================================
### Tool 1: 7. Max cells plot (coloured by depth) ###
# ============================================================
def plot_max_cells_depth(cfg, thr=0.9):
    """max cells on the Bz map, coloured by depth"""
    out = Out()
    base_dir = cfg.base_dir

    # 1. Bz data (QDM convention)
    x, y, bz = read_bz_obs(base_dir)

    # 2. max cells
    max_cells_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs",
                                  f"ALL_BLOBS_grain_max_meshrow_{thr:.1f}.csv")
    df = pd.read_csv(max_cells_path)

    mx = 0.5 * (df["X1"].values + df["X2"].values)
    my = 0.5 * (df["Y1"].values + df["Y2"].values)
    mz = 0.5 * (df["Z1"].values + df["Z2"].values)   # depth

    # 3. colour scaling
    bz_min = np.percentile(bz, 2)
    bz_max = np.percentile(bz, 98)

    # 4. plot
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    sc = ax.scatter(x, y, c=bz, s=5, cmap="seismic", vmin=bz_min, vmax=bz_max, rasterized=True)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.66, pad=0.02)
    cbar.set_label("Bz (nT)")

    # 5. max cells coloured by depth
    sc_max = ax.scatter(mx, my, s=55, c=mz, cmap="viridis", edgecolors="black",
                        linewidths=0.6, marker="s", zorder=3)

    plt.colorbar(sc_max, ax=ax, orientation="horizontal", pad=0.10)

    # 6. labels and save
    ax.set_xlabel("X coordinate (µm)")
    ax.set_ylabel("Y coordinate (µm)")
    ax.set_aspect("equal")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    out_png = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs",
                           f"BZ_max_cells_THR_{thr:.1f}.png")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    out.show()
    out.end()


# ============================================================
### Tool 2: 1. Create Parfile ###
# ============================================================
# Parfile template for the single cell inversions
PAR_TEMPLATE_SINGLE_CELL = """==================================================================================
GLOBAL
==================================================================================
global.outputFolderPath         = ./{run_id}/output/Blob_{blob_id:03d}
global.description              = QDM inversion – Blob {blob_id}

==================================================================================
MODEL GRID parameters
==================================================================================
# nx ny nz
modelGrid.size                        = 1 1 1
modelGrid.magn.file                   = {run_id}/data/meshgrid_blob{blob_id:03d}.txt
modelGrid.magn.nModelComponents       = 3

==================================================================================
DATA parameters
==================================================================================
forward.data.magn.nData                = {ndata}
forward.data.magn.dataGridFile         = {run_id}/data/Blob_{blob_id:03d}_RECT.obs
forward.data.magn.nDataComponents      = {ncomp}

==================================================================================
MAGNETIC FIELD constants
==================================================================================
forward.magneticField.inclination              = 90.0
forward.magneticField.declination              = 0.0
forward.magneticField.XaxisDeclination         = 0.d0

==================================================================================
DEPTH WEIGHTING
==================================================================================
# 1-depth weight, 2-distance weight.
forward.depthWeighting.type                    = 1
forward.depthWeighting.magn.Z0             = {z0} #adjust z0 paramter to data level (5)

==================================================================================
SENSITIVITY KERNEL
==================================================================================
sensit.readFromFiles                  = 0
sensit.folderPath                     = ./{run_id}/output/SENSIT/

==================================================================================
MATRIX COMPRESSION
==================================================================================
forward.matrixCompression.type                        = 0
forward.matrixCompression.rate                        = 0.1

==================================================================================
PRIOR MODEL
==================================================================================
inversion.priorModel.type                  = 1
inversion.priorModel.magn.value            = 0.000000

==================================================================================
INVERSION parameters
==================================================================================
inversion.nMajorIterations              = 3
inversion.nMinorIterations              = 100
inversion.targetMisfit = 100 #RMS_nT target misfit (measurement error) to avoid overfitting


==================================================================================
MODEL DAMPING (m - m_prior)
==================================================================================
inversion.modelDamping.magn.weight      = 0.0

==================================================================================
JOINT INVERSION parameters
==================================================================================
inversion.joint.grav.problemWeight      = 0.00000000
inversion.joint.magn.problemWeight      = 1.00000000
"""


def create_single_cell_parfiles(cfg, n_data_components=3, z0=10.0):
    """
    n_data_components: 3 if using Bx, By, Bz, or 1 if using Bz only
    z0: depth weighting Z0, the instrument height (matching the Z column in the data)
    """
    base_dir = cfg.base_dir
    out = Out()

    # input: window summary (blob_id nx ny nz ndata)
    summary_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "blob_window_summary.txt")

    # output: parfiles
    par_out_dir = os.path.join(base_dir, "0-Parfiles")
    os.makedirs(par_out_dir, exist_ok=True)

    summary = np.loadtxt(summary_path, comments="#", ndmin=2)

    print("Creating Parfiles...")

    for row in summary:
        blob_id = int(row[0])
        nx = int(row[1])
        ny = int(row[2])
        nz = int(row[3])
        ndata = int(row[4])

        par_text = PAR_TEMPLATE_SINGLE_CELL.format(
            run_id=cfg.run_id,
            blob_id=blob_id,
            nx=nx,
            ny=ny,
            nz=nz,
            ndata=ndata,
            ncomp=n_data_components,
            z0=z0
        )

        out_path = os.path.join(par_out_dir, f"Parfile_QDM_Inversion_Blob_{blob_id:03d}.txt")

        with open(out_path, "w") as f:
            f.write(par_text)

        out.say(f"  Blob {blob_id:03d}: nx={nx}, ny={ny}, nz={nz}, nData={ndata}")
        out.say("    ->", out_path)

    print("\nDone. Parfiles written to:", par_out_dir)
    out.end()


# ============================================================
### Tool 2: 2. Run Tomofast for minimum Data Cost Detection ###
# ============================================================
# neighbour offsets (6 neighbours), in steps of (dx,dy,dz)
NEIGH = [
    (-1, 0, 0),  # -x
    (1, 0, 0),   # +x
    (0, -1, 0),  # -y
    (0, 1, 0),   # +y
    (0, 0, -1),  # -z
    (0, 0, 1),   # +z
]


def run_steepest_descent(cfg,
                         start_mode="seed",   # "seed" = anomaly centers, "maxs" = max cells of the 3D inversion
                         thr=0.9,             # threshold of the max cells (only used when start_mode="maxs")
                         max_steps=200,
                         n_workers=None):     # blobs run at the same time (None = NPROC)
    """
    Steepest-descent search for each blob: finds the single cell position with the
    minimum Tomofast-x data cost. Blobs already in best_per_blob_steepest.txt are
    skipped, so an interrupted run can be started again.
    """
    base_dir = cfg.base_dir
    home = cfg.home
    run_id = cfg.run_id
    out = Out()

    START_MODE = start_mode
    MAX_STEPS = max_steps
    in_name = f"grain_max_meshrow_{thr:.1f}.txt"
    N_WORKERS = cfg.nproc if n_workers is None else n_workers

    # folders
    BLOB_ROOT = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")
    obs_dir = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Blob_Clipped_obs_RECT")
    par_dir = os.path.join(base_dir, "0-Parfiles")

    out_dir = os.path.join(base_dir, "4-Source_Detection", "Results_DataCost_Steepest")
    os.makedirs(out_dir, exist_ok=True)

    mesh_temp_dir = os.path.join(base_dir, "4-Source_Detection", "Mesh_Temp")
    os.makedirs(mesh_temp_dir, exist_ok=True)

    # Tomofast output (data and model folders) of the winning fit of each blob
    SAVE_DIR = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output_Steepest")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Base dir:", base_dir)
    print("BLOB_ROOT:", BLOB_ROOT)
    print("obs_dir :", obs_dir)
    print("par_dir :", par_dir)
    print("out_dir :", out_dir)
    print("mesh_temp_dir:", mesh_temp_dir)
    print("SAVE_DIR:", SAVE_DIR)

    # seed file (used only if START_MODE="seed")
    seed_meshrow_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "blob_seed_meshrow.txt")

    blob_dirs = sorted(
        f.replace("_RECT.obs", "") for f in os.listdir(obs_dir)
        if f.startswith("Blob_") and f.endswith("_RECT.obs")
    )

    # read seed file once (if selected)
    seed_map = {}
    if START_MODE == "seed":
        lines = [ln.strip() for ln in open(seed_meshrow_path, "r").readlines() if ln.strip()]

        for ln in lines[1:]:
            c = ln.split()

            # c[0] = "Blob_001"
            blob_id = int(c[0].split("_")[1])

            # Blob_001 1 xmin xmax ymin ymax zmin zmax 0 0 0 i j k
            xmin0 = float(c[2])
            xmax0 = float(c[3])
            ymin0 = float(c[4])
            ymax0 = float(c[5])
            zmin0 = float(c[6])
            zmax0 = float(c[7])

            seed_map[blob_id] = (xmin0, xmax0, ymin0, ymax0, zmin0, zmax0)

    t0 = time.time()

    run_quiet(cfg, f"mkdir -p '{home}/{run_id}/data' '{home}/{run_id}/output' '{home}/{run_id}/parfile'")

    # checkpoint: read any blobs already completed from a previous run
    summary_path = os.path.join(out_dir, "best_per_blob_steepest.txt")
    completed_blobs = set()

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for ln in f:
                if ln.startswith("#") or not ln.strip():
                    continue
                completed_blobs.add(ln.split()[0])
        print(f"Checkpoint found: {len(completed_blobs)} blob(s) already done -> will skip these")
    else:
        with open(summary_path, "w") as f:
            f.write("# blob grain meshfile xmin xmax ymin ymax zmin zmax final_data_cost\n")

    best_per_blob = []
    summary_lock = threading.Lock()

    # one blob; dispatched to N_WORKERS threads below
    def _run_one_blob(blob):
        if blob in completed_blobs:
            out.say(f"Skipping {blob} (already completed, found in checkpoint)")
            return None

        out.say("\n===================================================")
        out.say("Processing", blob)
        out.say("===================================================")

        blob_id = int(blob.split("_")[1])

        # temp mesh file (per blob, so parallel blobs don't collide)
        tmp_mesh_win = os.path.join(mesh_temp_dir, f"_meshgrid_tmp_blob{blob_id:03d}.txt")
        tmp_mesh_wsl = host_path(cfg, tmp_mesh_win)
        meshgrid_target_wsl = f"{home}/{run_id}/data/meshgrid_blob{blob_id:03d}.txt"

        # 1) read start cell bounds
        if START_MODE == "maxs":
            startfile = os.path.join(BLOB_ROOT, blob, blob, "Maxs_meshs", in_name)
            lines = [ln.strip() for ln in open(startfile, "r").readlines() if ln.strip()]
            cols = lines[1].split()  # line0=count, line1=grain row
            grain_id = int(float(cols[0]))
            xmin0, xmax0 = float(cols[1]), float(cols[2])
            ymin0, ymax0 = float(cols[3]), float(cols[4])
            zmin0, zmax0 = float(cols[5]), float(cols[6])

        else:  # START_MODE == "seed"
            grain_id = 1
            xmin0, xmax0, ymin0, ymax0, zmin0, zmax0 = seed_map[blob_id]

        dx, dy, dz = (xmax0 - xmin0), (ymax0 - ymin0), (zmax0 - zmin0)
        cx, cy, cz = 0.5 * (xmin0 + xmax0), 0.5 * (ymin0 + ymax0), 0.5 * (zmin0 + zmax0)

        out.say(f"Seed center=({cx:.3f},{cy:.3f},{cz:.3f}), cell size=({dx:.3f},{dy:.3f},{dz:.3f})")

        best_cost = float("inf")
        best_center = (cx, cy, cz)
        best_bounds = (xmin0, xmax0, ymin0, ymax0, zmin0, zmax0)
        best_meshname = "meshgrid_seed.txt"

        # 2) copy parfile + obs to the Tomofast machine (filename must match Parfile)
        obs_win = os.path.join(obs_dir, f"Blob_{blob_id:03d}_RECT.obs")
        par_win = os.path.join(par_dir, f"Parfile_QDM_Inversion_Blob_{blob_id:03d}.txt")

        run_quiet(cfg, f"cp '{host_path(cfg, par_win)}' '{home}/{run_id}/parfile/'")
        run_quiet(cfg, f"cp '{host_path(cfg, obs_win)}' '{home}/{run_id}/data/Blob_{blob_id:03d}_RECT.obs'")

        par_name = f"Parfile_QDM_Inversion_Blob_{blob_id:03d}.txt"

        log_path = os.path.join(out_dir, f"{blob}_steepest_log.txt")
        with open(log_path, "w") as f:
            f.write("# step meshfile cx cy cz xmin xmax ymin ymax zmin zmax cost\n")

        # 3) steepest descent
        for step in range(1, MAX_STEPS + 1):
            move_best_cost = float("inf")
            move_best_center = None
            move_best_bounds = None

            for (ix, iy, iz) in NEIGH:
                tx = cx + ix * dx
                ty = cy + iy * dy
                tz = cz + iz * dz

                xmin = tx - dx / 2
                xmax = tx + dx / 2
                ymin = ty - dy / 2
                ymax = ty + dy / 2
                zmin = tz - dz / 2
                zmax = tz + dz / 2

                with open(tmp_mesh_win, "w") as f:
                    f.write("1\n")
                    f.write(
                        f"{xmin:.6f} {xmax:.6f} {ymin:.6f} {ymax:.6f} {zmin:.6f} {zmax:.6f} "
                        "1 1 1\n"
                    )

                # overwrite the tomofast meshgrid
                rc, cost, text = run_quiet(cfg, f"cp '{tmp_mesh_wsl}' '{meshgrid_target_wsl}'")
                if rc != 0:
                    out.say(f"\n   [cp meshgrid FAILED, rc={rc}] {text[-500:]}")
                    continue

                # run tomofast
                rc, cost, text = run_quiet(cfg, f"cd '{home}' && ./tomofastx -p ./{run_id}/parfile/{par_name}")
                if rc != 0 or cost is None:
                    err = [ln.strip() for ln in text.splitlines() if "error" in ln.lower()][:3]
                    out.say(f"[{blob}] tomofastx FAILED rc={rc} -> {err}")
                    with open(os.path.join(out_dir, f"{blob}_tomofast_error.txt"), "w") as f:
                        f.write(text)
                    continue

                if math.isnan(cost) or math.isinf(cost):
                    continue

                if cost < move_best_cost:
                    move_best_cost = cost
                    move_best_center = (tx, ty, tz)
                    move_best_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

            if move_best_center is None:
                out.say(f"step {step:03d}: all NaN -> STOP")
                break

            cx, cy, cz = move_best_center
            meshname = f"meshgrid_step{step:03d}.txt"

            out.say(f"[{blob}] step {step:03d}: best cost={move_best_cost:.6e} at center=({cx:.3f},{cy:.3f},{cz:.3f})")

            xmin, xmax, ymin, ymax, zmin, zmax = move_best_bounds
            with open(log_path, "a") as f:
                f.write(
                    f"{step} {meshname} {cx:.6f} {cy:.6f} {cz:.6f} "
                    f"{xmin:.6f} {xmax:.6f} {ymin:.6f} {ymax:.6f} {zmin:.6f} {zmax:.6f} "
                    f"{move_best_cost:.16e}\n"
                )

            if move_best_cost >= best_cost:
                out.say(f"no improvement (best={best_cost:.6e}) -> STOP")
                break

            best_cost = move_best_cost
            best_center = (cx, cy, cz)
            best_bounds = move_best_bounds
            best_meshname = meshname

        # 4) re-fit the winning cell once more and keep its Tomofast output
        if not math.isinf(best_cost):
            xmin, xmax, ymin, ymax, zmin, zmax = best_bounds
            with open(tmp_mesh_win, "w") as f:
                f.write("1\n")
                f.write(f"{xmin:.6f} {xmax:.6f} {ymin:.6f} {ymax:.6f} {zmin:.6f} {zmax:.6f} 1 1 1\n")

            run_quiet(cfg, f"cp '{tmp_mesh_wsl}' '{meshgrid_target_wsl}'")
            rc, cost, text = run_quiet(cfg, f"cd '{home}' && ./tomofastx -p ./{run_id}/parfile/{par_name}")

            if rc != 0:
                out.say(f"   WARNING: re-fit run failed for {blob}, output not saved")
            else:
                src_out = f"{home}/{run_id}/output/Blob_{blob_id:03d}"
                final_dir = host_path(cfg, os.path.join(SAVE_DIR, blob))
                run_quiet(cfg, f"rm -rf '{final_dir}' && mkdir -p '{final_dir}'")
                for sub in ("data", "model"):
                    rc_cp, _, out_cp = run_quiet(cfg, f"cp -r '{src_out}/{sub}' '{final_dir}/'")
                    if rc_cp != 0:
                        out.say(f"   WARNING: could not copy {sub} for {blob}: {out_cp[-300:]}")

        # 5) save best per blob
        if math.isinf(best_cost):
            result = (blob, grain_id, "meshgrid_NONE.txt", (float("nan"),) * 6, float("nan"))
            out.say(f">>> BEST for {blob}: NONE")
        else:
            out.say(
                f">>> BEST for {blob}: cost={best_cost:.6e} "
                f"center=({best_center[0]:.3f},{best_center[1]:.3f},{best_center[2]:.3f})"
            )
            result = (blob, grain_id, best_meshname, best_bounds, best_cost)

        # checkpoint immediately -- this blob is now safely on disk even if the
        # process is interrupted before the next blob finishes
        _b, _g, _m, _bnds, _cost = result
        _xmin, _xmax, _ymin, _ymax, _zmin, _zmax = _bnds
        with summary_lock:
            with open(summary_path, "a") as f:
                f.write(
                    f"{_b} grain_{_g} {_m} "
                    f"{_xmin:.6f} {_xmax:.6f} {_ymin:.6f} {_ymax:.6f} {_zmin:.6f} {_zmax:.6f} "
                    f"{_cost:.16e}\n"
                )

        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(_run_one_blob, blob) for blob in blob_dirs]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            if r is not None:
                best_per_blob.append(r)

    print("\nSummary file:", summary_path)
    print(f"Total runtime: {(time.time() - t0) / 60:.2f} minutes")
    out.end()
    return best_per_blob


# ============================================================
### Tool 2: Clean grains ###
# ============================================================
def clean_grains(cfg,
                 max_step_to_remove=1000,   # remove grains whose best step == this number (the run's max_steps); a number bigger than max_steps removes none
                 cost_threshold=1,          # keep only grains at or below this data cost (0, 1)
                 bz_file=None,              # file to read the map extent from (e.g. mag_observed.txt), or None to use the numbers below
                 x_map=(0, 2079),           # map (Bz) extent, same units as the mesh (µm): min, max
                 y_map=(0, 1287),
                 z_map=(0, 10000)):         # model z range in the mesh sign convention; None on a side turns it off
    """
    Removes duplicate picks, grains that hit the max steepest-descent step,
    grains outside the map box (+ D) and grains above the cost threshold.
    Run it after the steepest-descent step and before the global inversion.
    The original file is kept as best_per_blob_steepest.txt.original_backup.txt
    """
    base_dir = cfg.base_dir
    MAX_STEP_TO_REMOVE = max_step_to_remove
    COST_THRESHOLD = cost_threshold

    # D = tolerance outside the map. A grain is removed if its center is outside
    # the model box (plus D) in x, y or z
    D = cfg.d

    X_MAP_MIN, X_MAP_MAX = x_map
    Y_MAP_MIN, Y_MAP_MAX = y_map
    Z_MAP_MIN, Z_MAP_MAX = z_map

    if bz_file is not None:
        _xy = pd.read_csv(bz_file, sep=r"\s+", header=None, usecols=[0, 1]).apply(pd.to_numeric, errors="coerce").dropna()
        X_MAP_MIN, X_MAP_MAX = float(_xy[0].min()), float(_xy[0].max())
        Y_MAP_MIN, Y_MAP_MAX = float(_xy[1].min()), float(_xy[1].max())
        print(f"Map extent read from {bz_file}")

    # allowed ranges: (lower, upper); None = that side off
    X_RANGE = (X_MAP_MIN - D, X_MAP_MAX + D)
    Y_RANGE = (Y_MAP_MIN - D, Y_MAP_MAX + D)
    Z_RANGE = (None if Z_MAP_MIN is None else Z_MAP_MIN - D,
               None if Z_MAP_MAX is None else Z_MAP_MAX + D)

    txt_path = os.path.join(base_dir, "4-Source_Detection", "Results_DataCost_Steepest", "best_per_blob_steepest.txt")
    backup_path = txt_path + ".original_backup.txt"

    colnames = ["blob", "grain", "meshfile", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "final_data_cost"]

    # 1) make the permanent backup, only if it doesn't exist yet
    if not os.path.exists(backup_path):
        shutil.copy(txt_path, backup_path)
        print(f"Backup created: {backup_path}")
    else:
        print(f"Using existing backup (original file is safe): {backup_path}")

    # 2) always read from the backup, never from the working file
    df = pd.read_csv(backup_path, sep=r"\s+", names=colnames, comment="#")
    n_original = len(df)

    # 3) remove duplicate picks: same x,y center, keep lowest cost
    df["x_center"] = 0.5 * (df["xmin"] + df["xmax"])
    df["y_center"] = 0.5 * (df["ymin"] + df["ymax"])
    df["z_center"] = 0.5 * (df["zmin"] + df["zmax"])

    df = df.sort_values("final_data_cost")
    df = df.drop_duplicates(subset=["x_center", "y_center"], keep="first")
    n_after_dedup = len(df)
    n_duplicates_removed = n_original - n_after_dedup

    # 4) remove grains that hit the max step (didn't finish converging)
    # the step number is in the meshfile name, e.g. "meshgrid_step050.txt" -> 50
    # "meshgrid_seed.txt" (never moved) and "meshgrid_NONE.txt" (no valid fit) have no
    # step number, so they are never removed here.
    def get_step_number(meshfile):
        m = re.search(r"step(\d+)", meshfile)
        return int(m.group(1)) if m else None

    df["step_number"] = df["meshfile"].apply(get_step_number)
    hit_max_step = df["step_number"] == MAX_STEP_TO_REMOVE
    n_maxstep_removed = int(hit_max_step.sum())
    df = df[~hit_max_step]
    n_after_maxstep = len(df)

    # 5) remove grains outside the allowed X / Y / Z range
    outside = pd.Series(False, index=df.index)
    n_removed_by_axis = {}
    for axis, (lo, hi) in (("x", X_RANGE), ("y", Y_RANGE), ("z", Z_RANGE)):
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[f"{axis}_center"] < lo
        if hi is not None:
            bad |= df[f"{axis}_center"] > hi
        n_removed_by_axis[axis] = int(bad.sum())
        outside |= bad
    n_location_removed = int(outside.sum())
    df = df[~outside]
    n_after_location = len(df)

    # 6) keep only rows at or below the cost threshold
    df = df[df["final_data_cost"] <= COST_THRESHOLD]
    n_final = len(df)
    n_cost_removed = n_after_location - n_final

    # 7) overwrite the working file with the cleaned result
    df_out = df[colnames].sort_values(["blob", "grain"]).reset_index(drop=True)
    df_out.to_csv(txt_path, sep=" ", header=False, index=False)

    # 8) summary
    approx_r2 = 1 - COST_THRESHOLD ** 2

    def _rng(r):
        lo, hi = r
        return "off" if (lo is None and hi is None) else f"[{'-inf' if lo is None else round(lo, 3)}, {'+inf' if hi is None else round(hi, 3)}]"

    print("=" * 55)
    print("CLEAN GRAINS SUMMARY")
    print("=" * 55)
    print(f"Original grains:              {n_original}")
    print(f"Removed (duplicates):         {n_duplicates_removed}")
    print(f"Removed (hit step {MAX_STEP_TO_REMOVE}):       {n_maxstep_removed}")
    print(f"Removed (location):           {n_location_removed}")
    print(f"   x {_rng(X_RANGE)}: {n_removed_by_axis['x']}   "
          f"y {_rng(Y_RANGE)}: {n_removed_by_axis['y']}   "
          f"z {_rng(Z_RANGE)}: {n_removed_by_axis['z']}   (a grain can count in several)")
    print(f"Removed (cost > {COST_THRESHOLD}):         {n_cost_removed}")
    print(f"Remaining grains:             {n_final}")
    print(f"Approx. R^2 at threshold:     {approx_r2:.4f}  (R^2 = 1 - cost^2)")
    print("=" * 55)
    print(f"Overwrote: {txt_path}")
    print(f"Original always recoverable from: {backup_path}")


# ============================================================
### Tool 2: 3. Analysis Plots ###
# ============================================================
def plot_blob_costs(cfg):
    """Bz observed map with the blob centers coloured by their steepest-descent final data cost"""
    out = Out()
    base_dir = cfg.base_dir

    # blob results
    txt_path = os.path.join(base_dir, "4-Source_Detection", "Results_DataCost_Steepest", "best_per_blob_steepest.txt")

    colnames = ["blob", "grain", "meshfile", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "final_data_cost"]
    df = pd.read_csv(txt_path, sep=r"\s+", names=colnames, comment="#")

    df = df[np.isfinite(df["final_data_cost"])].copy()
    df["x_center"] = 0.5 * (df["xmin"] + df["xmax"])
    df["y_center"] = 0.5 * (df["ymin"] + df["ymax"])

    print(f"Loaded {len(df)} blob(s) with valid data cost")

    # Bz observed map
    obs_path = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Mag_Inversion.obs")
    obs_df = load_field_table(obs_path)
    comp = "bz" if "bz" in obs_df.columns else "value"

    nx = len(np.unique(obs_df["x"]))
    ny = len(np.unique(obs_df["y"]))
    bz_grid = obs_df[comp].to_numpy().reshape(ny, nx)
    extent = [obs_df["x"].min(), obs_df["x"].max(), obs_df["y"].min(), obs_df["y"].max()]

    # plot: Bz map + blob points coloured by data cost
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    vmax = np.nanmax(np.abs(bz_grid))
    ax.imshow(bz_grid, origin="lower", cmap="seismic", extent=extent, aspect="equal", vmin=-vmax, vmax=vmax)

    sc = ax.scatter(df["x_center"], df["y_center"], c=df["final_data_cost"], cmap="viridis",
                    s=60, edgecolors="black", linewidths=0.5)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Final data cost (steepest)")

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("Bz observed - blob locations colored by data cost")

    plt.tight_layout()
    out_dir = os.path.join(base_dir, "4-Source_Detection", "Results_DataCost_Steepest")
    out_png = os.path.join(out_dir, "Bz_with_DataCost_points.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    out.show()
    print("Saved:", out_png)
    out.end()


# ============================================================
### Tool 3: 1. Combine best locations ###
# ============================================================
def combine_best_meshes(cfg, has_header=False, skip_nans=False):
    """
    Best location per blob (lowest data cost) written as one mesh file. Returns the number of blobs.
    has_header=True  : the summary file still has its "# blob grain ..." header line (not cleaned yet)
    skip_nans=True   : drop blobs without a valid data cost
    """
    base_dir = cfg.base_dir
    out = Out()

    summary_path = os.path.join(base_dir, "4-Source_Detection", "Results_DataCost_Steepest", "best_per_blob_steepest.txt")
    out_path = os.path.join(base_dir, "4-Source_Detection", "combined_best_blobs_mesh.txt")

    # summary table (blob, grain, meshfile, bounds, cost)
    colnames = ["blob", "grain", "meshfile", "xmin", "xmax", "ymin", "ymax",
                "zmin", "zmax", "final_data_cost"]

    df = pd.read_csv(summary_path, sep=r"\s+", names=colnames, skiprows=1 if has_header else None)

    # best mesh per blob (skip NaNs only)
    if skip_nans:
        df["final_data_cost"] = pd.to_numeric(df["final_data_cost"], errors="coerce")
        df = df.dropna(subset=["blob", "final_data_cost"])

    best = df.loc[df.groupby("blob")["final_data_cost"].idxmin()].reset_index(drop=True)

    n_blobs = len(best)
    print("Number of blobs detected:", n_blobs)

    # simple indices: i = 1 ... n_blobs
    best = best.reset_index(drop=True)
    best["i"] = best.index + 1
    best["j"] = 1
    best["k"] = 1

    best_sorted = best.copy()

    # write combined mesh file
    with open(out_path, "w") as f:
        f.write(f"{n_blobs}\n")

        out.say("\nWriting mesh entries (best per blob):", always=True)
        for _, row in best_sorted.iterrows():
            xmin, xmax = row["xmin"], row["xmax"]
            ymin, ymax = row["ymin"], row["ymax"]
            zmin, zmax = row["zmin"], row["zmax"]
            i = int(row["i"]); j = 1; k = 1

            out.say(f"  {row['blob']}  {row['meshfile']}  -> i={i}, j={j}, k={k}")
            f.write(f"{xmin} {xmax} {ymin} {ymax} {zmin} {zmax} {i} {j} {k}\n")

    # max i, j, k
    imax = int(best_sorted["i"].max()) if n_blobs else 0

    print("\n====================================")
    print(" Mesh Index Summary")
    print("====================================")
    print(f"Max i = {imax}")
    print("j is fixed to 1")
    print("k is fixed to 1")
    print("====================================")

    print("\nCombined mesh file saved to:")
    print(out_path)
    out.end()
    return n_blobs


# ============================================================
### Tool 3: 2. Create Parfile ###
# ============================================================
PAR_TEMPLATE_DIPOLE = """==================================================================================
GLOBAL
==================================================================================
global.outputFolderPath         = ./{run_id}/output
global.description              = QDM inversion data

==================================================================================
MODEL GRID parameters
==================================================================================
# nx ny nz
modelGrid.size                        = {nblobs} 1 1
modelGrid.magn.file                   = ./{run_id}/data/meshgrid.txt
modelGrid.magn.nModelComponents       = 3
==================================================================================
DATA parameters
==================================================================================
forward.data.magn.nData                = {ndata}
forward.data.magn.dataGridFile         = ./{run_id}/data/Mag_Inversion.obs
forward.data.magn.nDataComponents      = {ncomp}

==================================================================================
MAGNETIC FIELD constants
==================================================================================
forward.magneticField.inclination              = 90.0
forward.magneticField.declination              = 0.0
forward.magneticField.XaxisDeclination         = 0.d0

==================================================================================
DEPTH WEIGHTING
==================================================================================
# 1-depth weight, 2-distance weight.
forward.depthWeighting.type                    = 1
forward.depthWeighting.magn.Z0             = {z0} #adjust z0 paramter to data level (5)

==================================================================================
SENSITIVITY KERNEL
==================================================================================
# 0-calculate the sensitivity, 1-read from files at the specified folder.
sensit.readFromFiles                  = 0
sensit.folderPath                     = ./{run_id}/output/SENSIT/

==================================================================================
MATRIX COMPRESSION
==================================================================================
# 0-none, 1-wavelet compression.
forward.matrixCompression.type                        = 0
# The minimum compressed sensitivity absolute value.
forward.matrixCompression.rate                        = 0.1

==================================================================================
PRIOR MODEL
==================================================================================
# 1-set value, 2-read from file.
inversion.priorModel.type                  = 1
inversion.priorModel.magn.value            = 0.000000

==================================================================================
INVERSION parameters
==================================================================================
inversion.nMajorIterations              = 3
inversion.nMinorIterations              = 100
inversion.targetMisfit = 100 #RMS_nT target misfit (measurement error) to avoid overfitting
#inversion.modelDamping.normPower        = 1.4d0

==================================================================================
MODEL DAMPING (m - m_prior)
==================================================================================
inversion.modelDamping.magn.weight      = 0.d0


==================================================================================
JOINT INVERSION parameters
==================================================================================
inversion.joint.grav.problemWeight      = 0.00000000
inversion.joint.magn.problemWeight      = 1.00000000

===================================================================================
ADMM constraints
===================================================================================
inversion.admm.enableADMM           = 0
inversion.admm.nLithologies         = 1
inversion.admm.magn.bounds          = 0. 1.d10
inversion.admm.magn.weight          = 50.d0

#inversion.admm.magn.bounds          = -1.d10 0.d0
"""


def create_dipole_parfile(cfg, n_blobs, n_data_components=3, z0=10.0):
    """
    n_blobs: number of dipoles, from combine_best_meshes()
    n_data_components: 3 if using Bx, By, Bz, or 1 if using Bz only
    z0: depth weighting Z0, the instrument height (matching the Z column in the data)
    """
    base_dir = cfg.base_dir

    output_path = os.path.join(base_dir, "0-Parfiles", "Parfile_QDM_Dipole.txt")

    # full dataset size straight from the obs file header (not the per-blob leftover)
    with open(os.path.join(base_dir, "4-3D_Inversion_Filtered", "Mag_Inversion.obs")) as f:
        ndata_full = int(f.readline().split()[0])

    par_text = PAR_TEMPLATE_DIPOLE.format(run_id=cfg.run_id, nblobs=n_blobs, ndata=ndata_full,
                                          ncomp=n_data_components, z0=z0)

    print(f"[Sanity check] nblobs = {n_blobs}, ndata = {ndata_full}")

    with open(output_path, "w") as f:
        f.write(par_text)

    print("Parfile created:", output_path)


# ============================================================
### Tool 3: 3. Run Tomofast inversion for Dipole Fitting ###
# ============================================================
def run_dipole_inversion(cfg):
    """fit the detected locations with the full data in one Tomofast-x run (WSL or Linux, set by cfg.use_wsl)"""
    base_dir = cfg.base_dir
    home = cfg.home
    run_id = cfg.run_id
    out = Out()

    script_start_time = time.time()

    print("=== COPYING FILES TO TOMOFAST-x ===\n")

    run_live(cfg, f"mkdir -p '{home}/{run_id}/data' '{home}/{run_id}/output' '{home}/{run_id}/parfile'", out)

    # 1) copy parfile + Mag_Inversion.obs to Tomofast-x
    base_dir_host = host_path(cfg, base_dir)
    run_live(cfg, f"cp '{base_dir_host}/0-Parfiles/Parfile_QDM_Dipole.txt'  {home}/{run_id}/parfile/", out)
    run_live(cfg, f"cp '{base_dir_host}/4-3D_Inversion_Filtered/Mag_Inversion.obs'   {home}/{run_id}/data/", out)

    # 2) copy combined mesh file as meshgrid.txt (many grains, one run)
    combined_mesh_win = os.path.join(base_dir, "4-Source_Detection", "combined_best_blobs_mesh.txt")
    combined_mesh_wsl = host_path(cfg, combined_mesh_win)

    print("Using combined mesh file:")
    print("Local:  ", combined_mesh_win)
    print("Tomofast machine:", combined_mesh_wsl)

    run_live(cfg, f"cp '{combined_mesh_wsl}' '{home}/{run_id}/data/meshgrid.txt'", out)

    # 3) run TomoFast-x once
    print(f"\n=== RUNNING TOMOFAST-X ({run_id}) ===\n")

    if cfg.use_wsl:
        run_live(
            cfg,
            f"cd {home} && "
            f"{{ time ./tomofastx -p ./{run_id}/parfile/Parfile_QDM_Dipole.txt; }} "   # time
            f"2>&1 | tee ./{run_id}/output/QDM_log.txt",
            out
        )
    else:
        run_live(cfg, f"cd {home} && ./tomofastx -p ./{run_id}/parfile/Parfile_QDM_Dipole.txt | tee ./{run_id}/output/QDM_log.txt", out)

    # 4) copy Tomofast-x output back
    output_win = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output", "combined_grains")
    os.makedirs(output_win, exist_ok=True)

    output_wsl = host_path(cfg, output_win)

    print("Copying Tomofast-x output back:")
    print("Source:  ", f"{home}/{run_id}/output/")
    print("Target:  ", output_win)

    run_live(cfg, f"cp -r {home}/{run_id}/output/data '{output_wsl}/'", out)
    run_live(cfg, f"cp -r {home}/{run_id}/output/model '{output_wsl}/'", out)
    run_live(cfg, f"cp -r {home}/{run_id}/output/Paraview '{output_wsl}/'", out)
    run_live(cfg, f"cp -r {home}/{run_id}/output/QDM_log.txt '{output_wsl}/'", out)
    run_live(cfg, f"cp -r {home}/{run_id}/output/Parfile_copy.txt '{output_wsl}/'", out)
    run_live(cfg, f"cp -r {home}/{run_id}/output/costs.txt '{output_wsl}/'", out)
    # run_live(cfg, f"cp -r {home}/{run_id}/output/SENSIT '{output_wsl}/'", out)  # optional, very large

    print("\nDone: combined mesh inversion finished and outputs copied.\n")

    total_time = time.time() - script_start_time
    print(f"\nTotal runtime: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    out.end()


# ============================================================
### Tool 3: 4. Vectors Directions Analysis ###
# ============================================================
def check_vector_directions(cfg):
    """Direction and moment of every cell of the combined model (double check step; each cell = one grain)"""
    out = Out()
    base_dir = cfg.base_dir

    # folder containing Tomofast outputs
    root_folder = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output")

    # combined-grains run
    grain_folders = ["combined_grains"]
    print("Found model folder(s):", grain_folders)

    for grain in grain_folders:
        out.say("\n=====================================")
        out.say("Processing", grain)
        out.say("=====================================\n")

        Model = os.path.join(root_folder, grain, "model", "mag_final_model_full.txt")

        # first line = number of grains, then Mx My Mz per grain
        Ngrains, data = read_model_vectors(Model)
        print("Number of grains in model:", Ngrains)

        for grain_idx, (Mx, My, Mz_raw) in enumerate(data, start=1):

            out.say(f"\n--- Grain {grain_idx} ---")

            # sign flip
            Mz = -Mz_raw

            dec, inc, theta, phi = dec_inc_from_vector(Mx, My, Mz)

            out.say("phi=", phi, "theta=", theta)
            out.say("dec=", dec, "inc=", inc)

            # cell size (m)
            dx = cfg.d * 1e-6
            dy = cfg.d * 1e-6
            dz = cfg.d * 1e-6
            V = dx * dy * dz  # m^3

            # magnetization amplitude (A/m)
            M_amp = np.sqrt(Mx**2 + My**2 + Mz**2)

            # magnetic moment amplitude (A·m^2)
            moment_amp = M_amp * V

            out.say("Magnetization amplitude:", M_amp * 1e-6, "A/m")
            out.say("Magnetic moment amplitude:", moment_amp, "A·m²")

    out.end()


# ============================================================
### Tool 3: 4. Stereonet plot (combined grains) ###
# ============================================================
def plot_stereonet(cfg):
    out = Out()
    base_dir = cfg.base_dir

    root_folder = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output")

    # combined_grains model
    grain_folders = ["combined_grains"]
    print("Using combined-grains model:", grain_folders)

    results = []  # (grain_index, dec, inc)

    for folder in grain_folders:

        print("\nProcessing:", folder)

        Model = os.path.join(root_folder, folder, "model", "mag_final_model_full.txt")

        # first line = Ngrains, rest = Mx My Mz
        Ngrains, data = read_model_vectors(Model)
        print("Number of grains detected:", Ngrains)

        # dec/inc for each grain cell
        for idx, (Mx, My, Mz_raw) in enumerate(data, start=1):

            Mz = -Mz_raw  # same sign flip
            dec, inc, theta, phi = dec_inc_from_vector(Mx, My, Mz)

            out.say(f"Grain {idx}: dec={dec:.2f}, inc={inc:.2f}")
            results.append((idx, dec, inc))

    # plot on stereonet
    plt.figure(figsize=(9, 9))
    ipmag.plot_net()

    ax = plt.gca()

    colors_pts = ["r"] * len(results)   # all points same color

    for (grain_idx, dec, inc), col in zip(results, colors_pts):
        ipmag.plot_di(dec, inc, color=col, marker='o', markersize=200)

        plt.plot([], [], color="r", linestyle="None",
                 label=f"Dipole {grain_idx}: D={dec:.1f}, I={inc:.1f}")

    out_png = os.path.join(base_dir, "4-Source_Detection", "aAll_Combined_Grains_DecInc.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')

    out.show()

    print("Saved combined plot:", out_png)
    out.end()


# ============================================================
### Tool 3: 4. Stereonet plot (combined grains), color by moment ###
# ============================================================
def plot_stereonet_by_moment(cfg):
    out = Out()
    base_dir = cfg.base_dir

    # cell size (m)
    dx = cfg.d * 1e-6
    dy = cfg.d * 1e-6
    dz = cfg.d * 1e-6
    volume = dx * dy * dz

    root_folder = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output")

    grain_folders = ["combined_grains"]
    print("Using combined-grains model:", grain_folders)

    results = []  # (grain_index, dec, inc, moment)

    for folder in grain_folders:

        print("\nProcessing:", folder)
        Model = os.path.join(root_folder, folder, "model", "mag_final_model_full.txt")

        Ngrains, data = read_model_vectors(Model)
        print("Number of grains detected:", Ngrains)

        for idx, (Mx, My, Mz_raw) in enumerate(data, start=1):

            Mz = -Mz_raw
            mag = np.sqrt(Mx**2 + My**2 + Mz**2)     # magnetization magnitude
            moment = mag * volume                    # magnetic moment [A*m^2], assuming mag in A/m and volume in m^3

            dec, inc, theta, phi = dec_inc_from_vector(Mx, My, Mz)

            out.say(f"Grain {idx}: dec={dec:.2f}, inc={inc:.2f}, moment={moment:.3e}")
            results.append((idx, dec, inc, moment))

    # plot on stereonet
    plt.figure(figsize=(9, 9))
    ipmag.plot_net()
    ax = plt.gca()

    # colour by moment
    moments = np.array([r[3] for r in results], dtype=float)
    good = np.isfinite(moments)

    vmin = np.nanmin(moments[good])
    vmax = np.nanmax(moments[good])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.jet

    for (grain_idx, dec, inc, moment) in results:
        if not (np.isfinite(dec) and np.isfinite(inc) and np.isfinite(moment)):
            continue
        col = cmap(norm(moment))
        ipmag.plot_di(dec, inc, color=col, marker='o', markersize=400)

    # colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.05)
    cbar.set_label("Magnetic moment (A·m²)", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)

    # legend entries
    for (grain_idx, dec, inc, moment) in results:
        plt.plot([], [], color="k", linestyle="None",
                 label=f"Dipole {grain_idx}: D={dec:.1f}°, I={inc:.1f}°, m={moment:.2e} A·m²")

    out_png = os.path.join(base_dir, "4-Source_Detection", "All_Combined_Grains_DecInc_byMoment.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    out_svg = os.path.join(base_dir, "4-Source_Detection", "All_Combined_Grains_DecInc_byMoment.svg")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")

    out.show()

    print("Saved combined plot:", out_png)
    out.end()


# ============================================================
### Tool 3: 5. Final Report and plots ###
# ============================================================
def build_final_report(cfg, has_header=False, skip_nans=False, cell_size_from_bounds=False):
    """
    Position, direction and moment of every dipole, as a table (one model, many blob locations).
    has_header=True            : the summary file still has its header line (not cleaned yet)
    skip_nans=True             : drop blobs without a valid data cost
    cell_size_from_bounds=True : dx, dy, dz from each blob's bounds (non-uniform model), instead of d
    """
    out = Out()
    base_dir = cfg.base_dir

    # 1) paths
    summary_path = os.path.join(base_dir, "4-Source_Detection", "Results_DataCost_Steepest", "best_per_blob_steepest.txt")

    Model = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output", "combined_grains",
                         "model", "mag_final_model_full.txt")

    out_csv = os.path.join(base_dir, "4-Source_Detection", "Blob_Vector_Summary.csv")

    # 2) summary + best per blob
    df = pd.read_csv(
        summary_path,
        sep=r"\s+",
        skiprows=1 if has_header else None,
        names=["blob", "grain", "meshfile", "xmin", "xmax", "ymin", "ymax",
               "zmin", "zmax", "final_data_cost"]
    )

    if skip_nans:
        df["final_data_cost"] = pd.to_numeric(df["final_data_cost"], errors="coerce")
        df = df.dropna(subset=["blob", "final_data_cost"])

    # best mesh row per blob
    best = df.loc[df.groupby("blob")["final_data_cost"].idxmin()].reset_index(drop=True)

    # sort blobs Blob_001, Blob_002, ...
    best = best.sort_values("blob", key=lambda s: s.map(blob_num)).reset_index(drop=True)

    print("Number of blobs (best rows):", len(best))
    print("First blobs:", best["blob"].head().tolist())

    # centers
    best["x_center"] = 0.5 * (best["xmin"] + best["xmax"])
    best["y_center"] = 0.5 * (best["ymin"] + best["ymax"])
    best["z_center"] = 0.5 * (best["zmin"] + best["zmax"])

    # 3) the ONE model file (Mx My Mz rows)
    print("\nModel file:", Model)
    Nmodel, data = read_model_vectors(Model)

    print("Number of cells in model:", Nmodel)

    # check matching
    if Nmodel != len(best):
        print("WARNING: model cells != number of blobs in best table")
        print("  Nmodel =", Nmodel)
        print("  Nblobs =", len(best))
        print("Fix by: (1) ensuring blob list matches model order, (2) removing extra blobs, or (3) using the correct model file.")

    # 4) directions per blob (row i = blob i after sorting)
    dx = dy = dz = cfg.d * 1e-6
    V = dx * dy * dz

    # xmin/xmax/... are in micrometers (µm)
    UM_TO_M = 1e-6

    table_rows = []

    n = min(Nmodel, len(best))  # safe loop

    for i in range(n):
        blob_name = best.loc[i, "blob"]
        Mx, My, Mz_raw = data[i]

        if cell_size_from_bounds:
            dx = abs(best.loc[i, "xmax"] - best.loc[i, "xmin"]) * UM_TO_M
            dy = abs(best.loc[i, "ymax"] - best.loc[i, "ymin"]) * UM_TO_M
            dz = abs(best.loc[i, "zmax"] - best.loc[i, "zmin"]) * UM_TO_M
            V = dx * dy * dz

            out.say(f"\n[CHECK] {blob_name}  dx={dx/UM_TO_M:.6f} µm  dy={dy/UM_TO_M:.6f} µm  dz={dz/UM_TO_M:.6f} µm  V={V:.6e} m^3")

        # sign flip
        Mz = -Mz_raw

        dec, inc, theta, phi = dec_inc_from_vector(Mx, My, Mz)

        M_amp = np.sqrt(Mx**2 + My**2 + Mz**2)
        moment_amp = M_amp * V

        table_rows.append({
            "blob": blob_name,
            "meshfile": best.loc[i, "meshfile"],
            "final_data_cost": best.loc[i, "final_data_cost"],
            "x_center": best.loc[i, "x_center"],
            "y_center": best.loc[i, "y_center"],
            "z_center": best.loc[i, "z_center"],
            "dec": dec,
            "inc": inc,
            "moment(A·m²)": moment_amp
        })

    results_table = pd.DataFrame(table_rows)

    print("\n================= FINAL Report (PER BLOB) =================\n")
    out.table(results_table, index=False)

    results_table.to_csv(out_csv, index=False)
    print("\n Saved:", out_csv)
    out.end()
    return results_table


# ============================================================
### Tool 3: 5. Plot detected locations on Bz ###
# ============================================================
def plot_centres_on_bz(cfg):
    out = Out()
    base_dir = cfg.base_dir

    # 1. mag.obs (back to QDM convention)
    x, y, bz = read_bz_obs(base_dir)

    # 2. grain centres
    grain_path = os.path.join(base_dir, "4-Source_Detection", "Blob_Vector_Summary.csv")
    grains = pd.read_csv(grain_path, sep=None, engine="python")

    gx = grains["x_center"].values
    gy = grains["y_center"].values

    # 3. colour scaling (zero-centered)
    bz_max = np.nanmax(np.abs(bz))
    bz_min = -bz_max

    # 4. style
    plt.style.use("seaborn-v0_8-white")

    # 5. plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    sc = ax.scatter(x, y, c=bz, s=10, cmap="seismic", vmin=bz_min, vmax=bz_max, rasterized=True)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.67, pad=0.01)
    cbar.set_label("Bz (nT)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # grain centres
    ax.scatter(gx, gy, c="yellow", s=55, edgecolors="white", linewidths=0.7,
               label="Detected Dipole centres", zorder=3)

    # 6. labels and styling
    ax.set_xlabel("X (µm)", fontsize=12)
    ax.set_ylabel("Y (µm)", fontsize=12)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.margins(0)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    out_png = os.path.join(base_dir, "4-Source_Detection", "BZ_grains_centers.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out.show()
    out.end()


# ============================================================
### Tool 3: 5. Plot detected locations and depths on Bz ###
# ============================================================
def plot_centres_depth(cfg):
    out = Out()
    base_dir = cfg.base_dir

    # 1. mag.obs (QDM convention)
    x, y, bz = read_bz_obs(base_dir)

    # 2. grain centres
    grain_path = os.path.join(base_dir, "4-Source_Detection", "Blob_Vector_Summary.csv")
    grains = pd.read_csv(grain_path, sep=None, engine="python")

    gx = grains["x_center"].values
    gy = grains["y_center"].values
    gz = grains["z_center"].values   # depth

    # 3. colour scaling (clip 2% outliers)
    bz_min = np.percentile(bz, 2)
    bz_max = np.percentile(bz, 98)

    # 4. style
    plt.style.use("seaborn-v0_8-white")

    # 5. plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    sc = ax.scatter(x, y, c=bz, s=10, cmap="seismic", vmin=bz_min, vmax=bz_max, rasterized=True)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.52, pad=0.0)
    cbar.set_label("Bz (nT)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # grain centres coloured by depth
    scz = ax.scatter(gx, gy, c=gz, cmap="viridis", s=60, edgecolors="white", linewidths=0.7,
                     label="Detected dipole centres", zorder=3)

    cbarz = plt.colorbar(scz, ax=ax, shrink=0.52, pad=0.02)
    cbarz.set_label("Depth (µm)", fontsize=12)
    cbarz.ax.tick_params(labelsize=10)

    # scale bar (e.g., 100 µm)
    bar_len = 100.0  # µm
    x0 = np.min(x) + 0.05 * (np.max(x) - np.min(x))
    y0 = np.min(y) + 0.05 * (np.max(y) - np.min(y))

    ax.plot([x0, x0 + bar_len], [y0, y0], color="k", linewidth=3, solid_capstyle="butt", zorder=5)
    ax.text(x0 + 0.5 * bar_len, y0 + 0.02 * (np.max(y) - np.min(y)),
            f"{int(bar_len)} µm", ha="center", va="bottom", fontsize=10)

    # 6. labels and styling
    ax.set_xlabel("X (µm)", fontsize=12)
    ax.set_ylabel("Y (µm)", fontsize=12)

    ax = plt.gca()
    ax.tick_params(axis="both", length=4, labelsize=11)

    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

    plt.subplots_adjust(right=0.8)

    out_png = os.path.join(base_dir, "4-Source_Detection", "BZ_grains_centers_depth.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')

    out.show()
    out.end()


# ============================================================
### Tool 3: 5. Plot full results (locations, moment, inc, dec) ###
# ============================================================
def plot_results_panels(cfg):
    out = Out()
    base_dir = cfg.base_dir

    # 1) mag.obs
    x, y, bz = read_bz_obs(base_dir)

    # 2) grain centres and properties
    grain_path = os.path.join(base_dir, "4-Source_Detection", "Blob_Vector_Summary.csv")
    grains = pd.read_csv(grain_path)

    if "x_center" not in grains.columns or "y_center" not in grains.columns:
        raise ValueError(f"CSV must contain x_center and y_center. Found: {list(grains.columns)}")
    if "dec" not in grains.columns or "inc" not in grains.columns:
        raise ValueError(f"CSV must contain dec and inc. Found: {list(grains.columns)}")

    gx = grains["x_center"].to_numpy()
    gy = grains["y_center"].to_numpy()
    dec = grains["dec"].to_numpy()
    inc = grains["inc"].to_numpy()

    # moment column name can vary
    moment_col = get_moment_column(grains.columns)
    if moment_col is None:
        raise ValueError(f"Could not find moment column in CSV. Found: {list(grains.columns)}")
    moment = grains[moment_col].to_numpy()

    # 3) clip ranges
    bz_min = np.nanpercentile(bz, 2)
    bz_max = np.nanpercentile(bz, 98)

    m_min = np.nanmin(moment)
    m_max = np.nanmax(moment)

    # 4) figure
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), dpi=140, constrained_layout=True)

    def plot_bz_background(ax):
        # grey background
        sc = ax.scatter(x, y, c=bz, s=10, cmap="gray", vmin=bz_min, vmax=bz_max, rasterized=True)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return sc

    # Bz + centres
    ax = axs[0]
    sc_bz0 = plot_bz_background(ax)
    ax.scatter(gx, gy, c="red", s=60, edgecolors="white", linewidths=0.7, zorder=3)
    ax.set_title("Bz + Grain Centres", fontsize=13)

    # moment
    ax = axs[1]
    plot_bz_background(ax)
    sc_m = ax.scatter(gx, gy, c=moment, s=75, cmap="coolwarm", vmin=m_min, vmax=m_max,
                      edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_title("Moment", fontsize=13)

    # inclination
    ax = axs[2]
    plot_bz_background(ax)
    sc_i = ax.scatter(gx, gy, c=inc, s=75, cmap="coolwarm", vmin=-90, vmax=90,
                      edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_title("Inclination", fontsize=13)

    # declination
    ax = axs[3]
    plot_bz_background(ax)
    sc_d = ax.scatter(gx, gy, c=dec, s=75, cmap="coolwarm", vmin=0, vmax=360,
                      edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_title("Declination", fontsize=13)

    # colorbars next to their own axis
    cb_bz = fig.colorbar(sc_bz0, ax=axs[0], fraction=0.031, pad=0.01)
    cb_bz.set_label("Bz (nT)", fontsize=11)

    cb_m = fig.colorbar(sc_m, ax=axs[1], fraction=0.031, pad=0.01)
    cb_m.set_label(f"{moment_col}", fontsize=11)

    cb_i = fig.colorbar(sc_i, ax=axs[2], fraction=0.031, pad=0.01)
    cb_i.set_label("Inclination (°)", fontsize=11)

    cb_d = fig.colorbar(sc_d, ax=axs[3], fraction=0.031, pad=0.01)
    cb_d.set_label("Declination (°)", fontsize=11)

    out_png = os.path.join(base_dir, "4-Source_Detection", "BZ_grains_centers_results.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out.show()
    out.end()


# ============================================================
### Tool 3: 5. Plot declination arrows coloured by moment on Bz ###
# ============================================================
def plot_declination_arrows(cfg):
    out = Out()
    base_dir = cfg.base_dir

    # 1) mag.obs
    x, y, bz = read_bz_obs(base_dir)

    # 2) grain centres and properties
    grain_path = os.path.join(base_dir, "4-Source_Detection", "Blob_Vector_Summary.csv")
    grains = pd.read_csv(grain_path)

    for c in ["x_center", "y_center", "dec"]:
        if c not in grains.columns:
            raise ValueError(f"Missing column {c} in Blob_Vector_Summary.csv")

    gx = grains["x_center"].to_numpy()
    gy = grains["y_center"].to_numpy()
    dec = grains["dec"].to_numpy()

    moment_col = get_moment_column(grains.columns)
    if moment_col is None:
        raise ValueError(f"Could not find moment column in CSV. Found: {list(grains.columns)}")
    moment = grains[moment_col].to_numpy()

    # 3) clip ranges
    bz_min = np.nanpercentile(bz, 2)
    bz_max = np.nanpercentile(bz, 98)

    m_min = np.nanmin(moment)
    m_max = np.nanmax(moment)

    # 4) figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300, constrained_layout=True)

    # Bz background
    sc_bz = ax.scatter(x, y, c=bz, s=10, cmap="seismic", vmin=bz_min, vmax=bz_max, rasterized=True)

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.margins(0)
    ax.set_xlabel("X (µm)", fontsize=14)
    ax.set_ylabel("Y (µm)", fontsize=14)
    ax.tick_params(axis="both", length=4, labelsize=14)

    # declination arrows: 0° = North (+Y), clockwise positive
    dec_rad = np.deg2rad(dec)
    u = np.sin(dec_rad)
    v = np.cos(dec_rad)

    # fixed arrow length in data units
    dx = np.ptp(x[np.isfinite(x)])
    dy = np.ptp(y[np.isfinite(y)])
    arrow_len = 0.09 * max(dx, dy)

    # colour normalization for moment
    norm_m = Normalize(vmin=m_min, vmax=m_max)

    q = ax.quiver(
        gx, gy,
        u * arrow_len, v * arrow_len,
        moment,
        cmap="jet",
        norm=norm_m,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.008,
        headwidth=7,
        headlength=8,
        edgecolor="white",
        linewidth=0.6,
        zorder=4
    )

    # reference arrow (0° = North, pointing up)
    x0 = np.nanmax(x) - 0.10 * dx
    y0 = np.nanmax(y) - 0.22 * dy

    ax.quiver(
        x0, y0,
        0.0, arrow_len,
        color="black",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
        headwidth=6.0,
        headlength=7.0,
        zorder=4
    )

    ax.text(
        0.98, 0.98,
        "Declination reference: 0° (North)",
        transform=ax.transAxes,
        fontsize=14,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        zorder=5
    )

    # colorbars
    cb_bz = fig.colorbar(sc_bz, ax=ax, fraction=0.031, pad=0.01)
    cb_bz.set_label("Bz (nT)", fontsize=18)
    cb_bz.ax.tick_params(labelsize=14)

    cb_q = fig.colorbar(q, ax=ax, fraction=0.031, pad=-0.01)
    cb_q.set_label("Dipole moment (A·m²)", fontsize=18)
    cb_q.ax.tick_params(labelsize=14)
    cb_q.ax.yaxis.get_offset_text().set_fontsize(14)  # scientific notation

    out_png = os.path.join(base_dir, "4-Source_Detection", "BZ_grains_Dec.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out.show()
    out.end()


# ============================================================
### Tool 3: 5. Combined figure: Bz, centres by depth, declination arrows by moment ###
# ============================================================
def plot_combined_figure(cfg):
    out = Out()
    base_dir = cfg.base_dir

    # 1) mag.obs
    x, y, bz = read_bz_obs(base_dir)

    # 2) grain centres and properties
    grain_path = os.path.join(base_dir, "4-Source_Detection", "Blob_Vector_Summary.csv")
    grains = pd.read_csv(grain_path, sep=None, engine="python")

    for c in ["x_center", "y_center", "z_center", "dec"]:
        if c not in grains.columns:
            raise ValueError(f"Missing column {c} in Blob_Vector_Summary.csv")

    gx = grains["x_center"].to_numpy()
    gy = grains["y_center"].to_numpy()
    gz = grains["z_center"].to_numpy()
    dec = grains["dec"].to_numpy()

    moment_col = get_moment_column(grains.columns)
    if moment_col is None:
        raise ValueError(f"Could not find moment column in CSV. Found: {list(grains.columns)}")
    moment = grains[moment_col].to_numpy()

    # 3) colour scaling
    bz_max = np.nanmax(np.abs(bz))
    bz_min = -bz_max

    m_min = np.nanmin(moment)
    m_max = np.nanmax(moment)

    # 4) figure
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    # Bz background
    sc_bz = ax.scatter(x, y, c=bz, s=10, cmap="seismic", vmin=bz_min, vmax=bz_max, rasterized=True)

    # grain centres, coloured by depth
    sc_z = ax.scatter(gx, gy, c=gz, cmap="viridis", s=60, edgecolors="white", linewidths=0.7,
                      label="Detected dipole centres", zorder=3)

    # 5) declination arrows, coloured by moment (0 deg = North (+Y), clockwise positive)
    dec_rad = np.deg2rad(dec)
    u = np.sin(dec_rad)
    v = np.cos(dec_rad)

    dx = np.ptp(x[np.isfinite(x)])
    dy = np.ptp(y[np.isfinite(y)])
    arrow_len = 0.09 * max(dx, dy)

    norm_m = Normalize(vmin=m_min, vmax=m_max)

    q = ax.quiver(
        gx, gy,
        u * arrow_len, v * arrow_len,
        moment,
        cmap="jet",
        norm=norm_m,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.007,
        headwidth=7,
        headlength=8,
        edgecolor="black",
        linewidth=0.4,
        zorder=4
    )

    # 6) reference arrow (declination = 0 deg, North)
    x0 = np.nanmax(x) - 0.10 * dx
    y0 = np.nanmax(y) - 0.22 * dy

    ax.quiver(
        x0, y0, 0.0, arrow_len,
        color="black",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
        headwidth=6.0,
        headlength=7.0,
        zorder=5
    )

    ax.text(
        0.98, 0.98,
        "Declination reference: 0° (North)",
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"),
        zorder=6
    )

    # 7) scale bar
    bar_len = 100.0  # micrometres
    x_bar0 = np.min(x) + 0.05 * dx
    y_bar0 = np.min(y) + 0.05 * dy

    ax.plot([x_bar0, x_bar0 + bar_len], [y_bar0, y_bar0],
            color="k", linewidth=3, solid_capstyle="butt", zorder=5)
    ax.text(x_bar0 + 0.5 * bar_len, y_bar0 + 0.02 * dy, f"{int(bar_len)} µm",
            ha="center", va="bottom", fontsize=10, zorder=5)

    # 8) axes styling
    ax.set_xlabel("X (µm)", fontsize=13)
    ax.set_ylabel("Y (µm)", fontsize=13)
    ax.tick_params(axis="both", length=4, labelsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

    # 9) colourbars
    divider = make_axes_locatable(ax)

    cax_bz = divider.append_axes("right", size="4%", pad=0.15)
    cb_bz = fig.colorbar(sc_bz, cax=cax_bz)
    cb_bz.set_label("Bz (nT)", fontsize=13)
    cb_bz.ax.tick_params(labelsize=11)

    cax_z = divider.append_axes("right", size="4%", pad=0.8)
    cb_z = fig.colorbar(sc_z, cax=cax_z)
    cb_z.set_label("Depth (µm)", fontsize=13)
    cb_z.ax.tick_params(labelsize=11)

    cax_q = divider.append_axes("right", size="4%", pad=0.62)
    cb_q = fig.colorbar(q, cax=cax_q)
    cb_q.set_label("Dipole moment (A·m²)", fontsize=13)
    cb_q.ax.tick_params(labelsize=11)
    cb_q.ax.yaxis.get_offset_text().set_fontsize(11)

    plt.tight_layout()

    out_png = os.path.join(base_dir, "4-Source_Detection", "BZ_grains_centers_depth_dec.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    out.show()
    out.end()


# ============================================================
### Tool 3: 5. Plot dipole locations on LED on Bz ###
# ============================================================
def plot_centres_on_led(cfg):
    out = Out()
    base_dir = cfg.base_dir

    # mat file (Bz + LED)
    mat_file = glob.glob(os.path.join(base_dir, "*.mat"))[0]
    mat = loadmat(mat_file)

    Bz = np.squeeze(mat["Bz"]) * 1e9          # nT
    led = np.squeeze(mat["newLED"])

    # grain centres
    grain_path = os.path.join(base_dir, "4-Source_Detection", "Blob_Vector_Summary.csv")
    grains = pd.read_csv(grain_path, sep=None, engine="python")

    gx = grains["x_center"].values
    gy = grains["y_center"].values
    gz = grains["z_center"].values   # depth (µm)

    # pixel size (µm / pixel)
    pix = cfg.d
    ny, nx = Bz.shape

    x = np.linspace(0, nx * pix, nx)
    y = np.linspace(0, ny * pix, ny)

    extent = [x.min(), x.max(), y.min(), y.max()]

    # colour scaling (clip 2% outliers)
    bz_min = np.percentile(Bz, 2)
    bz_max = np.percentile(Bz, 98)

    plt.style.use("seaborn-v0_8-white")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Bz base
    im = ax.imshow(Bz, cmap="seismic", vmin=bz_min, vmax=bz_max, origin="upper", extent=extent)

    cbar = plt.colorbar(im, ax=ax, shrink=0.48, pad=0.01)
    cbar.set_label("Bz (nT)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # LED overlay
    ax.imshow(led, cmap="gray", alpha=0.7, origin="upper", extent=extent,
              vmin=np.percentile(led, 5), vmax=np.percentile(led, 95))

    # grain centres coloured by depth
    scz = ax.scatter(gx, gy, c=gz, cmap="viridis", s=60, edgecolors="white", linewidths=0.7,
                     label="Detected dipole centres", zorder=3)

    cbarz = plt.colorbar(scz, ax=ax, shrink=0.48, pad=0.01)
    cbarz.set_label("Depth (µm)", fontsize=12)
    cbarz.ax.tick_params(labelsize=10)

    # scale bar
    bar_len = 50.0  # µm
    x0 = x.min() + 0.05 * (x.max() - x.min())
    y0 = y.min() + 0.05 * (y.max() - y.min())

    ax.plot([x0, x0 + bar_len], [y0, y0], color="k", linewidth=3, solid_capstyle="butt", zorder=5)
    ax.text(x0 + 0.5 * bar_len, y0 + 0.02 * (y.max() - y.min()),
            f"{int(bar_len)} µm", ha="center", va="bottom", fontsize=10, zorder=5)

    ax.set_xlabel("X (µm)", fontsize=12)
    ax.set_ylabel("Y (µm)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

    plt.legend(frameon=True, fontsize=12, loc="upper right")
    plt.subplots_adjust(right=0.75)

    out_png = os.path.join(base_dir, "Bz_LED_grain_centers_depth.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    out.show()
    out.end()


# ============================================================
### Tool 3: 5. Plot obs, calc, misfit (Bx, By, Bz if present, else Bz only) ###
# ============================================================
def plot_obs_calc_misfit(cfg):
    out = Out()
    base_dir = cfg.base_dir

    data_dir = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output", "combined_grains", "data")

    obs_file = os.path.join(data_dir, "mag_observed.txt")
    calc_file = os.path.join(data_dir, "mag_final.txt")
    misfit_file = os.path.join(data_dir, "mag_misfit.txt")

    # check files
    if not os.path.exists(obs_file):
        raise FileNotFoundError(f"Missing file: {obs_file}")
    if not os.path.exists(calc_file):
        raise FileNotFoundError(f"Missing file: {calc_file}")
    if not os.path.exists(misfit_file):
        raise FileNotFoundError(f"Missing file: {misfit_file}")

    # load
    obs_df = clean_df(load_field_table(obs_file))
    calc_df = clean_df(load_field_table(calc_file))
    misfit_df = clean_df(load_field_table(misfit_file))

    if obs_df.empty or calc_df.empty or misfit_df.empty:
        raise ValueError("Empty/invalid data after cleaning.")

    # components present: bx/by/bz (vector) or just "value" (Bz only)
    components = [c for c in ("bx", "by", "bz") if c in obs_df.columns]
    if not components:
        components = ["value"]

    for comp in components:
        label = comp.upper() if comp != "value" else "Bz"

        # grid
        obs_grid, xs, ys = df_to_grid(obs_df, comp)
        calc_grid, xs2, ys2 = df_to_grid(calc_df, comp)
        misfit_grid, xs3, ys3 = df_to_grid(misfit_df, comp)

        if obs_grid is None or calc_grid is None or misfit_grid is None:
            raise ValueError(f"No valid grid (x/y unique values are empty) for {label}.")

        extent = [xs.min(), xs.max(), ys.min(), ys.max()]
        if not np.all(np.isfinite(extent)):
            raise ValueError(f"Bad extent (NaN/Inf) for {label}.")

        # plot (3 panels)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

        # same zero-centered scale across all three panels
        vmax = np.nanmax(np.abs(np.concatenate([obs_grid.ravel(), calc_grid.ravel(), misfit_grid.ravel()])))
        vmin = -vmax

        im0 = axes[0].imshow(-obs_grid, origin="lower", cmap="seismic", extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
        axes[0].set_xlabel("X (µm)")
        axes[0].set_ylabel("Y (µm)")
        axes[0].margins(0)
        cbar0 = plt.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.1, shrink=0.8)
        cbar0.set_label(f"Observed  {label} (nT)")

        im1 = axes[1].imshow(-calc_grid, origin="lower", cmap="seismic", extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
        axes[1].set_xlabel("X (µm)")
        axes[1].set_ylabel("Y (µm)")
        axes[1].margins(0)
        cbar1 = plt.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.1, shrink=0.7)
        cbar1.set_label(f"Calculated  {label} (nT)")

        im2 = axes[2].imshow(-misfit_grid, origin="lower", cmap="seismic", extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
        axes[2].set_xlabel("X (µm)")
        axes[2].set_ylabel("Y (µm)")
        axes[2].margins(0)
        cbar2 = plt.colorbar(im2, ax=axes[2], orientation="horizontal", pad=0.1, shrink=0.7)
        cbar2.set_label(f"Misfit (Obs − Calc)  {label} (nT)")

        plt.tight_layout(pad=0.2)

        fig_path = os.path.join(data_dir, f"obs_calc_misfit_{comp}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        out.show()

        # data cost, R2, RMS
        data_cost, r2, rms = fit_quality(obs_df[comp].to_numpy(), calc_df[comp].to_numpy())
        print(f"[{label}] Data cost = {data_cost:.6f}")
        print(f"[{label}] R2 = {r2:.6f}")
        print(f"[{label}] RMS = {rms:.6f}")

    out.end()


# ============================================================
### Analysis: Compare the QDM grain scale results to the SQUID bulk sample scale ###
# ============================================================
# colors for the two populations in row 2 (m^p / m^f) and the IRM reference
IRM_LINE_COLOR = "black"
MP_LINE_COLOR = "tab:blue"
MF_LINE_COLOR = "tab:green"

LETTER_BBOX = dict(facecolor="white", edgecolor="black", alpha=0.85,
                   boxstyle="round,pad=0.25", linewidth=0.8)


def fisher_mean(dec_deg, inc_deg):
    """Standard Fisher (1953) mean direction. dec/inc jointly define each
    unit vector (x=north, y=east, z=down); returns (mean_dec, mean_inc, R, kappa).
    NaNs if fewer than 2 valid directions or R==0."""
    dec_deg = np.asarray(dec_deg, dtype=float)
    inc_deg = np.asarray(inc_deg, dtype=float)
    ok = np.isfinite(dec_deg) & np.isfinite(inc_deg)
    dec_deg, inc_deg = dec_deg[ok], inc_deg[ok]
    n = len(dec_deg)
    if n < 2:
        return np.nan, np.nan, 0.0, np.nan

    dec_rad = np.radians(dec_deg)
    inc_rad = np.radians(inc_deg)
    x = np.cos(inc_rad) * np.cos(dec_rad)
    y = np.cos(inc_rad) * np.sin(dec_rad)
    z = np.sin(inc_rad)

    X, Y, Z = x.sum(), y.sum(), z.sum()
    R = np.sqrt(X**2 + Y**2 + Z**2)
    if R == 0:
        return np.nan, np.nan, R, np.nan

    mean_dec = np.degrees(np.arctan2(Y, X)) % 360
    mean_inc = np.degrees(np.arcsin(np.clip(Z / R, -1.0, 1.0)))
    kappa = (n - 1) / (n - R) if n > R else np.inf
    return mean_dec, mean_inc, R, kappa


def net_moment(dec_deg, inc_deg, moment):
    """Vector sum of a population of (Dec, Inc, moment) triples, returning
    the magnitude of the net (bulk-equivalent) moment vector, using the
    same x=north, y=east, z=down convention as fisher_mean()."""
    dec_deg = np.asarray(dec_deg, dtype=float)
    inc_deg = np.asarray(inc_deg, dtype=float)
    moment = np.asarray(moment, dtype=float)
    ok = np.isfinite(dec_deg) & np.isfinite(inc_deg) & np.isfinite(moment)
    dec_deg, inc_deg, moment = dec_deg[ok], inc_deg[ok], moment[ok]
    if len(moment) == 0:
        return np.nan

    dec_rad = np.radians(dec_deg)
    inc_rad = np.radians(inc_deg)
    Mx = moment * np.cos(inc_rad) * np.cos(dec_rad)
    My = moment * np.cos(inc_rad) * np.sin(dec_rad)
    Mz = moment * np.sin(inc_rad)
    return np.sqrt(Mx.sum()**2 + My.sum()**2 + Mz.sum()**2)


def plot_double_hist(ax, mp_vals, mf_vals, bins, xlabel, panel_letter,
                     vlines=None, show_legend=False):
    """Step-histogram overlay of m^p / m^f on one axis, with a
    transparent fill under each line so overlapping populations stay
    visible, plus optional vertical reference lines (Fisher mean,
    median, net moment, or the user-defined bulk IRM value)."""
    mp_vals = pd.Series(mp_vals).dropna()
    mf_vals = pd.Series(mf_vals).dropna()

    mp_counts, edges = np.histogram(mp_vals, bins=bins, density=False)
    mf_counts, _ = np.histogram(mf_vals, bins=bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.fill_between(centers, 0, mf_counts, step="mid", color=MF_LINE_COLOR, alpha=0.20, zorder=2)
    ax.fill_between(centers, 0, mp_counts, step="mid", color=MP_LINE_COLOR, alpha=0.20, zorder=3)

    ax.plot(centers, mf_counts, color=MF_LINE_COLOR, linewidth=1.8,
            drawstyle="steps-mid", label=f"$m^f$ (n={len(mf_vals)})", zorder=5)
    ax.plot(centers, mp_counts, color=MP_LINE_COLOR, linewidth=1.8,
            drawstyle="steps-mid", label=f"$m^p$ (n={len(mp_vals)})", zorder=6)

    if vlines:
        for value, color, ls, vlabel in vlines:
            if np.isfinite(value):
                ax.axvline(value, color=color, linestyle=ls, linewidth=1.8, zorder=7, label=vlabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("N dipoles")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(direction="in", length=5, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.text(0.03, 0.95, f"({panel_letter})", transform=ax.transAxes,
            fontsize=17, fontweight="bold", va="top", ha="left", bbox=LETTER_BBOX)

    if show_legend:
        ax.legend(loc="upper right", fontsize=7,
                  frameon=True, framealpha=0.85, edgecolor="black")


def net_direction(dec_deg, inc_deg, moment):
    """Dec/Inc of the vector sum of (Dec, Inc, moment) triples (moment-weighted mean direction)."""
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    inc = np.radians(np.asarray(inc_deg, dtype=float))
    m = np.asarray(moment, dtype=float)
    ok = np.isfinite(dec) & np.isfinite(inc) & np.isfinite(m)
    dec, inc, m = dec[ok], inc[ok], m[ok]
    X = (m * np.cos(inc) * np.cos(dec)).sum()
    Y = (m * np.cos(inc) * np.sin(dec)).sum()
    Z = (m * np.sin(inc)).sum()
    R = np.sqrt(X**2 + Y**2 + Z**2)
    if R == 0:
        return np.nan, np.nan
    return np.degrees(np.arctan2(Y, X)) % 360, np.degrees(np.arcsin(Z / R))


def _net_xy(dec, inc):
    """Equal-area (Schmidt) position of a direction; the point is plotted by |inc|."""
    r = np.sqrt(2.0) * np.sin(np.radians(90.0 - np.abs(np.asarray(inc, dtype=float))) / 2.0)
    d = np.radians(np.asarray(dec, dtype=float))
    return r * np.sin(d), r * np.cos(d)


def plot_stereonet_panel(ax, dec, inc, moment, norm, cmap, fisher_dec, fisher_inc,
                         irm_dec, irm_inc, title, panel_letter):
    """Equal-area net with the grains coloured by moment, the Fisher mean, the
    moment-weighted mean (vector sum) and the bulk IRM direction."""
    t = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.sin(t) * np.sqrt(2), np.cos(t) * np.sqrt(2), color="black", linewidth=1.2)
    for rr in (np.sqrt(2) * np.sin(np.radians(a) / 2) for a in (30, 60)):
        ax.plot(rr * np.sin(t), rr * np.cos(t), color="0.75", linewidth=0.6)
    ax.plot([0, 0], [-np.sqrt(2), np.sqrt(2)], color="0.75", linewidth=0.6)
    ax.plot([-np.sqrt(2), np.sqrt(2)], [0, 0], color="0.75", linewidth=0.6)
    ax.text(0, np.sqrt(2) + 0.04, "N", ha="center", va="bottom", fontsize=11)

    dec = np.asarray(dec, dtype=float)
    inc = np.asarray(inc, dtype=float)
    moment = np.asarray(moment, dtype=float)
    ok = np.isfinite(dec) & np.isfinite(inc) & np.isfinite(moment)
    x, y = _net_xy(dec[ok], inc[ok])
    down = inc[ok] >= 0
    col = cmap(norm(moment[ok]))
    ax.scatter(x[down], y[down], c=col[down], s=45, edgecolors="black", linewidths=0.4, zorder=4)
    ax.scatter(x[~down], y[~down], facecolors="white", edgecolors=col[~down], s=45, linewidths=1.2, zorder=4)

    wdec, winc = net_direction(dec, inc, moment)
    for (ddec, dinc, mk, size, fc, lab) in (
        (irm_dec, irm_inc, "*", 330, "red", f"Bulk IRM ({irm_dec % 360:.0f}°/{irm_inc:.0f}°)"),
        (fisher_dec, fisher_inc, "X", 150, "black", f"Fisher mean ({fisher_dec:.0f}°/{fisher_inc:.0f}°)"),
        (wdec, winc, "D", 110, "white", f"Moment-weighted ({wdec:.0f}°/{winc:.0f}°)"),
    ):
        if np.isfinite(ddec) and np.isfinite(dinc):
            px, py = _net_xy(ddec, dinc)
            ax.scatter(px, py, marker=mk, s=size, facecolors=fc, edgecolors="black",
                       linewidths=1.4, zorder=8, label=lab)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=8, frameon=True,
              framealpha=0.85, edgecolor="black")
    ax.text(0.0, 1.0, f"({panel_letter})", transform=ax.transAxes, fontsize=17, fontweight="bold",
            va="top", ha="left", bbox=LETTER_BBOX)


def compare_mp_mf(cfg,
                  irm_dec=270,             # bulk IRM measurement (natural sample reference): declination (deg)
                  irm_inc=0.5,             # inclination (deg)
                  irm_moment_am2=1.94e-5,  # net moment of the bulk sample (A*m^2)
                  mf_threshold=0.91,       # fixed threshold used for the m^f comparison
                  bz_vmax=4000):           # colour limit of the Bz maps (nT)
    """Combined figure of the m^p (global multi-dipole fit) and m^f (3D inversion) results, against the bulk IRM."""
    base_dir = cfg.base_dir

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 18,
    })

    IRM_DEC = irm_dec
    IRM_INC = irm_inc
    IRM_MOMENT_AM2 = irm_moment_am2
    MF_THRESHOLD = mf_threshold

    # load m^p data (single combined-grains fit)
    mp_data_dir = os.path.join(base_dir, "4-Source_Detection", "Tomofast_Output", "combined_grains", "data")
    mp_obs_df = clean_df(load_field_table(os.path.join(mp_data_dir, "mag_observed.txt")))
    mp_calc_df = clean_df(load_field_table(os.path.join(mp_data_dir, "mag_final.txt")))
    comp = "bz" if "bz" in mp_obs_df.columns else "value"
    comp_label = comp.upper() if comp != "value" else "Bz"

    mp_obs_grid, xs_p, ys_p = df_to_grid(mp_obs_df, comp)
    mp_calc_grid, _, _ = df_to_grid(mp_calc_df, comp)
    mp_extent = [xs_p.min(), xs_p.max(), ys_p.min(), ys_p.max()]

    mp_data_cost, mp_r2, mp_rms = fit_quality(mp_obs_df[comp].to_numpy(), mp_calc_df[comp].to_numpy())
    print(f"[m^p] Data cost = {mp_data_cost:.6f}  R2 = {mp_r2:.6f}  RMS = {mp_rms:.6f}")

    mp_rec_path = os.path.join(base_dir, "4-Source_Detection", "Blob_Vector_Summary.csv")
    mp_rec_df = pd.read_csv(mp_rec_path)
    mp_moment_col = get_moment_column(mp_rec_df.columns)
    if mp_moment_col is None:
        raise ValueError("Could not find a moment column in Blob_Vector_Summary.csv")
    mp_rec_df = mp_rec_df.dropna(subset=["dec", "inc", mp_moment_col]).copy()
    mp_rec_df = mp_rec_df[mp_rec_df[mp_moment_col] > 0].copy()

    # load m^f data (stitched across all blobs; grains at threshold = MF_THRESHOLD only)
    BLOB_ROOT = os.path.join(base_dir, "4-3D_Inversion_Filtered", "Tomofast_Output_Blobs")
    blob_dirs = list_blobs(BLOB_ROOT)

    mf_obs_all, mf_calc_all = [], []
    for blob in blob_dirs:
        data_dir = os.path.join(BLOB_ROOT, blob, blob, "data")
        obs_file = os.path.join(data_dir, "mag_observed.txt")
        calc_file = os.path.join(data_dir, "mag_final.txt")
        if not (os.path.exists(obs_file) and os.path.exists(calc_file)):
            continue
        mf_obs_all.append(clean_df(load_field_table(obs_file)))
        mf_calc_all.append(clean_df(load_field_table(calc_file)))

    mf_obs_df = pd.concat(mf_obs_all, ignore_index=True)
    mf_calc_df = pd.concat(mf_calc_all, ignore_index=True)

    mf_calc_grid, xs_f, ys_f = stitch_to_grid(mf_calc_df, comp)
    mf_extent = [xs_f.min(), xs_f.max(), ys_f.min(), ys_f.max()]

    mf_data_cost, mf_r2, mf_rms = fit_quality(mf_obs_df[comp].to_numpy(), mf_calc_df[comp].to_numpy())
    print(f"[m^f] Data cost = {mf_data_cost:.6f}  R2 = {mf_r2:.6f}  RMS = {mf_rms:.6f}")

    # m^f recovered grains: only threshold = MF_THRESHOLD, all grains, all blobs
    mf_rows = []
    all_thresholds_seen = set()
    for blob in blob_dirs:
        grains_folder = os.path.join(BLOB_ROOT, blob, blob, "Grains_detection")
        if not os.path.isdir(grains_folder):
            continue
        for fname in sorted(os.listdir(grains_folder)):
            if not fname.startswith("grain_dec_inc_") or "largest" in fname:
                continue
            df = pd.read_csv(os.path.join(grains_folder, fname))
            if df.empty:
                continue
            df["Threshold"] = df["Threshold"].round(2)
            all_thresholds_seen.update(df["Threshold"].unique().tolist())
            df = df[np.isclose(df["Threshold"], MF_THRESHOLD, atol=0.005)]
            if not df.empty:
                mf_rows.append(df)

    if not mf_rows:
        raise RuntimeError(
            f"No grain_dec_inc_*.csv rows found at threshold={MF_THRESHOLD}. "
            f"Thresholds actually present in these files: {sorted(all_thresholds_seen)}"
        )
    mf_rec_df = pd.concat(mf_rows, ignore_index=True)
    mf_rec_df = mf_rec_df[mf_rec_df["Net_Moment_Amplitude"] > 0].copy()
    print(f"[m^f] recovered grains at threshold={MF_THRESHOLD}: {len(mf_rec_df)}")

    # Fisher means, medians, and net (vector-summed) moments
    mp_fdec, mp_finc, _, _ = fisher_mean(mp_rec_df["dec"], mp_rec_df["inc"])
    mf_fdec, mf_finc, _, _ = fisher_mean(mf_rec_df["Dec"], mf_rec_df["Inc"])

    log_mp_moment = np.log10(mp_rec_df[mp_moment_col])
    log_mf_moment = np.log10(mf_rec_df["Net_Moment_Amplitude"])

    mp_median_log = np.nanmedian(log_mp_moment)
    mf_median_log = np.nanmedian(log_mf_moment)

    mp_net = net_moment(mp_rec_df["dec"], mp_rec_df["inc"], mp_rec_df[mp_moment_col])
    mf_net = net_moment(mf_rec_df["Dec"], mf_rec_df["Inc"], mf_rec_df["Net_Moment_Amplitude"])
    mp_net_log = np.log10(mp_net) if mp_net > 0 else np.nan
    mf_net_log = np.log10(mf_net) if mf_net > 0 else np.nan

    irm_moment_log = np.log10(IRM_MOMENT_AM2)

    print(f"\nm^p  Fisher mean:  Dec={mp_fdec:.2f}, Inc={mp_finc:.2f}")
    print(f"m^f  Fisher mean:  Dec={mf_fdec:.2f}, Inc={mf_finc:.2f}")
    print(f"IRM bulk:          Dec={IRM_DEC:.2f}, Inc={IRM_INC:.2f}, Moment={IRM_MOMENT_AM2:.3e} A*m^2")
    print(f"m^p  median log10(moment): {mp_median_log:.3f}   net log10(moment): {mp_net_log:.3f}")
    print(f"m^f  median log10(moment): {mf_median_log:.3f}   net log10(moment): {mf_net_log:.3f}")

    # combined 2x3 figure
    # rows: 0 maps, 1 histograms, (spacer), 2 stereonets coloured by moment
    fig = plt.figure(figsize=(19, 17))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 0.22, 1.25], hspace=0.0, wspace=0.32)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in (0, 1, 3)]

    # (a) observed Bz (m^p data), no location overlay (natural sample)
    ax_a, ax_b, ax_c = axes[0]

    vmax = bz_vmax
    im_a = ax_a.imshow(-mp_obs_grid, origin="lower", cmap="seismic", extent=mp_extent,
                       aspect="equal", vmin=-vmax, vmax=vmax)

    # (b) calculated Bz, m^f (stitched, all blobs)
    im_b = ax_b.imshow(-mf_calc_grid, origin="lower", cmap="seismic", extent=mf_extent,
                       aspect="equal", vmin=-vmax, vmax=vmax)
    leg_b = ax_b.legend(
        [plt.Line2D([0], [0], color="none")] * 3,
        [f"Data cost = {mf_data_cost:.3f}", f"R² = {mf_r2:.3f}", f"RMS = {mf_rms:.1f} nT"],
        loc="upper right", fontsize=9, frameon=True, framealpha=0.85, edgecolor="black",
        handlelength=0, handletextpad=0, title="$m^f$"
    )
    for t in leg_b.get_texts():
        t.set_color("black")
        t.set_fontweight("bold")

    # (c) calculated Bz, m^p (combined_grains)
    im_c = ax_c.imshow(-mp_calc_grid, origin="lower", cmap="seismic", extent=mp_extent,
                       aspect="equal", vmin=-vmax, vmax=vmax)
    leg_c = ax_c.legend(
        [plt.Line2D([0], [0], color="none")] * 3,
        [f"Data cost = {mp_data_cost:.3f}", f"R² = {mp_r2:.3f}", f"RMS = {mp_rms:.1f} nT"],
        loc="upper right", fontsize=9, frameon=True, framealpha=0.85, edgecolor="black",
        handlelength=0, handletextpad=0, title="$m^p$"
    )
    for t in leg_c.get_texts():
        t.set_color("black")
        t.set_fontweight("bold")

    for ax, letter in zip((ax_a, ax_b, ax_c), ["a", "b", "c"]):
        ax.set_xlabel(r"X ($\mu$m)")
        ax.set_ylabel(r"Y ($\mu$m)")
        ax.margins(0)
        ax.tick_params(direction="in", length=5, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.text(0.03, 0.95, f"({letter})", transform=ax.transAxes,
                fontsize=17, fontweight="bold", va="top", ha="left", color="black", bbox=LETTER_BBOX)

    # row 2: (d) declination, (e) inclination, (f) moment, m^p vs m^f, against the bulk IRM
    dec_bins = np.linspace(0, 360, 41)
    plot_double_hist(
        axes[1][0], mp_rec_df["dec"], mf_rec_df["Dec"],
        dec_bins, "Declination (°)", "d",
        vlines=[
            (IRM_DEC, IRM_LINE_COLOR, "--", f"Bulk IRM Dec={IRM_DEC:.1f}°"),
            (mf_fdec, MF_LINE_COLOR, "--", f"$m^f$ Fisher Dec={mf_fdec:.1f}°"),
            (mp_fdec, MP_LINE_COLOR, "--", f"$m^p$ Fisher Dec={mp_fdec:.1f}°"),
        ],
        show_legend=False,
    )
    axes[1][0].legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="black", fontsize=7)

    inc_bins = np.linspace(-90, 90, 41)
    plot_double_hist(
        axes[1][1], mp_rec_df["inc"], mf_rec_df["Inc"],
        inc_bins, "Inclination (°)", "e",
        vlines=[
            (IRM_INC, IRM_LINE_COLOR, "--", f"Bulk IRM Inc={IRM_INC:.1f}°"),
            (mf_finc, MF_LINE_COLOR, "--", f"$m^f$ Fisher Inc={mf_finc:.1f}°"),
            (mp_finc, MP_LINE_COLOR, "--", f"$m^p$ Fisher Inc={mp_finc:.1f}°"),
        ],
        show_legend=False,
    )
    axes[1][1].legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="black", fontsize=7)

    moment_bins = np.linspace(
        min(log_mp_moment.min(), log_mf_moment.min(), irm_moment_log),
        max(log_mp_moment.max(), log_mf_moment.max(), irm_moment_log),
        41,
    )
    plot_double_hist(
        axes[1][2], log_mp_moment, log_mf_moment,
        moment_bins, r"log$_{10}$(Moment / A$\cdot$m$^2$)", "f",
        vlines=[
            (irm_moment_log, IRM_LINE_COLOR, "--", f"Bulk IRM={irm_moment_log:.2f}"),
            (mf_median_log, MF_LINE_COLOR, ":", f"$m^f$ median={mf_median_log:.2f}"),
            (mf_net_log, MF_LINE_COLOR, "-.", f"$m^f$ net={mf_net_log:.2f}"),
            (mp_median_log, MP_LINE_COLOR, ":", f"$m^p$ median={mp_median_log:.2f}"),
            (mp_net_log, MP_LINE_COLOR, "-.", f"$m^p$ net={mp_net_log:.2f}"),
        ],
        show_legend=True,
    )

    # row 3: stereonets coloured by moment (one shared colour scale), under the dec / inc histograms
    all_moment = np.concatenate([mp_rec_df[mp_moment_col].to_numpy(), mf_rec_df["Net_Moment_Amplitude"].to_numpy()])
    norm_m = LogNorm(vmin=all_moment.min(), vmax=all_moment.max())
    cmap_m = plt.cm.jet
    plot_stereonet_panel(axes[2][0], mf_rec_df["Dec"], mf_rec_df["Inc"], mf_rec_df["Net_Moment_Amplitude"],
                         norm_m, cmap_m, mf_fdec, mf_finc, IRM_DEC, IRM_INC, "$m^f$", "g")
    plot_stereonet_panel(axes[2][1], mp_rec_df["dec"], mp_rec_df["inc"], mp_rec_df[mp_moment_col],
                         norm_m, cmap_m, mp_fdec, mp_finc, IRM_DEC, IRM_INC, "$m^p$", "h")
    axes[2][2].axis("off")
    cax_m = axes[2][2].inset_axes([0.15, 0.45, 0.06, 0.5])
    cb_m = fig.colorbar(ScalarMappable(norm=norm_m, cmap=cmap_m), cax=cax_m)
    cb_m.set_label("Moment (A$\\cdot$m$^2$)")

    print(f"m^p  moment-weighted mean: Dec={net_direction(mp_rec_df['dec'], mp_rec_df['inc'], mp_rec_df[mp_moment_col])[0]:.2f}, "
          f"Inc={net_direction(mp_rec_df['dec'], mp_rec_df['inc'], mp_rec_df[mp_moment_col])[1]:.2f}")
    mfw = net_direction(mf_rec_df["Dec"], mf_rec_df["Inc"], mf_rec_df["Net_Moment_Amplitude"])
    print(f"m^f  moment-weighted mean: Dec={mfw[0]:.2f}, Inc={mfw[1]:.2f}")

    # shared vertical colorbar for row 1
    fig.canvas.draw()
    pos_c = ax_c.get_position()
    cbar_gap = 0.015
    cbar_width = 0.012
    cbar_shrink = 0.75
    cbar_height = pos_c.height * cbar_shrink
    cbar_bottom = pos_c.y0 + (pos_c.height - cbar_height) / 2
    cbar_left = pos_c.x1 + cbar_gap
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(im_c, cax=cax, orientation="vertical")
    cbar.set_label(f"{comp_label} (nT)")

    out_dir = os.path.join(base_dir, "4-Source_Detection")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "Merged_mp_mf_NaturalSample_Comparison.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print("\nSaved:", out_png)
