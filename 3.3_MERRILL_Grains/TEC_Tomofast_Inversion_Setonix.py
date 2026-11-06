"""
TEC Tomofast Inversion - Setonix Version
Replaces WSL calls with direct Linux commands on Setonix.
Runs AFTER the forward modelling script (uses its output as input data).

Directory structure expected on Setonix:
  /scratch/pawsey1147/ahussein/tomofast/
      Tomofast-x/                       <- compiled tomofastx lives here
      data/
          tomofast_forward/
              body_118_24_mult/
                  Parfile_body_118_24_mult_inversion.txt
                  Magnetization_body_118_24_mult.txt
                  Meshgrid_body_118_24_mult.txt
      output/
          TEC_body_118_24_mult/          <- forward modelling results (input to this step)
              data/mag_prior.txt
      output_inversion/                  <- inversion results will appear here
"""

import subprocess
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on Setonix
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# USER CONFIG - only change these lines
# ============================================================
SCRATCH        = "/scratch/pawsey1147/ahussein/tomofast"
TOMOFAST_HOME  = os.path.join(SCRATCH, "Tomofast-x")
DATA_DIR       = os.path.join(SCRATCH, "data", "tomofast_forward")
FORWARD_OUTPUT = os.path.join(SCRATCH, "output")            # output of forward run
OUTPUT_DIR     = os.path.join(SCRATCH, "output_inversion")  # results of THIS run
N_CORES        = 64  # number of MPI cores per inversion
# ============================================================

def run(cmd):
    """Run a shell command and print output live."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] command failed with code {result.returncode}")

# --- ensure folders exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
run(f"mkdir -p {TOMOFAST_HOME}/data/TEC")

# --- loop over every body folder ---
if not os.path.isdir(DATA_DIR):
    print(f"[ERROR] Data folder not found: {DATA_DIR}")
    exit(1)

bodies = sorted([d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d))])

if not bodies:
    print(f"[ERROR] No body folders found in {DATA_DIR}")
    exit(1)

print(f"\nFound {len(bodies)} bodies to invert: {bodies}\n")

for stem in bodies:
    body_dir       = os.path.join(DATA_DIR, stem)
    forward_result = os.path.join(FORWARD_OUTPUT, stem, "data", "mag_prior.txt")
    out_dir        = os.path.join(OUTPUT_DIR, stem)

    print(f"\n{'='*50}")
    print(f"  Inverting: {stem}")
    print(f"{'='*50}")

    if not os.path.isfile(forward_result):
        print(f"[SKIP] Forward modelling output not found: {forward_result}")
        print("       Run the forward modelling script first!")
        continue

    # copy input files into Tomofast directory
    run(f"cp '{body_dir}/Parfile_{stem}_inversion.txt' {TOMOFAST_HOME}/parfiles/")
    run(f"cp '{forward_result}' {TOMOFAST_HOME}/data/TEC/Forward_{stem}.obs")
    run(f"cp '{body_dir}/Magnetization_{stem}.txt' {TOMOFAST_HOME}/data/TEC/")
    run(f"cp '{body_dir}/Meshgrid_{stem}.txt'      {TOMOFAST_HOME}/data/TEC/")

    # create output folder
    run(f"mkdir -p {TOMOFAST_HOME}/output/TEC_{stem}_inversion")
    run(f"mkdir -p {out_dir}")

    # run Tomofast inversion
    run(
        f"cd {TOMOFAST_HOME} && "
        f"srun -n {N_CORES} ./tomofastx -p ./parfiles/Parfile_{stem}_inversion.txt "
        f"| tee ./output/TEC_{stem}_inversion/log.txt"
    )

    # copy results to output folder (including SENSIT for reuse)
    run(
        f"rsync -av --exclude='SENSIT' "
        #f"rsync -av "
        f"{TOMOFAST_HOME}/output/TEC_{stem}_inversion/ "
        f"{out_dir}/"
    )

print("\n\nAll inversions done!")

# ============================================================
# PLOTTING - Bxyz misfits (observed / calculated / misfit)
# ============================================================

def plot_bxyz(x_coords, y_coords, data, sfx, out_dir, label):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{sfx} - {label}", fontsize=14)

    bz_col = data[:, 5]
    scale = np.nanmax(np.abs(bz_col))

    for ax, col, component in zip(axes, [3, 4, 5], ["Bx", "By", "Bz"]):
        b_values = -data[:, col] if component == "Bz" else data[:, col]  # flip Bz, tomofast -> qdm convention
        df = pd.DataFrame({"x": x_coords, "y": y_coords, "b": b_values})
        df = df.sort_values(["y", "x"]).reset_index(drop=True)
        unique_x = np.sort(df["x"].unique())
        unique_y = np.sort(df["y"].unique())
        nx, ny = len(unique_x), len(unique_y)
        if len(df) != nx * ny:
            print(f"[SKIP] {sfx} {label} {component}: not a regular grid")
            continue
        X = df["x"].values.reshape(ny, nx)
        Y = df["y"].values.reshape(ny, nx)
        Z = df["b"].values.reshape(ny, nx)
        im = ax.contourf(X, Y, Z, levels=100, cmap="seismic", vmin=-scale, vmax=scale)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{component} (nT)", fontsize=12)
        cbar.ax.tick_params(labelsize=11)
        ax.set_xlabel("x (µm)", fontsize=12)
        ax.set_ylabel("y (µm)", fontsize=12)
        ax.set_title(component, fontsize=13)
        ax.tick_params(labelsize=11)
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    plt.tight_layout()
    filename = os.path.join(out_dir, f"Bxyz_{sfx}_{label}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_bz_only(x_coords, y_coords, data, sfx, out_dir, label):
    """For 4-column files: X Y Z Bz (single component only)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(f"{sfx} - {label}", fontsize=14)

    bz_col = -data[:, 3]  # flip Bz, tomofast -> qdm convention
    scale = np.nanmax(np.abs(bz_col))

    df = pd.DataFrame({"x": x_coords, "y": y_coords, "b": bz_col})
    df = df.sort_values(["y", "x"]).reset_index(drop=True)
    unique_x = np.sort(df["x"].unique())
    unique_y = np.sort(df["y"].unique())
    nx, ny = len(unique_x), len(unique_y)
    if len(df) != nx * ny:
        print(f"[SKIP] {sfx} {label} Bz: not a regular grid")
        return
    X = df["x"].values.reshape(ny, nx)
    Y = df["y"].values.reshape(ny, nx)
    Z = df["b"].values.reshape(ny, nx)
    im = ax.contourf(X, Y, Z, levels=100, cmap="seismic", vmin=-scale, vmax=scale)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Bz (nT)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    ax.set_xlabel("x (µm)", fontsize=12)
    ax.set_ylabel("y (µm)", fontsize=12)
    ax.set_title("Bz", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    plt.tight_layout()
    filename = os.path.join(out_dir, f"Bz_{sfx}_{label}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


print("\nGenerating plots...")
for sfx in sorted(os.listdir(OUTPUT_DIR)):
    sample_dir = os.path.join(OUTPUT_DIR, sfx)
    if not os.path.isdir(sample_dir):
        continue

    data_dir = os.path.join(sample_dir, "data")
    files_to_plot = {
        "observed":   os.path.join(data_dir, "mag_observed.txt"),
        "calculated": os.path.join(data_dir, "mag_final.txt"),
        "misfit":     os.path.join(data_dir, "mag_misfit.txt"),
    }

    for label, filepath in files_to_plot.items():
        if not os.path.isfile(filepath):
            print(f"[missing] {filepath}")
            continue
        data = np.loadtxt(filepath, skiprows=1)
        x_coords = data[:, 0]
        y_coords = data[:, 1]

        if data.shape[1] >= 6:
            plot_bxyz(x_coords, y_coords, data, sfx, sample_dir, label)
        elif data.shape[1] == 4:
            plot_bz_only(x_coords, y_coords, data, sfx, sample_dir, label)
        else:
            print(f"[SKIP] {sfx} {label}: unexpected column count {data.shape[1]}")

print("\nAll plots saved!")
