"""
TEC Tomofast Forward Modelling - Setonix Version
Replaces WSL calls with direct Linux commands on Setonix.

Directory structure expected on Setonix:
  /scratch/pawsey1147/ahussein/tomofast/
      Tomofast-x/              <- compiled tomofastx lives here
      data/
          tomofast_forward/
              body_118_24_mult/
                  Parfile_body_118_24_mult.txt
                  Forward_body_118_24_mult.obs
                  Magnetization_body_118_24_mult.txt
                  Meshgrid_body_118_24_mult.txt
              body_27_6_mult/
                  ...
      output/                  <- results will appear here
"""

import subprocess
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed on Setonix
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# USER CONFIG - only change these lines
# ============================================================
SCRATCH       = "/scratch/pawsey1147/ahussein/tomofast"
TOMOFAST_HOME = os.path.join(SCRATCH, "Tomofast-x")
DATA_DIR      = os.path.join(SCRATCH, "data", "tomofast_forward")
OUTPUT_DIR    = os.path.join(SCRATCH, "output")
N_CORES       = 64   # number of MPI cores per inversion
# ============================================================

def run(cmd):
    """Run a shell command and print output live."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, text=True,
                            capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] command failed with code {result.returncode}")

# --- ensure output and data folders exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
run(f"mkdir -p {TOMOFAST_HOME}/data/TEC")

# --- loop over every body folder ---
if not os.path.isdir(DATA_DIR):
    print(f"[ERROR] Data folder not found: {DATA_DIR}")
    print("Please upload your data files first!")
    exit(1)

bodies = sorted([d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d))])

if not bodies:
    print(f"[ERROR] No body folders found in {DATA_DIR}")
    exit(1)

print(f"\nFound {len(bodies)} bodies to process: {bodies}\n")

for stem in bodies:
    body_dir = os.path.join(DATA_DIR, stem)
    #out_dir  = os.path.join(OUTPUT_DIR, f"TEC_{stem}")
    out_dir  = os.path.join(OUTPUT_DIR, stem)

    print(f"\n{'='*50}")
    print(f"  Processing: {stem}")
    print(f"{'='*50}")

    # copy input files into Tomofast directory
    run(f"cp '{body_dir}/Parfile_{stem}.txt'      {TOMOFAST_HOME}/parfiles/")
    run(f"cp '{body_dir}/Forward_{stem}.obs'       {TOMOFAST_HOME}/data/TEC/")
    run(f"cp '{body_dir}/Magnetization_{stem}.txt' {TOMOFAST_HOME}/data/TEC/")
    run(f"cp '{body_dir}/Meshgrid_{stem}.txt'      {TOMOFAST_HOME}/data/TEC/")

    # create output folder
    run(f"mkdir -p {out_dir}")

    # run Tomofast inversion
    run(
        f"cd {TOMOFAST_HOME} && "
        f"srun -n {N_CORES} ./tomofastx -p ./parfiles/Parfile_{stem}.txt "
        f"| tee {out_dir}/log.txt"
    )

    # copy results to output folder (skip large SENSIT files)
    run(
        f"rsync -av --exclude='SENSIT' "
        #f"rsync -av "
        f"{TOMOFAST_HOME}/output/TEC_{stem}/ "
        f"{out_dir}/"
    )

print("\n\nAll inversions done!")

# ============================================================
# PLOTTING - Bxyz maps for each body
# ============================================================

def plot_bxyz(x_coords, y_coords, data, sfx, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(sfx, fontsize=14)

    # compute ONE shared scale from Bz (column 5), matching the published code
    bz_col = data[:, 5]
    scale = np.nanmax(np.abs(bz_col))

    for ax, col, component in zip(axes, [3, 4, 5], ["Bx", "By", "Bz"]):
        #df = pd.DataFrame({"x": x_coords, "y": y_coords, "b": -data[:, col]})
        b_values = -data[:, col] if component == "Bz" else data[:, col]  # flip only Bz from tomofast to qdm convention
        df = pd.DataFrame({"x": x_coords, "y": y_coords, "b": b_values})
        df = df.sort_values(["y", "x"]).reset_index(drop=True)
        unique_x = np.sort(df["x"].unique())
        unique_y = np.sort(df["y"].unique())
        nx, ny = len(unique_x), len(unique_y)
        if len(df) != nx * ny:
            print(f"[SKIP] {sfx} {component}: not a regular grid")
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
    filename = os.path.join(out_dir, f"Bxyz_{sfx}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show() 
    plt.close()
    print("Saved:", filename)


print("\nGenerating plots...")
for sfx in sorted(os.listdir(OUTPUT_DIR)):
    sample_dir = os.path.join(OUTPUT_DIR, sfx)
    if not os.path.isdir(sample_dir):
        continue
    filepath = os.path.join(sample_dir, "data", "mag_prior.txt")
    if not os.path.isfile(filepath):
        print(f"[missing] {filepath}")
        continue
    data = np.loadtxt(filepath, skiprows=1)
    plot_bxyz(data[:, 0], data[:, 1], data, sfx, sample_dir)

print("\nAll plots saved!")
