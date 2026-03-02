# Quick Start Guide

Get up and running with tomofast_x_mu in 5 minutes.

## Installation

```bash
cd tomofast_x_mu
pip install -e .
```

For development with tests:

```bash
pip install -e ".[dev]"
```

## Verify Installation

Run the smoke test (no Tomofast-x required):

```bash
tomofast_x_mu smoke --h-units auto
```

This processes a sample QDM scan and generates meshes/parfiles. Output goes to `./tomofast_x_mu_smoke_out/`.

## Basic Python Usage

### 1. Load QDM Data

```python
from pathlib import Path
import tomofast_x_mu as tfmu

# Load .mat file -> xarray Dataset
mat_path = Path("scan.mat")
ds = tfmu.load_qdm_mat(mat_path, save_csv=True, h_units="auto")

# Inspect
print(ds)
print(f"Pixel size: {ds.attrs['pixel_size_um']:.2f} µm")
print(f"Sensor distance: {ds.attrs['sensor_sample_distance_um']:.2f} µm")
```

### 2. Convert to Tomofast Format

```python
# CSV -> Tomofast observation file
csv_path = mat_path.with_suffix(".csv")
obs_path = Path("Mag_Inversion.obs")

tfmu.format_tomofast_obs(csv_path, obs_path, flip_bz=True)
```

### 3. Detect Magnetic Anomalies

```python
# Find blobs and create per-blob observation files
blobs = tfmu.detect_anomalies(
    str(obs_path),
    threshold_factor=2.75,
    pad_x=40,
    pad_y=10,
    min_blob_cells=150,
    max_blob_cells=100000,
    nz_fixed=120,
    out_obs_dir="blob_obs",
    windows_path="blob_windows.txt",
    summary_path="blob_summary.txt"
)

print(f"Found {len(blobs)} magnetic anomalies")
```

### 4. Generate Meshes

```python
# Create meshes for each blob
meshes = tfmu.build_mesh_per_blob(
    "blob_windows.txt",
    "meshgrids",
    "mesh_summary.txt",
    npad_x=20,
    npad_y=20,
    nz=120,
    dz0=4.4
)
```

### 5. Create Parfiles

```python
# Generate Tomofast parameter files
parfiles = tfmu.create_inversion_parfiles("mesh_summary.txt", "parfiles")
print(f"Created {len(parfiles)} parfiles")
```

## Running Inversions

To actually run inversions, you need:
- Tomofast-x executable (`tomofastx`)
- MPI runtime (e.g., OpenMPI)

```python
# Run inversions (requires Tomofast-x)
output_dirs = tfmu.run_tomofast_blobs(
    base_dir="/path/to/project",
    tomofast_home="/path/to/Tomofast-x",
    nproc=7,
    wsl_exe=None  # or r"C:\Windows\System32\wsl.exe" for WSL
)
```

## Important: Units Convention

QDM `.mat` files often use **mixed units**:
- `step` (pixel size) in **meters** (e.g., `4.4e-06` m)
- `h` (sensor distance) in **micrometers** (e.g., `5` µm)

Always use `h_units="auto"` (default) when loading real data:

```python
ds = tfmu.load_qdm_mat(mat_path, h_units="auto")  # ✓ Correct
ds = tfmu.load_qdm_mat(mat_path, h_units="m")     # ✗ Wrong for typical QDM data
```

Using `h_units="m"` on real data will place observations at meter-scale distances (10⁶× error), causing meaningless inversions.

## Next Steps

- Read the full documentation in `DOCUMENTATION.md` for the complete workflow
- See `docs/architecture.md` for pipeline details
- Check `docs/h_units_bug_summary.md` for units background

## CLI Commands

```bash
# Check version
tomofast_x_mu version

# View architecture diagram location
tomofast_x_mu diagram

# Run smoke test with custom parameters
tomofast_x_mu smoke --mat data.mat --out results/ --h-units auto
```

## Troubleshooting

**Problem:** `FileNotFoundError` when running smoke test

**Solution:** Ensure you're in the repository root or specify `--mat` path explicitly.

---

**Problem:** Inversion results look wrong (unrealistic magnetizations)

**Solution:** Check that you used `h_units="auto"` when loading the `.mat` file. Incorrect units are the most common cause of bad inversions.

---

**Problem:** `ValueError: CSV data is not in expected row-major order`

**Solution:** The CSV must be sorted with y varying slowly and x varying fast. Use `load_qdm_mat()` which handles this automatically.
