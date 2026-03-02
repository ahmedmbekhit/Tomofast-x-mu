# Notebook to Package Migration Guide

## Overview

The `tomofast_x_mu` Python package was refactored from the original Jupyter notebook `Tomofast-x-µ(v1).ipynb`. This document explains the relationship between the notebook and the package, and how to migrate from notebook-based workflows to the package.

## Notebook Structure

The original notebook contains **75 cells** organized into three main tools:

1. **Tool 1: 3D Inversion** (Cells 10-37)
   - Read QDM .mat files
   - Format data for Tomofast
   - Detect anomalies
   - Create meshes and parfiles
   - Run inversions
   - Extract 3D grains

2. **Tool 2: Source Location Detection** (Cells 39-48)
   - Create single-cell meshes
   - Run grid search inversions
   - Find minimum data cost positions

3. **Tool 3: Vector Direction & Intensity** (Cells 50-71)
   - Combine best locations
   - Run dipole fitting
   - Compute magnetization vectors
   - Generate final reports

## Package Mapping

### Notebook Cell → Package Module

| Notebook Section | Package Module | Function |
|------------------|----------------|----------|
| Cell 12: Read Bz mat | `io.py` | `load_qdm_mat()` |
| Cell 16: Format data | `format.py` | `format_tomofast_obs()` |
| Cell 18: Detect anomalies | `detect.py` | `detect_anomalies()` |
| Cell 20: Create meshes | `mesh.py` | `build_mesh_per_blob()` |
| Cell 22: Create parfiles | `parfile.py` | `create_inversion_parfiles()` |
| Cell 24: Run Tomofast | `runner.py` | `run_tomofast_blobs()` |
| Cell 29: Extract grains | `detect.py` | `extract_grains()` |
| Cell 35: Max cell extraction | `detect.py` | `extract_max_cells()` |
| Cell 31: Grain statistics | `analysis.py` | `grain_statistics()` |
| Cell 41: Single-cell meshes | `mesh.py` | `create_single_cell_meshes()` |
| Cell 43: Source parfiles | `parfile.py` | `create_source_parfiles()` |
| Cell 45: Run source detection | `runner.py` | `run_source_detection()` |
| Cell 52: Combine locations | `analysis.py` | `combine_best_locations()` |
| Cell 55: Fitting parfile | `parfile.py` | `create_fitting_parfile()` |
| Cell 57: Run dipole fitting | `runner.py` | `run_dipole_fitting()` |
| Cell 60: Vector analysis | `analysis.py` | `vector_analysis()` |
| Cells 8, 27, 49, 62: Plotting | `plotting.py` | Various plot functions |

## Key Conventions from Notebook

### 1. Coordinate System Convention

**Critical:** QDM and Tomofast use opposite Z-axis conventions.

| System | Z-axis Direction | Conversion Rule |
|--------|------------------|-----------------|
| QDM | Positive **up** | Flip both Bz and Mz |
| Tomofast | Positive **down** | Flip both Bz and Mz |

**Implementation in package:**
- `format_tomofast_obs()`: `flip_bz=True` (default) converts QDM → Tomofast
- `analysis.py`: `Mz = -Mz_raw` converts Tomofast → QDM for reporting

### 2. Default Parameters

The notebook established these recommended defaults (now used in package):

```python
# Anomaly detection
threshold_factor = 2.75  # Threshold = factor × std(Bz)
PAD_X = 40              # Padding cells in X
PAD_Y = 10              # Padding cells in Y
MIN_BLOB_CELLS = 150    # Minimum blob size
MAX_BLOB_CELLS = 100000 # Maximum blob size
NZ_FIXED = 120          # Fixed Z dimension

# Mesh generation
dx_core = 4.4  # µm (must match data spacing)
dy_core = 4.4  # µm
npad_x = 20    # Padding cells
npad_y = 20
nz = 120       # Z cells
dz0 = 4.4      # First Z cell thickness (µm)
rx = 1.2       # X padding growth factor
ry = 1.2       # Y padding growth factor
rz = 1.05      # Z padding growth factor

# Grain detection
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Tomofast compression
compression_rate = 0.05  # Range: 0.01-0.1 (lower = more accurate, slower)
```

### 3. User Attention Points from Notebook

The notebook highlighted these critical user decisions:

1. **Pixel size** (Cell 7): Must match QDM scan resolution (typically 4.4 µm)

2. **Vmin/Vmax for plots** (Cell 11): Important for visualizing tiny dipoles

3. **Threshold factor** (Cell 17): Adjust by ±0.25 to capture reliable anomalies

4. **Core cell sizes** (Cell 19): Must match data spacing (dx_core = dy_core = pixel_size)

5. **Padding cells** (Cell 19): Ensure full 3D model recovery and overcome edge effects

6. **Compression rate** (Cell 21): Balance between speed and accuracy (0.01-0.1)

7. **Amplitude threshold** (Cell 28, 34): 0.9 is recommended for max cell extraction

8. **Source search range** (Cell 39): Define RX, RY ranges around maximum cell

9. **Cell volume** (Cells 59, 63): For dipole moment calculation (dx × dy × dz)

### 4. Directory Structure Convention

The notebook established this directory structure (preserved in package):

```
project_root/
├── 0-Parfiles/
│   ├── Parfile_QDM_Inversion_Blob_001.txt
│   ├── Parfile_QDM_Inversion_Blob_002.txt
│   └── Parfile_QDM_Dipole.txt
├── 4-3D_Inversion_Filtered/
│   ├── Mag_Inversion.obs
│   ├── Blob_Clipped_obs_RECT/
│   │   ├── Blob_001_RECT.obs
│   │   └── Blob_002_RECT.obs
│   ├── Blob_Meshgrids/
│   │   ├── Meshgrid_Blob_001.txt
│   │   └── Meshgrid_Blob_002.txt
│   ├── Tomofast_Output_Blobs/
│   │   ├── Blob_001/
│   │   │   ├── model/
│   │   │   ├── data/
│   │   │   └── Paraview/
│   │   └── Blob_002/
│   └── blob_windows.txt
└── 4-Source_Detection/
    ├── Single_cell_meshs/
    │   └── Blob_001/
    │       └── grain_1/
    ├── combined_best_blobs_mesh.txt
    └── Tomofast_Output/
        └── combined_grains/
```

## Migration Examples

### From Notebook Cell to Package Function

**Notebook (Cell 12):**
```python
# Read Bz mat file
mat = scipy.io.loadmat(mat_path)
Bz = np.squeeze(mat["Bz"]) * 1e9
step = float(mat["step"]) * 1e6
h = float(mat["h"]) * 1e6
# ... manual CSV creation ...
```

**Package:**
```python
import tomofast_x_mu as tfmu
ds = tfmu.load_qdm_mat(mat_path, save_csv=True, h_units="auto")
```

---

**Notebook (Cell 18):**
```python
# Detect anomalies (manual implementation with scipy.ndimage)
threshold = threshold_factor * np.std(bz_grid)
mask = np.abs(bz_grid) > threshold
# ... 50+ lines of blob detection code ...
```

**Package:**
```python
blobs = tfmu.detect_anomalies(
    obs_path,
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
```

---

**Notebook (Cell 24):**
```python
# Run Tomofast (manual WSL commands)
for blob_id in blob_ids:
    cmd = f"wsl bash -lc 'cd {tomofast_home} && mpirun -np {nproc} ./tomofastx -p ./parfiles/{parfile}'"
    subprocess.run(cmd, shell=True)
    # ... manual file copying ...
```

**Package:**
```python
outputs = tfmu.run_tomofast_blobs(
    base_dir=".",
    tomofast_home="/path/to/Tomofast-x",
    nproc=7,
    wsl_exe=r"C:\Windows\System32\wsl.exe"  # or None for native
)
```

## Improvements in Package vs Notebook

### 1. Error Handling
- **Notebook:** Minimal error checking, crashes on edge cases
- **Package:** Comprehensive validation, clear error messages

### 2. Units Handling
- **Notebook:** Hardcoded `h * 1e6` conversion (assumes meters)
- **Package:** `h_units="auto"` detects mixed-units convention

### 3. Code Organization
- **Notebook:** 75 cells, ~2000 lines, mixed logic
- **Package:** 8 modules, clear separation of concerns

### 4. Reusability
- **Notebook:** Copy-paste cells, manual parameter editing
- **Package:** Import functions, programmatic control

### 5. Testing
- **Notebook:** No tests
- **Package:** 59 unit tests, smoke test CLI

### 6. Documentation
- **Notebook:** Markdown cells with "User Attention" notes
- **Package:** Complete API docs, quick start guide, troubleshooting

## Backward Compatibility

The package preserves all notebook functionality:

✅ Same default parameters
✅ Same directory structure
✅ Same file formats
✅ Same coordinate conventions
✅ Same Tomofast-x interface

**Breaking changes:**
- `h_units` parameter added (default `"m"` for backward compatibility, but `"auto"` recommended)
- Some internal function signatures changed (but high-level workflow is identical)

## Recommended Migration Path

1. **Keep using notebook for exploration** - It's great for interactive analysis
2. **Use package for production workflows** - Better error handling, testing, automation
3. **Hybrid approach** - Use package functions inside notebook cells:

```python
# In notebook cell
import tomofast_x_mu as tfmu

# Replace manual code with package functions
ds = tfmu.load_qdm_mat("scan.mat", h_units="auto")
blobs = tfmu.detect_anomalies(obs_path, threshold_factor=2.75)
meshes = tfmu.build_mesh_per_blob(windows_path, out_dir, summary_path)

# Continue with notebook-style plotting and exploration
import matplotlib.pyplot as plt
fig, ax = tfmu.plot_bz_led("scan.mat")
plt.show()
```

## Notebook-Specific Features Not in Package

Some notebook features are intentionally not in the package:

1. **Interactive widgets** - Use Jupyter for interactive parameter tuning
2. **Inline plots** - Notebook shows plots automatically; package returns fig/ax for manual display
3. **Progress bars** - Notebook had visual progress; package uses logging (add if needed)
4. **WSL path conversion UI** - Notebook had manual path entry; package requires programmatic paths

## Getting Help

- **Notebook questions:** See original notebook markdown cells
- **Package questions:** See `DOCUMENTATION.md` and `QUICKSTART.md`
- **Migration issues:** Compare notebook cell code with package function in this guide

## Future Work

Potential enhancements to bridge notebook and package:

1. **Jupyter integration:** Add `%%tomofast` magic commands
2. **Progress bars:** Add `tqdm` support for long-running operations
3. **Interactive mode:** Add `interactive=True` flag for parameter tuning
4. **Notebook templates:** Provide example notebooks using package functions
