# Tomofast-x-µ: 3-D Magnetization Reconstruction of Micro-Magnetic Imaging

A. M. Bekhit, V. Ogarko, M. W. Jessell, Y. Liu, K. Evans, Z. Li, U. Kirscher

An open-source tool for full 3-D magnetic vector inversion of quantum diamond microscope (QDM) scans, built on the [Tomofast-x](https://github.com/TOMOFAST/Tomofast-x) inversion code (Giraud et al., 2021; Ogarko et al., 2024).

---

## Three-Stage Pipeline

1. **3-D Micromagnetic Modelling** — Full 3-D magnetic vector inversion to recover a volumetric image of the QDM scan.
2. **Magnetic Source Location Detection** — Fast micro-scale inversions within the grain-discretized 3-D volume to find true source locations.
3. **Paleomagnetic Information Delineation** — Fit micro-mag data at detected source locations to recover magnetization vectors (declination, inclination, intensity).

---

## Two Ways to Use This Software

### 1. Jupyter Notebooks (Reproduce the Paper)

The notebooks are the canonical way to reproduce the results from the manuscript. They are self-contained and require no package installation beyond the conda environment.

```bash
conda env create -f environment.yml
conda activate Tomofast-x-mu
jupyter lab
```

Then open:
- `Tomofast-x-µ(v1).ipynb` — main workflow notebook
- `Natural_Applications/` — real QDM data examples
- `Synthetic_Applications/` — synthetic data examples

You must also install Tomofast-x separately:
- [Installation guide](https://github.com/TOMOFAST/Tomofast-manual/blob/main/install.md)
- [Full manual](https://github.com/TOMOFAST/Tomofast-manual)

### 2. Python Package (Easier Programmatic Use)

The `tomofast_x_mu` package provides the same pipeline as a reusable Python API — useful if you want to process multiple datasets, integrate into your own scripts, or run from the command line.

**Installation** (from repo root, inside the conda environment):

```bash
pip install -e .
```

**Smoke test** (no Tomofast-x required):

```bash
tomofast_x_mu smoke --h-units auto
```

**Minimal example:**

```python
import tomofast_x_mu as tfmu

# Load QDM .mat file
ds = tfmu.load_qdm_mat("scan.mat", h_units="auto")

# Convert to Tomofast observation format
tfmu.format_tomofast_obs("scan.csv", "Mag_Inversion.obs", flip_bz=True)

# Detect magnetic anomalies
blobs = tfmu.detect_anomalies("Mag_Inversion.obs", threshold_factor=2.75)

# Run inversions (requires Tomofast-x)
outputs = tfmu.run_tomofast_blobs(
    base_dir=".",
    tomofast_home="/path/to/Tomofast-x",
    nproc=7
)
```

See `docs/DOCUMENTATION.md` for the full API reference and `QUICKSTART.md` to get started in 5 minutes.

---

## Important: Units Convention

QDM `.mat` files commonly store mixed units:

| Field | Units | Example |
|---|---|---|
| `step` (pixel size) | meters | `4.4e-06` → 4.4 µm |
| `h` (sensor–sample distance) | **micrometers** | `5.0` → 5 µm |

Always use `h_units="auto"` when loading real data — this detects the convention automatically. Using `h_units="m"` on typical QDM data places the observation plane 10⁶× too far away, making inversion results meaningless.

```python
ds = tfmu.load_qdm_mat("scan.mat", h_units="auto")  # correct
```

---

## Package Modules

| Module | Purpose |
|---|---|
| `io.py` | Load QDM `.mat` files, Tomofast outputs, mesh bounds |
| `format.py` | Convert data to Tomofast `.obs` and meshgrid formats |
| `detect.py` | Anomaly detection, grain extraction, max-cell identification |
| `mesh.py` | Mesh generation (uniform core + geometric padding, single-cell grids) |
| `parfile.py` | Generate Tomofast parameter files for each stage |
| `runner.py` | Execute Tomofast via subprocess/MPI with timing and progress reporting |
| `analysis.py` | Post-inversion analysis (grain statistics, data costs, magnetization vectors) |
| `plotting.py` | Visualization (Bz/LED overlays, misfits, stereonets, result maps) |
| `cli.py` | Command-line interface (`smoke`, `version`, `diagram`) |

---

## Dependencies

All dependencies are managed via the conda environment:

```bash
conda env create -f environment.yml
conda activate Tomofast-x-mu
```

Core: NumPy, SciPy, pandas, xarray, matplotlib, PmagPy.
See `environment.yml` for the full specification.

---

## Documentation

- `QUICKSTART.md` — Get started in 5 minutes
- `docs/DOCUMENTATION.md` — Full API reference with diagrams
- `docs/NOTEBOOK_MIGRATION.md` — Relationship between notebooks and package
- `CHANGELOG.md` — Version history

---

## Citation

If you use this software, please cite:

- V. Ogarko et al. (2024), "Tomofast-x 2.0: an open-source parallel code for inversion of potential field data with topography using wavelet compression", *Geosci. Model Dev.*, 17, 2325–2345, https://doi.org/10.5194/gmd-17-2325-2024

- J. Giraud et al. (2021), "Structural, petrophysical, and geological constraints in potential field inversion using the Tomofast-x v1.0 open-source code", *Geosci. Model Dev.*, 14, 6681–6709, https://doi.org/10.5194/gmd-14-6681-2021

---

## Authors and Contact

Ahmed Bekhit, Vitaliy Ogarko, Mark Jessell, Yebo Liu, Katy Evans, Zheng-Xiang Li, Uwe Kirscher

Questions welcome — Ahmed Bekhit: a.hussein4@postgrad.curtin.edu.au or ahmed.m.bekhit@alexu.edu.eg
