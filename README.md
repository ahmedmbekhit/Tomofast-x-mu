# Tomofast-x-µ: 3-D Magnetization Vector Inversion of Micro-Magnetic Quantum Imaging

This repository contains the code and data of:

1. Preprint: Bekhit, A. M., Ogarko, V., Kirscher, U., Liu, Y., Jessell, M., Li, Z.-X., & Evans, K. (2026). *Tomofast-x-µ: 3-D magnetization reconstruction of micromagnetic imaging* (Version 1) [Preprint]. https://doi.org/10.22541/essoar.15001802/v1
2. Zenodo archive: Bekhit, A., Ogarko, V., Kirscher, U., Liu, Y., Jessell, M., Li, Z.-X., & Evans, K. (2026). *Supplementary of "Tomofast-x-µ: 3-D Magnetization Reconstruction of Micromagnetic Imaging"* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19603350


<img width="734" height="292" alt="GITHUB_WORKFLOW" src="https://github.com/user-attachments/assets/18489dde-10cb-4f13-adca-b467d3da6521" />


## Description

Tomofast-x-µ is an open-source Python interface built on the [Tomofast-x](https://github.com/TOMOFAST/Tomofast-x) inversion code (Giraud et al., 2021; Ogarko et al., 2024). It turns a quantum diamond microscope (QDM) image into the magnetization direction (declination and inclination) and moment of the individual magnetic sources. The workflow has six steps, grouped into three independent tools:

| Step | What it does | Tool |
|------|--------------|------|
| 1. Data acquisition | Read the QDM remanent field (Bx, By, Bz) or (Bz only) and convert it to Tomofast-x input | Data preparation |
| 2. Anomaly detection and windowing | Threshold T = α·σ(Bz) applied to Bz; the anomalies are cut into spatial windows (blobs) | Data preparation |
| 3. 3-D vector magnetization inversion | Multi-cell inversion of Bx, By, Bz or Bz only for each blob (Tomofast-x) | Tool 1 |
| 4. Extracting magnetic volumes | Threshold τ on the model gives the grains (m^f, declination, inclination) | Tool 1 |
| 5. Locating equivalent dipole centres | Single-cell steepest-descent search for the dipole position (r_e) | Tool 2 |
| 6. Global multi-dipole inversion | Joint N-cell inversion for the moment, declination and inclination of all dipoles (m^p) | Tool 3 |

Step 5 can also start from the maximum-magnetization cell of step 4 instead of the anomaly centres (`start_mode="maxs"`).

1. **Tool 1, 3-D micromagnetic modelling (steps 3-4):** a full 3-D magnetic vector inversion of each window, then grain extraction.
2. **Tool 2, magnetic source location (step 5):** fast single-cell inversions inside the grain volume find the true location of each source.
3. **Tool 3, paleomagnetic information (step 6):** a joint multi-dipole fit at the detected locations recovers the magnetization vectors, from which the declination, inclination and intensity are calculated. Results can be compared with bulk (e.g. SQUID) measurements.

## Installation

1. Install Tomofast-x: [Tomofast-x install](https://github.com/TOMOFAST/Tomofast-manual/blob/main/install.md). The user manual is here: [Tomofast-x manual](https://github.com/TOMOFAST/Tomofast-manual).
2. Create the Python environment from `environment.yml` in this repository:
```
conda env create -f environment.yml
conda activate Tomofast-x-mu
jupyter lab
```
3. Windows users run Tomofast-x through WSL. Linux users run it natively.

## Dependencies
Data handling and numerical calculations used NumPy (Harris et al., 2020), SciPy (Virtanen et al., 2020) and pandas (McKinney, 2011). Grid-based data and input files were managed with xarray (Hoyer and Hamman, 2017) and pathlib. Figures used matplotlib (Hunter, 2007), and Paraview for 3-D views (Ahrens et al., 2005). Paleomagnetic data analysis used PmagPy (ipmag) (Tauxe et al., 2016). File handling and system operations used the standard Python libraries os, sys, glob, subprocess, re and math. Automatic anomaly detection and 3-D grain extraction used scipy.ndimage (labeling, dilation, structure generation).

## Simple instructions
1. Put `tomofastx_mu.py` and `Tomofast-x-mu_v1.1.ipynb` in the same folder, together with your `Bz.mat` (or `Bz.csv`) file.
2. Open the notebook and set the first cell:
   - `TOMOFAST_HOME`: where Tomofast-x is installed (and `TOMOFAST_HOME_WSL` for WSL).
   - `USE_WSL`: `True` on Windows with WSL, `False` on Linux.
   - `NPROC`: number of MPI processes.
   - `d`: cell size in µm.
   - `PRINT_LIMIT` and `PLOT_LIMIT`: how many lines and figures to show per step, to keep the output short (`0` shows none, `None` shows everything; files are always saved).
3. Run the cells from top to bottom. The notebook has short calls, and the functions are in `tomofastx_mu.py`. Cell headers (`### … ###`) match between the notebook and the `.py` so you can find the code behind each step.

The code can also be used as a Python interface to simplify running Tomofast-x.

## Repository structure

```
Tomofast-x-mu/
├── README.md
├── environment.yml
├── Tomofast-x-mu_v1.1.ipynb     # current workflow notebook (short calls)
├── tomofastx_mu.py              # all functions, same order and headers as the notebook
├── Code_Versions/               # all earlier versions of the code
├── Natural_Applications/        # natural sample application
└── Synthetic_Applications/      # synthetic data application
```

## Authors and contacts

Ahmed Mohamed Bekhit, Vitaliy Ogarko, Uwe Kirscher, Yebo Liu, Mark Jessell, Zheng-Xiang Li, Katy Evans

All questions are welcome: Ahmed Bekhit, a.hussein4@postgrad.curtin.edu.au or ahmed.m.bekhit@alexu.edu.eg
