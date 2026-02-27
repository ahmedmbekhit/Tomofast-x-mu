"""Parfile generation for Tomofast inversions."""

import os

import numpy as np

# Template for per-blob 3D inversion parfiles
_INVERSION_PAR_TEMPLATE = """==================================================================================
GLOBAL
==================================================================================
global.outputFolderPath         = ./output/QDM/Blob_{blob_id:03d}
global.description              = QDM inversion – Blob {blob_id}

==================================================================================
MODEL GRID parameters
==================================================================================
# nx ny nz
modelGrid.size                        = {nx} {ny} {nz}
modelGrid.magn.file                   = data/QDM/Meshgrid_Blob_{blob_id:03d}.txt
modelGrid.magn.nModelComponents       = 3

==================================================================================
DATA parameters
==================================================================================
forward.data.magn.nData                = {ndata}
forward.data.magn.dataGridFile         = data/QDM/Blob_{blob_id:03d}_RECT.obs
forward.data.magn.dataValuesFile       = data/QDM/Blob_{blob_id:03d}_RECT.obs
forward.data.magn.nDataComponents      = 1

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

==================================================================================
SENSITIVITY KERNEL
==================================================================================
sensit.readFromFiles                  = 0
sensit.folderPath                     = ./output/QDM/SENSIT/

==================================================================================
MATRIX COMPRESSION
==================================================================================
forward.matrixCompression.type                        = 1
forward.matrixCompression.rate                        = {compression_rate}

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

# Template for source detection parfiles (single-cell mesh)
_SOURCE_PAR_TEMPLATE = """==================================================================================
GLOBAL
==================================================================================
global.outputFolderPath         = ./output/QDM/Blob_{blob_id:03d}
global.description              = QDM inversion – Blob {blob_id}

==================================================================================
MODEL GRID parameters
==================================================================================
# nx ny nz
modelGrid.size                        = 1 1 1
modelGrid.magn.file                   = data/QDM/meshgrid.txt
modelGrid.magn.nModelComponents       = 3

==================================================================================
DATA parameters
==================================================================================
forward.data.magn.nData                = {ndata}
forward.data.magn.dataGridFile         = data/QDM/Blob_{blob_id:03d}_RECT.obs
forward.data.magn.dataValuesFile       = data/QDM/Blob_{blob_id:03d}_RECT.obs
forward.data.magn.nDataComponents      = 1

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

==================================================================================
SENSITIVITY KERNEL
==================================================================================
sensit.readFromFiles                  = 0
sensit.folderPath                     = ./output/QDM/SENSIT/

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

# Template for combined dipole fitting parfile
_FITTING_PAR_TEMPLATE = """==================================================================================
GLOBAL
==================================================================================
global.outputFolderPath         = ./output/QDM
global.description              = QDM inversion data

==================================================================================
MODEL GRID parameters
==================================================================================
# nx ny nz
modelGrid.size                        = {nx} 1 1
modelGrid.magn.file                   = ./data/QDM/meshgrid.txt
modelGrid.magn.nModelComponents       = 3
==================================================================================
DATA parameters
==================================================================================
forward.data.magn.nData                = {ndata}
forward.data.magn.dataGridFile         = ./data/QDM/Mag_Inversion.obs
forward.data.magn.dataValuesFile       = ./data/QDM/Mag_Inversion.obs
forward.data.magn.nDataComponents      = 1

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

==================================================================================
SENSITIVITY KERNEL
==================================================================================
# 0-calculate the sensitivity, 1-read from files at the specified folder.
sensit.readFromFiles                  = 0
sensit.folderPath                     = ./output/QDM/SENSIT/

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
"""


def create_inversion_parfiles(summary_path, par_out_dir, compression_rate=0.01):
    """Create one parfile per blob for 3D magnetic inversion.

    Parameters
    ----------
    summary_path : str
        Path to blob_window_summary.txt (columns: blob_id nx ny nz ndata).
    par_out_dir : str
        Output directory for parfiles.
    compression_rate : float
        Matrix compression rate.

    Returns
    -------
    list of str
        Paths to created parfiles.
    """
    os.makedirs(par_out_dir, exist_ok=True)
    summary = np.loadtxt(summary_path, comments="#", ndmin=2)

    paths = []
    for row in summary:
        blob_id = int(row[0])
        nx = int(row[1])
        ny = int(row[2])
        nz = int(row[3])
        ndata = int(row[4])

        par_text = _INVERSION_PAR_TEMPLATE.format(
            blob_id=blob_id,
            nx=nx,
            ny=ny,
            nz=nz,
            ndata=ndata,
            compression_rate=compression_rate,
        )

        out_path = os.path.join(
            par_out_dir, f"Parfile_QDM_Inversion_Blob_{blob_id:03d}.txt"
        )
        with open(out_path, "w") as f:
            f.write(par_text)

        paths.append(out_path)

    return paths


def create_source_parfiles(summary_path, par_out_dir):
    """Create parfiles for single-cell source detection inversions.

    Parameters
    ----------
    summary_path : str
        Path to blob_window_summary.txt.
    par_out_dir : str
        Output directory for parfiles.

    Returns
    -------
    list of str
        Paths to created parfiles.
    """
    os.makedirs(par_out_dir, exist_ok=True)
    summary = np.loadtxt(summary_path, comments="#", ndmin=2)

    paths = []
    for row in summary:
        blob_id = int(row[0])
        ndata = int(row[4])

        par_text = _SOURCE_PAR_TEMPLATE.format(
            blob_id=blob_id,
            ndata=ndata,
        )

        out_path = os.path.join(
            par_out_dir, f"Parfile_QDM_Inversion_Blob_{blob_id:03d}.txt"
        )
        with open(out_path, "w") as f:
            f.write(par_text)

        paths.append(out_path)

    return paths


def create_fitting_parfile(output_path, nx, ndata):
    """Create a parfile for combined dipole fitting.

    Parameters
    ----------
    output_path : str
        Path to write the parfile.
    nx : int
        Number of model cells (grains).
    ndata : int
        Number of data points.

    Returns
    -------
    str
        Path to the created parfile.
    """
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    par_text = _FITTING_PAR_TEMPLATE.format(nx=nx, ndata=ndata)

    with open(output_path, "w") as f:
        f.write(par_text)

    return output_path
