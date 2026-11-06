# =============================================================================
# QDM Noise Generation Utilities
# =============================================================================
# Extracted from:
#   micromag.py - Functions for processing and inversion of microscopy data.
#   Authors  : Gelson Ferreira de Souza-Junior, Leonardo Uieda
#   License  : MIT License
#   Copyright: (c) 2023 Gelson Ferreira de Souza-Junior, Leonardo Uieda
#   Source   : https://github.com/compgeolab/micromag
#
# Only the two functions required for QDM-realistic noise generation are
# included here. All other micromag functionality is intentionally omitted.
# If you use this noise model in published work, please cite the original
# micromag package and its authors.
# =============================================================================

import numpy as np
from scipy.fftpack import fftshift, ifft


def _generate_fourier_intensity(x, y, fit4a, fit4b):
    """
    Compute Fourier intensity based on a 2D power-law fit to real QDM spectra.

    Extracted from micromag.py (Souza-Junior & Uieda, 2023, MIT License).

    Parameters
    ----------
    x : float
        Fourier column index (spatial frequency in x).
    y : float
        Fourier row index (spatial frequency in y).
    fit4a : list of 3 floats
        Coefficients [a0, a1, a2] for the exponential part:
        transecta = exp(a0 * x^a1 + a2)
    fit4b : list of 3 floats
        Coefficients [b0, b1, b2] for the power-law modulation:
        transectb = b0 * x^b1 + b2

    Returns
    -------
    intensity : float
        Fourier power at frequency (x, y):  transecta * y^transectb
    """
    transecta = np.exp(fit4a[0] * (x ** fit4a[1]) + fit4a[2], dtype=np.float64)
    transectb = fit4b[0] * (x ** fit4b[1]) + fit4b[2]
    return transecta * (y ** transectb)


def qdm_noise(shape, std_dev=50, seed=None):
    """
    Generate a synthetic QDM-realistic noise map via spectral synthesis.

    Extracted from micromag.py (Souza-Junior & Uieda, 2023, MIT License).

    The noise is spatially correlated and spectrally coloured: its Fourier
    power spectrum follows a 2D power law whose coefficients were fitted to
    real QDM instrument measurements. This is more realistic than white
    Gaussian noise for training data that will later be compared against
    actual QDM observations.

    Algorithm
    ---------
    1. For every Fourier frequency (i, j), compute a frequency-dependent
       standard deviation sigma(i,j) from the fitted power-law model.
    2. Draw independent Gaussian samples N(0, sigma) for the real and
       imaginary parts of the Fourier coefficients.
    3. Enforce Hermitian symmetry (so the IFFT is real-valued).
    4. Apply an inverse FFT to obtain a spatially-correlated noise field.
    5. Crop to the requested shape and rescale so the peak equals 3*std_dev.

    Parameters
    ----------
    shape : tuple of int
        Output shape as (rows, cols) - should match your Bz grid shape.
    std_dev : float, optional
        Noise scale in nT. The output peak amplitude is 3 * std_dev.
        Default is 50 nT.
    seed : int or None, optional
        Random seed for reproducibility. Default is None (random).

    Returns
    -------
    noise : 2D numpy array, shape = (rows, cols)
        Spatially-correlated noise field in nT, ready to add to Bz_nT.

    Usage
    -----
        noise = qdm_noise(shape=Bz_nT.shape, std_dev=50, seed=42)
        Bz_nT += noise

    Citation
    --------
        Souza-Junior, G. F. & Uieda, L. (2023). micromag [Software].
        MIT License. https://github.com/compgeolab/micromag
    """
    rng = np.random.default_rng(seed)

    realmap = np.zeros((shape[0] * 2, shape[1] * 2))
    outdimensions = (shape[0] + 1, shape[1] + 1)

    # Best-fit spectral power parameters (fitted to real QDM measurements)
    cvalsa = [4.5666, -0.8939, -21.2259]
    cvalsb = [-1.5723, -0.5414, -0.0208]

    # ── Build real part of the half Fourier map ───────────────────────────────
    realmapout1 = np.zeros((realmap.shape[0] // 2, realmap.shape[1] // 2))
    realmapout2 = np.zeros((realmap.shape[0] // 2, realmap.shape[1] // 2))

    end_j = realmapout1.shape[1] - 1
    for i in range(realmapout1.shape[0]):
        for j in range(realmapout1.shape[1]):
            if i == 0 or j == 0:
                continue
            sigma = np.sqrt(
                _generate_fourier_intensity(j, i, cvalsa, cvalsb), dtype=np.float64
            )
            realmapout1[i, end_j - j + 1] = rng.normal(0, sigma)
            realmapout2[i, j]             = rng.normal(0, sigma)

    realmapout = np.hstack((realmapout1, realmapout2))

    # ── Build imaginary part of the half Fourier map ──────────────────────────
    imagmapout1 = np.zeros_like(realmapout1)
    imagmapout2 = np.zeros_like(realmapout2)

    for i in range(imagmapout1.shape[0]):
        for j in range(imagmapout1.shape[1]):
            if i == 0 or j == 0:
                continue
            sigma = np.sqrt(
                _generate_fourier_intensity(j, i, cvalsa, cvalsb), dtype=np.float64
            )
            imagmapout1[i, end_j - j + 1] = rng.normal(0, sigma)
            imagmapout2[i, j]             = rng.normal(0, sigma)

    imagmapout = np.hstack((imagmapout1, imagmapout2))

    # ── Construct complex Fourier map and enforce Hermitian symmetry ──────────
    halfmapout = realmapout + 1j * imagmapout
    fullmapout = np.zeros(
        (halfmapout.shape[0] * 2, halfmapout.shape[1]), dtype=complex
    )
    fullmapout[-halfmapout.shape[0]:, :] = halfmapout
    fullmapout[:-halfmapout.shape[0], :] = np.rot90(halfmapout, 2)

    # ── Inverse FFT → spatial noise field ────────────────────────────────────
    temp2 = ifft(fftshift(np.sqrt(fullmapout)))

    # ── Crop to requested shape ───────────────────────────────────────────────
    rows = [
        round(temp2.shape[0] / 2 - outdimensions[0] / 2),
        round(temp2.shape[0] / 2 - outdimensions[0] / 2 + outdimensions[0] - 1),
    ]
    cols = [
        round(temp2.shape[1] / 2 - outdimensions[1] / 2),
        round(temp2.shape[1] / 2 - outdimensions[1] / 2 + outdimensions[1] - 1),
    ]

    outBz = np.real(temp2[rows[0]:rows[1], cols[0]:cols[1]])

    # ── Rescale: peak = 3 * std_dev ───────────────────────────────────────────
    return (outBz / outBz.max()) * 3 * std_dev
