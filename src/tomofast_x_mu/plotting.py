"""Plotting utilities for QDM microscopy data and Tomofast results."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from tomofast_x_mu.io import load_field_table


def plot_bz_led(mat_path, pix=4.4, ax=None):
    """Plot Bz with LED overlay.

    Parameters
    ----------
    mat_path : str or Path
        Path to the .mat file containing Bz and newLED variables.
    pix : float
        Pixel size in um.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.  If None, creates a new figure.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    mat = loadmat(str(mat_path))

    Bz = np.squeeze(mat["Bz"]) * 1e9
    led = np.squeeze(mat["newLED"])

    ny, nx = Bz.shape
    # Use arange for pixel-center coordinates (not linspace which adds extra spacing)
    x = np.arange(nx) * pix
    y = np.arange(ny) * pix

    bz_min = np.percentile(Bz, 2)
    bz_max = np.percentile(Bz, 98)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    else:
        fig = ax.get_figure()

    im = ax.imshow(
        Bz,
        cmap="seismic",
        vmin=bz_min,
        vmax=bz_max,
        origin="upper",
        extent=[x.min(), x.max(), y.min(), y.max()],
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Bz", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.imshow(
        led,
        cmap="gray",
        alpha=0.7,
        origin="upper",
        extent=[x.min(), x.max(), y.min(), y.max()],
        vmin=np.percentile(led, 5),
        vmax=np.percentile(led, 95),
    )

    bar_len = 50
    x0 = x.min() + 0.05 * (x.max() - x.min())
    y0 = y.min() + 0.05 * (y.max() - y.min())
    ax.plot([x0, x0 + bar_len], [y0, y0], color="k", linewidth=3, solid_capstyle="butt")
    ax.text(
        x0 + bar_len / 2,
        y0 + 0.02 * (y.max() - y.min()),
        f"{int(bar_len)} um",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.set_xlabel("X (um)", fontsize=12)
    ax.set_ylabel("Y (um)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

    return fig, ax


def plot_misfits(root_out, ax=None):
    """Plot 3-panel obs/calc/misfit per blob.

    Parameters
    ----------
    root_out : str
        Root folder containing Blob_XXX subfolders with Tomofast outputs.
    ax : None
        Unused; provided for API consistency.

    Returns
    -------
    dict
        Mapping blob_name -> (fig, axes) tuple.
    """

    def _clean_df(df):
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["x", "y", "value"])
        df = df[np.isfinite(df["x"]) & np.isfinite(df["y"]) & np.isfinite(df["value"])]
        return df

    def _df_to_grid(df):
        xs = np.sort(df["x"].unique())
        ys = np.sort(df["y"].unique())
        if xs.size == 0 or ys.size == 0:
            return None, None, None
        grid = df.pivot_table(index="y", columns="x", values="value")
        grid = grid.reindex(index=ys, columns=xs)
        return grid.values, xs, ys

    blob_dirs = sorted(d for d in os.listdir(root_out) if d.startswith("Blob_"))
    figures = {}

    for blob in blob_dirs:
        data_dir = os.path.join(root_out, blob, blob, "data")

        obs_file = os.path.join(data_dir, "mag_observed_data.txt")
        calc_file = os.path.join(data_dir, "mag_calc_final_data.txt")
        misfit_file = os.path.join(data_dir, "mag_misfit_final_data.txt")

        if not all(os.path.exists(f) for f in [obs_file, calc_file, misfit_file]):
            continue

        obs_df = _clean_df(load_field_table(obs_file))
        calc_df = _clean_df(load_field_table(calc_file))
        misfit_df = _clean_df(load_field_table(misfit_file))

        if obs_df.empty or calc_df.empty or misfit_df.empty:
            continue

        obs_grid, xs, ys = _df_to_grid(obs_df)
        calc_grid, _, _ = _df_to_grid(calc_df)
        misfit_grid, _, _ = _df_to_grid(misfit_df)

        if obs_grid is None or calc_grid is None or misfit_grid is None:
            continue

        extent = [xs.min(), xs.max(), ys.min(), ys.max()]
        if not np.all(np.isfinite(extent)):
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

        for ax_i, grid, cbar_label in zip(
            axes,
            [obs_grid, calc_grid, misfit_grid],
            ["Observed", "Calculated", "Misfit (Obs - Calc)"],
        ):
            im = ax_i.imshow(
                grid, origin="lower", cmap="jet", extent=extent, aspect="equal"
            )
            ax_i.set_xlabel("X (um)")
            ax_i.set_ylabel("Y (um)")
            ax_i.margins(0)
            cbar = plt.colorbar(im, ax=ax_i, shrink=0.85)
            cbar.set_label(f"{cbar_label}  Bz (nT)")

        plt.tight_layout(pad=0.2)
        figures[blob] = (fig, axes)

    return figures


def plot_stereonet(results, color_by_moment=False, ax=None):
    """Plot declination/inclination on a stereonet.

    Parameters
    ----------
    results : list of tuple
        Each element is (grain_idx, dec, inc) or (grain_idx, dec, inc, moment)
        if color_by_moment is True.
    color_by_moment : bool
        If True, color points by moment magnitude.
    ax : matplotlib.axes.Axes or None
        Not used directly (pmagpy creates its own axes), but provided for
        API consistency.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import pmagpy.ipmag as ipmag
    except ImportError:
        raise ImportError("pmagpy is required for stereonet plots: pip install pmagpy")

    fig = plt.figure(figsize=(9, 9))
    ipmag.plot_net()
    ax = plt.gca()

    if color_by_moment:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        moments = np.array([r[3] for r in results], dtype=float)
        good = np.isfinite(moments)
        vmin = np.nanmin(moments[good])
        vmax = np.nanmax(moments[good])
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.jet

        for grain_idx, dec, inc, moment in results:
            if not (np.isfinite(dec) and np.isfinite(inc) and np.isfinite(moment)):
                continue
            col = cmap(norm(moment))
            ipmag.plot_di(dec, inc, color=col, marker="o", markersize=200)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.05)
        cbar.set_label("Magnetic moment (A.m2)", fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.get_offset_text().set_fontsize(16)

        for grain_idx, dec, inc, moment in results:
            plt.plot(
                [],
                [],
                color="k",
                linestyle="None",
                label=f"Dipole {grain_idx}: D={dec:.1f}, I={inc:.1f}, m={moment:.2e} A.m2",
            )
    else:
        for grain_idx, dec, inc in results:
            ipmag.plot_di(dec, inc, color="r", marker="o", markersize=200)
            plt.plot(
                [],
                [],
                color="r",
                linestyle="None",
                label=f"Dipole {grain_idx}: D={dec:.1f}, I={inc:.1f}",
            )

    plt.legend(loc="upper right", handlelength=0, handletextpad=0, fontsize=12, frameon=True)
    return fig


def plot_results_on_bz(mag_path, grain_path, ax=None):
    """Overlay detected grain locations on Bz map.

    Parameters
    ----------
    mag_path : str
        Path to Mag_Inversion.obs file.
    grain_path : str
        Path to Blob_Vector_Summary.csv.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.  If None, creates a new figure.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    mag = np.loadtxt(mag_path, skiprows=1)
    x = mag[:, 0]
    y = mag[:, 1]
    bz = -1 * mag[:, 3]  # back to QDM convention

    grains = pd.read_csv(grain_path, sep=None, engine="python")
    gx = grains["x_center"].values
    gy = grains["y_center"].values

    # Use 2nd and 98th percentiles for better contrast (not min/max)
    bz_min = np.percentile(bz, 2)
    bz_max = np.percentile(bz, 98)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    else:
        fig = ax.get_figure()

    sc = ax.scatter(
        x, y, c=bz, s=10, cmap="seismic", vmin=bz_min, vmax=bz_max, rasterized=True
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Bz (nT)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.scatter(
        gx,
        gy,
        c="yellow",
        s=55,
        edgecolors="white",
        linewidths=0.7,
        label="Detected Dipole centres",
        zorder=3,
    )

    ax.set_xlabel("X (um)", fontsize=12)
    ax.set_ylabel("Y (um)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.margins(0)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    return fig, ax
