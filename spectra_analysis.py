"""
spectra_analysis.py
===================

High-level analysis and visualization utilities for Raman spectra workflows.

This module contains helper functions used across the Raman4React pipeline for:
    - baseline estimation and subtraction
    - constrained polynomial baseline fitting using anchor points
    - MCR/NNLS component fitting against reference pure spectra
    - exporting, loading, and visualizing results 
    
The typical workflow this module supports is:

1) Load or generate spectra matrix
   ------------------------------
   The core data are usually represented as:

   - rxn_shift : 1D array (n_points,)
       Raman shift axis (cm^-1)

   - D : 2D array (n_samples, n_points)
       Spectral intensities, one row per sample spectrum

   - sample_names : list[str] (n_samples,)
       Names/labels of each sample spectrum (e.g. ["A", "B", "C", ...])

2) Optional preprocessing
   ----------------------
   Baseline correction may be applied to spectra prior to fitting:

   - rolling_circle_baseline(y, radius=...)
       Estimates a baseline using a morphological opening approximation
       (robust for slowly varying backgrounds).

   - polynomial_baseline_constrained(x, y, anchor_points, order=...)
       Fits a polynomial baseline passing through user-defined anchor points
       (useful when you have reliable "baseline-only" regions).

3) Reference-based fitting (NNLS / MCR-style coefficients)
   -------------------------------------------------------
   Using reference spectra interpolated to the same shiftgrid:

   - perform_mcr_nnls(D, rxn_shift, TCP_i, DCP_i, GLY_i, sample_names)
       Fits each spectrum row in D as a non-negative linear combination of
       the reference spectra (TCP_i, DCP_i, GLY_i) using NNLS.
       Returns:
         * dfC : DataFrame of coefficients and fractions per sample
         * reconstructed : fitted spectra (same shape as D)
         * residuals : difference spectra (D - reconstructed)

4) Visualization and reporting
   ---------------------------
   The module provides plotting functions to summarize both coefficient outputs
   and quality of reconstruction:

   - plot_concentration_fractions(dfC, sample_names, out_dir, ...)
       Stacked bar chart of TCP/DCP/GLY fractions (normalized contributions).

   - plot_actual_intensities(dfC, sample_names, out_dir, ...)
       Stacked bar chart of raw NNLS coeffs (absolute contribution levels).

   - plot_original_vs_reconstructed(dfC, reconstructed, D, rxn_shift, TCP_i, 
                                    DCP_i, GLY_i, ...)
       Subplot grid comparing original vs reconstructed spectra, including
       component contributions per sample and optional R² values.
       Uses fixed number of rows (default 3) and fills subplots row-by-row.

Input/Output conventions
------------------------
- Most functions assume tabular data in text/CSV outputs, where spectra are 
  aligned by shared RamanShift and intensity vectors.
- Plots are written to disk as PNGs into `out_dir` and figures are closed after 
  saving (safe for batch processing and non-interactive backends such as Agg).

Notes
-----
- All plotting functions are designed to be batch-safe: they do not require
  interactive windows and will not block execution.

Dependencies
------------
Expected imports used by functions in this module:
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

Optional dependencies (only used in baseline tools):
    from scipy.ndimage import minimum_filter1d, maximum_filter1d
    from scipy.optimize import nnls

"""

import numpy as np
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.optimize import nnls
import pandas as pd
import os

# IMPORTANT: use non-interactive backend 
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt


def rolling_circle_baseline(y, radius=100):
    """
    Estimate a smooth baseline using a 1D morphological "rolling circle" 
    approach. This function approximates baseline drift by applying 
    a morphological opening operation (erosion followed by dilation) to a 1D 
    signal. 

    The returned baseline can be subtracted from the original signal, e.g.:

        baseline = rolling_circle_baseline(y, radius=150)
        y_corr = y - baseline
        y_corr[y_corr < 0] = 0

    Parameters
    ----------
    y : array-like of shape (n_points,)
        Input spectrum (intensity values). Must be a 1D numeric array.

    radius : int, optional
        Window size (in number of points) controlling baseline smoothness.
        Larger values produce a smoother baseline that follows only broad 
        trends,while smaller values may track peaks more closely. 
        Default is 100.

    Returns
    -------
    baseline : ndarray of shape (n_points,)
        Estimated baseline (morphological opening result), same length as `y`.

    """
    # Ensure radius is integer
    r = int(radius)

    # Step 1: morphological opening (erosion then dilation)
    erosion = minimum_filter1d(y, size=r, mode='nearest')
    opening = maximum_filter1d(erosion, size=r, mode='nearest')

    # baseline = opening
    return opening


def polynomial_baseline_constrained(x, y, anchor_points, order=1):
    """
    Estimate a polynomial baseline constrained to user-defined anchor points.

    This function fits a polynomial baseline to the spectrum using only
    the values at specified *anchor points* (wavenumbers) that are assumed 
    to represent the baseline (i.e., regions without Raman peaks). 

    The returned baseline can be subtracted from the original spectrum, e.g.:

       baseline = polynomial_baseline_constrained(x, y, anchor_points, order=2)
       y_corr = y - baseline
       y_corr[y_corr < 0] = 0

    Parameters
    ----------
    x : array-like of shape (n_points,)
        Raman shift / wavenumber axis (e.g., in cm^-1). Must be 1D and numeric.

    y : array-like of shape (n_points,)
        Intensity values corresponding to `x`. 
        Must be 1D and same length as `x`.

    anchor_points : list[float] or array-like
        List of wavenumbers (in the same units as `x`) where the baseline 
        should be anchored. Each anchor point is mapped to the nearest position 
        in `x`.

    order : int, optional
        Polynomial order used for baseline fitting:
        - 0: constant baseline
        - 1: linear baseline
        - 2+: higher-order curvature
        Default is 1.

    Returns
    -------
    baseline : ndarray of shape (n_points,)
        Fitted polynomial baseline evaluated across all `x`.

    """
    # Ensure anchor_points are within range
    anchor_points = [ap for ap in anchor_points if x.min() <= ap <= x.max()]
    if len(anchor_points) < order + 1:
        raise ValueError(
            f"Polynomial of order {order} requires at least {order+1} points."
        )

    # Convert anchor wavenumbers → indices
    anchor_indices = [np.argmin(np.abs(x - ap)) for ap in anchor_points]

    # Extract anchor x and y values
    x_anchor = x[anchor_indices]
    y_anchor = y[anchor_indices]

    # Fit polynomial through anchors
    coeffs = np.polyfit(x_anchor, y_anchor, order)

    # Compute baseline everywhere
    baseline = np.polyval(coeffs, x)

    return baseline


def perform_mcr_nnls(D, rxn_shift, TCP_i, DCP_i, GLY_i, sample_names):
    """
    Perform constrained (non-negative) linear unmixing of spectra using NNLS.

    This function models each measured spectrum as a non-negative linear 
    combination of provided *pure component spectra* (here: TCP, DCP, GLY) 
    using **Non-Negative Least Squares (NNLS)**:

        y ≈ a_TCP * TCP_i + a_DCP * DCP_i + a_GLY * GLY_i,   with a_* >= 0

    For each sample (row of `D`), NNLS is solved independently. The function 
    returns:
      - the fitted coefficients and derived fractions,
      - reconstructed spectra from the fitted mixture model,
      - residual spectra (original - reconstructed).

    Parameters
    ----------
    D : array-like of shape (n_samples, n_points)
        Input matrix of spectra (rows = samples, columns = Raman shift points).

    rxn_shift : array-like of shape (n_points,)
        Raman shift axis corresponding to the columns of `D`. Included for 
        clarity and downstream consistency; not used in the NNLS computation 
        itself.

    TCP_i : array-like of shape (n_points,)
        Pure component spectrum for TCP, interpolated onto `rxn_shift`.

    DCP_i : array-like of shape (n_points,)
        Pure component spectrum for DCP, interpolated onto `rxn_shift`.

    GLY_i : array-like of shape (n_points,)
        Pure component spectrum for GLY, interpolated onto `rxn_shift`.

    sample_names : list[str] of length n_samples
        Names/labels for each sample (row of `D`). Used as the index of the 
        output coefficient table.

    Returns
    -------
    df : pandas.DataFrame
        Table of NNLS coefficients and diagnostics indexed by `sample_names`, 
        with columns:
          - "TCP", "DCP", "GLY" : non-negative mixture coefficients
          - "Total"            : sum of coefficients
          - "TCP_frac", "DCP_frac", "GLY_frac" : fraction of each component 
                                                 (coef / Total)
          - "Residual_norm"    : L2 norm of residual spectrum per sample

    reconstructed : ndarray of shape (n_samples, n_points)
        Reconstructed spectra computed from NNLS coefficients and pure spectra.

    residuals : ndarray of shape (n_samples, n_points)
        Residual spectra: D - reconstructed.

    """
    n_samples, n_vars = D.shape
    
    # pure spectra matrix (columns = components)
    A = np.vstack([TCP_i, DCP_i, GLY_i]).T 
    
    # initialize data containers
    coeffs = np.zeros((n_samples, 3))
    reconstructed = np.zeros_like(D)
    residuals = np.zeros_like(D)
    residual_norms = np.zeros(n_samples)
    
    # MCR-NNLS loop
    for i in range(n_samples):
        y = D[i, :]
        x, _ = nnls(A, y)
        coeffs[i] = x
        reconstructed[i] = x @ A.T
        residuals[i] = y - reconstructed[i]
        residual_norms[i] = np.linalg.norm(residuals[i])
    
    # create dataframes
    
    df = pd.DataFrame(
        coeffs, 
        columns=["TCP", "DCP", "GLY"], 
        index=sample_names
        )
    
    df["Total"] = df.sum(axis=1)
    df["TCP_frac"] = df["TCP"] / df["Total"]
    df["DCP_frac"] = df["DCP"] / df["Total"]
    df["GLY_frac"] = df["GLY"] / df["Total"]
    df["Residual_norm"] = residual_norms

    return df, reconstructed, residuals


def plot_concentration_fractions(
        df, 
        sample_names, 
        out_dir, 
        dpi=300, 
        font_scale=1.25
        ):
    
    """
    Plot stacked component fractions (TCP/DCP/GLY) for each sample. Produces 
    a stacked bar chart showing the fractional contribution of each component 
    per sample. The function expects fraction columns ("TCP_frac", "DCP_frac",
    "GLY_frac"). If they are not present, it will compute them from the absolte 
    coefficients ("TCP", "DCP", "GLY") as:

        frac = component / (TCP + DCP + GLY)

    The sample order is forced to match `sample_names` by reindexing `df`.

    Parameters
    ----------
    df : pandas.DataFrame or array-like
        Concentration/coefficient table. Expected columns:
          - Preferred: "TCP_frac", "DCP_frac", "GLY_frac"
          - Fallback: "TCP", "DCP", "GLY" (used to compute fractions)
        Index should match `sample_names` (or will be reindexed).

    sample_names : list of str
        Sample labels in the desired plotting order.

    out_dir : str
        Output directory where "fractions.png" will be saved.

    dpi : int, optional
        Resolution of the saved PNG. Default is 300.

    font_scale : float, optional
        Multiplier applied to base font sizes for readability. Default is 1.25.

    Returns
    -------
    None
        Saves "fractions.png" to `out_dir` and closes the figure.
    """
    # Create out_dir if missing
    os.makedirs(out_dir, exist_ok=True)

    # Ensure df is DataFrame and aligned to sample_names
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = df.copy()
    df = df.reindex(sample_names)

    # If fractions are missing, compute them from TCP/DCP/GLY
    frac_cols = ["TCP_frac", "DCP_frac", "GLY_frac"]
    if not all(c in df.columns for c in frac_cols):
        needed = ["TCP", "DCP", "GLY"]
        if not all(c in df.columns for c in needed):
            raise ValueError(
                f"Missing fraction columns {frac_cols} and cannot compute them"
                f" because TCP/DCP/GLY are not all present. "
                f"Columns: {list(df.columns)}"
            )

        for c in needed:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        total = df["TCP"] + df["DCP"] + df["GLY"]
        total = total.replace(0, np.nan)

        df["TCP_frac"] = df["TCP"] / total
        df["DCP_frac"] = df["DCP"] / total
        df["GLY_frac"] = df["GLY"] / total

    # Convert fraction columns to numeric
    for c in frac_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    n = len(sample_names)
    ind = np.arange(n)

    # Font scaling
    base_font = 10 * font_scale
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font * 1.2,
        "axes.labelsize": base_font * 1.05,
        "xtick.labelsize": base_font * 0.9,
        "ytick.labelsize": base_font * 0.9,
        "legend.fontsize": base_font * 0.95,
    })

    # Figure size scales with sample count
    fig_w = max(10, 0.45 * n)
    fig_h = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    bottom = np.zeros(n, dtype=float)

    # Stacking order
    stacks = [("TCP_frac", "TCP"), ("DCP_frac", "DCP"), ("GLY_frac", "GLY")]

    for col, label in stacks:
        vals = df[col].to_numpy(dtype=float)
        ax.bar(ind, vals, bottom=bottom, label=label, width=0.85)
        bottom += vals

    ax.set_ylabel("Fraction")
    ax.set_xlabel("Samples")
    ax.set_title("Component Fractions (NNLS-MCR)")
    ax.set_ylim(0, 1.02)

    ax.set_xticks(ind)
    ax.set_xticklabels(sample_names, rotation=45, ha="right")

    ax.grid(True, axis="y", linewidth=0.7, alpha=0.25)

    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.savefig(os.path.join(out_dir, "fractions.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    

def plot_actual_intensities(
        df, 
        sample_names, 
        out_dir, 
        dpi=300, 
        font_scale=1.25
        ):
    
    """
    Plot stacked NNLS coefficients (TCP/DCP/GLY) for each sample.

    Produces a stacked bar chart of the absolute NNLS coefficients returned by
    the unmixing step. The sample order is forced to match `sample_names` by
    reindexing `df`.

    Parameters
    ----------
    df : pandas.DataFrame or array-like
        Concentration/coefficient table containing columns:
        - "TCP", "DCP", "GLY"
        Index should match `sample_names` (or will be reindexed).

    sample_names : list of str
        Sample labels in the desired plotting order.

    out_dir : str
        Output directory where "intensity_stack.png" will be saved.

    dpi : int, optional
        Resolution of the saved PNG. Default is 300.

    font_scale : float, optional
        Multiplier applied to base font sizes for readability. Default is 1.25.


    Returns
    -------
    None
        Saves "intensity_stack.png" to `out_dir` and closes the figure.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Ensure df is DataFrame and aligned to sample_names
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = df.copy()
    df = df.reindex(sample_names)

    needed = ["TCP", "DCP", "GLY"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(
                f"Missing '{c}' in df. Columns: {list(df.columns)}"
                )
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    n = len(sample_names)
    ind = np.arange(n)

    # Font scaling
    base_font = 10 * font_scale
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font * 1.2,
        "axes.labelsize": base_font * 1.05,
        "xtick.labelsize": base_font * 0.9,
        "ytick.labelsize": base_font * 0.9,
        "legend.fontsize": base_font * 0.95,
    })

    # Figure size scales with sample count
    fig_w = max(10, 0.45 * n)
    fig_h = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    bottom = np.zeros(n, dtype=float)

    stacks = [("TCP", "TCP"), ("DCP", "DCP"), ("GLY", "GLY")]
    for col, label in stacks:
        vals = df[col].to_numpy(dtype=float)
        ax.bar(ind, vals, bottom=bottom, label=label, width=0.85)
        bottom += vals

    ax.set_ylabel("NNLS coefficient (intensity)")
    ax.set_xlabel("Samples")
    ax.set_title("Component Intensities (TCP / DCP / GLY)")

    ax.set_xticks(ind)
    ax.set_xticklabels(sample_names, rotation=45, ha="right")

    ax.grid(True, axis="y", linewidth=0.7, alpha=0.25)

    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.savefig(os.path.join(
        out_dir, 
        "intensity_stack.png"
        ), 
        dpi=dpi, 
        bbox_inches="tight"
        )
    
    plt.close(fig)


def plot_original_vs_reconstructed(
    dfC, reconstructed, D, rxn_shift,
    TCP_i, DCP_i, GLY_i,
    sample_names, out_dir,
    ncols=3,                 # FIXED COLUMNS (default 3)
    dpi=300,
    show_r2=True,
    font_scale=1.35
):
    """
    Plot original vs reconstructed spectra (NNLS/MCR) with component 
    contributions. Creates a grid of subplots where each subplot corresponds 
    to one sample and shows:
      - Original spectrum (row from D)
      - Reconstructed spectrum (row from reconstructed)
      - Component contributions: coefficient * pure spectrum (TCP/DCP/GLY)

    Layout rule:
    - Uses a fixed number of columns (ncols)
    - Number of rows is computed automatically
    - Subplots are filled row-by-row (A,B,C across the first row)

    X-axis direction:
    - Raman shift is plotted from lowest -> highest (left to right)

    Saves:
    - combined_mcr_matrix.png in out_dir

    Parameters
    ----------
    dfC : pandas.DataFrame
        Coefficient table with columns ["TCP", "DCP", "GLY"] indexed by sample 
        names.
        
    reconstructed : array-like, shape (n_samples, n_points)
        Reconstructed spectra.
    
    D : array-like, shape (n_samples, n_points)
        Original spectra.
    
    rxn_shift : array-like, shape (n_points,)
        Raman shift axis.
    
    TCP_i, DCP_i, GLY_i : array-like, shape (n_points,)
        Pure component spectra interpolated to rxn_shift.
    
    sample_names : list[str]
        Sample labels corresponding to rows of D and reconstructed.
    
    out_dir : str
        Directory to save the output plot.
    
    ncols : int, optional
        Number of subplot columns (default 3).
    
    dpi : int, optional
        Output image DPI (default 300).
    
    show_r2 : bool, optional
        Show R² annotation in each subplot (default True).
    
    font_scale : float, optional
        Global font scaling multiplier (default 1.35).

    Returns
    -------
    None
    """

    os.makedirs(out_dir, exist_ok=True)

    # Global font scaling
    base_font = 10 * font_scale
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font * 1.05,
        "axes.labelsize": base_font,
        "legend.fontsize": base_font * 0.95,
        "xtick.labelsize": base_font * 0.9,
        "ytick.labelsize": base_font * 0.9,
    })

    # Convert inputs
    rxn_shift = np.asarray(rxn_shift, dtype=float)
    D = np.asarray(D, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)

    n_samples = len(sample_names)

    if D.shape[0] != n_samples:
        raise ValueError(
            f"D has {D.shape[0]} rows but sample_names has {n_samples}")
    if reconstructed.shape[0] != n_samples:
        raise ValueError(
            f"reconstructed has {reconstructed.shape[0]} rows ",
            f"but sample_names has {n_samples}")

    # Ensure shift is ascending (low->high) and reorder all arrays accordingly
    TCP_i = np.asarray(TCP_i, dtype=float)
    DCP_i = np.asarray(DCP_i, dtype=float)
    GLY_i = np.asarray(GLY_i, dtype=float)

    # Align coefficients with sample_names
    dfC = dfC.reindex(sample_names)
    TCP_coeff = dfC["TCP"].to_numpy(dtype=float)
    DCP_coeff = dfC["DCP"].to_numpy(dtype=float)
    GLY_coeff = dfC["GLY"].to_numpy(dtype=float)

    # Fixed columns, compute rows
    ncols = int(max(1, ncols))
    nrows = int(np.ceil(n_samples / ncols))

    # Figure size scaling
    fig_w = 5.4 * ncols
    fig_h = 4.0 * nrows

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(nrows, ncols)

    legend_handles = None
    legend_labels = None

    for i, name in enumerate(sample_names):
        # Row-by-row fill
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])

        # Original + reconstructed
        h1, = ax.plot(
            rxn_shift, 
            D[i], 
            label="Original", 
            linewidth=2.2
            )
        h2, = ax.plot(
            rxn_shift, 
            reconstructed[i], 
            label="Reconstructed", 
            linewidth=2.2, 
            linestyle="--"
            )

        # Component contributions
        h3, = ax.plot(
            rxn_shift, 
            TCP_coeff[i] * TCP_i, 
            label="TCP", 
            linewidth=1.5, 
            alpha=0.9
            )
        h4, = ax.plot(
            rxn_shift, 
            DCP_coeff[i] * DCP_i, 
            label="DCP", 
            linewidth=1.5, 
            alpha=0.9
            )
        h5, = ax.plot(
            rxn_shift, 
            GLY_coeff[i] * GLY_i, 
            label="GLY", 
            linewidth=1.5, 
            alpha=0.9
            )

        if legend_handles is None:
            legend_handles = [h1, h2, h3, h4, h5]
            legend_labels = [h.get_label() for h in legend_handles]

        ax.set_title(str(name), pad=8)
        ax.grid(True, linewidth=0.7, alpha=0.25)

        # Auto y-limits with padding
        y_all = np.concatenate([
            D[i],
            reconstructed[i],
            TCP_coeff[i] * TCP_i,
            DCP_coeff[i] * DCP_i,
            GLY_coeff[i] * GLY_i,
        ])
        
        ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = 0.07 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        # R² label
        if show_r2:
            y_true = D[i]
            y_pred = reconstructed[i]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            ax.text(
                0.02, 0.96,
                f"$R^2$ = {r2:.3f}" if np.isfinite(r2) else "$R^2$ = NA",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=base_font * 0.9,
                bbox=dict(
                    boxstyle="round,pad=0.30", 
                    alpha=0.18, 
                    linewidth=0.8
                    ),
            )

        # Axis labels on outer plots only
        if r == nrows - 1:
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
        else:
            ax.set_xticklabels([])

        if c == 0:
            ax.set_ylabel("Intensity (a.u.)")
        else:
            ax.set_yticklabels([])

    # Turn off unused axes
    for j in range(n_samples, nrows * ncols):
        r = j // ncols
        c = j % ncols
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    # Shared legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0.1,
    )

    fig.suptitle(
        "Original vs Reconstructed Raman Spectra with Component Contributions",
        fontsize=base_font * 1.35,
        y=1.045,
    )

    out_path = os.path.join(out_dir, "combined_mcr_matrix.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

