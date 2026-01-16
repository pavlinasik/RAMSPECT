# -*- coding: utf-8 -*-
"""
Raman4React_refactored.py

Key ideas:
- __init__ stores configuration and validates inputs (fast, side-effect free).
- .run() executes the pipeline (preprocessing OR reference/MCR branch).
- Steps are split into small, testable methods.
- Figure saving avoids requiring an interactive backend (%matplotlib qt):
  we save using the current figure handle and immediately close it.

This file assumes your existing helper modules still exist:
- spectra_processing as r4rProcc
- spectra_analysis as r4rAnal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

import glob
import os

import numpy as np
import pandas as pd

# IMPORTANT: use non-interactive backend 
import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt

import spectra_processing as r4rProcc
import spectra_analysis as r4rAnal


# -----------------------------------------------------------------------------
# Configuration containers :: keep all preprocessing related inputs in one
# structured object, so the main Reactions class doesn't store 20 separate
# attributes.
# -----------------------------------------------------------------------------

@dataclass
class PrepSettings:
    """ Configuration container for the preprocessing branch (preps=True). """
    
    spectral_data_path: str
    blank_data_path: str

    c_lower: int = 23
    c_upper: int = 571
    s_lower: int = 200
    s_upper: int = 210
    range1: Tuple[int, int] = (270, 280)
    range2: Tuple[int, int] = (200, 210)

    show: bool = False
    save: bool = True
    export: bool = True

    specs: bool = True
    ratio: bool = True
    averages: bool = True
    
    
@dataclass(frozen=True)
class IOConfig:
    """ Immutable configuration container for the handling of directories. """
    export_path: str = "."
    output_folder: str = "output"
    save: bool = True
    show: bool = False
    export: bool = True
    messages: bool = True

    def out_dir(self) -> str:
        return os.path.join(self.export_path, self.output_folder)


@dataclass(frozen=True)
class CropConfig:
    """ Immutable configuration container for spectra cropping. """
    
    c_lower: int = 23
    c_upper: int = 571


@dataclass(frozen=True)
class SpecificsConfig:
    """ Immutable configuration container for specific operations during 
        preprocessing. """

    specs: bool = True
    ratio: bool = True
    averages: bool = True
    s_lower: int = 200
    s_upper: int = 210
    range1: Tuple[int, int] = (270, 280)
    range2: Tuple[int, int] = (200, 210)


@dataclass(frozen=True)
class PreprocessConfig:
    """ Immutable configuration container for the blank subtraction process."""

    enabled: bool = True
    spectral_data_path: str = ""
    blank_data_path: str = ""
    wlength: int = 7
    porder: int = 1


@dataclass(frozen=True)
class ReferenceMCRConfig:
    """ Immutable configuration container for MCR analysis. """

    enabled: bool = False
    reference_path: Optional[str] = None
    
    # folder with reaction *.txt files to analyze
    preps_path: Optional[str] = None  
    delimiter: str = "\t"

    apply_baseline_correction: bool = True
    apply_normalization: bool = False
    use_polynomial_baseline: bool = True
    poly_order: int = 4
    anchor_points: Tuple[float, ...] = (
        2700, 2710, 2720, 2730, 2740, 2750, 2760, 2770, 2780, 2790, 2800,
        3050, 3060, 3070, 3080, 3090, 3100
        

    )
    
    rfc_radius: int = 150
    
    # Which files to process inside preps_path
    # - "all": all *.txt
    # - "average_only": only files with 'average' in filename
    
    file_mode: str = "average_only"


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """ Creates a directory if it does not already exist. """
    os.makedirs(path, exist_ok=True)


def _msg(enabled: bool, text: str) -> None:
    """ Print a message only if logging is enabled. """
    if enabled:
        print(text)


def _save_current_figure(save: bool, out_path: str, dpi: int = 300) -> None:
    """ Save the currently active matplotlib figure WITHOUT requiring an
    interactive  backend. Safe to call even when show=False. """
    if not save:
        return
    
    fig = plt.gcf()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _list_txt_files(folder: str, mode: str = "all") -> List[str]:
    """ Return a sorted list of .txt files with optional filtering. """
    all_txt = sorted(glob.glob(os.path.join(folder, "*.txt")))
    
    if mode == "all":
        return all_txt
    
    if mode == "average_only":
        return [p for p in all_txt if "combo" in os.path.basename(p).lower()]
    
    raise ValueError(
        f"Unknown file_mode={mode!r}. Use 'all' or 'average_only'."
        )


def _load_struct_txt(path: str, delimiter: str = "\t") -> np.ndarray:
    """ Robust loader for tab-delimited structured arrays with header. Handles 
    BOM and stray whitespace in header names. """
    arr = np.genfromtxt(
        path, 
        delimiter=delimiter, 
        names=True, 
        dtype=None, 
        encoding=None
        )
    
    if arr is None or \
        getattr(arr, "dtype", None) is None or \
            arr.dtype.names is None:
        raise ValueError(f"Failed to load structured data from {path}")

    # Normalize field names (BOM/whitespace issues)
    names = list(arr.dtype.names)
    clean_map = {n.replace("\ufeff", "").strip(): n for n in names}

    # If RamanShift exists under a slightly different name, standardize by view 
    # copy
    
    if "RamanShift" not in clean_map:
        raise ValueError(
            f"'RamanShift' column not found in {path}. Columns: {names}"
            )

    if clean_map["RamanShift"] != "RamanShift":
        
        # Rebuild with corrected names
        new_names = []
        
        for n in names:
            c = n.replace("\ufeff", "").strip()
            new_names.append(c)
        arr = arr.copy()
        arr.dtype.names = tuple(new_names)

    return arr


def _to_spectra_matrix(
        rxn_data: np.ndarray
        ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    
    """ 
    Convert a structured Raman text dataset into a matrix suitable 
    for MCR/NNLS. 
    
    This helper takes a NumPy structured array (typically loaded 
    from a tab-delimited .txt file with a header row) and converts it into:
      1) A Raman shift axis (x-values)
      2) A list of sample/column names (excluding RamanShift)
      3) A 2D intensity matrix D_raw where each row corresponds to one sample
    
    Parameters
    ----------
    rxn_data : np.ndarray
        Structured NumPy array where:
          - One field must be named "RamanShift"
          - All other fields are interpreted as intensity columns (samples)

    Returns
    -------
    rxn_shift : np.ndarray, shape (n_points,)
        Raman shift axis extracted from the "RamanShift" column.

    sample_names : list[str]
        Names of all intensity columns (all fields except "RamanShift").

    D_raw : np.ndarray, shape (n_samples, n_points)
        Intensity matrix assembled from the intensity columns, stacked so that
        each row corresponds to one sample spectrum.
        
    """
    names = list(rxn_data.dtype.names)
    
    if "RamanShift" not in names:
        raise ValueError(f"Missing RamanShift. Columns: {names}")

    rxn_shift = np.asarray(rxn_data["RamanShift"], dtype=float)
    sample_names = [n for n in names if n != "RamanShift"]
    if not sample_names:
        raise ValueError(f"No sample columns found. Columns: {names}")

    D_raw = np.vstack(
        [np.asarray(rxn_data[n], dtype=float) for n in sample_names])
    
    return rxn_shift, sample_names, D_raw


def _filter_anchors(
        anchor_points: Sequence[float], x: np.ndarray) -> List[float]:
    """ Filter baseline anchor points to those that fall within the x-axis 
    range. """
    xmin, xmax = float(np.min(x)), float(np.max(x))
    
    return [float(ap) for ap in anchor_points if xmin <= ap <= xmax]


def export_spectra(
    shifts,
    spectra,
    output_dir: str = "output",
    prefix: str = "spectrum",
    combo_name: str = "spectra_combo.txt",
) -> None:
    """
    Export Raman spectra into individual two-column text files and one combined 
    table.

    This function writes:
      1) One text file per spectrum (sample), containing:
         - RamanShift (x-axis)
         - RamanIntensity (y-axis)
         with a header line: "RamanShift<TAB>RamanIntensity"

      2) One combined text file (`combo_name`) containing:
         - First column: RamanShift
         - Remaining columns: intensities of all spectra stacked side-by-side
           with Excel-style headers: A, B, C, ..., Z, AA, AB, ...

    The output files are tab-delimited and suitable for reading using:
        np.genfromtxt(..., delimiter="\\t", names=True)

    Parameters
    ----------
    shifts : array-like or list/tuple of array-like
        Raman shift axis (x-values). This may be passed either as:
          - a single 1D array of length n_points, or
          - a list/tuple whose first element is the shift array.
        The function will internally convert it to a NumPy array.

    spectra : array-like
        Raman intensity values for one or more spectra.
        Expected shape is (n_samples, n_points), but a common alternative
        orientation (n_points, n_samples) is handled automatically by
        transposing if needed.
        A single spectrum may also be provided as a 1D array of length n_points.

    output_dir : str, optional
        Output directory where files will be written. If it does not exist,
        it will be created. Default is "output".

    prefix : str, optional
        Prefix used for individual spectrum filenames. Files are written as:
            {prefix}_001.txt, {prefix}_002.txt, ...
        Default is "spectrum".

    combo_name : str, optional
        Filename for the combined spectrum table saved inside `output_dir`.
        Default is "spectra_combo.txt".

    Returns
    -------
    None
        This function writes files to disk and does not return any values.

    """
    _ensure_dir(output_dir)

    # Normalize shift
    shift = np.asarray(shifts[0] if isinstance(shifts, (list, tuple)) \
                       else shifts, dtype=float)
    shift = shift[0]
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)

    # Fix common orientation issues:
    # - expected: (n_samples, n_points)
    # - if transpose matches, transpose
    if spectra.shape[1] != shift.shape[0] and spectra.shape[0]==shift.shape[0]:
        spectra = spectra.T
    
    if spectra.shape[1] != shift.shape[0]:
        raise ValueError(
            f"Shift length ({shift[0].shape[0]}) does not match spectra ",
            "points ({spectra.shape[1]})."
        )

    n_samples = spectra.shape[0]

    # Individual files
    for i in range(n_samples):
        data = np.column_stack((shift, spectra[i]))
        filename = os.path.join(output_dir, f"{prefix}_{i+1:03d}.txt")
        np.savetxt(
            filename,
            data,
            fmt="%.6f\t%.6f",
            header="RamanShift\tRamanIntensity",
            comments="",
        )

    # Combined file
    def _excel_col(n: int) -> str:
        """0->A, 1->B, ... 25->Z, 26->AA, ..."""
        s = ""

        while True:
            n, r = divmod(n, 26)
            s = chr(ord("A") + r) + s
            if n == 0:
                break
            n -= 1
        return s

    col_headers = ["RamanShift"] + [_excel_col(i) for i in range(n_samples)]
    combo = np.column_stack([shift] + [spectra[i] for i in range(n_samples)])
    combo_path = os.path.join(output_dir, combo_name)
    
    np.savetxt(
        combo_path,
        combo,
        fmt="%.6f",
        delimiter="\t",
        header="\t".join(col_headers),
        comments="",
    )


def load_saved_D(folder: str):
    """
    Load previously saved D.npy and metadata from a processing output folder.

    Returns
    -------
    rxn_shift : np.ndarray (n_points,)
    sample_names : list[str] (n_samples,)
    D : np.ndarray (n_samples, n_points)
    """
    D_path = os.path.join(folder, "D.npy")
    meta_path = os.path.join(folder, "D_meta.npz")

    if not os.path.isfile(D_path):
        raise FileNotFoundError(f"Missing: {D_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing: {meta_path}")

    D = np.load(D_path)
    meta = np.load(meta_path, allow_pickle=True)

    rxn_shift = meta["rxn_shift"].astype(float)
    sample_names = [str(x) for x in meta["sample_names"].tolist()]

    return rxn_shift, sample_names, D


def load_pure_spectra(out_dir: str):
    """ Loads pure_spectra.csv and returns TCP_i, DCP_i, GLY_i arrays
    aligned to rxn_shift. """
    fpath = os.path.join(out_dir, "pure_spectra.csv")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Missing: {fpath}")

    dfPure = pd.read_csv(fpath, index_col=0)

    # rxn_shift is stored as column names -> strings -> convert to float
    rxn_shift = dfPure.columns.astype(float).to_numpy()

    # Rows -> arrays
    TCP_i = dfPure.loc["TCP"].to_numpy(dtype=float)
    DCP_i = dfPure.loc["DCP"].to_numpy(dtype=float)
    GLY_i = dfPure.loc["GLY"].to_numpy(dtype=float)

    return rxn_shift, TCP_i, DCP_i, GLY_i


def load_reconstructed(out_dir: str):
    """ Loads reconstructed_spectra.csv.
    
    Returns
    -------
    rxn_shift (1D float array)
    sample_names (list[str])
    reconstructed (2D float array, reconstructed spectra)
    """
    fpath = os.path.join(out_dir, "reconstructed_spectra.csv")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Missing: {fpath}")

    dfRecon = pd.read_csv(fpath, index_col=0)

    # rxn_shift was stored as column headers (strings) -> convert to float
    rxn_shift = dfRecon.columns.astype(float).to_numpy()

    # sample names from index
    sample_names = dfRecon.index.tolist()

    # 2D numeric matrix: shape = (n_samples, n_points)
    reconstructed = dfRecon.to_numpy(dtype=float)

    return rxn_shift, sample_names, reconstructed

# -----------------------------------------------------------------------------
# Main pipeline class
# -----------------------------------------------------------------------------

class Reactions:
    def __init__(
        self,
        spectral_data_path: str,
        blank_data_path: str,
        reference_path: str | None = None,
        preps_path: str | None = None,
        export_path: str | None = None,
        output_folder: str = "output2",
        reaction_type: str = "DCPdegradation",

        # flags
        preps: bool = True,
        mcr: bool | None = None,     # NEW
        autorun: bool = False,

        show: bool = False,
        save: bool = True,
        export: bool = True,
        messages: bool = True,

        # needed to avoid undefined variables in this snippet
        c_lower: int = 23,
        c_upper: int = 571,
        s_lower: int = 200,
        s_upper: int = 210,
        range1: tuple[int, int] = (270, 280),
        range2: tuple[int, int] = (200, 210),
        specs: bool = True,
        ratio: bool = True,
        averages: bool = True,
        wlength: int = 15,
        porder: int = 3,

        **kwargs,
    ):
        """
        Initialize a Raman4React processing reactor.

        This class supports two logical execution branches:

        1) Preprocessing branch ("preps"):
           - Loads raw reaction spectra and blank spectra from folders.
           - Crops to region of interest (c_lower/c_upper).
           - Applies smoothing/processing (wlength/porder) using your pipeline.
           - Subtracts blank spectra.
           - Optionally computes ratios/spec features and/or averaged spectra.
           - Optionally exports spectra and figures into out_dir.

        2) Reference/MCR branch ("mcr"):
           - Loads pure reference spectra file (reference_path) containing at 
             least: RamanShift, TCP, DCP, GLY  (tab-delimited header expected)
           - Loads processed/averaged spectra files from preps_path.
           - Optionally applies baseline correction / normalization.
           - Runs NNLS-MCR and exports concentration tables + reconstructions.

        Parameters
        ----------
        spectral_data_path : str
            Directory with reaction spectra (.txt) files. Used when preps=True.

        blank_data_path : str
            Directory with blank spectra (.txt) files. Used when preps=True.

        reference_path : str | None
            Path to pure reference spectra file for MCR. Used when MCR enabled.

        preps_path : str | None
            Directory containing spectra to be used for MCR (e.g., 
            spectra_combo.txt or averaged files). Used when MCR is enabled.

        export_path : str | None
            Base output directory. If None, current working directory is used.

        output_folder : str
            Folder name inside export_path where outputs are written.

        reaction_type : str
            Label used for your internal branching/naming.

        preps : bool
            If True, enable preprocessing branch. If False, preprocessing is 
            skipped.

        mcr : bool | None
            If None, keeps backwards compatibility:
                - preps=False implies MCR branch runs.
            If explicitly True/False, forces MCR on/off regardless of preps.

        autorun : bool
            If True, automatically executes enabled branches at the end of init

        show/save/export/messages : bool
            I/O controls:
              - show: allow plotting to screen (interactive backend required)
              - save: save figures where implemented
              - export: export spectra/tables
              - messages: print progress messages

        c_lower/c_upper : int
            Crop indices for extract_data (preps branch).

        s_lower/s_upper : int
            Secondary indices for your integration window (preps branch).

        range1/range2 : tuple[int, int]
            Raman shift ranges for ratio calculations (preps branch).

        specs/ratio/averages : bool
            Feature toggles for the preps branch.

        wlength/porder : int
            Processing parameters used by your process_data routine.

        Other Parameters
        ----------------
        delimiter : str (kwarg)
            Delimiter used in text loading for MCR branch. Default "\\t".

        file_mode : str (kwarg)
            Determines which files are used in preps_path for MCR. Suggested 
            values:
              - "average_only" : only files containing "average" in filename
              - "all"          : all .txt files
              - "combo_first"  : prefer spectra_combo.txt if present
              
        apply_baseline_correction : bool (kwarg)
            Flag to allow baseline correction in MCR branch.
        
        apply_normalization : bool (kwarg) 
            Flag to apply normalization in MCR branch.
        
        use_polynomial_baseline : bool (kwarg) 
            Flag to use polynomial baseline in MCR branch.
        
        poly_order : int (kwarg) 
            Order of the polynome to be used for background removal.
        
        anchor_points : list[float] (kwarg)
            Anchor points of the polynome to define background removal.
        
        rfc_radius : int (kwarg)
            Radius of the morphological disk for baclground removal.
            
        """
        # Store core inputs:
        self.spectral_data_path = spectral_data_path
        self.blank_data_path = blank_data_path
        self.reference_path = reference_path
        self.preps_path = preps_path
        self.export_path = export_path
        self.output_folder = output_folder
        self.reaction_type = reaction_type

        self.show = show
        self.save = save
        self.export = export
        self.messages = messages

        # Backward-compatible resolution of mcr flag:
        if mcr is None:
            self.mcr_enabled = (not preps)
        else:
            self.mcr_enabled = bool(mcr)

        self.preps_enabled = bool(preps)

        # "Results" placeholders to run steps later
        self.prep_results = None
        self.mcr_results = None

        # IO settings (messages / logging flags)
        self.io = type("IO", (), {})()
        self.io.messages = bool(messages)
        self.io.show = bool(show)      # needed by _run_preprocessing_branch
        self.io.save = bool(save)      # needed by plotting save
        self.io.export = bool(export)  # used in preprocessing export gate

        # Output directory used throughout
        base_out = self.export_path or os.getcwd()
        self.out_dir = os.path.join(base_out, self.output_folder)
        os.makedirs(self.out_dir, exist_ok=True)

        # PREP settings used by the preprocessing branch
        self.prep = PrepSettings(
            spectral_data_path=spectral_data_path,
            blank_data_path=blank_data_path,
            c_lower=c_lower,
            c_upper=c_upper,
            s_lower=s_lower,
            s_upper=s_upper,
            range1=range1,
            range2=range2,
            show=show,
            save=save,
            export=export,
            specs=specs,
            ratio=ratio,
            averages=averages,
        )
        
        # process_data expects these attributes in your snippet
        self.prep.wlength = wlength
        self.prep.porder = porder
        
        # MCR settings object (required by _run_reference_mcr_branch)
        class _MCRSettings:
            pass
        
        self.mcr = _MCRSettings()
        
        # Required paths (coming from your existing inputs)
        self.mcr.reference_path = self.reference_path
        self.mcr.preps_path = self.preps_path
        
        # File parsing
        self.mcr.delimiter = kwargs.pop("delimiter", "\t")        
        self.mcr.file_mode = kwargs.pop("file_mode", "average_only")         
        
        # Baseline / normalization
        self.mcr.apply_baseline_correction = \
            kwargs.pop("apply_baseline_correction", False)
        self.mcr.apply_normalization = \
            kwargs.pop("apply_normalization", False)
        
        # Polynomial baseline options
        self.mcr.use_polynomial_baseline = \
            kwargs.pop("use_polynomial_baseline", True)
        self.mcr.poly_order = kwargs.pop("poly_order", 2)
        self.mcr.anchor_points = kwargs.pop("anchor_points", [])
        self.mcr.rfc_radius = kwargs.pop("rfc_radius", 150)
        
        # If user passed unexpected kwargs, keep them visible rather than 
        # silently ignoring.
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unknown keyword arguments: {unknown}")
            
        # Autorun behavior
        # IMPORTANT: avoid double-running MCR. Let run() orchestrate based on 
        # flags.
        if autorun:
            self.run()
            self.run_mcr()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self) -> "Reactions":
        """ Runs whatever is enabled by flags. """
        if self.preps_enabled:
            self.run_preps()

        if self.mcr_enabled:
            self.run_mcr()

        return self

    def run_preps(self) -> "Reactions":
        """ Run preprocessing (load/crop/filter/subtract/plots/specs/etc).
        Stores results on self for later use. """
        
        self._run_preprocessing_branch()
        return self

    def run_mcr(self) -> "Reactions":
        """ Run reference/MCR branch. If possible, re-use preprocessing outputs
        if they exist; otherwise load from preps_path. """
        
        self._run_reference_mcr_branch()
        return self
    
    
    def plot_mcr(
        self,
        input_folder: str,
        output_folder: str | None = None,
        concentrations_name: str = "concentrations.csv",
        ncols: int = 3,
        dpi: int = 300,
        show_R2: bool = True,
        ) -> None:
        """
        Plot MCR results from an output folder containing saved MCR artifacts.
        
        Expects (in input_folder):
          - concentrations.csv
          - pure_spectra.csv
          - reconstructed_spectra.csv
          - D.npy + D_meta.npz (or whatever your loaders expect)
        
        Produces (in output_folder or input_folder):
          - fractions.png
          - intensity_stack.png
          - combined_mcr_matrix.png
        """
        # Validate inputs
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        in_path = os.path.join(input_folder, concentrations_name)
        if not os.path.isfile(in_path):
            raise FileNotFoundError(
                f"Missing '{concentrations_name}' in: {input_folder}")
        
        # Output folder for plots
        out_dir = output_folder or input_folder
        os.makedirs(out_dir, exist_ok=True)
        

        # Load concentrations
        dfC = pd.read_csv(in_path, index_col=0)
        
        # Normalize index labels (strip whitespace/BOM issues)
        dfC.index = dfC.index.map(
            lambda x: str(x).replace("\ufeff", "").strip())
        
        # Convert numeric columns safely (do NOT destroy index)
        for col in dfC.columns:
            dfC[col] = pd.to_numeric(dfC[col], errors="coerce")
        
        # Load saved artifacts
        rxn_shift, TCP_i, DCP_i, GLY_i = load_pure_spectra(input_folder)
        _, sample_names_D, D = load_saved_D(input_folder)
        _, _, reconstructed = load_reconstructed(input_folder)
        
        # Normalize sample names from D as well
        if sample_names_D is not None:
            sample_names_D = \
                [str(s).replace("\ufeff", "").strip() for s in sample_names_D]
        
        # Decide the sample order
        # Priority: D_meta sample_names (because D/reconstructed rows depend 
        # on it)
        if sample_names_D is not None and len(sample_names_D) == D.shape[0]:
            sample_names = sample_names_D
        else:
            # Fallback: use dfC index as sample_names
            sample_names = dfC.index.astype(str).tolist()
        
        # Align dfC to the chosen sample order.
        # IMPORTANT: only reindex if there is overlap; otherwise keep dfC order
        overlap = len(set(sample_names).intersection(set(dfC.index)))
        if overlap > 0:
            dfC = dfC.reindex(sample_names)
        else:
            # No overlap: keep dfC's own index to avoid all-NaN plots
            sample_names = dfC.index.astype(str).tolist()
        
        # Sanity checks (fail early with a useful message)
        if D.shape[0] != len(sample_names):
            raise ValueError(
                f"Mismatch: D has {D.shape[0]} rows but sample_names has ",
                f"{len(sample_names)}.\n"
                f"First 5 sample_names: {sample_names[:5]}"
            )
        if reconstructed.shape[0] != len(sample_names):
            raise ValueError(
                f"Mismatch: reconstructed has {reconstructed.shape[0]} rows ",
                f"but sample_names has {len(sample_names)}."
            )

        # Plot
        r4rAnal.plot_concentration_fractions(
            dfC, 
            sample_names, 
            out_dir, 
            dpi=dpi
            )
        
        r4rAnal.plot_actual_intensities(
            dfC, 
            sample_names, 
            out_dir, 
            dpi=dpi
            )
    
        r4rAnal.plot_original_vs_reconstructed(
            dfC,
            reconstructed,
            D,
            rxn_shift,
            TCP_i,
            DCP_i,
            GLY_i,
            sample_names,
            out_dir,
            ncols=ncols,
            dpi=dpi,
            show_r2=show_R2,
        )


    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate(self) -> None:
        """
        Validate required inputs before running preprocessing and/or MCR 
        branches.
    
        This method checks that the necessary paths and files exist depending 
        on which execution flags are enabled:
    
        - If `self.preps_enabled` is True (preprocessing branch):
            * `self.prep.spectral_data_path` must be set (folder with reaction 
               spectra)
            * `self.prep.blank_data_path` must be set (folder with blank 
               spectra)
    
        - If `self.mcr_enabled` is True (reference/MCR branch):
            * `self.reference_path` must be set and point to an existing file
              (tab-delimited reference file containing columns like RamanShift, 
               TCP, DCP, GLY)
            * `self.preps_path` must be set and point to an existing directory
              (folder containing the spectra to be analyzed by MCR;
               e.g., average/combined spectra file(s))
    
        Raises
        ------
        ValueError
            If a required path is missing or invalid type (e.g., tuple).
        FileNotFoundError
            If `reference_path` file or `preps_path` directory does not exist.
        """
        # Preprocessing branch checks
        if self.preps_enabled:
            if not getattr(self, "prep", None):
                raise ValueError(
                    "Internal error: self.prep is not initialized.")
    
            if not self.prep.spectral_data_path:
                raise ValueError(
                    "spectral_data_path is required when preps=True")
    
            if not self.prep.blank_data_path:
                raise ValueError(
                    "blank_data_path is required when preps=True")
    
        # MCR branch checks
        if self.mcr_enabled:
            # Catch accidental tuples early
            # (common when a trailing comma is present)
            if isinstance(self.reference_path, tuple):
                raise ValueError(
                    "reference_path must be a string path, got tuple: ",
                    f"{self.reference_path}. This often happens if you ",
                    "accidentally added a trailing comma."
                )
            if isinstance(self.preps_path, tuple):
                raise ValueError(
                    "preps_path must be a string path, got tuple: ",
                    f"{self.preps_path}. This happens if you accidentally",
                    " added a trailing comma."
                )
    
            if not self.reference_path:
                raise ValueError(
                    "reference_path must be provided when mcr=True"
                    )
    
            if not self.preps_path:
                raise ValueError("preps_path must be provided when mcr=True")
    
            if not os.path.isfile(self.reference_path):
                raise FileNotFoundError(
                    f"reference_path not found: {self.reference_path}"
                 )
    
            if not os.path.isdir(self.preps_path):
                raise FileNotFoundError(
                    f"preps_path folder not found: {self.preps_path}")

    # -------------------------------------------------------------------------
    # Branch 1: preprocessing
    # -------------------------------------------------------------------------

    def _run_preprocessing_branch(self) -> None:
        """
        Execute the preprocessing branch: 
            (raw spectra → processed/blank-subtracted → exports)
     
        Pipeline overview
        -----------------
        1) Load raw reaction spectra and blank spectra from folders.
        2) Crop both datasets to the requested Raman-shift index window 
            (c_lower:c_upper).
        3) Process both datasets 
            (e.g., smoothing/normalization inside r4rProcc.process_data).
        4) Subtract blank spectra from reaction spectra (group-aware 
           subtraction).
        5) Optional: generate diagnostic plots (only if self.io.show=True).
        6) Optional: export averaged spectra to:
             - individual two-column txt files (RamanShift / RamanIntensity)
             - combined multi-column table "spectra_combo.txt"
     
        Side effects / stored attributes
        -------------------------------
        This method stores intermediate and final arrays on `self`, including:
         - self.dfiles, self.bfiles      (lists of file paths)
         - self.dcropped, self.bcropped  (cropped raw arrays)
         - self.dshift, self.bshift      (shift axes returned by processing)
         - self.dprocc, self.bprocc      (processed spectra)
         - self.dsubtracted              (blank-subtracted spectra)
         - self.averages_idx             (index mapping from subtract_blank)
         - self.avgdt                    (averaged spectra, when enabled)
     
        """
        _msg(self.io.messages, "[INFO] Running preprocessing branch...")

        # (0) Load data -------------------------------------------------------
        self.dfiles = r4rProcc.load_data(self.prep.spectral_data_path)
        self.bfiles = r4rProcc.load_data(self.prep.blank_data_path)
        
        _msg(
            self.io.messages, 
            f"[INFO] Spectral data loaded, total of {len(self.dfiles)} files."
            )
        _msg(
            self.io.messages, 
            f"[INFO] Blank data loaded, total of {len(self.bfiles)} files."
            )

        # (1) Extract/crop ----------------------------------------------------
        # Cropping is done using index limits (c_lower/c_upper) that match 
        # the internal format expected by r4rProcc.extract_data().
        self.dcropped, self.dfull = r4rProcc.extract_data(
            self.dfiles, lower=self.prep.c_lower, upper=self.prep.c_upper
        )
        self.bcropped, self.bfull = r4rProcc.extract_data(
            self.bfiles, lower=self.prep.c_lower, upper=self.prep.c_upper
        )

        _msg(self.io.messages, "[INFO] Spectral/blank data cropped.")

        # (2) Process ---------------------------------------------------------
        # (e.g., filtering/smoothing, normalization inside your pipeline)
        self.dshift, self.dprocc, self.filtered = r4rProcc.process_data(
            self.dcropped, 
            wlength=self.prep.wlength, 
            porder=self.prep.porder, 
            show=self.io.show
        )
        
        self.bshift, self.bprocc, self.bfiltered = r4rProcc.process_data(
            self.bcropped, 
            wlength=self.prep.wlength, 
            porder=self.prep.porder, 
            show=self.io.show
        )
        
        _msg(self.io.messages, "[INFO] Spectral/blank data processed.")

        # (3) Subtract blank --------------------------------------------------
        # subtract_blank returns:
        #   - dsubtracted : blank-subtracted spectra
        #   - gidx, didx  : group/sample bookkeeping indices
        #   - averages_idx: mapping used later by avg_spectra
        self.dsubtracted, gidx, didx, self.averages_idx = \
            r4rProcc.subtract_blank(
                self.dfiles, 
                self.dprocc, 
                self.bfiles, 
                self.bprocc
            )
        _msg(self.io.messages, "[INFO] Blank subtracted.")

        # (4) Optional plots --------------------------------------------------
        # NOTE: If using a non-interactive backend (Agg), plt.show() may warn;
        # saving still works fine. The plotting functions may call plt.show() 
        # internally.
        # safe saving without interactive backend
        if self.io.show:
            _msg(self.io.messages, "[INFO] Plotting outputs...")

            r4rProcc.plot_spectra_subplots(
                didx, 
                self.dsubtracted, 
                self.dshift, 
                group_labels=None, 
                subplot_shape=(6, 6), 
                figsize=(24, 12)
            )
            _save_current_figure(
                self.io.save, 
                os.path.join(self.out_dir, "subplot_spectra.png")
                )

            idx = np.random.randint(0, self.dprocc.shape[1])
            r4rProcc.plot_spectra_subplots(
                didx, 
                self.dsubtracted, 
                self.dshift, 
                group_labels=None, 
                subplot_shape=(6, 6), 
                figsize=(24, 12),
                selected_column=idx
            )
            _save_current_figure(
                self.io.save, 
                os.path.join(self.out_dir, "subplot_intensities.png")
                )

        # (5) Export average spectra if requested -----------------------------
        # Only export if export is enabled AND averaging is enabled.

        if self.io.export:
            if self.prep.averages:
                grp_idx = r4rProcc.get_grp_idx(didx, self.dsubtracted.shape[0])
                self.avgdt = r4rProcc.avg_spectra(
                    self.dprocc, 
                    self.averages_idx, 
                    self.bshift, 
                    grp_idx, 
                    show=False
                    )
                
                export_spectra(
                    self.bshift,
                    self.avgdt,
                    output_dir=self.out_dir,
                    prefix="spectrum",
                    combo_name="spectra_combo.txt",
                )
                _msg(self.io.messages, "[INFO] Export complete.")

    # -------------------------------------------------------------------------
    # Branch 2: reference/MCR
    # -------------------------------------------------------------------------

    def _run_reference_mcr_branch(self) -> None:
        """
        Execute the reference/MCR branch (NNLS-MCR using pure reference 
        spectra).
        
        What this branch does
        ---------------------
        1) Loads the pure reference spectra file (tab-delimited) which must 
           contain: RamanShift, TCP, DCP, GLY (these are stored on `self` as:
           self.ref_shift, self.TCP, self.DCP, self.GLY).
        
        2) Selects reaction spectrum file(s) from `self.mcr.preps_path` 
           according to the file selection policy `self.mcr.file_mode` 
           (implemented in `_list_txt_files`).
        
        3) For each selected file, runs `_process_one_reaction_file(file_path)`
           which performs:
             - loading the spectra table
             - optional baseline correction / normalization
             - interpolation of pure spectra onto the reaction shift grid
             - NNLS fitting (MCR) per sample spectrum
             - saving outputs into a per-file output directory
        
        Preconditions
        -------------
        - `self.mcr` must be initialized in __init__ and define at least:
            * reference_path (str): file path to reference pure spectra (*.txt)
            * preps_path (str): folder with reaction spectra tables (*.txt)
            * delimiter (str): typically "\\t"
            * file_mode (str): selection policy used by `_list_txt_files`
        """
        _msg(self.io.messages, "[INFO] Running reference/MCR branch...")
        
        # Ensure settings exist
        if not getattr(self, "mcr", None):
            raise AttributeError(
                "self.mcr is not initialized. Create it in __init__."
                )
            
        # Validate required inputs
        if not self.mcr.reference_path:
            raise ValueError("reference_path must be provided for MCR branch.")
            
        if not self.mcr.preps_path:
            raise ValueError("preps_path must be provided for MCR branch.")
    
        delimiter = self.mcr.delimiter
    
        # Load reference spectra (pure components)
        ref = _load_struct_txt(self.mcr.reference_path, delimiter=delimiter)
        
        # Protect against BOM/whitespace in headers (common in exported .txt)
        ref_names = list(ref.dtype.names)
        norm = {n.replace("\ufeff", "").strip(): n for n in ref_names}
    
        required = ["RamanShift", "TCP", "DCP", "GLY"]
        missing = [c for c in required if c not in norm]
        if missing:
            raise KeyError(
                f"Missing required columns {missing} in reference file: ",
                f"{self.mcr.reference_path}. Columns found: {ref_names}"
            )       
            
        self.ref_shift = np.asarray(ref["RamanShift"], dtype=float)
        self.TCP = np.asarray(ref["TCP"], dtype=float)
        self.DCP = np.asarray(ref["DCP"], dtype=float)
        self.GLY = np.asarray(ref["GLY"], dtype=float)
    
        # Select reaction files to process
        files = _list_txt_files(self.mcr.preps_path, mode=self.mcr.file_mode)
        l = len(files)
        
        _msg(self.io.messages, 
             f"[INFO] Using preps_path: {self.mcr.preps_path}")
        _msg(self.io.messages, 
             f"[INFO] Found {l} reaction files ({self.mcr.file_mode}).")
    
        if not files:
            raise ValueError(
                "No .txt files found to process. "
                "Check preps_path='{self.mcr.preps_path}' and file_mode=",
                f"'{self.mcr.file_mode}'."
            )

        # Process each reaction file
        for file_path in files:
            self._process_one_reaction_file(file_path)


    def _process_one_reaction_file(self, file_path: str) -> None:
        """
        Process a single reaction spectra table and run NNLS/MCR against 
        reference spectra.
        
        Steps
        -----
        1) Load a tab-delimited spectra table from `file_path`.
           Expected format:
             - column "RamanShift"
             - remaining columns = sample spectra (e.g., A, B, C... or sample
               names)
        
        2) Convert the structured table into:
             - rxn_shift: shape (n_points,)
             - sample_names: list[str]
             - D_raw: shape (n_samples, n_points)
        
        3) Optional preprocessing per spectrum (row of D_raw):
             - polynomial baseline correction OR rolling-circle baseline 
               correction
             - optional min-max normalization
        
        4) Interpolate pure reference spectra (TCP/DCP/GLY) onto rxn_shift.
        
        5) Solve NNLS per sample spectrum:
             D[i] ≈ a_TCP * TCP_i + a_DCP * DCP_i + a_GLY * GLY_i
        
        6) Save outputs into a per-file output subdirectory:
             - D.npy + D_meta.npz (rxn_shift, sample_names)
             - concentrations.csv
             - reconstructed_spectra.csv
             - residuals.csv
             - pure_spectra.csv
        
        """

        delimiter = self.mcr.delimiter
        
        # Create an output folder for this file
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_dir = os.path.join(self.out_dir, base)
        _ensure_dir(out_dir)

        _msg(self.io.messages, 
             f"\n[INFO] Processing: {os.path.basename(file_path)}")

        # Load and convert to matrix form
        rxn_data = _load_struct_txt(
            file_path, 
            delimiter=delimiter
            )
        
        rxn_shift, self.sample_names, D_raw = _to_spectra_matrix(rxn_data)
        
        # Prepare anchors (for polynomial baseline)
        anchors = _filter_anchors(self.mcr.anchor_points, rxn_shift)
        if self.mcr.apply_baseline_correction and \
            self.mcr.use_polynomial_baseline:
            if len(anchors) < (self.mcr.poly_order + 1):
                raise ValueError(
                    f"Need at least {self.mcr.poly_order + 1} anchor points ",
                    f" for polynomial order {self.mcr.poly_order}. "
                    f"Have {len(anchors)} within range for file {file_path}."
                )
        
        # Preprocess spectra (row-wise)
        D = np.zeros_like(D_raw, dtype=float)
        for i in range(D_raw.shape[0]):
            y = D_raw[i].astype(float).copy()
            
            # polynomial baseline correction
            if self.mcr.apply_baseline_correction and \
                self.mcr.use_polynomial_baseline:
                    
                baseline = r4rAnal.polynomial_baseline_constrained(
                    rxn_shift, 
                    y, 
                    anchor_points=anchors, 
                    order=self.mcr.poly_order
                )
                
                y = y - baseline
                y[y < 0] = 0.0
                
            # rolling-circle baseline correction
            elif self.mcr.apply_baseline_correction and not \
                self.mcr.use_polynomial_baseline:
                    
                base_line = r4rAnal.rolling_circle_baseline(
                    y, 
                    radius=self.mcr.rfc_radius
                )
                
                y = y - base_line
                y[y < 0] = 0.0
            
            # optional normalization
            if self.mcr.apply_normalization:
                ymin, ymax = float(y.min()), float(y.max())
                if ymax - ymin > 0:
                    y = (y - ymin) / (ymax - ymin)

            D[i] = y
        
        # Interpolate pure spectra to reaction grid
        TCP_i = np.interp(rxn_shift, self.ref_shift, self.TCP)
        DCP_i = np.interp(rxn_shift, self.ref_shift, self.DCP)
        GLY_i = np.interp(rxn_shift, self.ref_shift, self.GLY)
        
        # Run NNLS/MCR
        dfC, reconstructed, residuals = r4rAnal.perform_mcr_nnls(
            D, rxn_shift, TCP_i, DCP_i, GLY_i, self.sample_names
        )
        
        
        # Save the processed spectra matrix and minimal metadata
        np.save(os.path.join(out_dir, "D.npy"), D)
        np.savez(
            os.path.join(out_dir, "D_meta.npz"),
            rxn_shift=np.asarray(rxn_shift, dtype=float),
            sample_names=np.asarray(self.sample_names, dtype=object),
        )
        
        # Save coefficients table
        dfC.to_csv(os.path.join(out_dir, "concentrations.csv"), index=True)
        
        # Keep last processed results on self (used by plot_mcr() if you rely
        # on it)
        self.dfC = dfC
        self.odir = out_dir
        
        # Save reconstructed + residuals
        pd.DataFrame(
            reconstructed, 
            index=self.sample_names, 
            columns=rxn_shift
            ).to_csv(
            os.path.join(out_dir, "reconstructed_spectra.csv"), index=True
        )
        pd.DataFrame(
            residuals, 
            index=self.sample_names, 
            columns=rxn_shift
            ).to_csv(
            os.path.join(out_dir, "residuals.csv"), index=True
        )
            
        # Save pure spectra (on the same rxn_shift grid)
        pd.DataFrame(
            [TCP_i, DCP_i, GLY_i], 
            index=["TCP", "DCP", "GLY"], 
            columns=rxn_shift).to_csv(
                os.path.join(out_dir, "pure_spectra.csv"), 
                index=True
                )
