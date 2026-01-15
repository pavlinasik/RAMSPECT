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

# IMPORTANT: use non-interactive backend so saving does NOT require %matplotlib qt
import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt

import spectra_processing as r4rProcc
import spectra_analysis as r4rAnal


# -----------------------------------------------------------------------------
# Configuration containers
# -----------------------------------------------------------------------------

@dataclass
class PrepSettings:
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
    c_lower: int = 23
    c_upper: int = 571


@dataclass(frozen=True)
class SpecificsConfig:
    specs: bool = True
    ratio: bool = True
    averages: bool = True
    s_lower: int = 200
    s_upper: int = 210
    range1: Tuple[int, int] = (270, 280)
    range2: Tuple[int, int] = (200, 210)


@dataclass(frozen=True)
class PreprocessConfig:
    enabled: bool = True
    spectral_data_path: str = ""
    blank_data_path: str = ""
    wlength: int = 7
    porder: int = 1


@dataclass(frozen=True)
class ReferenceMCRConfig:
    enabled: bool = False
    reference_path: Optional[str] = None
    preps_path: Optional[str] = None  # folder with reaction *.txt files to analyze
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


# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _msg(enabled: bool, text: str) -> None:
    if enabled:
        print(text)


def _save_current_figure(save: bool, out_path: str, dpi: int = 300) -> None:
    """
    Save the currently active matplotlib figure WITHOUT requiring an interactive backend.
    Safe to call even when show=False.
    """
    if not save:
        return
    fig = plt.gcf()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _list_txt_files(folder: str, mode: str = "all") -> List[str]:
    """
    Return a sorted list of .txt files with optional filtering.
    """
    all_txt = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if mode == "all":
        return all_txt
    if mode == "average_only":
        return [p for p in all_txt if "average" in os.path.basename(p).lower()]
    raise ValueError(f"Unknown file_mode={mode!r}. Use 'all' or 'average_only'.")


def _load_struct_txt(path: str, delimiter: str = "\t") -> np.ndarray:
    """
    Robust loader for tab-delimited structured arrays with header.
    Handles BOM and stray whitespace in header names.
    """
    arr = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=None, encoding=None)
    if arr is None or getattr(arr, "dtype", None) is None or arr.dtype.names is None:
        raise ValueError(f"Failed to load structured data from {path}")

    # Normalize field names (BOM/whitespace issues)
    names = list(arr.dtype.names)
    clean_map = {n.replace("\ufeff", "").strip(): n for n in names}

    # If RamanShift exists under a slightly different name, standardize by view copy
    if "RamanShift" not in clean_map:
        raise ValueError(f"'RamanShift' column not found in {path}. Columns: {names}")

    if clean_map["RamanShift"] != "RamanShift":
        # Rebuild with corrected names
        new_names = []
        for n in names:
            c = n.replace("\ufeff", "").strip()
            new_names.append(c)
        arr = arr.copy()
        arr.dtype.names = tuple(new_names)

    return arr


def _to_spectra_matrix(rxn_data: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    From structured rxn_data, return:
      - rxn_shift: shape (n_points,)
      - sample_names: list[str]
      - D_raw: shape (n_samples, n_points)
    """
    names = list(rxn_data.dtype.names)
    if "RamanShift" not in names:
        raise ValueError(f"Missing RamanShift. Columns: {names}")

    rxn_shift = np.asarray(rxn_data["RamanShift"], dtype=float)
    sample_names = [n for n in names if n != "RamanShift"]
    if not sample_names:
        raise ValueError(f"No sample columns found. Columns: {names}")

    D_raw = np.vstack([np.asarray(rxn_data[n], dtype=float) for n in sample_names])
    return rxn_shift, sample_names, D_raw


def _filter_anchors(anchor_points: Sequence[float], x: np.ndarray) -> List[float]:
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
    Export individual spectra files with headers + a combined table.

    - Individual files: two columns with header "RamanShift\tRamanIntensity"
    - Combined file: first column RamanShift, remaining columns A, B, C... (Excel-like)
    """
    _ensure_dir(output_dir)

    # Normalize shift
    shift = np.asarray(shifts[0] if isinstance(shifts, (list, tuple)) else shifts, dtype=float)
    shift = shift[0]
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)

    # Fix common orientation issues:
    # - expected: (n_samples, n_points)
    # - if transpose matches, transpose
    if spectra.shape[1] != shift.shape[0] and spectra.shape[0] == shift.shape[0]:
        spectra = spectra.T
    
    if spectra.shape[1] != shift.shape[0]:
        raise ValueError(
            f"Shift length ({shift[0].shape[0]}) does not match spectra points ({spectra.shape[1]})."
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
    """
    Loads pure_spectra.csv and returns TCP_i, DCP_i, GLY_i arrays
    aligned to rxn_shift.
    """
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
    """
    Loads reconstructed_spectra.csv and returns:
      - rxn_shift (1D float array)
      - sample_names (list[str])
      - reconstructed (2D float array, reconstructed spectra)
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

# -------------------------------------------------------------------------
# Main pipeline class
# -------------------------------------------------------------------------

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

        # >>> ADDED: only what is needed to avoid undefined variables in this snippet <<<
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
        # <<< END ADDED <<<

        # ... keep the rest of your parameters as-is
        **kwargs,
    ):
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
            self.mcr_enabled = (not preps)   # old behavior: preps=False implied MCR branch
        else:
            self.mcr_enabled = bool(mcr)

        self.preps_enabled = bool(preps)

        # "Results" placeholders so you can run steps later
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

        # PREP settings used by your preprocessing branch
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
        self.mcr.apply_baseline_correction = kwargs.pop("apply_baseline_correction", False)
        self.mcr.apply_normalization = kwargs.pop("apply_normalization", False)
        
        # Polynomial baseline options
        self.mcr.use_polynomial_baseline = kwargs.pop("use_polynomial_baseline", True)
        self.mcr.poly_order = kwargs.pop("poly_order", 2)
        self.mcr.anchor_points = kwargs.pop("anchor_points", [])
        self.mcr.rfc_radius = kwargs.pop("rfc_radius", 150)

        if autorun:
            self.run()
            self.run_mcr()

    # -----------------------------
    # Public API
    # -----------------------------

    def run(self) -> "Reactions":
        """
        Runs whatever is enabled by flags.
        """
        if self.preps_enabled:
            self.run_preps()

        if self.mcr_enabled:
            self.run_mcr()

        return self

    def run_preps(self) -> "Reactions":
        """
        Run preprocessing (load/crop/filter/subtract/plots/specs/etc).
        Stores results on self for later use.
        """
        self._run_preprocessing_branch()
        return self

    def run_mcr(self) -> "Reactions":
        """
        Run reference/MCR branch. If possible, re-use preprocessing outputs
        if they exist; otherwise load from preps_path.
        """
        self._run_reference_mcr_branch()
        return self
    
    
    def plot_mcr(
        self,
        input_folder: str,
        output_folder: str | None = None,
        concentrations_name: str = "concentrations.csv",
        ncols: int = 3,
        dpi: int=300,
        show_R2: bool=True,
    ) -> None:
        """
        Plot MCR results from a folder containing concentrations.csv.
    
        Parameters
        ----------
        input_folder : str
            Folder that contains concentrations.csv
            
        output_folder : str | None
            Where to save plots. If None, uses input_folder.
            
        concentrations_name : str
            Filename of concentration table (default: concentrations.csv).
        """
        import os
        import pandas as pd
    
        in_path = os.path.join(
            input_folder, 
            concentrations_name
            )
        if not os.path.isfile(in_path):
            raise FileNotFoundError(
                f"Missing '{concentrations_name}' in: {input_folder}")
    
        odir = output_folder or input_folder
        os.makedirs(odir, exist_ok=True)
    
        # Most likely: index is sample names, columns are components (TCP/DCP/GLY)
        try:
            dfC = pd.read_csv(in_path, index_col=0)
        except Exception:
            # Fallback: no index column
            dfC = pd.read_csv(in_path)
    
        # Ensure numeric where possible
        dfC = dfC.apply(pd.to_numeric)
    
        
        rxn_shift, TCP_i, DCP_i, GLY_i = load_pure_spectra(odir)
        _, sample_names, D = load_saved_D(odir)
        _, _, dfR = load_reconstructed(odir)
        
        # Call your plotting utilities on loaded data
        r4rAnal.plot_concentration_fractions(dfC, sample_names, odir)
        r4rAnal.plot_actual_intensities(dfC, sample_names, odir)
        r4rAnal.plot_original_vs_reconstructed(
            dfC, 
            dfR,
            D, 
            rxn_shift, 
            TCP_i, 
            DCP_i, 
            GLY_i, 
            sample_names, 
            odir,
            ncols=ncols,
            dpi=dpi,
            show_r2=show_R2)

    # -----------------------------
    # Validation
    # -----------------------------

    def _validate(self) -> None:
        # >>> FIXED: do not reference self.prep.enabled / self.mcr.* (they are not created in this snippet) <<<
        if self.preps_enabled:
            if not self.prep.spectral_data_path:
                raise ValueError("spectral_data_path is required when preps=True")
            if not self.prep.blank_data_path:
                raise ValueError("blank_data_path is required when preps=True")

        if self.mcr_enabled:
            if not self.reference_path:
                raise ValueError("reference_path must be provided when mcr=True")
            if not self.preps_path:
                raise ValueError("preps_path must be provided when mcr=True")
            if not os.path.isfile(self.reference_path):
                raise FileNotFoundError(f"reference_path not found: {self.reference_path}")
            if not os.path.isdir(self.preps_path):
                raise FileNotFoundError(f"preps_path folder not found: {self.preps_path}")

    # -----------------------------
    # Branch 1: preprocessing
    # -----------------------------

    def _run_preprocessing_branch(self) -> None:
        _msg(self.io.messages, "[INFO] Running preprocessing branch...")

        # (0) Load
        self.dfiles = r4rProcc.load_data(self.prep.spectral_data_path)
        self.bfiles = r4rProcc.load_data(self.prep.blank_data_path)
        _msg(self.io.messages, f"[INFO] Spectral data loaded, total of {len(self.dfiles)} files.")
        _msg(self.io.messages, f"[INFO] Blank data loaded, total of {len(self.bfiles)} files.")

        # (1) Extract/crop
        self.dcropped, self.dfull = r4rProcc.extract_data(
            self.dfiles, lower=self.prep.c_lower, upper=self.prep.c_upper
        )
        self.bcropped, self.bfull = r4rProcc.extract_data(
            self.bfiles, lower=self.prep.c_lower, upper=self.prep.c_upper
        )
        # <<< END FIXED <<<
        _msg(self.io.messages, "[INFO] Spectral/blank data cropped.")

        # (2) Process
        self.dshift, self.dprocc, self.filtered = r4rProcc.process_data(
            self.dcropped, wlength=self.prep.wlength, porder=self.prep.porder, show=self.io.show
        )
        self.bshift, self.bprocc, self.bfiltered = r4rProcc.process_data(
            self.bcropped, wlength=self.prep.wlength, porder=self.prep.porder, show=self.io.show
        )
        _msg(self.io.messages, "[INFO] Spectral/blank data processed.")

        # (3) Subtract blank
        self.dsubtracted, gidx, didx, self.averages_idx = r4rProcc.subtract_blank(
            self.dfiles, self.dprocc, self.bfiles, self.bprocc
        )
        _msg(self.io.messages, "[INFO] Blank subtracted.")

        # (4) Optional plots (safe saving without interactive backend)
        if self.io.show:
            _msg(self.io.messages, "[INFO] Plotting outputs...")

            r4rProcc.plot_spectra_subplots(
                didx, self.dsubtracted, self.dshift, group_labels=None, subplot_shape=(6, 6), figsize=(24, 12)
            )
            _save_current_figure(self.io.save, os.path.join(self.out_dir, "subplot_spectra.png"))

            idx = np.random.randint(0, self.dprocc.shape[1])
            r4rProcc.plot_spectra_subplots(
                didx, self.dsubtracted, self.dshift, group_labels=None, subplot_shape=(6, 6), figsize=(24, 12),
                selected_column=idx
            )
            _save_current_figure(self.io.save, os.path.join(self.out_dir, "subplot_intensities.png"))

        # (5) Export average spectra if requested
        if self.io.export:
            # >>> FIXED: self.spec.averages does not exist; use self.prep.averages <<<
            if self.prep.averages:
                grp_idx = r4rProcc.get_grp_idx(didx, self.dsubtracted.shape[0])
                self.avgdt = r4rProcc.avg_spectra(self.dprocc, self.averages_idx, self.bshift, grp_idx, show=False)
                export_spectra(
                    self.bshift,
                    self.avgdt,
                    output_dir=self.out_dir,
                    prefix="spectrum",
                    combo_name="spectra_combo.txt",
                )
                _msg(self.io.messages, "[INFO] Export complete.")

    # -----------------------------
    # Branch 2: reference/MCR
    # -----------------------------

    def _run_reference_mcr_branch(self) -> None:
        _msg(self.io.messages, "[INFO] Running reference/MCR branch...")
    
        if not getattr(self, "mcr", None):
            raise AttributeError("self.mcr is not initialized. Create it in __init__.")
    
        if not self.mcr.reference_path:
            raise ValueError("reference_path must be provided for MCR branch.")
        if not self.mcr.preps_path:
            raise ValueError("preps_path must be provided for MCR branch.")
    
        delimiter = self.mcr.delimiter
    
        # Load reference spectra
        ref = _load_struct_txt(self.mcr.reference_path, delimiter=delimiter)
        self.ref_shift = np.asarray(ref["RamanShift"], dtype=float)
        self.TCP = np.asarray(ref["TCP"], dtype=float)
        self.DCP = np.asarray(ref["DCP"], dtype=float)
        self.GLY = np.asarray(ref["GLY"], dtype=float)
    
        files = _list_txt_files(self.mcr.preps_path, mode=self.mcr.file_mode)
        _msg(self.io.messages, f"[INFO] Found {len(files)} reaction files ({self.mcr.file_mode}).")
        if not files:
            raise ValueError("No .txt files found to process.")
    
        for file_path in files:
            self._process_one_reaction_file(file_path)


    def _process_one_reaction_file(self, file_path: str) -> None:
        delimiter = self.mcr.delimiter
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_dir = os.path.join(self.out_dir, base)
        _ensure_dir(out_dir)

        _msg(self.io.messages, f"\n[INFO] Processing: {os.path.basename(file_path)}")

        rxn_data = _load_struct_txt(file_path, delimiter=delimiter)
        rxn_shift, self.sample_names, D_raw = _to_spectra_matrix(rxn_data)

        anchors = _filter_anchors(self.mcr.anchor_points, rxn_shift)
        if self.mcr.apply_baseline_correction and self.mcr.use_polynomial_baseline:
            if len(anchors) < (self.mcr.poly_order + 1):
                raise ValueError(
                    f"Need at least {self.mcr.poly_order + 1} anchor points for polynomial order {self.mcr.poly_order}. "
                    f"Have {len(anchors)} within range for file {file_path}."
                )

        D = np.zeros_like(D_raw, dtype=float)
        for i in range(D_raw.shape[0]):
            y = D_raw[i].astype(float).copy()

            if self.mcr.apply_baseline_correction and self.mcr.use_polynomial_baseline:
                baseline = r4rAnal.polynomial_baseline_constrained(
                    rxn_shift, y, anchor_points=anchors, order=self.mcr.poly_order
                )
                y = y - baseline
                y[y < 0] = 0.0
            elif self.mcr.apply_baseline_correction and not self.mcr.use_polynomial_baseline:
                base_line = r4rAnal.rolling_circle_baseline(y, radius=self.rcf_radius)
                y = y - base_line
                y[y < 0] = 0.0

            if self.mcr.apply_normalization:
                ymin, ymax = float(y.min()), float(y.max())
                if ymax - ymin > 0:
                    y = (y - ymin) / (ymax - ymin)

            D[i] = y

        TCP_i = np.interp(rxn_shift, self.ref_shift, self.TCP)
        DCP_i = np.interp(rxn_shift, self.ref_shift, self.DCP)
        GLY_i = np.interp(rxn_shift, self.ref_shift, self.GLY)

        dfC, reconstructed, residuals = r4rAnal.perform_mcr_nnls(
            D, rxn_shift, TCP_i, DCP_i, GLY_i, self.sample_names
        )
        
        
        np.save(os.path.join(out_dir, "D.npy"), D)

        np.savez(
            os.path.join(out_dir, "D_meta.npz"),
            rxn_shift=np.asarray(rxn_shift, dtype=float),
            sample_names=np.asarray(self.sample_names, dtype=object),
        )
        
        
        dfC.to_csv(os.path.join(out_dir, "concentrations.csv"), index=True)
        self.dfC = dfC
        self.odir = out_dir

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

        pd.DataFrame(
            [TCP_i, DCP_i, GLY_i], index=["TCP", "DCP", "GLY"], columns=rxn_shift
        ).to_csv(os.path.join(out_dir, "pure_spectra.csv"), index=True)
