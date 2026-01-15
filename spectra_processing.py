# -*- coding: utf-8 -*-
"""
spectra_processing.py
====================

Core utilities for Raman spectra preprocessing, group handling, export, and
basic exploratory visualization.

This module provides a lightweight workflow for working with Raman datasets
stored as individual text files, where each file contains a two-column 
spectrum:

    RamanShift   RamanIntensity

It supports:
    - loading spectra files from a folder
    - cropping spectral regions by index
    - smoothing and normalization
    - blank subtraction (group-wise blank averaging)
    - group visualization utilities
    - averaging spectra per group
    - exporting individual spectra + a combined multi-column dataset

The functions are designed to be used as building blocks in higher-level
workflows (e.g., in Raman4React), but also work standalone for quick 
inspection.

Expected Data Layout
--------------------
1) Raw spectra and blank spectra
   Files are expected to be stored in directories, with each file containing
   numerical data in two columns:

       shift[cm^-1]    intensity[a.u.]

   The loader expects ".txt" files and uses NumPy to read them.

2) Grouping convention (important for subtraction)
   Blank subtraction assumes spectra belong to groups identified by the leading
   number in their filename, e.g.:

       "1_sample_01.txt"   -> group "1"
       "1_sample_02.txt"   -> group "1"
       "2_sample_01.txt"   -> group "2"
       ...

   This grouping is used to:
       - average blank spectra within each group
       - subtract the appropriate group blank from each data spectrum

Main Processing Steps
---------------------
Typical pipeline:

    1) load_data(path)
       Collect all spectrum files in a folder.

    2) extract_data(files, lower, upper)
       Load each file with NumPy and crop by index bounds.

    3) process_data(cropped_spectra, wlength, porder, show)
       Apply Savitzky–Golay smoothing (scipy.signal.savgol_filter) and min–max
       normalization.

    4) subtract_blank(dfiles, data, bfiles, blank)
       Build group indices from filenames, compute mean blank per group, and
       subtract that mean from the corresponding data spectra.

    5) avg_spectra(data, blank, shift, grp_idx, show)
       Compute average spectra per group (optionally plotting).

    6) export_spectra(shifts, spectra, output_dir, prefix, combo_name)
       Export spectra to individual text files and optionally create one 
       combined matrix-style file.

Export Output Formats
---------------------
The `export_spectra(...)` function produces two outputs:

(1) Individual spectra files (one per spectrum):
    Example filename: spectrum_001.txt

    Format (tab-separated):
        RamanShift<TAB>RamanIntensity
        2699.900000<TAB>0.000123
        ...

(2) A combined spectra file (default: spectra_combo.txt):
    Format (tab-separated):
        RamanShift<TAB>A<TAB>B<TAB>C<...>

    Where:
        - RamanShift is the shared x-axis
        - Each additional column is one exported spectrum intensity trace
        - Column names follow Excel-like alphabet headers (A, B, .., Z, AA, ..)

This combined file is intended for downstream matrix-based methods
(e.g., multivariate fitting, decomposition, MCR/NNLS workflows).


Dependencies
------------
Required:
    numpy
    scipy (savgol_filter)
    matplotlib

Optional / workflow-dependent:
    - external code may wrap these utilities in higher-level classes
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.signal import savgol_filter

import re
import os

def load_data(path):
    """
    Loads all .txt files from the specified directory.

    Parameters
    ----------
    path : str
        The directory path from which .txt files should be loaded.

    Returns
    -------
    data : list of str
        A list containing the full file paths of all .txt files in a specified
        directory.
    """
    # Load filenames of all .txt files from the input path
    files = glob.glob(path + r"\*.*")
    data = [f for f in files if f.lower().endswith(".txt")]
    
    return data


def extract_data(data, lower=23, upper=571):
    """
    Extracts and crops data from a list of text files.
    
    Parameters
    ----------
    data : list of str
        List of file paths to text files containing numerical data.
    lower : int, optional
        Number of elements to remove from the beginning of each dataset. 
        Default is 23.
    upper : int, optional
        Number of elements to remove from the end of each dataset. 
        Default is 571.
    
    Returns
    -------
    cropped : list of np.ndarray
        List of cropped numerical arrays, where each array corresponds 
        to a file with the first `lower` elements and last `upper` elements 
        removed.
    full : list of np.ndarray
        List of full numerical arrays, where each array corresponds to 
        the original file data.
    """

    # Create empty lists
    cropped, full = [], []
    
    # Iterate through input data
    for f in data:
        # Load datafile
        s = np.loadtxt(f)
        
        # Full data length
        full.append(s)
        
        # Cropped data length
        s = s[lower:s.shape[0]-upper]
        cropped.append(s)
    
    return cropped, full


def plot_random_spectrum(data, no=1, dtype="spectrum", grp_idx=None):
    """
    Plots random spectra from the given dataset.
    
    Parameters
    ----------
    data : list of np.ndarray or np.ndarray
        List of spectra or arrays containing spectra to plot.
        - If `dtype` is "spectrum" or "blank", each entry in `data` should be 
          a 2D numpy array where the first column represents the Raman shift 
          values and the second column represents the corresponding intensity 
          values.
        - If `dtype` is "subtracted", each entry in `data` should be a 1D numpy 
          array representing a subtracted spectrum.
    
    no : int, optional
        Number of random spectra to plot. Default is 1.
        If `no` exceeds the available number of spectra, it will plot as many 
        as possible.
    
    dtype : str, optional
        Type of data to plot. Can be "spectrum", "blank", or "subtracted". 
        - "spectrum" for original spectra.
        - "blank" for blank spectra.
        - "subtracted" for spectra after background subtraction.
    
    grp_idx : list of int, optional
        List of group indices to assign specific colors to spectra based on 
        their group. If provided, the function will use the `tab20` and `Set3` 
        colormaps for color assignment. If not provided, spectra will be
        assigned random colors based on their index.
    
    Returns
    -------
    None
        The function directly displays the plot with random spectra.
    
    Notes
    -----
    - If `grp_idx` is provided, each spectrum will be assigned a color 
      corresponding to its group.
    - If `grp_idx` is not provided, random colors are assigned to the spectra
      based on their index.
    - The legend displays group labels (or indices if no groups are specified) 
      and is positioned outside the plot.
    - Up to 36 spectra can be plotted with distinct colors (based on available 
      colormap options).
    """

    # Generate random indices of spectra to be displayed
    rand_idx = np.random.randint(0, len(data), no)
    
    # If group indices exist, assign colors based on groups
    if grp_idx is not None and len(grp_idx) > 0:
        # Unique colors for each spectrum
        # Combine tab20 and Set3 colormaps
        colors1 = plt.cm.get_cmap("tab20", 20)  # 20 colors from tab20
        colors2 = plt.cm.get_cmap("Set3", 16)   # 16 colors from Set3
        
        # Combine them
        colors = np.concatenate([colors1(np.linspace(0, 1, 20)), 
                                 colors2(np.linspace(0, 1, 16))])
        
        # Determine group membership for each random spectrum
        group_labels = []
        for idx in rand_idx:
            for i, g_idx in enumerate(grp_idx):
                if idx < g_idx:
                    group_labels.append(i - 1)
                    break
            else:
                group_labels.append(len(grp_idx) - 1)
    else:
        # Unique colors for each spectrum
        # Combine tab20 and Set3 colormaps
        colors1 = plt.cm.get_cmap("tab20", 20)  # 20 colors from tab20
        colors2 = plt.cm.get_cmap("Set3", 16)   # 16 colors from Set3
        
        # Combine them
        colors = np.concatenate([colors1(np.linspace(0, 1, 20)), 
                                 colors2(np.linspace(0, 1, 16))])
        
    # Create figure
    plt.figure()
    plotted_groups = set()  # To ensure each group appears in legend only once

    for i, idx in enumerate(rand_idx):
        if grp_idx is not None and len(grp_idx) > 0:
            group = group_labels[i]
            color = colors[group]  # Assign color based on group
            
            label = f"Group {group+1}" if group not in plotted_groups else ""
            plotted_groups.add(group)
        else:
            label = rf"ID {idx}"
            color = colors[i]  # Normalize index for distinct colors

        if dtype == "spectrum":
            plt.plot(
                data[idx][:, 0], 
                data[idx][:, 1], 
                color=color, 
                label=label
                )
            plt.xlabel("Raman shift [cm$^{-1}$]")
            plt.title("Random spectra")
        
        elif dtype == "blank":
            plt.plot(
                data[idx][:, 0], 
                data[idx][:, 1], 
                color=color, 
                label=label
                )
            plt.xlabel("Raman shift [cm$^{-1}$]")
            plt.title("Random blank spectra")
        
        elif dtype == "subtracted":
            plt.plot(data[idx,:], color=color, label=label)
            plt.xlabel("series index [-]")
            plt.title("Random subtracted spectra")

    # Ensure legend is always present, placed outside the plot
    plt.legend(loc='upper left', 
               bbox_to_anchor=(1.02, 1), 
               ncol=2, fontsize=10, frameon=True)

    plt.ylabel("Raman intensity [-]")
    plt.tight_layout()
    plt.show()


def process_data(data, wlength=7, porder=1, show=False):
    """
    Processes a list of spectra by applying a Savitzky-Golay filter, 
    normalizing the intensities, and optionally displaying a comparison 
    of a randomly selected spectrum before and after processing.
    
    Parameters
    ----------
    data : list of np.ndarray
        List of 2D arrays where each array represents a spectrum with the first 
        column as the Raman shift and the second column as the corresponding 
        intensity values.
    
    wlength : int, optional
        The window length for the Savitzky-Golay filter.
        Must be an odd integer. Default is 7.
    
    porder : int, optional
        The polynomial order for the Savitzky-Golay filter. 
        Default is 1.
    
    show : bool, optional
        Whether to display a comparison plot for a randomly selected spectrum, 
        showing both the raw and processed spectra. Default is False.
    
    Returns
    -------
    shift : np.ndarray
        Array containing the Raman shift values for each spectrum.
    
    normalized : np.ndarray
        Array containing the normalized spectra after Savitzky-Golay filtering 
        and min-max normalization.
    
    filtered : np.ndarray
        Array containing the filtered intensities using the Savitzky-Golay 
        filter.

    """

    # Convert list to numpy.array
    data = [np.array(arr) for arr in data]
    
    # Extract columns /intensity, shift/ separately
    shift = np.array([arr[:, 0] for arr in data])
    intensity = np.array([arr[:, 1] for arr in data])
    
    # Filter intensity
    filtered = savgol_filter(intensity, 
                             window_length=wlength, 
                             polyorder=porder, 
                             axis=1)
    
    # Normalize data (fit and transform on filtered)
    normalized = min_max_normalize(filtered)
    
    if show:
        rand_idx = np.random.randint(0, len(data))
        
        # Transform the intensity of the randomly selected spectrum
        spectrum = min_max_normalize(intensity[rand_idx, :])
        
        plt.figure()
        plt.plot(shift[rand_idx, :], spectrum, label="raw")
        plt.plot(shift[rand_idx, :], normalized[rand_idx, :], label="filtered")
        plt.xlabel("Raman shift [cm$^{-1}$]")
        plt.ylabel("Norm. Raman intensity [%]")
        plt.title(rf"Cropped normalized spectra #{rand_idx}")
        plt.tight_layout()
        plt.show()

    return shift, normalized, filtered


def show_shift(data, idx=205, dtype="raw", grp_idx=None, yaxis=None, avg=False):
    """
    Plots the intensity of the spectra at a specific Raman shift index and 
    optionally marks the group separations and labels them between vertical 
    lines, based on the provided group indices.
    
    Parameters
    ----------
    data : np.ndarray
        2D array where each row represents a spectrum, and each column 
        corresponds to a specific Raman shift value. The function will plot 
        the intensity of the spectrum at the given shift index.
    
    idx : int, optional
        The index of the Raman shift to be plotted. 
        Default is 205.
    
    dtype : str, optional
        Specifies the type of data to display in the plot. It can either be 
        "raw" (intensity before blank subtraction) or "subtracted" (intensity 
        after blank subtraction). Default is "raw".
    
    grp_idx : list of int, optional
        List of indices indicating the boundaries of different groups.
        If provided, vertical lines are drawn at these indices, and each group 
        is labeled between the lines. Default is None.
    
    Returns
    -------
    None
        The function does not return any value but displays a plot with 
        the intensity at the given Raman shift index, with group separators and 
        labels if `grp_idx` is provided.
    
    """
    
    int_idx = data[:,idx]

    plt.figure(figsize=(16,5))
    plt.plot(int_idx, label="Intensity")
    plt.xlabel("No. of Raman spectrum [-]")
    plt.ylabel("Relative Raman intensity [%]")
    if yaxis:
        plt.ylim(yaxis[0],yaxis[1])
    
    # Draw vertical lines at the group indices if provided
    if grp_idx is not None:
        for g_idx in grp_idx:
            plt.axvline(x=g_idx, color="r", linestyle='--', 
                        label=r'Group separators' \
                            if g_idx == grp_idx[0] else "")
        
        # Add group number labels above the plot
        ax = plt.gca()  # Get current axes
        ymin, ymax = ax.get_ylim()
        label_y = ymax - 0.1 * (ymax - ymin)  # slightly above top

        for i in range(len(grp_idx) - 1):
            midpoint = (grp_idx[i] + grp_idx[i + 1]) / 2
            ax.text(midpoint, label_y, str(i + 1),
                    ha='center', va='bottom',
                    fontsize=10, color='blue')
                    
            
    if dtype=="raw":
        plt.title(rf"36 ratios at shift index {idx}, blank not subtracted")
        
    elif dtype=="subtracted":
        plt.title(rf"36 ratios at shift index {idx}, blank subtracted")
    
    if avg:
        # Compute and plot the average intensity per group
        group_means = []
        group_positions = []
        
        for i in range(len(grp_idx) - 1):
            start, end = grp_idx[i], grp_idx[i + 1]
            mean_val = np.mean(int_idx[start:end])
            group_means.append(mean_val)
            group_positions.append(start)
        
        # Add the final point for the last segment
        group_positions.append(grp_idx[-1])
        
        # Repeat the last mean value to keep the length consistent
        group_means.append(group_means[-1])
        
        # Plot the stepped average line
        plt.step(group_positions, group_means, where='post', color='black',
                 label='Group average', linewidth=2)


    plt.legend()
    plt.tight_layout()
    plt.show()


def subtract_blank(dfiles, data, bfiles, blank):
    """
     Subtracts the blank spectra from the data spectra, averaging the blank 
     spectra by group and applying the corresponding blank subtraction to each 
     data spectrum.
    
     Parameters
     ----------
     dfiles : list of str
         List of file paths for the data files to be processed. The filenames 
         are expected to have a leading number representing the group number.
    
     data : np.ndarray
         2D array where each row represents a spectrum from the data files and 
         each column represents a Raman shift value. The blank spectra are 
         subtracted from these spectra.
    
     bfiles : list of str
         List of file paths for the blank files, with filenames including
         a leading number that indicates the group to which each blank belongs.
    
     blank : np.ndarray
         2D array where each row represents a blank spectrum, and each column 
         represents a Raman shift value.
    
     Returns
     -------
     final : np.ndarray
         2D array containing the data spectra after the blank subtraction, with 
         each row representing a subtracted spectrum.
    
     group_indices : dict
         A dictionary where keys are group numbers (as strings), and the values 
         are lists of indices corresponding to the files belonging to each 
         group in the blank file list.
    
     data_indices : dict
         A dictionary where keys are group numbers (as strings), and the values 
         are lists of indices corresponding to the files belonging to each 
         group in the data file list.
    
     averages : list of np.ndarray
         A list of the averaged blank spectra for each group. Each entry is 
         the mean of the blank spectra for a particular group.
    
     Notes
     -----
     - The function extracts the group numbers from the filenames of the blank
       and data files.
     - Blank spectra are averaged by group, and the corresponding average is 
       subtracted from each data spectrum.
     - The resulting subtracted spectra are returned along with the group 
       indices and the blank averages.
     """
 
    # Dictionaries to store group indices
    group_indices = {} 
    data_indices = {}
    blank_avg = {}
    
    # List to store subtracted data
    subtracted = []

    # (1a) Iterate through the file paths of blanks to extract the group number 
    #      and index
    for i, file_path in enumerate(bfiles):
        # Extract filename
        filename = os.path.basename(file_path)  
        
        # Extract leading number
        match = re.match(r"(\d+)", filename) 
        if match:
            group_number = match.group(1)
            
            if group_number not in group_indices:
                group_indices[group_number] = []
            
            group_indices[group_number].append(i)
        else:
            # Debugging output
            print(f"No match found for: {filename}")  
    
    # (1b) Iterate through the file paths of data to extract the group number 
    #      and index
    for i, file_path in enumerate(dfiles):
        # Extract filename
        filename = os.path.basename(file_path)  
        
        # Extract leading number
        match = re.match(r"(\d+)", filename)  
        if match:
            group_number = match.group(1)
            
            if group_number not in data_indices:
                data_indices[group_number] = []
            
            data_indices[group_number].append(i)
        else:
            # Debugging output
            print(f"No match found for: {filename}") 
    
    # (2) Averaging blanks per group
    # Iterate over actual group numbers
    for group_number in group_indices.keys(): 
        # Get indices
        indices = group_indices[group_number]  
        
        if indices:  # Ensure indices exist
            # Extract relevant blank spectra
            b1 = blank[indices]  
            # Store in dictionary
            blank_avg[group_number] = np.mean(b1, axis=0)  
     
    
    # (3) Subtract averages from its respective spectra
    for group_number in group_indices.keys():  
        # Skip missing groups
        if group_number not in blank_avg:  
            continue
        
        # Retrieve average for this group
        sub = blank_avg[group_number] 
        
        # Get corresponding data indices
        indices = data_indices.get(group_number, []) 
        if indices:
            # Subtract blank
            s1 = data[indices]-sub  
            subtracted.append(s1)
    
    # (4) Convert to an array
    final = np.concatenate(subtracted) if subtracted else np.array([])
    
    # (5) Convert blank averages to lists
    averages = list(blank_avg.values())
    
    return final, group_indices, data_indices, averages


def plot_spectra_subplots(data_indices, spectra_final, shift, 
                          group_labels=None, 
                          subplot_shape=(6, 6), figsize=(15, 15), 
                          selected_column=None):

    """
    Plots the spectra from different groups in subplots, allowing for 
    individual spectra visualization or the selection of a specific column 
    of the spectra.

    Parameters
    ----------
    data_indices : dict
        A dictionary where the keys are group numbers (as strings), and 
        the values are lists of indices corresponding to the data spectra 
        of that group.

    spectra_final : np.ndarray
        2D array containing the final spectra data, where each row corresponds 
        to a spectrum.

    shift : np.ndarray
        2D array containing the Raman shift values corresponding to each 
        spectrum in `spectra_final`.

    group_labels : list of str, optional
        List of labels for each group, used as titles for the subplots. 
        If not provided, group numbers are used as titles.

    subplot_shape : tuple of int, optional
        The shape of the subplot grid. Default is (6, 6), creating a 6x6 grid 
        of subplots.

    figsize : tuple of int, optional
        Size of the figure in inches. Default is (15, 15).

    selected_column : int, optional
        If provided, only this column (spectrum index) is plotted for each 
        group. Default is None, which plots the entire spectrum.

    Returns
    -------
    None
        The function does not return any values but displays the plot 
        with the subplots.
    """
    
    # Create the subplot grid
    fig, axes = plt.subplots(subplot_shape[0], subplot_shape[1], 
                             figsize=figsize, 
                             sharex=True, sharey=True)
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
        
    # Plot each group in its subplot
    for idx, (group_number, indices) in enumerate(data_indices.items()):
        # Ensure we don't exceed the grid size
        if idx >= subplot_shape[0] * subplot_shape[1]:  
            break
        
        # Get the current subplot
        ax = axes[idx] 
        
        # Extract corresponding spectra, of shape (N, spectrum_length)
        group_spectra = spectra_final[indices] 
        group_shift = shift[indices]
        group_mean = np.mean(group_spectra)
        
        # Plot the selected column of the spectrum for each group
        # Transpose to plot correctly
        if selected_column:
            ax.plot(group_spectra[:, selected_column].T, alpha=0.7)  
        else:
            ax.plot(group_shift.T, group_spectra.T, alpha=0.7)  
        
        # Add title, using group number or label if provided
        title = group_labels[idx] if group_labels \
            else f"Group {group_number}: mean value = {group_mean:.2f}"
        ax.set_title(title, fontsize=10)
    
    # Hide empty subplots if less than the total grid
    for i in range(len(data_indices), subplot_shape[0] * subplot_shape[1]):
        fig.delaxes(axes[i])
    
    plt.axis("on")
    # Adjust layout
    plt.tight_layout()
    
    # Prevents blocking, so you can close it manually
    plt.show()  
    
    return


def get_grp_idx(data_indices, last):
    """
    Generates a list of group indices from the data_indices dictionary, ending
    with the specified `last` value.
    
    Parameters
    ----------
    data_indices : dict
        A dictionary where keys are group identifiers (e.g., group numbers) 
        and values are lists of indices corresponding to the data of that group
    
    last : int
        The last value to append to the group indices list. Typically used 
        to mark the end of the last group.
    
    Returns
    -------
    grp_idx : list of int
        A list of group indices, where each index corresponds to the first data 
        index of a group in `data_indices`, and the final element is the `last` 
        value.
    """
    
    grp_idx = []
    
    for key in data_indices:
        first_value = data_indices[key][0]
        grp_idx.append(first_value)
    
    grp_idx.append(last)
    
    return grp_idx


def min_max_normalize(data, mrange=100):
    """
    Min-max normalize data to a target range.

    Parameters
    ----------
    data : array-like
        1D (single spectrum) or 2D (n_spectra x n_points).
    mrange : float or (float, float)
        If float/int: interpreted as (0, mrange).
        If tuple/list length 2: interpreted as (low, high).

    Returns
    -------
    np.ndarray
        Normalized array with same shape as input.
    """
    x = np.asarray(data, dtype=float)

    # Interpret mrange
    if isinstance(mrange, (int, float, np.number)):
        a, b = 0.0, float(mrange)
    else:
        if len(mrange) != 2:
            raise ValueError(
                "mrange must be a scalar or a (low, high) tuple/list."
                )
            
        a, b = float(mrange[0]), float(mrange[1])

    if x.ndim == 1:
        mn = x.min()
        mx = x.max()
        if mx == mn:
            return np.full_like(x, a)
        y = (x - mn) / (mx - mn)
        return y * (b - a) + a

    if x.ndim == 2:
        mn = x.min(axis=1, keepdims=True)
        mx = x.max(axis=1, keepdims=True)
        denom = np.where(mx == mn, 1.0, mx - mn)
        y = (x - mn) / denom
        return y * (b - a) + a

    raise ValueError(f"Expected 1D or 2D input, got {x.ndim}D.")


def plot_group(subtracted, idx1, idx2, group):
    """
    Plots subtracted spectra for a specified group within a given index range.
    
    Parameters
    ----------
    subtracted : np.ndarray
        A 2D array where each row represents a subtracted spectrum.
    
    idx1 : int
        The starting index of the spectra to plot (inclusive).
    
    idx2 : int
        The ending index of the spectra to plot (exclusive).
    
    group : int
        The group number to be used in the plot title.
    
    Returns
    -------
    None
        This function does not return any value. It generates a plot of 
        the subtracted spectra for the specified group.
    """

    plt.figure(figsize=(8, 5))
    
    data = subtracted[idx1:idx2,:]
    
    # Plot each spectrum (each row in subtracted)
    for i in range(data.shape[0]):
        plt.plot(data[i, :], label=f"Spectrum {idx1 + i}")

    plt.xlabel("Index [-]")
    plt.ylabel("Intensity [-]")
    plt.title(f"Subtracted Spectra for Group {group}")
    plt.tight_layout()
    plt.show()
    return


def sum_intensities(
        data, 
        lower=200, 
        upper=210, 
        show=True, 
        grp_idx=None, 
        yaxis=None, 
        avg=False
        ):
    
    """
    Computes and optionally visualizes the sum of intensities within 
    a specified spectral range (between `lower` and `upper` indices) for each 
    Raman spectrum.

    Parameters
    ----------
    data : np.ndarray
        2D array where each row is a Raman spectrum and each column represents 
        an intensity value at a specific Raman shift index.

    lower : int, optional
        Lower index of the spectral range to be summed (inclusive).
        Default is 200.

    upper : int, optional
        Upper index of the spectral range to be summed (exclusive).
        Default is 210.

    show : bool, optional
        If True, the function will display a plot of the summed intensities 
        for each spectrum. Default is True.

    grp_idx : list of int, optional
        List of indices indicating the boundaries of different groups. 
        If provided, vertical dashed lines are drawn at the boundaries, and 
        group numbers are labeled above the plot. Default is None.

    yaxis : tuple of float, optional
        y-axis limits for the plot in the form (ymin, ymax). 
        If None, matplotlib's default limits are used. Default is None.

    avg : bool, optional
        If True and `grp_idx` is provided, the function will compute and 
        overlay a stepped line representing the average summed intensity for 
        each group.

    Returns
    -------
    range_sums : np.ndarray
        1D array containing the summed intensities for each spectrum 
        over the specified index range.

    """
    range_sums = []
    
    for d in data:
        segment = d[lower:upper]
        sum_segment = np.sum(segment)
        range_sums.append(sum_segment)
    
    range_sums = np.array(range_sums)
        
    
    if show:
        plt.figure(figsize=(32,10))
        plt.plot(range_sums, label="Intensity sum")
        plt.title(rf"Intensity sum per spectral range ({lower}, {upper})")
        plt.xlabel("No. of Raman spectrum [-]")
        plt.ylabel("Intensity sum")
        if yaxis:
            plt.ylim(yaxis[0],yaxis[1])
        
        # Draw vertical lines at the group indices if provided
        if grp_idx is not None:
            for g_idx in grp_idx:
                plt.axvline(x=g_idx, color="r", linestyle='--', 
                            label=r'Group separators' \
                                if g_idx == grp_idx[0] else "")
            
            # Add group number labels above the plot
            ax = plt.gca()  # Get current axes
            ymin, ymax = ax.get_ylim()
            label_y = ymax - 0.1 * (ymax - ymin)  # slightly above top

            for i in range(len(grp_idx) - 1):
                midpoint = (grp_idx[i] + grp_idx[i + 1]) / 2
                ax.text(midpoint, label_y, str(i + 1),
                        ha='center', va='bottom',
                        fontsize=10, color='blue')
                        
        if avg:
            # Compute and plot the average intensity per group
            group_means = []
            group_positions = []
            
            for i in range(len(grp_idx) - 1):
                start, end = grp_idx[i], grp_idx[i + 1]
                mean_val = np.mean(range_sums[start:end])
                group_means.append(mean_val)
                group_positions.append(start)
            
            # Add the final point for the last segment
            group_positions.append(grp_idx[-1])
            
            # Repeat the last mean value to keep the length consistent
            group_means.append(group_means[-1])
            
            # Plot the stepped average line
            plt.step(group_positions, group_means, where='post', color='black',
                     label='Group average', linewidth=2)


        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return range_sums
        

def show_peaksum_ratio(
        data, 
        range1=(270,280), 
        range2=(200,210), 
        order=1, 
        show=True, 
        yaxis=None, 
        grp_idx=None, 
        avg=False
        ):
    
    """
    Computes and optionally plots the ratio of intensity sums between two 
    user-defined spectral index ranges across a series of Raman spectra.

    Parameters
    ----------
    data : np.ndarray
        2D array where each row is a Raman spectrum and each column represents 
        an intensity value at a specific Raman shift index.

    range1 : tuple of int, optional
        Index range (lower, upper) of the first spectral peak to be summed.
        Default is (270, 280).

    range2 : tuple of int, optional
        Index range (lower, upper) of the second spectral peak to be summed.
        Default is (200, 210).

    order : int, optional
        Determines the ratio calculation:
        - If 1 (default), computes ratio = sum(range1) / sum(range2)
        - If 2, computes ratio = sum(range2) / sum(range1)

    show : bool, optional
        If True, displays a plot of the intensity sum ratio for each spectrum. 
        Default is True.

    yaxis : tuple of float, optional
        y-axis limits for the plot in the form (ymin, ymax). 
        If None, matplotlib uses automatic scaling. Default is None.

    grp_idx : list of int, optional
        List of indices marking group boundaries. If provided, the plot will 
        include dashed vertical lines at these points and group numbers above 
        the plot. Default is None.

    avg : bool, optional
        If True and `grp_idx` is provided, overlays the group-wise average 
        ratio as a stepped black line. Default is False.

    Returns
    -------
    ratio : np.ndarray
        1D array containing the ratio of intensity sums for each spectrum.

    Notes
    -----
    - The intensity in each range is summed using the `sum_intensities()` 
      function.
    - This function is useful for analyzing changes in relative peak 
      intensities, e.g., for monitoring concentration changes or peak 
      suppression/enhancement.
    """

    drange1 = sum_intensities(data, 
                              lower=range1[0], upper=range1[1], 
                              show=False)
    drange2 = sum_intensities(data, 
                              lower=range2[0], upper=range2[1], 
                              show=False)
    
    if order == 1:
        ratio = drange1 / drange2
        ratio_text = f"{range1} / {range2}"
    elif order == 2:
        ratio = drange2 / drange1
        ratio_text = f"{range2} / {range1}"
    
    if show:
        plt.figure(figsize=(32,10))
        plt.plot(ratio, label="Intensity sum ratio")
        plt.xlabel("No. of Raman spectrum [-]")
        plt.ylabel("Raman intensity sum ratio")
        plt.title(f"Intensity sum ratio: {ratio_text}")
        
        if yaxis:
            plt.ylim(yaxis[0], yaxis[1])
        
        # Draw vertical lines at the group indices if provided
        if grp_idx is not None:
            for g_idx in grp_idx:
                plt.axvline(
                    x=g_idx, 
                    color="r", 
                    linestyle='--', 
                    label=r'Group separators' if g_idx == grp_idx[0] else "")
            
            # Add group number labels above the plot
            ax = plt.gca()
            ymin, ymax = ax.get_ylim()
            label_y = ymax - 0.1 * (ymax - ymin)

            for i in range(len(grp_idx) - 1):
                midpoint = (grp_idx[i] + grp_idx[i + 1]) / 2
                ax.text(midpoint, label_y, str(i + 1),
                        ha='center', va='bottom',
                        fontsize=10, color='blue')
        
        if avg:
            group_means = []
            group_positions = []
            
            for i in range(len(grp_idx) - 1):
                start, end = grp_idx[i], grp_idx[i + 1]
                mean_val = np.mean(ratio[start:end])
                group_means.append(mean_val)
                group_positions.append(start)
            
            group_positions.append(grp_idx[-1])
            group_means.append(group_means[-1])
            
            plt.step(
                group_positions, 
                group_means, 
                where='post', 
                color='black',
                label='Group average', 
                linewidth=2
                )

        plt.legend()
        plt.tight_layout()
        plt.show()

    return ratio


def avg_spectra(data, blank, shift, grp_idx, show=True):
    """
    Computes the average Raman spectrum for each group of spectra, subtracts
    the corresponding mean blank signal, and optionally plots the results.

    Parameters
    ----------
    data : np.ndarray
        2D array where each row corresponds to a single Raman spectrum and each 
        column corresponds to an intensity value at a specific Raman shift 
        index.

    blank : list or np.ndarray
        List or 1D array of blank spectra values, one per group. These are 
        subtracted from the group average spectra.

    shift : np.ndarray
        2D array representing the Raman shift values. Should have shape (1, N), 
        where N is the number of spectral points.

    grp_idx : list of int
        List of indices that define the boundaries of the groups. The function 
        computes
        group averages between successive index pairs: [grp_idx[0]:grp_idx[1]], 
        [grp_idx[1]:grp_idx[2]], etc.

    show : bool, optional
        If True, plots the average spectra for each group with the blank 
        spectrum subtracted. Default is True.

    Returns
    -------
    data_avg : list of np.ndarray
        List containing the blank-subtracted average spectrum for each group.

    """
    
    data_avg = []
    
    for i in range(0,len(grp_idx)-1):
        group_mean = np.mean(data[grp_idx[i]:grp_idx[i+1]],axis=0)
        group_mean = group_mean-blank[i]
        data_avg.append(group_mean)
    
    if show:
        plt.figure(figsize=(8,5))
        group=1
        for d in data_avg:
            plt.plot(shift[0,:], d, label=f"G{group}")
            group+=1
            
        plt.xlabel("Raman shift [cm$^{-1}$]")    
        plt.ylabel("Raman intensity [-]")
        
        # Ensure legend is always present, placed outside the plot
        plt.legend(loc='upper left', 
                   bbox_to_anchor=(1.02, 1), 
                   ncol=2, fontsize=10, frameon=True)
        
        plt.title("Subtraction of mean blank from mean spectrum per group")
        plt.tight_layout()
        plt.show()
    
    return data_avg


def export_spectra(shifts, spectra, output_dir="output", prefix="spectrum",
                  combo_name="spectra_combo.txt"):
    """
    Exports Raman spectra to:
      (1) individual txt files: spectrum_001.txt, spectrum_002.txt, ...
      (2) one combined txt file: spectra_combo.txt

    Combined format:
        RamanShift<TAB>A<TAB>B<TAB>C...
    """
    def _excel_col_name(n: int) -> str:
        """
        Convert 0-based index to Excel-like column names:
        0->A, 1->B, ..., 25->Z, 26->AA, 27->AB, ...
        """
        name = ""
        n = n + 1  # 1-based for calculation
        while n > 0:
            n, r = divmod(n - 1, 26)
            name = chr(65 + r) + name
        return name

    # shifts can be list-of-arrays or a single array
    shift = shifts[0] if isinstance(shifts, (list, tuple)) else shifts
    shift = np.asarray(shift, dtype=float)

    spectra = np.asarray(spectra, dtype=float)

    os.makedirs(output_dir, exist_ok=True)

    # Ensure spectra is 2D: (n_samples, n_points)
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)

    n_samples, n_points = spectra.shape

    # (1) Export individual files
    for i in range(n_samples):
        data = np.column_stack((shift[0], spectra[i]))
        filename = os.path.join(output_dir, f"{prefix}_{i+1:03d}.txt")

        np.savetxt(
            filename,
            data,
            fmt="%.6f\t%.6f",
            header="RamanShift\tRamanIntensity",
            comments=""
        )

    # (2) Export combined file: RamanShift + all spectra as columns
    combo_path = os.path.join(output_dir, combo_name)

    # Column headers: RamanShift, A, B, C, ...
    headers = ["RamanShift"] + [_excel_col_name(i) for i in range(n_samples)]
    header_line = "\t".join(headers)

    # Combined data shape = (n_points, 1 + n_samples)
    combo_data = np.column_stack(
        [shift[0]] + [spectra[i] for i in range(n_samples)]
        )

    np.savetxt(
        combo_path,
        combo_data,
        fmt="%.6f" + "\t%.6f" * n_samples,
        header=header_line,
        comments=""
    )




