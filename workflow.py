# -*- coding: utf-8 -*-
"""
RAMSPECT example workflow

Fill in paths below. Typical use:
1) Run preprocessing from raw reaction + blank spectra
2) Run MCR on the averaged spectrum output
3) Plot MCR results from an output folder
"""

import os
import Raman4React
import time

#%% Directories management
# Raw input folders
spectral_data_path=r""
blank_data_path=r""

# Reference pure spectra (tab-delimited with header: RamanShift, TCP, DCP, GLY)
reference_path=r""

# Output base + run folder
export_path=r""
output_folder="output"

# If you already have preprocessed/averaged spectra and want MCR only:
preps_path = os.path.join(export_path, output_folder)

#%% Settings

# Run modes
preps   = True   # preprocessing branch
autorun = True   # run immediately upon init

# Preprocessing (INDEX-based)
c_lower, c_upper = 23, 571
s_lower, s_upper = 200, 210

# Ratio ranges (cm^-1)
range1 = (270, 280)
range2 = (200, 210)

# MCR preprocessing (cm^-1 anchors)
apply_baseline_correction = True
apply_normalization       = False
use_polynomial_baseline   = True
poly_order = 4
anchor_points = [2700, 2710, 2720, 2730, 2740,
                 2750, 2760, 2770, 2780, 2790, 2800,
                 3050, 3060, 3070, 3080, 3090, 3100]
rfc_radius = 150  # used only if use_polynomial_baseline=False

# Output control
show     = False
save     = True
export   = True
messages = True
  
#%% Initialization
t0 = time.time()
reactor = Raman4React.Reactions(
    spectral_data_path=spectral_data_path,
    blank_data_path=blank_data_path,
    reference_path=reference_path,
    export_path=export_path,
    output_folder=output_folder,
    preps_path=preps_path,

    preps=preps,
    autorun=autorun,

    c_lower=c_lower, c_upper=c_upper,
    s_lower=s_lower, s_upper=s_upper,
    range1=range1, range2=range2,

    apply_baseline_correction=apply_baseline_correction,
    apply_normalization=apply_normalization,
    use_polynomial_baseline=use_polynomial_baseline,
    poly_order=poly_order,
    anchor_points=anchor_points,
    rfc_radius=rfc_radius,

    show=show, save=save, export=export, messages=messages,
)

if autorun==False:
   reactor.run()

print(f"[INFO] Done. Elapsed time: {time.time()-t0} s.")
#%% Plotting
# Plot results from an output folder containing concentrations.csv etc.
reactor.plot_mcr(
    input_folder=r"",
    ncols=4, dpi=300, show_R2=True
)
