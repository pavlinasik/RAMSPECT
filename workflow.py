# -*- coding: utf-8 -*-
"""
RAMSPECT example workflow

Fill in paths below. Typical use:
1) Run preprocessing from raw reaction + blank spectra
2) Run MCR on the averaged spectrum output
3) Plot MCR results from an output folder
"""

import Raman4React

#%% Directories management
# Raw input folders
spectral_data_path=r""
blank_data_path=r""
root_folder=r""

# Reference pure spectra (tab-delimited with header: RamanShift, TCP, DCP, GLY)
reference_path=r""

# Output base + run folder
export_path=r""
output_folder="run_01"

preps_path = None # set only if preps=False

#%% Settings

# Run modes
preps    = True   # preprocessing branch
mcr      = True   # analysis branch
autorun  = True   # run immediately upon init
multiple = False  # process all reactions x blanks combinations

# Folder detection tokens (for multiple = True)
reaction_token = "reaction"
blank_token    = "blank"
    
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
header   = "0"    # "0" for numbers in column header, "A" for letters (excel style)
messages = True
  
#%% Initialization
reactor = Raman4React.Reactions(
    # root folder mode
    root_folder=root_folder,
    multiple=multiple,

    # These can be left as "" or any placeholder if the class still requires 
    # them.
    spectral_data_path=spectral_data_path,
    blank_data_path=blank_data_path,

    reference_path=reference_path,
    export_path=export_path,
    output_folder=output_folder,

    # preps_path can be None; the class should set it to average_spectra 
    # automatically
    preps_path=preps_path,

    preps=preps,         # preprocessing must run to generate combo averages
    mcr=mcr,             # explicitly run MCR after preps
    autorun=autorun,

    # folder detection tokens (defaults shown)
    reaction_token=reaction_token,
    blank_token=blank_token,

    # export config
    header=header,

    # preprocessing config
    c_lower=c_lower, c_upper=c_upper,
    s_lower=s_lower, s_upper=s_upper,
    range1=range1, range2=range2,

    # MCR preprocessing options
    apply_baseline_correction=apply_baseline_correction,
    apply_normalization=apply_normalization,
    use_polynomial_baseline=use_polynomial_baseline,
    poly_order=poly_order,
    anchor_points=anchor_points,
    rfc_radius=rfc_radius,

    show=show, save=save, export=export, messages=messages,

    # optional: keep individual combination outputs (debug)
    keep_combos=True,   
)

if not autorun:
    reactor.run()

#%% Plotting
#Plot results from an output folder containing concentrations.csv etc.
if export:
    reactor.plot_mcr(
        input_folder=reactor.out_dir+"\spectra_combo",
        ncols=3, dpi=300, show_R2=True
    )
