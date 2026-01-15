# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 13:00:07 2026

@author: p-sik
"""

import Raman4React
import matplotlib.pyplot as plt


reactor = Raman4React.Reactions(
    ### REQUIRED I/O ----------------------------------------------------------
    spectral_data_path=r"C:\_isibrno\help_ramanovci\project03_subtract_blank\36",
    blank_data_path=r"C:\_isibrno\help_ramanovci\project03_subtract_blank\36 blank",

    # Used only when preps=False (reference/MCR branch)
    reference_path=r"C:\_isibrno\help_ramanovci\project03_subtract_blank\neat\All_neat.txt",

    # Output / folders
    export_path=r"C:\MY_PYLIB\RAMSPECT",
    output_folder="test",

    # Reaction type label (used mainly for your naming/logic)
    reaction_type="DCPdegradation",

    ### Execution switches ----------------------------------------------------
    preps=True,        # True = preprocessing branch, False = reference/MCR branch
    autorun=True,      # if True -> runs automatically in __init__

    show=False,        # True = show plots (interactive backend)
    save=True,         # True = save figures (png) + other outputs
    export=True,       # True = export spectra / tables to files

    messages=True,     # print progress info

    ### Feature toggles for preps=True branch ---------------------------------
    specs=True,
    ratio=True,
    averages=True,

    # Used only when preps=False branch
    preps_path=r"C:\MY_PYLIB\RAMSPECT\test",

    # Crop range for extract_data (preps=True only) ---------------------------
    c_lower=23,
    c_upper=571,

    # Integration/specific region (preps=True only) ---------------------------
    s_lower=200,
    s_upper=210,

    # Ratio ranges (preps=True only) ------------------------------------------
    range1=(270, 280),
    range2=(200, 210),

    # Preprocessing parameters (if used in your processing pipeline) ----------
    apply_baseline_correction=True,
    apply_normalization=False,

    # polynomial baseline configuration
    use_polynomial_baseline=True,
    poly_order=4,

    anchor_points=[
        2700, 2710, 2720, 2730, 2740, 2750, 2760, 2770, 2780, 2790, 2800,
        3050, 3060, 3070, 3080, 3090, 3100
    ],
    
    rcf_radius = 150,
)

# if autorun=False:
# reactor.run()

#%%
reactor.plot_mcr(
    input_folder=r"C:\MY_PYLIB\RAMSPECT\test\average_spectra1",
    ncols=4, dpi=300, show_R2=True
    )