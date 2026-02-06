# \# RAMspect

# 

# Raman spectra analysis and NNLS-MCR workflow.

# 

# This package supports:

# \- preprocessing of raw Raman spectra (cropping, smoothing, blank subtraction),

# \- optional averaging across reaction × blank combinations,

# \- NNLS-MCR analysis using pure reference spectra,

# \- automated plotting of concentrations and reconstructions.

# 

# ---

# 

# \## Required inputs for MCR

# 

# The following inputs are \*\*mandatory\*\* when running MCR:

# 

# \- \*\*`reference\_path`\*\*  

# &nbsp; Path to a pure spectra `.txt` file containing columns:  

# &nbsp; `RamanShift`, `TCP`, `DCP`, `GLY`

# 

# \- \*\*`export\_path`\*\*  

# &nbsp; Base directory where results will be written

# 

# \- \*\*`output\_folder`\*\*  

# &nbsp; Name of the run directory inside `export\_path`  

# &nbsp; (recommended short name, e.g. `run\_01`)

# 

# ---

# 

# \## General rules

# 

# \- `reference\_path` is \*\*always required\*\* for `mcr=True`

# \- `export\_path + output\_folder` are required if `export=True`

# \- `preps\_path` is used when preprocessing is skipped

# \- `root\_folder` is only used when `multiple=True`

# 

# ---

# 

# \## Quick decision table

# 

# | Situation | `multiple` | `preps` | `mcr` | reaction / blank paths | `root\_folder` | `preps\_path` | `reference\_path` |

# |---------:|:----------:|:-------:|:-----:|:----------------------:|:-------------:|:------------:|:----------------:|

# | \*\*1)\*\* One raw reaction–blank → full pipeline | ❌ | ✅ | ✅ | ✅ yes | ❌ | optional | ✅ |

# | \*\*2)\*\* One preprocessed pair → MCR only | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |

# | \*\*3)\*\* Many raw reactions + blanks → preps + avg + MCR | ✅ | ✅ | ✅ | ❌ | ✅ | optional | ✅ |

# | \*\*4)\*\* Many processed combos → avg + MCR | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |

# | \*\*5)\*\* Preprocess one pair only (no MCR) | ❌ | ✅ | ❌ | ✅ yes | ❌ | optional | ❌ |

# | \*\*6)\*\* Preprocess many reactions + blanks only | ✅ | ✅ | ❌ | ❌ | ✅ | optional | ❌ |

# 

# ---

