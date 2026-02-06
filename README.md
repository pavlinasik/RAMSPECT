# RAMspect




Raman spectra analysis and NNLS-MCR workflow. RAMspect provides a complete pipeline for:



- preprocessing Raman spectra (cropping, smoothing, blank subtraction)

- averaging reaction × blank combinations

- NNLS-MCR analysis using pure reference spectra

- visualization of concentrations and reconstructions





## Required inputs for MCR



The following inputs are mandatory when running MCR:



- `reference_path`: path to a `.txt` file with columns RamanShift, TCP, DCP, GLY

- `export_path`: base directory where results will be written

- `output_folder`: name of the run directory inside `export_path`





## General rules



- `reference_path` is always required when `mcr=True`

- `export_path` and `output_folder` are required when `export=True`

- `preps_path` is used when preprocessing is skipped

- `root_folder` is only used when `multiple=True`





## Quick decision table



The table below summarizes which inputs and flags are required for common workflows.



| Situation | `multiple` | `preps` | `mcr` | reaction / blank paths | `root_folder` | `preps_path` | `reference_path` |

|---|---|---|---|---|---|---|---|

| **1)** One raw reaction–blank → full pipeline | ❌ | ✅ | ✅ | ✅ | ❌ | optional | ✅ |

| **2)** One preprocessed pair → MCR only | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |

| **3)** Many raw reactions + blanks → preps + avg + MCR | ✅ | ✅ | ✅ | ❌ | ✅ | optional | ✅ |

| **4)** Many processed combinations → avg + MCR | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |

| **5)** Preprocess one raw pair only (no MCR) | ❌ | ✅ | ❌ | ✅ | ❌ | optional | ❌ |

| **6)** Preprocess many reactions + blanks only | ✅ | ✅ | ❌ | ❌ | ✅ | optional | ❌ |



