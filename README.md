# RAMspect

Raman spectra analysis. 

Below is a user-guide.





Required inputs for MCR

-----------------------

* `reference\_path` : path to the pure spectra .TXT with columns RamanShift, TCP, DCP, GLY
* `export\_path`    : base folder where results are written
* `output\_folder`  : name of a run folder inside export\_path (recommend a short name like "run\_01")





Quick decision table

--------------------

* `reference\_path` required for mcr analysis
* `export\_path` + `output\_folder` required if `export=True`



| Situation                                       | `multiple` | `preps`  | `mcr` | reaction/blank paths?  | `root\_folder`?  | `preps\_path?`  | `reference\_path?`  |

| ----------------------------------------------: | ---------: | -------: | ----: | ---------------------: | --------------: | -------------: | -----------------: |

| 1) One pair raw → full pipeline                 |  False     | True     | True  | ✅ yes                 | ❌             | optional       | ✅ yes             |

| 2) One pair already preprocessed → MCR only     |  False     | False    | True  | ❌                     | ❌             | ✅ yes         | ✅ yes            |     

| 3) Many reactions+blanks raw → preps+avg+MCR    |  True      | True     | True  | ❌                     | ✅ yes         | optional       | ✅ yes             |

| 4) Many combos already processed → avg+MCR      |  True      | False    | True  | ❌                     | ❌             | ✅ yes         | ✅ yes            |

| 5) Preprocess one pair raw, no MCR              |  False     | True     | False | ✅ yes                 | ❌             | optional       | ❌                 |

| 6) Preprocess Many reactions+blanks raw, no MCR |  True      | True     | False | ❌                     | ✅ yes         | optional       | ❌                 |



