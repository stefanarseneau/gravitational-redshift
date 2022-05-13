# gravitational-redshift
---

Python notebooks designed for the identification of white dwarf main sequence wide binaries and calculation of their gravitational redshifts.
* `widebinaries.ipynb` : Data selection. Uses information from [Kepler et al, 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.2169K/abstract) and wide binaries from [El Badry et al, 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.2269E/abstract) to select a sample of wide binaries containing one white dwarf and one main sequence star. Needs to be proofed & validated.
* `compare_binaries.ipynb` : Working with spectra from `widebinaries.ipynb`, very early stages.
* `star_redshift.ipynb` and `wd_redshift.ipynb` : Playing around with spectra and fitting techniques for atmospheric parameters and radial velocities.

Notes:
* `data/white_dwarves.csv` contains all the white dwarves & associated spectra from Kepler.
* `data/wide_binaries.csv` contains all the aboce information for stars in wide binaries as well as information on their companions.
