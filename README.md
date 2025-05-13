<p align="center">
  <img src="https://github.com/alexandergagliano/Prost/blob/main/static/logo.png" style="width: 50%;" alt="Logo">
</p>

## Prost is a library for rapidly associating transients with their host galaxies. 
The code calculates the posterior probability that each galaxy in a search region is the true host galaxy, by considering
the transient's fractional offset, redshift (or the prior for the survey), and brightness. The code supports the following catalogs:
* <a href="https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.1403D">GLADE+</a>  
* <a href="https://ui.adsabs.harvard.edu/abs/2019AJ....157..168D">DECaLS (DR9 and DR10)</a>
* <a href="https://ui.adsabs.harvard.edu/abs/2016arXiv161205560C">PanSTARRS (DR1 and DR2)</a>
* <a href="https://ui.adsabs.harvard.edu/abs/2024PASA...41...61O">SkyMapper (DR4)</a>

The code also estimates the posterior probability that the true host lies outside of the search cone or is missing from the search catalog. The priors and likelihoods for each property can be customized according to the transient survey. Using the code is straightforward:

```
import pandas as pd
from astro_prost import associate_sample
from scipy.stats import gamma, halfnorm, uniform

# define a transient catalog 
transient_catalog = pd.DataFrame({
    'name': ['MyTransient'],
    'ra': [237.1981094],
    'dec': [9.2000414]
})

# define a set of catalogs to search -- options are glade, decals, panstarrs, and skymapper
catalogs = ["decals"]

# define priors and likelihoods
priorfunc_offset = uniform(loc=0, scale=10)
likefunc_offset = gamma(a=0.75)

priors = {"offset": priorfunc_offset}
likes = {"offset": likefunc_offset}

# associate
hosts = \
    associate_sample(
        transient_catalog,
        priors=priors,
        likes=likes,
        catalogs=catalogs,
        name_col='name',
        coord_cols=('ra', 'dec'),
        save=False
)

```

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/848607796.svg)](https://doi.org/10.5281/zenodo.15397885)
[![PyPI](https://img.shields.io/pypi/v/prost?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/prost/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/alexandergagliano/prost/smoke-test.yml)](https://github.com/alexandergagliano/prost/actions/workflows/smoke-test.yml)
[![Read The Docs](https://img.shields.io/readthedocs/astro_prost)](https://astro-prost.readthedocs.io/)

If you find Prost useful for your work, please cite the Zenodo release: 

```
@software{Gagliano2025_Prost,
  author       = {Alex Gagliano and
                  Kaylee de Soto and
                  Adam Boesky and
                  T. Andrew Manning},
  title        = {alexandergagliano/Prost: v1.2.11},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.2.11},
  doi          = {10.5281/zenodo.15397886},
  url          = {https://doi.org/10.5281/zenodo.15397886},
}
```

Questions? Functionality you'd like to see? [Report an issue](https://github.com/alexandergagliano/Prost/issues/new?title=New%20Issue&body=Please%20describe%20the%20issue%20here) or reach out at gaglian2[at]mit.edu.

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).
