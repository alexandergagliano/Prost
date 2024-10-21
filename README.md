
# Prost: Probabilistic Host Galaxy Association
## Prost is a code for rapidly associating transients with their host galaxies. 
The code calculates the posterior probability that each galaxy in a search region is the true host galaxy, by considering
the transient's fractional offset, redshift (or the prior for the survey), and brightness. The code queries the following catalogs in order: 
* <a href="https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.1403D">GLADE+</a>  
* <a href="https://ui.adsabs.harvard.edu/abs/2019AJ....157..168D">DECaLS</a>
* <a href="https://ui.adsabs.harvard.edu/abs/2016arXiv161205560C">PanSTARRS Data Release 2</a>

The code also estimates the posterior probability that the true host lies outside of the search cone or is missing from the search catalog. The priors and likelihoods for each property can be customized according to the transient survey.

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/prost?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/prost/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/alexandergagliano/prost/smoke-test.yml)](https://github.com/alexandergagliano/prost/actions/workflows/smoke-test.yml)
[![Read The Docs](https://img.shields.io/readthedocs/astro_prost)](https://astro-prost.readthedocs.io/)

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).
