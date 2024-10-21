
.. astro_prost documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========================================================================================

Pröst is a code for host-galaxy identification of extragalactic transients. It's 
fast, probabilistic, and highly customizable. 

Pröst was developed to improve the quality of associations using prior information about 
the survey being conducted and the nature of the transients targeted. The user defines 
priors on the redshift and absolute magnitude of an expected host, and on the fractional 
offset (angular separation scaled by galaxy radius) between a discovered 
transient and its host. Likelihoods are also defined for these properties. Each 
candidate is assigned a posterior probability of hosting the transient in 
question, and the source with the highest probability is chosen as host. 

Each association is made multiple times while propagating uncertainties in each candidate's 
photometric and morphological properties, thereby providing uncertainty estimates on each 
final association. The probability that the true host was missed is also provided. 

Multiple catalogs are currently supported: 

- `GLADE+ v1 <https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII/275>`_
- `The Dark Energy Camera Legacy Survey (DECaLS) Data Release 9 <https://www.legacysurvey.org/dr9/description/>`_
- `Pan-STARRS Data Release 2 <https://outerspace.stsci.edu/display/PANSTARRS/>`_

Associations with Pröst can be made in serial or parallelized across multiple cores, 
making it scalable for the next generation of synoptic surveys.

Installation
========================================================================================

Installation can be done via pip (ideally in a conda environment): 

.. code-block:: bash

   >> conda create env -n <env_name> python=3.10
   >> conda activate <env_name>
   >> pip install astro-prost

Once installed, head over to the Notebooks tab for a tutorial on how to use the code.

.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Notebooks <notebooks>
