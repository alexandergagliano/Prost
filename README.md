# Prost -- A probabilistic Host-Galaxy Association Method
## Prost is developed for rapid association of transients. 
The code calculates the posterior probability that each galaxy in a search region is the true host galaxy, by considering
the transient's fractional offset, redshift (or the prior for the survey), and brightness. The code queries the following catalogs in order: 
* GLADE (CITE)
* DECaLS (CITE)
* PanSTARRS Data Release 2
The code also estimates the posterior probability that the true host lies outside of the search cone or is missing from the search catalog. The priors and likelihoods for each property can be customized according to the transient survey.

## TODO list: 
# include delta-mag prior & likelihood --  host is probably not more than 2mag dimmer than the supernoa at peak
# add in brightnesses of the host galaxies
# write out first host as second-most-likely host if presumed hostless
# write out probability of hostless vs first host vs second host
# add in SDSS next instead of 3Pi? P_unobserved as a threshold for moving between catalogs
# see if shreds have low photo-zs with high errors
