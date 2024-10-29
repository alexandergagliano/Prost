import pandas as pd
import numpy as np
from scipy.stats import gamma, halfnorm, uniform

from astro_prost.associate import associate_sample, prepare_catalog
from astro_prost.helpers import SnRateAbsmag

source = "ZTF BTS"

transient_catalog = pd.read_csv(
    "/Users/alexgagliano/Documents/Research/multimodal-supernovae/data/ZTFBTS/ZTFBTS_TransientTable.csv"
)

#only take the first 10 events
transient_catalog = transient_catalog.sample(n=10)

# define priors for properties
priorfunc_z = halfnorm(loc=0.0001, scale=0.5)

# if you want the redshift prior to be based
# on an observed distribution of transients within a given absmag range
# transients are uniformly distributed in cosmological volume
# between zmin and zmax and the subset
# peaking above mag_cutoff sets the z distribution
# cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# priorfunc_z = PriorzObservedTransients(z_min=0, z_max=1, mag_cutoff=19,
#       absmag_mean=-19, absmag_min=-24, absmag_max=-17, cosmo=cosmo)

# look at your distribution (only available for the above experiment)
# priorfunc_z.plot()

priorfunc_offset = uniform(loc=0, scale=10)
priorfunc_absmag = uniform(loc=-30, scale=20)

likefunc_offset = gamma(a=0.75)
likefunc_absmag = SnRateAbsmag(a=-30, b=-10)

priors = {"offset": priorfunc_offset, "absmag": priorfunc_absmag, "z": priorfunc_z}
likes = {"offset": likefunc_offset, "absmag": likefunc_absmag}

# set up properties of the association run
verbose = 2
parallel = False
save = False
# if not parallel, results can be returned directly
progress_bar = False
cat_cols = True

# list of catalogs to search -- options are (in order) glade, decals, panstarrs
catalogs = ["decals"]

# the name of the coord columns in the dataframe
transient_coord_cols = ("RA", "Dec")

# the column containing the transient names
transient_name_col = "IAUID"

transient_catalog = prepare_catalog(
    transient_catalog, transient_name_col=transient_name_col, transient_coord_cols=transient_coord_cols
)

# cosmology can be specified, else flat lambdaCDM is assumed with H0=70, Om0=0.3, Ode0=0.7
associate_sample(
    transient_catalog,
    priors=priors,
    likes=likes,
    catalogs=catalogs,
    parallel=parallel,
    verbose=verbose,
    save=save,
    progress_bar=progress_bar,
    cat_cols=cat_cols,
)
