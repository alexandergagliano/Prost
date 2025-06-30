import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
from astro_prost.associate import associate_sample
from astro_prost.helpers import SnRateAbsmag
from astropy.coordinates import SkyCoord
import sys
if sys.version_info >= (3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources
import astropy.units as u
import numpy as np
import time

def test_rsp():
    np.random.seed(18)

    transient_catalog = pd.DataFrame({'IAUID':['SN1'], 'RA':[61.804905], 'Dec':[-36.058397]}, index=[0])

    # define priors for properties
    priorfunc_redshift = halfnorm(loc=0.0001, scale=0.5)
    priorfunc_offset = uniform(loc=0, scale=10)
    priorfunc_absmag = uniform(loc=-30, scale=20)

    likefunc_offset = gamma(a=0.75)
    likefunc_absmag = SnRateAbsmag(a=-30, b=-10)

    priors = {"offset": priorfunc_offset}
    likes = {"offset": likefunc_offset}

    # set up properties of the association run
    verbose = 1
    parallel = False
    save = False
    progress_bar = False
    cat_cols = False

    # list of catalogs to search -- options are glade, decals, panstarrs, and rubin
    catalogs = ["rubin"]

    # the columns containing the transient names, coordinates, and (optionally) redshift info
    name_col = "IAUID"
    coord_cols = ("RA", "Dec")

    # cosmology can be specified, else flat lambdaCDM is assumed with H0=70, Om0=0.3, Ode0=0.7
    hostTable = associate_sample(
        transient_catalog,
        run_name="rubin_test",
        priors=priors,
        likes=likes,
        catalogs=catalogs,
        parallel=parallel,
        verbose=verbose,
        save=save,
        name_col=name_col,
        coord_cols=coord_cols,
        log_path='./',
        progress_bar=progress_bar,
        cat_cols=cat_cols,
    )
 
   
    assert hostTable['host_objID'].values[0] == '1651536833663761248'
