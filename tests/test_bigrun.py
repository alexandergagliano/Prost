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

def test_bigrun():
    np.random.seed(18)

    pkg = pkg_resources.files("astro_prost")
    pkg_data_file = pkg / "data" / "ZTFBTS_TransientTable.csv"
    with pkg_resources.as_file(pkg_data_file) as csvfile:
        transient_catalog = pd.read_csv(csvfile)

    transient_catalog = transient_catalog.sample(n=10)

    # define priors for properties
    priorfunc_offset = uniform(loc=0, scale=10)
    likefunc_offset = gamma(a=0.75)

    priors = {"offset": priorfunc_offset}
    likes = {"offset": likefunc_offset}

    # set up properties of the association run
    verbose = 0
    parallel = True
    save = False
    progress_bar = False
    cat_cols = False

    # list of catalogs to search -- options are (in order) glade, decals, panstarrs
    catalogs = ["glade"]

    # the columns containing the transient names, coordinates, and (optionally) redshift info
    name_col = "IAUID"
    coord_cols = ("RA", "Dec")

    # cosmology can be specified, else flat lambdaCDM is assumed with H0=70, Om0=0.3, Ode0=0.7
    hostTable = associate_sample(
        transient_catalog,
        run_name="bigrun_test",
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

    assert len(hostTable) > 8
