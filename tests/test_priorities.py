import pandas as pd
from scipy.stats import gamma, halfnorm, uniform

from astro_prost.associate import associate_sample, prepare_catalog
from astro_prost.helpers import SnRateAbsmag
from astropy.coordinates import SkyCoord
import importlib.resources as pkg_resources
import astropy.units as u
import time
import numpy as np

def test_cat_priority():
    np.random.seed(18)

    pkg = pkg_resources.files("astro_prost")
    pkg_data_file = pkg / "data" / "ZTFBTS_TransientTable.csv"
    with pkg_resources.as_file(pkg_data_file) as csvfile:
        transient_catalog = pd.read_csv(csvfile)
    transient_catalog = transient_catalog[transient_catalog['IAUID'] == 'SN2022yei']

    # define priors for properties
    priorfunc_z = halfnorm(loc=0.0001, scale=0.5)
    priorfunc_offset = uniform(loc=0, scale=10)
    priorfunc_absmag = uniform(loc=-30, scale=20)

    likefunc_offset = gamma(a=0.75)
    likefunc_absmag = SnRateAbsmag(a=-30, b=-10)

    priors = {"offset": priorfunc_offset}
    likes = {"offset": likefunc_offset}

    # set up properties of the association run
    verbose = 2
    parallel = False
    save = False
    progress_bar = False
    cat_cols = False

    # list of catalogs to search -- options are (in order) glade, decals, panstarrs
    catalogs = ["glade", "decals"]
    cat_priority = {"decals":1, "glade":2}

    # the name of the coord columns in the dataframe
    transient_coord_cols = ("RA", "Dec")

    # the column containing the transient names
    transient_name_col = "IAUID"

    transient_catalog = prepare_catalog(
        transient_catalog, transient_name_col=transient_name_col, transient_coord_cols=transient_coord_cols
    )

    # cosmology can be specified, else flat lambdaCDM is assumed with H0=70, Om0=0.3, Ode0=0.7
    hostTable = associate_sample(
        transient_catalog,
        run_name="decals_test",
        priors=priors,
        likes=likes,
        catalogs=catalogs,
        cat_priority=cat_priority,
        parallel=parallel,
        verbose=verbose,
        save=save,
        log_path='./',
        progress_bar=progress_bar,
        cat_cols=cat_cols,
    )

    assert (hostTable['best_cat'].values[0] == 'decals') and (hostTable['best_cat_release'].values[0] == 'dr9')

