import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
import requests
import pytest

from astro_prost.associate import associate_sample
from astro_prost.helpers import SnRateAbsmag
from astropy.coordinates import SkyCoord
import sys
if sys.version_info >= (3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources
import astropy.units as u
import time
import numpy as np

def test_ngc():
    np.random.seed(42)

    # Use SN2020oi in M100/NGC4321 instead
    transient_catalog = pd.DataFrame({
        'IAUID': ['SN2020oi'],
        'RA': [185.7288697],
        'Dec': [15.8235796],
        'redshift': [0.005240]  # M100 redshift
    })

    # define priors for properties
    priorfunc_z = halfnorm(loc=0.0001, scale=0.5)
    priorfunc_offset = uniform(loc=0, scale=10)
    priorfunc_absmag = uniform(loc=-30, scale=20)

    likefunc_offset = gamma(a=0.75)
    likefunc_absmag = SnRateAbsmag(a=-30, b=-10)

    priors = {"offset": priorfunc_offset, "absmag": priorfunc_absmag, "redshift": priorfunc_z}
    likes = {"offset": likefunc_offset, "absmag": likefunc_absmag}

    # set up properties of the association run
    verbose = 2
    parallel = False
    save = False
    progress_bar = False
    cat_cols = True  # Need catalog columns to get host_name

    # list of catalogs to search -- options are (in order) glade, decals, panstarrs
    catalogs = ["glade"]

    # the columns containing the transient names, coordinates, and redshift info
    name_col = "IAUID"
    coord_cols = ("RA", "Dec")
    redshift_col = 'redshift'

    # cosmology can be specified, else flat lambdaCDM is assumed with H0=70, Om0=0.3, Ode0=0.7
    try:
        hostTable = associate_sample(
        transient_catalog,
        priors=priors,
        likes=likes,
        catalogs=catalogs,
        name_col=name_col,
        coord_cols=coord_cols,
        redshift_col=redshift_col,
        parallel=parallel,
        verbose=verbose,
        save=save,
        progress_bar=progress_bar,
        cat_cols=cat_cols,
    )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pytest.skip("Service timeout")

    # Check that we got a host and have the host_name column
    assert 'host_name' in hostTable.columns
    assert len(hostTable) > 0
    assert hostTable['host_name'].values[0] is not None
