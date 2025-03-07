import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
import pytest
from astro_prost.associate import associate_sample
from astro_prost.helpers import SnRateAbsmag, is_service_available
from astropy.coordinates import SkyCoord
import sys
if sys.version_info >= (3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources
import astropy.units as u
import time
import numpy as np

@pytest.mark.skipif(
    not is_service_available("https://catalogs.mast.stsci.edu"),
    reason="Remote service is unavailable"
)
def test_associate_parallel():
    np.random.seed(6)

    pkg = pkg_resources.files("astro_prost")
    pkg_data_file = pkg / "data" / "ZTFBTS_TransientTable.csv"
    with pkg_resources.as_file(pkg_data_file) as csvfile:
        transient_catalog = pd.read_csv(csvfile)
    transient_catalog = transient_catalog.sample(n=5)

    # define priors for properties
    priorfunc_z = halfnorm(loc=0.0001, scale=0.5)
    priorfunc_offset = uniform(loc=0, scale=10)
    priorfunc_absmag = uniform(loc=-30, scale=20)

    likefunc_offset = gamma(a=0.75)
    likefunc_absmag = SnRateAbsmag(a=-30, b=-10)

    priors = {"offset": priorfunc_offset, "absmag": priorfunc_absmag, "redshift": priorfunc_z}
    likes = {"offset": likefunc_offset, "absmag": likefunc_absmag}

    # set up properties of the association run
    verbose = 1
    save = False
    progress_bar = False
    cat_cols = False

    # list of catalogs to search -- options are (in order) glade, decals, panstarrs
    catalogs = ["decals"]

    # the columns containing the transient names, coordinates, and redshift info
    name_col = "IAUID"
    coord_cols = ("RA", "Dec")
    redshift_col = 'redshift'

    # cosmology can be specified, else flat lambdaCDM is assumed with H0=70, Om0=0.3, Ode0=0.7
    hostTable_serial  = associate_sample(
        transient_catalog,
        priors=priors,
        likes=likes,
        catalogs=catalogs,
        name_col=name_col,
        coord_cols=coord_cols,
        redshift_col=redshift_col,
        parallel=False,
        verbose=verbose,
        save=save,
        progress_bar=progress_bar,
        cat_cols=cat_cols,
    )

    # cosmology can be specified, else flat lambdaCDM is assumed with H0=70, Om0=0.3, Ode0=0.7
    hostTable_parallel = associate_sample(
        transient_catalog,
        priors=priors,
        likes=likes,
        catalogs=catalogs,
        name_col=name_col,
        coord_cols=coord_cols,
        redshift_col=redshift_col,
        parallel = True,
        verbose=verbose,
        save=save,
        progress_bar=progress_bar,
        cat_cols=cat_cols,
    )
  
    hostIDs_serial = np.sort(hostTable_serial['host_objID'].values)
    hostIDs_parallel = np.sort(hostTable_parallel['host_objID'].values)

    assert np.allclose(hostIDs_serial, hostIDs_parallel, equal_nan=True)
