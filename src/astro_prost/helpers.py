import os
import time
import sys
if sys.version_info >= (3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources
import pickle
import requests
import matplotlib.pyplot as plt
import numpy as np
#hacky monkey-patch for python 3.8
if not hasattr(np, 'int'):
    np.int = int
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord, match_coordinates_sky
from astropy.io import ascii
from dl import queryClient as qC
from scipy import stats as st
from scipy.integrate import quad
from scipy.stats import halfnorm, norm, gaussian_kde, rv_continuous
from io import StringIO
import pandas as pd
import logging
import re

from .photoz_helpers import evaluate, load_lupton_model, preprocess

# Precision & default values
PROB_FLOOR = np.finfo(float).eps
DUMMY_FILL_VAL = -999

# Conversion constants
RAD_TO_ARCSEC = 206265
MAX_RAD_DEG = 0.14 # 500'' search radius max

# Redshift & galaxy properties
REDSHIFT_FLOOR = 0.001  # minimum redshift of z=0.001
SIZE_FLOOR = 0.25  # 1 pixel, for Pan-STARRS
ABSMAG_FLOOR = -10  # guess at a minimum absolute magnitude for a galaxy?
OFFSET_FLOOR = SHAPE_FLOOR = 1.e-10

# Uncertainty floors & ceilings
SIGMA_ABSMAG_FLOOR = SIGMA_SIZE_FLOOR = SIGMA_REDSHIFT_FLOOR = 0.05  # 5% minimum uncertainty
SIGMA_ABSMAG_CEIL = SIGMA_SIZE_CEIL = SIGMA_REDSHIFT_CEIL = 0.5  # 50% maximum uncertainty

# Default settings for catalogs
DEFAULT_LIMITING_MAG = {"panstarrs": 22, "decals": 26, "glade": 17}
CATALOG_SHRED_SETTINGS = {
    "panstarrs": True,  # Only enable for Pan-STARRS by default
    "decals": False,
    "glade": False,
}

# Default paths
DEFAULT_DUST_PATH = "."

# Data Structure Definitions
PROP_DTYPES = [
        ("objID", np.int64),
        ("objID_info", "U20"),
        ("name", "U20"),
        ("ra", float),
        ("dec", float),
        ("redshift_samples", object),
        ("redshift_mean", float),
        ("redshift_std", float),
        ("redshift_posterior", float),
        ("redshift_info", "U10"),
        ("offset_samples", object),
        ("offset_mean", float),
        ("offset_std", float),
        ("offset_posterior", float),
        ("offset_info", str),
        ("absmag_samples", object),
        ("absmag_mean", float),
        ("absmag_std", float),
        ("absmag_posterior", float),
        ("absmag_info", "U10"),
        ("dlr_samples", object),
        ("total_posterior", float),
    ]

# Define a TRACE level below DEBUG -- for lengthy printouts
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

# Boilerplate trace method for the logger class
def trace(self, message, *args, **kws):
    """Logs a message at the TRACE level, which is lower than DEBUG (verbose = 3).

    Parameters
    ----------
    message : str
        The message to be logged.
    *args : tuple
        Additional positional arguments to be passed to the logging call.
    **kws : dict
        Additional keyword arguments to be passed to the logging call.

    Returns
    -------
    None
    """
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kws)

logging.Logger.trace = trace

def sanitize_input(cat_name):
    """Cleans up catalog names for use by Prost by removing spaces, underscores, and hyphens, and converting to lowercase.

    Parameters
    ----------
    cat_name : str
        The catalog name to be sanitized. Supported catalog names include 'decals', 'glade', and 'panstarrs'.

    Returns
    -------
    str
        The sanitized catalog name.
    """
    cat_name_clean = re.sub(r"[_\-\s]", "", cat_name)
    return cat_name_clean.lower()


def add_console_handler(logger, formatter):
    """Attach a console (stream) handler to the logger.

    Parameters
    ----------
    logger : logging.Logger
        The logger to which the console handler will be added.
    formatter : logging.Formatter
        The formatter used to format log messages.

    Returns
    -------
    None
    """
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def add_file_handler(logger, log_file, formatter):
    """Attach a file handler to the logger.

    Parameters
    ----------
    logger : logging.Logger
        The logger to which the file handler will be added.
    log_file : str
        The file path where log messages will be written.
    formatter : logging.Formatter
        The formatter used to format log messages.

    Returns
    -------
    None
    """
    log_path = os.path.dirname(log_file)
    os.makedirs(log_path, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def setup_logger(log_file=None, verbose=1, is_main=False):
    """
    Sets up a logger that logs messages to both the console and a file (if specified).

    Parameters
    ----------
    log_file : str, optional
        Path to the log file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("Prost_logger")

    # If logger already exists and has handlers, return it (prevents duplicates)
    if logger.hasHandlers():
        if log_file is None:
            return logger
        elif any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in logger.handlers):
            return logger

    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG, 3: TRACE_LEVEL}
    logger.setLevel(log_levels.get(verbose))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Main process: Set up log file and store its path
    log_file = log_file if is_main else os.environ.get('LOG_PATH_ENV')

    if log_file:
        add_file_handler(logger, log_file, formatter)
        if is_main:
            # Store log path for workers
            os.environ['LOG_PATH_ENV'] = log_file
    else:
        add_console_handler(logger, formatter)
    return logger

def fetch_decals_sources(search_pos, search_rad, cat_cols, calc_host_props, release='dr9'):
    """Queries the decals catalogs (https://www.legacysurvey.org/decamls/).

    Parameters
    ----------
    search_pos : astropy SkyCoord object
        Search position (transient location in this code).
    search_rad : astropy Angle object
        Size of the search radius.
    cat_cols : boolean
        If True, concatenates all columns from the catalog to the final output.
    calc_host_props : list
        Properties to calculate internally for each host ('offset', 'redshift', 'absmag').
    release : str
        Data release of the catalog; can be 'dr9' or 'dr10'.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the retrieved sources.

    """
    if release not in ["dr9", "dr10"]:
        raise ValueError(f"Invalid DECaLS version '{release}'. Please choose 'dr9' or 'dr10'.")

    if search_rad is None:
        search_rad = Angle(60 * u.arcsec)

    rad_deg = search_rad.deg

    result = qC.query(
        sql=f"""SELECT
        t.ls_id,
        t.shape_r,
        t.shape_r_ivar,
        t.shape_e1,
        t.shape_e1_ivar,
        t.shape_e2,
        t.shape_e2_ivar,
        t.ra,
        t.type,
        t.dec,
        t.dered_mag_r,
        t.mag_r,
        t.flux_r,
        t.flux_ivar_r,
        t.nobs_g,
        t.nobs_r,
        t.nobs_z,
        t.fitbits,
        t.ra_ivar,
        t.dec_ivar,
        t.dered_flux_r,
        pz.z_phot_mean,
        pz.z_phot_median,
        pz.z_phot_std,
        pz.z_spec
    FROM
        ls_{release}.tractor t
    INNER JOIN
        ls_{release}.photo_z pz
    ON
        t.ls_id= pz.ls_id
    WHERE
        q3c_radial_query(t.ra, t.dec, {search_pos.ra.deg:.5f}, {search_pos.dec.deg:.5f}, {rad_deg})
    AND (t.nobs_r > 0) AND (t.dered_flux_r > 0) AND (t.snr_r > 0)
    AND nullif(t.dered_mag_r, 'NaN') is not null AND (t.fitbits != 8192)
    AND ((pz.z_spec > 0) OR (pz.z_phot_mean > 0))"""
    )

    candidate_hosts = pd.read_csv(StringIO(result))
    candidate_hosts.rename(columns={'ls_id':'objID'}, inplace=True)

    if len(candidate_hosts) < 1:
        return None
    else:
        return candidate_hosts

def fetch_panstarrs_sources(search_pos, search_rad, cat_cols, calc_host_props, logger=None, release='dr2'):
    """Queries the panstarrs catalogs (https://catalogs.mast.stsci.edu/panstarrs/).

    Parameters
    ----------
    search_pos : astropy SkyCoord object
        Search position (transient location in this code).
    search_rad : astropy Angle object
        Size of the search radius.
    cat_cols : boolean
        If True, concatenates all columns from the catalog to the final output.
    calc_host_props : list
        Properties to calculate internally for each host ('offset', 'redshift', 'absmag').
    release : str
        Data release of the catalog; can be 'dr1' or 'dr2'.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the retrieved sources (after some basic quality cuts).

    """
    if release not in ["dr1", "dr2"]:
        raise ValueError(f"Invalid Pan-STARRS version '{release}'. Please choose 'dr1' or 'dr2'.")
    elif (release == 'dr1') and (('redshift' in calc_host_props) or ('absmag' in calc_host_props)):
        raise ValueError("Redshift estimation with Pan-STARRS data can only be done with release 'dr2'.")

    if logger is None:
        logger = setup_logger()

    if search_rad is None:
        search_rad = Angle(60 * u.arcsec)

    rad_deg = search_rad.deg
    if rad_deg > MAX_RAD_DEG:
        logger.warning("Search radius at this distance >500''! Reducing to ensure a fast pan-starrs query.")
        rad_dec = MAX_RAD_DEG

    # load table metadata to avoid a query
    pkg_data_file = pkg_resources.files('astro_prost') / 'data' / 'panstarrs_metadata.pkl'

    with pkg_resources.as_file(pkg_data_file) as metadata_path:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

    if not cat_cols:
        source_cols = [
            "objID",
            "raMean",
            "decMean",
            "gmomentXX",
            "rmomentXX",
            "imomentXX",
            "zmomentXX",
            "ymomentXX",
            "gmomentYY",
            "rmomentYY",
            "imomentYY",
            "zmomentYY",
            "ymomentYY",
            "gmomentXY",
            "rmomentXY",
            "imomentXY",
            "zmomentXY",
            "ymomentXY",
            "nDetections",
            "primaryDetection",
            "gKronRad",
            "rKronRad",
            "iKronRad",
            "zKronRad",
            "yKronRad",
            "gKronMag",
            "rKronMag",
            "iKronMag",
            "zKronMag",
            "yKronMag",
            "gKronMagErr",
            "rKronMagErr",
            "iKronMagErr",
            "zKronMagErr",
            "yKronMagErr",
        ]

        result = panstarrs_cone(metadata, search_pos.ra.deg, search_pos.dec.deg, rad_deg, columns=source_cols,
        release=release)
    else:
        result = panstarrs_cone(metadata, search_pos.ra.deg, search_pos.dec.deg, rad_deg, release=release)

    if not result:
        logging.debug(f"Found no pan-starrs {release} sources.")
        return None

    candidate_hosts = pd.read_csv(StringIO(result))

    if len(candidate_hosts) < 1:
        return None

    if ('redshift' in calc_host_props) or ('absmag' in calc_host_props):
        candidate_hosts = candidate_hosts.set_index("objID")

        #columns needed for photometric redshift inference
        photoz_cols = [
            "objID",
            "raMean",
            "decMean",
            "gFKronFlux",
            "rFKronFlux",
            "iFKronFlux",
            "zFKronFlux",
            "yFKronFlux",
            "gFPSFFlux",
            "rFPSFFlux",
            "iFPSFFlux",
            "zFPSFFlux",
            "yFPSFFlux",
            "gFApFlux",
            "rFApFlux",
            "iFApFlux",
            "zFApFlux",
            "yFApFlux",
            "gFmeanflxR5",
            "rFmeanflxR5",
            "iFmeanflxR5",
            "zFmeanflxR5",
            "yFmeanflxR5",
            "gFmeanflxR6",
            "rFmeanflxR6",
            "iFmeanflxR6",
            "zFmeanflxR6",
            "yFmeanflxR6",
            "gFmeanflxR7",
            "rFmeanflxR7",
            "iFmeanflxR7",
            "zFmeanflxR7",
            "yFmeanflxR7",
        ]

        result_photoz = panstarrs_cone(
            metadata,
            search_pos.ra.deg,
            search_pos.dec.deg,
            rad_deg,
            columns=photoz_cols,
            table="forced_mean",
            release=release,
        )

        candidate_hosts_pzcols = pd.read_csv(StringIO(result_photoz))

        candidate_hosts_pzcols = candidate_hosts_pzcols.set_index("objID")

        candidate_hosts = (
            candidate_hosts.join(candidate_hosts_pzcols, lsuffix="_DROP")
            .filter(regex="^(?!.*DROP)")
            .reset_index()
        )

    candidate_hosts.replace(DUMMY_FILL_VAL, np.nan, inplace=True)
    candidate_hosts.reset_index(inplace=True, drop=True)

    for prop in ['momentXX', 'momentYY', 'momentXY', 'KronRad', 'KronMag', 'KronMagErr']:
        prop_list = [f"{flt}{prop}" for flt in 'grizy']
        candidate_hosts[prop] = candidate_hosts[prop_list].median(axis=1)

    # some VERY basic filtering to say that it's confidently detected
    candidate_hosts = candidate_hosts[candidate_hosts["nDetections"] > 2]
    candidate_hosts = candidate_hosts[candidate_hosts["primaryDetection"] == 1]

    drop_cols = ['raMean', 'decMean']

    if 'absmag' in calc_host_props:
        drop_cols.append('KronMag')
    if 'offset' in calc_host_props:
        drop_cols.append('KronRad')

    candidate_hosts.dropna(subset=drop_cols, inplace=True)
    candidate_hosts.rename(columns={'raMean':'ra', 'decMean':'dec'}, inplace=True)
    candidate_hosts.reset_index(drop=True, inplace=True)

    return candidate_hosts

def calc_shape_props_panstarrs(candidate_hosts):
    """Wrapper to calculate the shape parameters for pan-starrs galaxies.

    Parameters
    ----------
    candidate_hosts : pandas.DataFrame
        Dataset containing sources with shape information.

    Returns
    -------
    tuple of np.ndarray
        (temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std),
        where:
          - temp_sizes: Semi-major axes (arcsec)
          - temp_sizes_std: Uncertainty in semi-major axes
          - a_over_b: axis ratio
          - a_over_b_std: Uncertainty in axis ratio
          - phi: Position angles (radians)
          - phi_std: Uncertainty in position angles

    """
    temp_sizes = candidate_hosts["KronRad"].values

    # assume some fiducial shape floor
    temp_sizes_std = SIGMA_SIZE_FLOOR * candidate_hosts["KronRad"].values
    temp_sizes_std = np.maximum(temp_sizes_std, SHAPE_FLOOR)  # Prevent division by zero

    temp_sizes_std[temp_sizes_std < (SIGMA_SIZE_FLOOR * temp_sizes)] = SIGMA_SIZE_FLOOR* temp_sizes[temp_sizes_std <  (SIGMA_SIZE_FLOOR * temp_sizes)]

    gal_u = candidate_hosts["momentXY"].values
    gal_q = candidate_hosts["momentXX"].values - candidate_hosts["momentYY"].values

    phi = 0.5 * np.arctan2(gal_u, gal_q)
    phi_std = 0.05 * np.abs(phi)
    phi_std = np.maximum(SHAPE_FLOOR, phi_std)
    kappa = gal_q**2 + gal_u**2
    kappa = np.minimum(kappa, 0.99)
    a_over_b = (1 + kappa + 2 * np.sqrt(kappa)) / (1 - kappa)
    a_over_b = np.clip(a_over_b, 0.1, 10)
    a_over_b_std = SIGMA_SIZE_FLOOR * np.abs(a_over_b)  # uncertainty floor

    #return result
    return temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std

def calc_shape_props_decals(candidate_hosts):
    """Wrapper to calculate the shape parameters for decals galaxies.

    Parameters
    ----------
    candidate_hosts : pandas.DataFrame
        Dataset containing sources with shape information.

    Returns
    -------
    tuple of np.ndarray
        (temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std),
        where:
          - temp_sizes: Semi-major axes (arcsec)
          - temp_sizes_std: Uncertainty in semi-major axes
          - a_over_b: axis ratio
          - a_over_b_std: Uncertainty in axis ratio
          - phi: Position angles (radians)
          - phi_std: Uncertainty in position angles
    """
    temp_sizes = candidate_hosts["shape_r"].values
    temp_sizes[temp_sizes < SIZE_FLOOR] = SIZE_FLOOR
    temp_sizes_ivar = np.maximum(1/(SIGMA_SIZE_FLOOR*temp_sizes)**2, candidate_hosts["shape_r_ivar"].values)
    temp_sizes_std = np.sqrt(1 / temp_sizes_ivar)

    temp_e1 = candidate_hosts["shape_e1"].astype(float).values
    temp_e1_ivar = np.maximum(1/(SIGMA_SIZE_FLOOR*temp_e1)**2, candidate_hosts["shape_e1_ivar"].astype(float).values)

    temp_e2 = candidate_hosts["shape_e2"].astype(float).values
    temp_e2_ivar = np.maximum(1/(SIGMA_SIZE_FLOOR*temp_e2)**2, candidate_hosts["shape_e2_ivar"].astype(float).values)

    temp_e1_std = np.sqrt(1.0 / temp_e1_ivar)
    temp_e2_std = np.sqrt(1.0 / temp_e2_ivar)

    mask_e1_floor = (temp_e1_std < SIGMA_SIZE_FLOOR * np.abs(temp_e1))
    temp_e1_std[mask_e1_floor] = SIGMA_SIZE_FLOOR * np.abs(temp_e1[mask_e1_floor])

    mask_e2_floor = (temp_e2_std < SIGMA_SIZE_FLOOR * np.abs(temp_e2))
    temp_e2_std[mask_e2_floor] = SIGMA_SIZE_FLOOR * np.abs(temp_e2[mask_e2_floor])

    # 3) Also clamp to a hard floor so that no std is < SHAPE_FLOOR
    temp_e1_std = np.maximum(temp_e1_std, SHAPE_FLOOR)
    temp_e2_std = np.maximum(temp_e2_std, SHAPE_FLOOR)

    # Calculate ellipticity and axis ratio for all samples
    e = np.sqrt(temp_e1**2 + temp_e2**2)
    e = np.maximum(e, SHAPE_FLOOR)

    a_over_b = (1 + e) / (1 - e)

    # Compute uncertainty in e (sigma_e)
    e_std = (1 / e) * np.sqrt(temp_e1**2 * temp_e1_std**2 + temp_e2**2 * temp_e2_std**2)

    # Compute uncertainty in a_over_b (sigma_a_over_b)
    a_over_b_std = (2 / (1 - e) ** 2) * e_std

    # Position angle and angle calculations for all samples
    phi = -np.arctan2(temp_e2, temp_e1) / 2

    # now propagate uncertainty from the shape params -- this is a bit messy because
    # it requires partial derivatives d/de1 and d/de2 of arctan2, but let's try:
    # d/de2(arctan2) = e1/(e1^2 + e2^2)
    # d/de1(arctan2) = -e2/(e1^2 + e2^2)

    # so d/de2(-arctan2/2)= -e1/2*(e1^2 + e2^2)
    # so d/de1(-arctan2/2)= e2/2*(e1^2 + e2^2)
    59000+2400000.5
    denom = temp_e1**2 + temp_e2**2
    denom = np.maximum(denom, SHAPE_FLOOR)

    partial_phi_e2 = -temp_e1 / (2.0 * denom)   # d(phi)/d(e2)
    partial_phi_e1 =  temp_e2 / (2.0 * denom)   # d(phi)/d(e1)

    # Now propagate uncertainties from e1 and e2
    phi_std = np.sqrt((partial_phi_e1**2) * (temp_e1_std**2) + (partial_phi_e2**2) * (temp_e2_std**2))

    # First clamp phi_std to be at least SIGMA_SIZE_FLOOR * |phi|
    phi_std = np.maximum(phi_std, SIGMA_SIZE_FLOOR * np.abs(phi))
    phi_std = np.maximum(phi_std, SHAPE_FLOOR)

    return temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std

def calc_shape_props_glade(candidate_hosts):
    """Wrapper to calculate the shape parameters for pan-starrs galaxies.

    Parameters
    ----------
    candidate_hosts : pandas.DataFrame
        Dataset containing sources with shape information.

    Returns
    -------
    tuple of np.ndarray
        (temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std),
        where:
          - temp_sizes: Semi-major axes (arcsec)
          - temp_sizes_std: Uncertainty in semi-major axes
          - a_over_b: axis ratio
          - a_over_b_std: Uncertainty in axis ratio
          - phi: Position angles (radians)
          - phi_std: Uncertainty in position angles
    """
    temp_pa = candidate_hosts["PAHyp"].values

    # assume no position angle for unmeasured gals
    temp_pa[temp_pa != temp_pa] = 0

    # (n) HyperLEDA decimal logarithm of the length of the projected major axis
    # of a galaxy at the isophotal level 25mag/arcsec2 in the B-band,
    # to semi-major half-axis (half-light radius) in arcsec
    temp_sizes = 0.5 * 3 * 10 ** (candidate_hosts["logd25Hyp"].values)

    temp_sizes = np.maximum(SIZE_FLOOR, temp_sizes)

    temp_sizes_std = np.minimum(
        temp_sizes, np.abs(temp_sizes) * np.log(10) * candidate_hosts["e_logd25Hyp"].values
    )

    temp_sizes_std[temp_sizes_std != temp_sizes_std] = SIGMA_SIZE_FLOOR * temp_sizes[temp_sizes_std != temp_sizes_std]
    temp_sizes_std[temp_sizes_std < SIGMA_SIZE_FLOOR * temp_sizes] = SIGMA_SIZE_FLOOR * temp_sizes[temp_sizes_std < SIGMA_SIZE_FLOOR * temp_sizes]

    a_over_b = 10 ** (candidate_hosts["logr25Hyp"].values)
    a_over_b_std = a_over_b * np.log(10) * candidate_hosts["e_logr25Hyp"].values

    # set uncertainty floor
    nanbool = a_over_b_std != a_over_b_std
    a_over_b_std[nanbool] = SIGMA_SIZE_FLOOR * a_over_b[nanbool]

    temp_pa = candidate_hosts["PAHyp"].values

    # assume no position angle for unmeasured gals
    # (round is a decent assumption for the most distant ones)
    temp_pa[temp_pa != temp_pa] = SHAPE_FLOOR

    phi = np.radians(temp_pa)
    phi_std = SIGMA_SIZE_FLOOR * phi  # uncertainty floor

    return temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std

def build_galaxy_array(candidate_hosts, cat_cols, transient_name, catalog, release, logger):
    """Builds a structured NumPy array of galaxy properties for host association.

    Parameters
    ----------
    candidate_hosts : pandas.DataFrame
        DataFrame with candidate host galaxy properties.
    cat_cols : bool
        If True, include extra catalog columns not in PROP_DTYPES.
    transient_name : str
        Name of the transient (for logging purposes).
    catalog : str
        Name of the catalog (e.g., 'panstarrs', 'decals', 'glade').
    release : str
        Catalog release version (e.g., 'dr2').
    logger : logging.Logger
        Logger instance for messages to console or output file.

    Returns
    -------
    galaxies : numpy.ndarray
        Structured array of galaxy properties with dtype defined by PROP_DTYPES
        (extended to include catalog properties if cat_cols is True).
    cat_col_fields : list
        List of extra catalog column names from the catalog.
    """
    n_galaxies = len(candidate_hosts)
    base_fields = ['objID', 'ra','dec']
    calc_fields = [x[0] for x in PROP_DTYPES]
    dtype = list(PROP_DTYPES) #local copy of global var

    if n_galaxies < 1:
        logger.info(f"No sources found around {transient_name} in {catalog} {release}! "
        "Double-check that the SN coords overlap the survey footprint.")
        return None, []

    if cat_cols:
        # Identify new fields to add from candidate_hosts
        cat_col_fields = list(set(candidate_hosts.columns) - set(calc_fields))

        # Extend the dtype with new fields and their corresponding data types
        for col in cat_col_fields:
            dtype.append((col, candidate_hosts[col].dtype))  # Append (column name, column dtype)

        # Create galaxies array with updated dtype
        galaxies = np.zeros(n_galaxies, dtype=dtype)

        # Populate galaxies array with data from candidate_hosts
        for col in candidate_hosts.columns:
            galaxies[col] = candidate_hosts[col].values
    else:
        galaxies = np.zeros(n_galaxies, dtype=dtype)
        cat_col_fields = []

    # Populate galaxies array with data from candidate_hosts
    for col in base_fields:
        galaxies[col] = candidate_hosts[col].values
    return galaxies, cat_col_fields

def fetch_catalog_data(self, transient, search_rad, cosmo, logger, cat_cols, calc_host_props):
    """
    Fetch and process catalog data for a given transient.

    This function selects the appropriate catalog function based on the catalog name
    (stored in `self.catalog_functions`), sets default parameters (e.g. limiting magnitude),
    and passes common parameters (transient info, search radius, cosmology, etc.) to the
    catalog-specific function. For the 'glade' catalog, it requires that `self.data` is provided.

    Parameters
    ----------
    transient : object
        A transient object with at least `name` and `position` attributes.
    search_rad : astropy.units.Angle
        Cone search radius.
    cosmo : astropy.cosmology
        Cosmological model for distance conversions.
    logger : logging.Logger
        Logger instance for messages to console or output file.
    cat_cols : bool
        If True, include extra catalog columns in the returned data.
    calc_host_props : list
        Properties to calculate internally for each host ('offset', 'redshift', 'absmag').

    Returns
    -------
    tuple
        A tuple (catalog_data, extra_columns) where:
          - catalog_data is the processed candidate host data (e.g. a pandas.DataFrame or structured array),
          - extra_columns is a list of additional catalog column names (empty if `cat_cols` is False).

    Raises
    ------
    ValueError
        If the catalog name is unknown or if required catalog data (e.g. for 'glade') is missing.
    """
    if self.name not in self.catalog_functions:
        raise ValueError(f"Unknown catalog: {self.name}. Open a pull request to add functionality for other catalogs!")

    catalog_func = self.catalog_functions[self.name]
    self.limiting_mag = DEFAULT_LIMITING_MAG.get(self.name, None)

    # Common parameters for all catalogs
    init_params = {
        "transient": transient,
        "search_rad": search_rad,
        "cosmo": cosmo,
        "logger": logger,
        "calc_host_props": calc_host_props,
        "n_samples": self.n_samples,
        "cat_cols": cat_cols,
        "release": self.release
    }

    # Add extra parameters if needed (e.g., `glade_catalog` for `glade`)
    if self.name == "glade":
        if self.data is not None:
            init_params["glade_catalog"] = self.data
        else:
            raise ValueError("Please provide GLADE catalog as 'data' when initializing GalaxyCatalog object.")
            return np.array([]), np.array([])

    # quality cut -- if True, removes candidate galaxy shreds identified at the catalog level
    init_params['shred_cut'] = self.shred_cut

    # Run the function and store results
    return catalog_func(**init_params)

class GalaxyCatalog:
    """Class for a catalog containing candidate transient host galaxies.

    Parameters
    ----------
    name : str
        Name of the transient catalog. Currently 'glade', 'decals', and 'panstarrs' are supported.
    data : pandas.DataFrame or None
        Locally-saved catalog data (e.g., GLADE) used for host association at low redshift.
    n_samples : int
        Number of Monte Carlo samples to draw during the association process.

    Attributes
    ----------
    name : str
        Identifier for the catalog. Current supported values are 'glade', 'decals', and 'panstarrs'.
    data : pandas.DataFrame or None
        The catalog data provided to the object. Used for catalogs that do not require an external query.
    n_samples : int
        The number of samples used for probabilistic host association calculations.
    release : str or None
        The version or data release of the catalog (e.g., 'dr1/dr2' for pan-starrs; 'dr9/dr100' for decals).
        Can be None if not applicable.
    shred_cut : bool
        A flag indicating whether to automatically remove likely source shreds (duplicate detections)
        from the candidate list. The default is determined by the catalog type (only pan-starrs is True).
    """

    def __init__(self, name, release=None, data=None, n_samples=1000):
        self.name = name
        self.data = data
        self.n_samples = n_samples
        self.release = release
        self.shred_cut = CATALOG_SHRED_SETTINGS.get(self.name, False)

        # Define catalog function mapping as an instance attribute
        self.catalog_functions = {
            "panstarrs": build_panstarrs_candidates,
            "decals": build_decals_candidates,
            "glade": build_glade_candidates,
        }

    def get_candidates(self, transient, cosmo, logger, time_query=False, calc_host_props=['offset' ,'absmag', 'redshift'], cat_cols=False):
        """Hydrates the catalog attribute catalog.galaxies with a list of candidates.

        Parameters
        ----------
        transient : Transient
            Source to associate, of custom class Transient.
        cosmo : astropy cosmology
            Assumed cosmology for conversions (defaults to LambdaCDM if not set).
        logger : logging.Logger
            Logger instance for messages to console or output file.
        time_query : boolean
            If True, times the catalog query and stores the result in self.query_time.
        calc_host_props : list
            Properties to calculate internally for each host ('offset', 'redshift', 'absmag').
        cat_cols : boolean
            If True, contatenates catalog columns to resulting DataFrame.
        """
        search_rad = Angle(300 * u.arcsec)

        if transient.redshift == transient.redshift:
            search_rad = Angle(
                np.nanmax(
                    [
                        100 / cosmo.angular_diameter_distance(transient.redshift).to(u.kpc).value * RAD_TO_ARCSEC,
                        search_rad.arcsec,
                    ]
                )
                * u.arcsec
            )

        self.search_rad = search_rad
        self.search_pos = transient.position
        if time_query:
            start_time = time.time()

        self.galaxies, self.cat_col_fields = fetch_catalog_data(self, transient, search_rad, cosmo, logger, cat_cols, calc_host_props)

        if self.galaxies is None:
            self.ngals = 0
        else:
            self.ngals = len(self.galaxies)

        if time_query:
            end_time = time.time()
            elapsed = end_time - start_time
            self.query_time = elapsed

class Transient:
    """Class for transient source to be associated.

    Parameters
    ----------
    name : str
        Name of transient.
    position : astropy.coord.SkyCoord object
        Position of transient.
    position_err : tuple of quantity objects
        Positional uncertainty of transient.
    logger : logging.Logger
        Logger instance for messages to console or output file.
    redshift : float
        Photometric or spectroscopic redshift of transient.
    spec_class : str
        Spectroscopic class of transient, if available.
    phot_class : str
        Photometric class of transient, if available.
    n_samples : int
        Number of iterations for Monte Carlo association.

    Attributes
    ----------
    best_host : int
        Catalog index of host with the highest posterior probability of association.
        Set to -1 if no valid host.
    second_best_host : int
        Catalog index of host with the second highest posterior probability of association.
    redshift : float
        Redshift of the transient. This is either the spectroscopic/photometric
        redshift or an inferred value from sampling from the prior.
    redshift_std : float
        Uncertainty in transient redshift.
    gen_z_samples : function
        Generates samples for the transient's redshift from the measured redshift (if available)
        or a user-specified prior distribution.
    priors : dict
        Prior distributions for host properties used in association.
        Keys are a subset of 'offset', 'absmag', and 'redshift'.
    likes : dict
        Likelihood functions for host properties used in association.
        Keys are a subset of 'offset', 'absmag', and 'redshift'.
    name : str
        Name of the transient.
    position : astropy.coordinates.SkyCoord
        The transient's position.
    spec_class : str
        Spectroscopic classification of the transient, if available. Defaults to an empty string.
    phot_class : str
        Photometric classification of the transient, if available. Defaults to an empty string.
    n_samples : int
        Number of samples used for probabilistic association. Defaults to 1000.

    """

    def __init__(self, name, position,  logger, redshift=np.nan, redshift_std=np.nan, spec_class="", position_err=(0.1*u.arcsec, 0.1*u.arcsec), phot_class="", n_samples=1000):
        self.name = name
        self.position = position
        self.position_err = position_err
        self.redshift = redshift
        self.redshift_std = redshift_std
        self.spec_class = spec_class
        self.phot_class = phot_class
        self.n_samples = n_samples
        self.best_host = -1
        self.logger = logger
        self.second_best_host = -1

        if (redshift == redshift) and (redshift_std != redshift_std):
            redshift_std = SIGMA_REDSHIFT_FLOOR * self.redshift
            self.redshift_std = redshift_std
            self.logger.info(f"Setting redshift uncertainty for {name} to floor of {redshift_std:.5f}.")

        self.priors = {}
        self.likes = {}

        # draw n_samples positional samples
        ra_samples = np.random.normal(self.position.ra.deg, self.position_err[0].to(u.deg).value, size=n_samples)
        dec_samples = np.random.normal(self.position.dec.deg, self.position_err[1].to(u.deg).value, size=n_samples)

        self.position_samples = SkyCoord(
            ra=ra_samples*u.deg,
            dec=dec_samples*u.deg,
        )

    def __str__(self):
        # Define what should be shown when printing the Transient object
        redshift_str = f"redshift={self.redshift:.4f}" if (self.redshift == self.redshift) else "redshift=N/A"
        class_str = (
            f"spec. class = {self.spec_class}"
            if len(self.spec_class) > 0
            else f"phot. class = {self.phot_class}"
            if len(self.phot_class) > 0
            else "unclassified"
        )
        return f"Transient(name={self.name}, position={self.position}, {redshift_str}, {class_str}"

    def get_prior(self, type):
        """Retrieves the transient host's prior for a given property.

        Parameters
        ----------
        type : str
            Type of prior to retrieve (can be redshift, offset, or absmag).

        Returns
        -------
        prior : scipy stats continuous distribution
            The prior for 'type' property.

        """
        try:
            prior = self.priors[type]
        except KeyError:
            prior = None
        return prior

    def get_likelihood(self, type):
        """Retrieves the transient host's likelihood for a given property.

        Parameters
        ----------
        type : str
            Type of prior to retrieve (can be redshift, offset, absmag).

        Returns
        -------
        prior : scipy stats continuous distribution
            The likelihood for 'type' property.

        """
        try:
            like = self.likes[type]
        except KeyError:
            like = None
        return like

    def set_likelihood(self, type, func):
        """Sets the transient's host prior for a given property.

        Parameters
        ----------
        type : str
            Type of likelihood to set (can be redshift, offset, absmag).
        func : scipy stats continuous distribution
            The likelihood to set for 'type' property.

        """
        self.likes[type] = func

    def set_prior(self, type, func):
        """Sets the transient host's prior for a given property.

        Parameters
        ----------
        type : str
            Type of prior to set (can be redshift, offset, absmag).
        func : scipy stats continuous distribution
            The prior to set for 'type' property.

        """
        self.priors[type] = func
        if (type == "redshift") and (self.redshift != self.redshift):
            self.gen_z_samples()

    def gen_z_samples(self, ret=False, n_samples=None):
        """Generates transient redshift samples for Monte Carlo association.
            If redshift is not measured, samples are drawn from the prior.

        Parameters
        ----------
        ret : boolean
            If true, returns the samples.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        samples : np.ndarray
            If ret=True, redshift samples.

        """
        if n_samples is None:
            n_samples = self.n_samples

        # Sample initially based on whether redshift is NaN or not
        if np.isnan(self.redshift):
            samples = self.get_prior("redshift").rvs(size=n_samples)
            self.redshift = np.nanmean(samples)
            self.redshift_std = np.nanstd(samples)
        else:
            samples = norm.rvs(self.redshift, self.redshift_std, size=n_samples)

        # Resample only those below the floor
        while np.any(samples < REDSHIFT_FLOOR):
            mask = samples < REDSHIFT_FLOOR
            samples[mask] = (self.get_prior("redshift").rvs(size=np.sum(mask))
                             if np.isnan(self.redshift)
                             else norm.rvs(self.redshift, self.redshift_std, size=np.sum(mask)))

        # now return or assign redshift samples
        if ret:
            return samples
        else:
            self.redshift_samples = samples

    def calc_prior_redshift(self, redshift_samples, reduce="mean"):
        """Calculates the prior probability of the transient redshift samples.

        Parameters
        ----------
        redshift_samples : np.ndarray
            Array of transient redshift samples.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        pdf : float or np.ndarray
            prior probability point estimate or samples.

        """
        pdf = self.get_prior("redshift").pdf(redshift_samples)
        if reduce == "mean":
            return np.nanmean(pdf, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == "median":
            return np.nanmedian(pdf, axis=1)
        else:
            return pdf

    def calc_prior_offset(self, fractional_offset_samples, reduce="mean"):
        """Calculates the prior probability of the transient's fractional offset.

        Parameters
        ----------
        fractional_offset_samples : np.ndarray
            Array of transient fractional offset samples.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        type
            Description of returned object.

        """

        pdf = self.get_prior("offset").pdf(fractional_offset_samples)

        if reduce == "mean":
            return np.nanmean(pdf, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == "median":
            return np.nanmedian(pdf, axis=1)
        else:
            return pdf

    def calc_prior_absmag(self, absmag_samples, reduce="mean"):
        """Computes the prior probability for host galaxy absolute magnitude.

        Parameters
        ----------
        absmag_samples : np.ndarray
            2D array of (n_galaxies, n_samples).
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        np.ndarray
            The collapsed prior probability across samples for all galaxies.

        """
        pdf = self.get_prior("absmag").pdf(absmag_samples)
        if reduce == "mean":
            return np.nanmean(pdf, axis=1)
        elif reduce == "median":
            return np.nanmedian(pdf, axis=1)
        else:
            return pdf

    def calc_like_redshift(self, redshift_mean, redshift_std, reduce="mean"):
        """Calculates the likelihood of a redshift with uncertainties.

        Parameters
        ----------
        redshift_mean : np.ndarray
            Redshift means for candidate hosts.
        redshift_std : np.ndarray
            Redshift stds for candidate hosts.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        np.ndarray
            The collapsed likelihood probability across samples for all galaxies.

        """
        z_sn_samples = self.redshift_samples[np.newaxis, :]  # Shape: (n_sn_samples, 1)
        redshift_mean = redshift_mean[:, np.newaxis]  # Shape: (1, n_galaxies)
        redshift_std = redshift_std[:, np.newaxis]  # Shape: (1, n_galaxies)

        # Calculate the likelihood of each SN redshift sample across each galaxy
        likelihoods = norm.pdf(
            z_sn_samples, loc=redshift_mean, scale=redshift_std
        )  # Shape: (n_sn_samples, n_galaxies)

        if reduce == "mean":
            return np.nanmean(likelihoods, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == "median":
            return np.nanmedian(likelihoods, axis=1)
        else:
            return likelihoods

    def calc_like_offset(self, fractional_offset_samples, reduce="mean"):
        """Calculates the likelihoods of a set of fractional offsets.

        Parameters
        ----------
        fractional_offset_samples : np.ndarray
            Array of (n_galaxies, n_samples) fractional offsets.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        np.ndarray
            The collapsed likelihood probability across samples for all galaxies.

        """
        likelihoods = self.get_likelihood("offset").pdf(fractional_offset_samples)
        if reduce == "mean":
            return np.nanmean(likelihoods, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == "median":
            return np.nanmedian(likelihoods, axis=1)
        else:
            return likelihoods

    def calc_like_absmag(self, absmag_samples, reduce="mean"):
        """Calculates the likelihoods of a set of absolute magnitudes.

        Parameters
        ----------
        fractional_offset_samples : np.ndarray
            Array of (n_galaxies, n_samples) absmag values.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        np.ndarray
            The collapsed likelihood probability across samples for all galaxies.

        """
        # assuming a typical 0.1 SN/century/10^10 Lsol (in K-band)
        # TODO -- convert to K-band luminosity of the host!
        # https://www.aanda.org/articles/aa/pdf/2005/15/aa1411.pdf
        likelihoods = self.get_likelihood("absmag").pdf(absmag_samples)
        if reduce == "mean":
            return np.nanmean(likelihoods, axis=1)
        elif reduce == "median":
            return np.nanmedian(likelihoods, axis=1)
        else:
            return likelihoods

    def associate(self, galaxy_catalog, cosmo, condition_host_props=['offset']):
        """Runs the main transient association module.

        Parameters
        ----------
        galaxy_catalog : GalaxyCatalog object
            The catalog populated with candidate hosts and their attributes.
        cosmo : astropy cosmology
            Assumed cosmology.

        Returns
        -------
        galaxy_catalog : GalaxyCatalog object
            The catalog, with additional attributes including posterior probabilities,
            best host, and unobserved probability.

        """
        ngals = galaxy_catalog.ngals
        search_rad = galaxy_catalog.search_rad
        limiting_mag = galaxy_catalog.limiting_mag
        galaxies = galaxy_catalog.galaxies
        n_gals = len(galaxies)
        n_samples = galaxy_catalog.n_samples

        post_set = []

        if 'redshift' in condition_host_props:
            # Extract arrays for all galaxies from the catalog
            redshift_mean = np.array(galaxies["redshift_mean"])
            redshift_std = np.array(galaxies["redshift_std"])
            redshift_samples = np.vstack(galaxies["redshift_samples"])

            prior_redshift = self.calc_prior_redshift(redshift_samples, reduce=None)
            like_redshift = self.calc_like_redshift(redshift_mean, redshift_std, reduce=None)

            if np.isnan(self.redshift):
                # Marginalize over the sampled supernova redshifts
                # by integrating the likelihood over the redshift prior
                sorted_indices = np.argsort(redshift_samples, axis=1)
                sorted_redshift_samples = np.take_along_axis(redshift_samples, sorted_indices, axis=1)
                sorted_integrand = np.take_along_axis(prior_redshift * like_redshift, sorted_indices, axis=1)

                # Perform integration using simps or trapz
                post_redshift = np.trapz(sorted_integrand, sorted_redshift_samples, axis=1)
                post_redshift = post_redshift[:, np.newaxis]
            else:
                post_redshift = prior_redshift * like_redshift
            post_set.append(post_redshift)

        if 'absmag' in condition_host_props:
            absmag_mean = np.array(galaxies["absmag_mean"])
            absmag_std = np.array(galaxies["absmag_std"])
            absmag_samples = np.vstack(galaxies["absmag_samples"])

            prior_absmag = self.calc_prior_absmag(absmag_samples, reduce=None)
            like_absmag = self.calc_like_absmag(absmag_samples, reduce=None)
            post_absmag = prior_absmag * like_absmag
            post_set.append(post_absmag)

        if 'offset' in condition_host_props:
            offset_mean = np.array(galaxies["offset_mean"])
            offset_std = np.array(galaxies["offset_std"])
            offset_samples = np.vstack(galaxies["offset_samples"])
            galaxy_dlr_samples = np.vstack(galaxies["dlr_samples"])

            # Calculate angular diameter distances for all samples
            fractional_offset_samples = offset_samples / galaxy_dlr_samples

            prior_offset = self.calc_prior_offset(fractional_offset_samples, reduce=None) # Shape (N,)
            like_offset = self.calc_like_offset(fractional_offset_samples, reduce=None)  # Shape (N,)

            logging.info("prior_offset:", prior_offset)
            logging.info("like_offset", like_offset)

            post_offset = prior_offset * like_offset
            post_set.append(post_offset)

        # Compute the posterior probabilities for all galaxies, multiplying host properties
        # shape -> (n_properties, n_gals, n_samples)
        post_gals_stacked = np.stack(post_set, axis=0)
        # shape -> (n_gals, n_samples)
        post_gals = np.prod(post_gals_stacked, axis=0)

        # some very low value that the SN is actually hostless, across all samples.
        post_hostless = np.ones(n_samples)*PROB_FLOOR

        if self.redshift == self.redshift:
            post_outside = self.probability_host_outside_cone(search_rad=search_rad, n_samples=n_samples, cosmo=cosmo, condition_host_props=condition_host_props)
        else:
            post_outside = PROB_FLOOR

        post_unobs = self.probability_of_unobserved_host(search_rad=search_rad, limiting_mag=limiting_mag, n_samples=n_samples, cosmo=cosmo, condition_host_props=condition_host_props)

        # sum across all galaxies
        post_tot = np.nansum(post_gals, axis=0) + post_hostless + post_outside + post_unobs

        # floor to machine precision
        post_tot[post_tot < PROB_FLOOR] = PROB_FLOOR

        p_none_norm = (post_outside + post_hostless + post_unobs) / post_tot
        p_any_norm = np.nansum(post_gals, axis=0) / post_tot
        post_gals_norm = post_gals / post_tot
        post_outside_norm = post_outside / post_tot
        post_unobs_norm = post_unobs / post_tot
        post_hostless_norm = post_hostless / post_tot

        if 'offset' in condition_host_props:
            post_offset_norm = post_offset / post_tot
        if 'redshift' in condition_host_props:
            post_redshift_norm = post_redshift / post_tot
        if 'absmag' in condition_host_props:
            post_absmag_norm = post_absmag / post_tot

        # calculate the galaxies with the most matches in the MCMC
        all_posts = np.vstack([
        post_gals_norm,
        post_outside_norm,
        post_unobs_norm,
        post_hostless_norm
        ])

        winner_indices = np.argmax(all_posts, axis=0)
        counts = np.bincount(winner_indices, minlength=n_gals+3)
        win_fractions = counts / float(n_samples)
        top_idxs = np.argsort(win_fractions)[::-1]

        self.associated_catalog = galaxy_catalog.name
        self.any_posterior = np.nanmedian(p_any_norm)
        self.none_posterior = np.nanmedian(p_none_norm)
        self.smallcone_posterior = np.nanmedian(post_outside_norm)
        self.missedcat_posterior = np.nanmedian(post_unobs_norm)

        best_idx = top_idxs[0]

        if best_idx < n_gals:
            # This is a real galaxy
            self.logger.info("Association successful!")
            self.best_host = best_idx
        else:
            # no galaxy found
            self.best_host = -1
            if best_idx == n_gals:
                self.logger.info("Association failed. Host is likely outside the search cone.")
            elif best_idx == (n_gals + 1):
                self.logger.info("Association failed. Host is likely missing from the catalog.")
            elif best_idx == (n_gals + 2):
                self.logger.info("Association failed. Host is likely hostless.")

        # Now figure out second best index
        if len(top_idxs) > 1:
            second_idx = top_idxs[1]
            if second_idx < n_gals:
                self.second_best_host = second_idx
            else:
                self.second_best_host = -1
        else:
            # No second candidate galaxy
            self.second_best_host = -1

        # consolidate across samples
        galaxy_catalog.galaxies["total_posterior"] = np.nanmedian(post_gals_norm, axis=1)

        if 'offset' in condition_host_props:
            galaxy_catalog.galaxies["offset_posterior"] = np.nanmedian(post_offset_norm, axis=1)
        if 'redshift' in condition_host_props:
            galaxy_catalog.galaxies["redshift_posterior"] = np.nanmedian(post_redshift_norm, axis=1)
        if 'absmag' in condition_host_props:
            absmag_best_post = np.nanmedian(post_absmag_norm, axis=1)[self.best_host]
            galaxy_catalog.galaxies["absmag_posterior"] = np.nanmedian(post_absmag_norm, axis=1)

        return galaxy_catalog

    def probability_of_unobserved_host(self, search_rad, cosmo, limiting_mag=30, n_samples=1000, condition_host_props=['offset']):
        """Calculates the posterior probability of the host being either dimmer than the
           limiting magnitude of the catalog or not in the catalog at all.

        Parameters
        ----------
        search_rad : float
            Cone search radius, in arcsec.
        limiting_mag : float
            Limiting magnitude of the survey, in AB mag.
        n_samples : int
            Number of samples for Monte Carlo association.
        cosmo : astropy cosmology
            Assumed cosmology for the run.

        Returns
        -------
        post_unobserved : np.ndarray
            n_samples of posterior probabilities of the host not being in the catalog.

        """
        # only set if we have absmag and redshift priors -- otherwise set to 0!
        if ('absmag' not in condition_host_props) or ('redshift' not in condition_host_props):
            return np.ones(n_samples)*PROB_FLOOR

        post_set = []

        n_gals = int(0.5 * n_samples)
        z_sn = self.redshift
        z_sn_std = self.redshift_std
        sn_distance = cosmo.luminosity_distance(z_sn).to(u.pc).value  # in pc

        if np.isnan(z_sn):
            # draw galaxies from the same distribution
            redshift_mean = self.gen_z_samples(n_samples=n_gals, ret=True)
            redshift_std = SIGMA_REDSHIFT_FLOOR * redshift_mean
            redshift_samples = np.maximum(
                REDSHIFT_FLOOR,
                norm.rvs(
                    loc=redshift_mean[:, np.newaxis], scale=redshift_std[:, np.newaxis], size=(n_gals, n_samples)
                ),
            )
            prior_redshift = self.calc_prior_redshift(redshift_samples, reduce=None)
            like_redshift = self.calc_like_redshift(redshift_mean, redshift_std, reduce=None)

            sorted_indices = np.argsort(redshift_samples, axis=1)
            sorted_redshift_samples = np.take_along_axis(redshift_samples, sorted_indices, axis=1)
            sorted_integrand = np.take_along_axis(prior_redshift * like_redshift, sorted_indices, axis=1)

            # Perform integration using simps or trapz
            post_z = np.trapz(sorted_integrand, sorted_redshift_samples, axis=1)
            # Shape: (n_galaxies, 1)
            post_z = post_z[:, np.newaxis]
        else:
            # Use the known supernova redshift
            redshift_mean = np.maximum(REDSHIFT_FLOOR, norm.rvs(loc=z_sn, scale=z_sn_std, size=(n_gals)))
            # assume all well-constrained redshifts
            redshift_std = SIGMA_REDSHIFT_FLOOR * redshift_mean
            redshift_samples = np.maximum(
                REDSHIFT_FLOOR,
                norm.rvs(
                    loc=redshift_mean[:, np.newaxis], scale=redshift_std[:, np.newaxis], size=(n_gals, n_samples)
                ),
            )

            prior_redshift = self.calc_prior_redshift(redshift_samples, reduce=None)
            like_redshift = self.calc_like_redshift(redshift_mean, redshift_std, reduce=None)
            post_z = prior_redshift * like_redshift

        post_set.append(post_z)

        absmag_lim = limiting_mag - 5 * (np.log10(sn_distance / 10))
        absmag_mean = np.linspace(absmag_lim, ABSMAG_FLOOR, n_gals)
        absmag_std = SIGMA_ABSMAG_FLOOR*np.abs(absmag_mean)
        absmag_samples = norm.rvs(
            loc=absmag_mean[:, np.newaxis],
            scale=absmag_std[:, np.newaxis],
            size=(n_gals, n_samples)
        )

        prior_absmag = self.calc_prior_absmag(absmag_samples, reduce=None)
        like_absmag = self.calc_like_absmag(absmag_samples, reduce=None)

        post_absmag = prior_absmag*like_absmag
        post_set.append(post_absmag)

        if 'offset' in condition_host_props:
            galaxy_physical_radius_prior_means = halfnorm.rvs(size=n_gals, loc=1.0, scale=10)  # in kpc
            galaxy_physical_radius_prior_std = SIGMA_SIZE_FLOOR * galaxy_physical_radius_prior_means
            galaxy_physical_radius_prior_samples = norm.rvs(
                loc=galaxy_physical_radius_prior_means[:, np.newaxis],
                scale=galaxy_physical_radius_prior_std[:, np.newaxis],
                size=(n_gals, n_samples),
            )

            min_phys_rad = 1.0
            max_phys_rad = (search_rad.arcsec / RAD_TO_ARCSEC) * sn_distance / 1.0e3  # in kpc

            physical_offset_mean = np.linspace(min_phys_rad, max_phys_rad, n_gals)
            physical_offset_std = SIGMA_SIZE_FLOOR * physical_offset_mean

            physical_offset_samples = norm.rvs(
                physical_offset_mean[:, np.newaxis], physical_offset_std[:, np.newaxis], size=(n_gals, n_samples)
            )

            # Shape: (n_gals, n_samples)
            fractional_offset_samples = (
                physical_offset_samples / galaxy_physical_radius_prior_samples
            )

            prior_offset_unobs = self.calc_prior_offset(fractional_offset_samples, reduce=None)
            l_offset_unobs = self.calc_like_offset(fractional_offset_samples, reduce=None)

            post_offset = prior_offset_unobs * l_offset_unobs
            post_set.append(post_offset)

        # Compute the posterior probabilities for all galaxies
        # shape -> (n_properties, n_gals, n_samples)
        prob_unobs_stacked = np.stack(post_set, axis=0)
        # shape -> (n_gals, n_samples)
        post_unobs = np.prod(prob_unobs_stacked, axis=0)
        # average over all galaxies -- keep n_samples
        post_unobs = np.nanmean(post_unobs, axis=0)

        return post_unobs

    def probability_host_outside_cone(self, cosmo, search_rad=60, n_samples=1000, condition_host_props=['offset']):
        """Calculates the posterior probability of the host being outside the cone search chosen
           for the catalog query. Primarily set by the fractional offset and redshift prior.

        Parameters
        ----------
        search_rad : float
            Cone search radius, in arcsec.
        n_samples : int
            Number of samples to draw for Monte Carlo association.
        cosmo : astropy cosmology
            Assumed cosmology.
        condition_host_props : list
            List of galaxy properties to use for association.

        Returns
        -------
        post_outside : np.ndarray
            An array of n_samples posterior probabilities of the host being outside the search cone.

        """

        # only calculate if we have redshift and offset priors -- otherwise fix to PROB_FLOOR
        if 'redshift' not in condition_host_props:
            return np.ones(n_samples)*PROB_FLOOR

        n_gals = int(n_samples / 2)

        post_set = []

        z_sn = self.redshift
        z_sn_std = self.redshift_std

        if np.isnan(z_sn):
            # draw galaxies from the same distribution
            redshift_mean = self.gen_z_samples(
                n_samples=n_gals, ret=True
            )  # draw from prior if redshift is missing
            redshift_std = SIGMA_REDSHIFT_FLOOR * redshift_mean

            # scatter around some nominal uncertainty
            redshift_samples = np.maximum(
                REDSHIFT_FLOOR,
                norm.rvs(
                    loc=redshift_mean[:, np.newaxis], scale=redshift_std[:, np.newaxis], size=(n_gals, n_samples)
                ),
            )

            prior_redshift = self.calc_prior_redshift(redshift_samples, reduce=None)
            like_redshift = self.calc_like_redshift(redshift_mean, redshift_std, reduce=None)

            sorted_indices = np.argsort(redshift_samples, axis=1)
            sorted_redshift_samples = np.take_along_axis(redshift_samples, sorted_indices, axis=1)
            sorted_integrand = np.take_along_axis(prior_redshift * like_redshift, sorted_indices, axis=1)

            # Perform integration using trapezoidal integration
            p_z = np.trapz(sorted_integrand, sorted_redshift_samples, axis=1)
            p_z = p_z[:, np.newaxis]
        else:
            # Use the known supernova redshift
            # some higher spread for host redshift photo-zs
            redshift_mean = np.maximum(REDSHIFT_FLOOR, norm.rvs(loc=z_sn, scale=z_sn_std, size=(n_gals)))
            redshift_std = SIGMA_REDSHIFT_FLOOR * redshift_mean  # assume all well-constrained redshifts
            redshift_samples = np.maximum(
                REDSHIFT_FLOOR,
                norm.rvs(
                    loc=redshift_mean[:, np.newaxis], scale=redshift_std[:, np.newaxis], size=(n_gals, n_samples)
                ),
            )

            prior_redshift = self.calc_prior_redshift(redshift_samples, reduce=None)
            like_redshift = self.calc_like_redshift(redshift_mean, redshift_std, reduce=None)

            p_z = prior_redshift * like_redshift
            post_set.append(p_z)

        # Calculate the distance to the supernova for each sampled redshift
        sn_distances = cosmo.comoving_distance(self.redshift_samples).value  # in Mpc

        # Convert angular cutout radius to physical offset at each sampled redshift
        min_phys_rad = (search_rad.arcsec / RAD_TO_ARCSEC) * sn_distances * 1e3  # in kpc
        max_phys_rad = 5 * min_phys_rad

        galaxy_physical_radius_prior_means = halfnorm.rvs(size=n_gals, loc=0, scale=10)  # in kpc
        galaxy_physical_radius_prior_std = SIGMA_SIZE_FLOOR * galaxy_physical_radius_prior_means
        galaxy_physical_radius_prior_samples = norm.rvs(
            loc=galaxy_physical_radius_prior_means[:, np.newaxis],
            scale=galaxy_physical_radius_prior_std[:, np.newaxis],
            size=(n_gals, n_samples),
        )

        physical_offset_samples = np.linspace(min_phys_rad, max_phys_rad, n_gals)
        fractional_offset_samples = physical_offset_samples / galaxy_physical_radius_prior_samples

        prior_offset = self.calc_prior_offset(fractional_offset_samples, reduce=None)
        l_offset = self.calc_like_offset(fractional_offset_samples, reduce=None)
        post_offset = prior_offset * l_offset
        post_set.append(post_offset)

        if 'absmag' in condition_host_props:
            # sample brightnesses
            absmag_mean = self.get_prior("absmag").rvs(size=n_gals)
            absmag_std = 0.05 * np.abs(absmag_mean)
            absmag_samples = np.maximum(REDSHIFT_FLOOR,
                norm.rvs(loc=absmag_mean[:, np.newaxis], scale=absmag_std[:, np.newaxis], size=(n_gals, n_samples)),
            )

            prior_absmag = self.calc_prior_absmag(absmag_samples, reduce=None)

            like_absmag = self.calc_like_absmag(absmag_samples, reduce=None)
            post_absmag = prior_absmag * like_absmag
            post_set.append(post_absmag)

        # Compute the posterior probabilities for all galaxies
         # shape -> (n_properties, n_gals, n_samples)
        prob_outside_stacked = np.stack(post_set, axis=0)
        # shape -> (n_gals, n_samples)
        post_outside = np.prod(prob_outside_stacked, axis=0)

        # average over all simulated galaxies -- keep the samples
        post_outside = np.nanmean(post_outside, axis=0)

        return post_outside

class PriorzObservedTransients(rv_continuous):
    """A continuous probability distribution for a redshift prior defined by
       an observed sample of transients with a given limiting magnitude, volumetric rate,
       and brightness distribution.

    Parameters
    ----------
    z_min : float
        Minimum redshift to draw transients from.
    z_max : float
        Maximum redshift to draw transients from.
    n_bins : int
        Number of bins with which to fit the observed sample to a PDF.
    mag_cutoff : float
        Maximum apparent magnitude of the transient survey.
    absmag_mean : float
        Expected absolute brightness of the transient.
    absmag_min : float
        Description of parameter `absmag_min`.
    absmag_max : type
        Description of parameter `absmag_max`.
    r_transient : float
        Transient volumetric rate, in units of N/Mpc^3/yr.
        (This gets normalized, so this is not too important).
    t_obs : float
        The observing time in years.
    **kwargs : dict
        Any other params.

    Attributes
    ----------
    cosmo : astropy cosmology
        Assumed cosmology.
    _generate_distribution : function
        Runs the experiment to build the distribution of observed transients.
    z_min
    z_max
    n_bins
    mag_cutoff
    absmag_mean
    absmag_min
    absmag_max
    r_transient
    t_obs

    """

    def __init__(
        self,
        cosmo,
        z_min=0,
        z_max=1,
        n_bins=100,
        mag_cutoff=22,
        absmag_mean=-19,
        absmag_min=-24,
        absmag_max=-17,
        r_transient=1e-5,
        t_obs=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Assign the parameters
        self.z_min = z_min
        self.z_max = z_max
        self.n_bins = n_bins
        self.mag_cutoff = mag_cutoff
        self.absmag_mean = absmag_mean
        self.absmag_min = absmag_min
        self.absmag_max = absmag_max
        self.r_transient = r_transient
        self.t_obs = t_obs
        self.cosmo = cosmo

        # Automatically run the internal function to generate the distribution
        self._generate_distribution()

    def _generate_distribution(self):
        """Generate and store the empirical redshift distribution for the prior.

        This method:
          - Creates redshift bins between self.z_min and self.z_max.
          - Computes the comoving volume element (dV/dz) for each bin using self.cosmo.
          - Estimates the number of supernovae per bin based on the transient rate (self.r_transient) and survey solid angle.
          - Samples redshifts uniformly within each bin according to the estimated counts.
          - Calculates luminosity distances for the sampled redshifts.
          - Samples and clips absolute magnitudes, then computes apparent magnitudes.
          - Filters the sample to include only supernovae with apparent magnitudes ≤ self.mag_cutoff.
          - Fits a Gaussian KDE to the resulting observed redshifts, storing the KDE in self.bestFit and the observed redshifts in self.observed_redshifts.

        Returns
        -------
        None
            The distribution is stored internally.
        """
        # Create redshift bins
        z_bins = np.linspace(self.z_min, self.z_max, self.n_bins + 1)
        # Centers of redshift bins
        z_centers = (z_bins[:-1] + z_bins[1:])/2

        # Calculate the comoving volume element dV/dz for each redshift bin
        # in Mpc^3 per steradian per dz
        dv_dz = self.cosmo.differential_comoving_volume(z_centers).value

        # Full sky solid angle (4 pi steradians)
        # full sky in steradians
        solid_angle = 4 * np.pi

        # Supernovae per redshift bin (for full sky)
        supernovae_per_bin = (self.r_transient * dv_dz * solid_angle * np.diff(z_bins)).astype(int)

        # Generate random redshifts for all supernovae
        z_scattered = np.hstack(
            [
                np.random.uniform(z_bins[i], z_bins[i + 1], size=n)
                for i, n in enumerate(supernovae_per_bin)
                if n > 0
            ]
        )

        # Calculate luminosity distance (in parsecs) for all supernovae
        d_l = self.cosmo.luminosity_distance(z_scattered).to(u.pc).value

        # Sample absolute magnitudes from a Gaussian and clip the values to the range [Mmin, Mmax]
        absolute_magnitudes = np.random.normal(loc=self.absmag_mean, scale=1.5, size=len(z_scattered))
        absolute_magnitudes = np.clip(absolute_magnitudes, self.absmag_min, self.absmag_max)

        # Calculate apparent magnitudes using distance modulus for all supernovae
        m_apparent = absolute_magnitudes + 5 * np.log10(d_l / 10)

        # Filter supernovae based on apparent magnitude cutoff
        observed_indices = m_apparent <= self.mag_cutoff
        self.observed_redshifts = z_scattered[observed_indices]

        # Calculate the best fit KDE for observed redshifts
        self.bestFit = gaussian_kde(self.observed_redshifts)

    def pdf(self, z):
        """
        Return the PDF (KDE) based on observed redshifts.
        Handles 1D and 2D arrays.

        Parameters
        ----------
        z : np.ndarray
            List of input redshifts.

        Returns
        -------
        flat_pdf : the pdf from a kde fit to the input redshifts.

        """
        if z.ndim == 2:
            flat_z = z.flatten()
            flat_pdf = self.bestFit(flat_z)
            return flat_pdf.reshape(z.shape)
        else:
            return self.bestFit(z)

    def rvs(self, size=None):
        """Generate random variables from the empirical distribution.

        Parameters
        ----------
        size : int
            Number of samples to draw from the distribution

        Returns
        -------
        samples : np.ndarray
            The redshift samples from the distribution.

        """
        samples = self.bestFit.resample(size=size).reshape(-1)
        return samples

    def plot(self):
        """Plots the empirical redshift distribution.
        """
        z_bins = np.linspace(self.z_min, self.z_max, self.n_bins + 1)

        plt.figure(figsize=(8, 6))
        plt.hist(
            self.observed_redshifts,
            bins=z_bins,
            edgecolor="k",
            alpha=0.7,
            density=True,
            label="Observed Histogram",
        )
        plt.xlabel("Redshift (z)")
        plt.ylabel("Density")

        # Plot the KDE over a fine grid of redshift values
        z_fine = np.linspace(self.z_min, self.z_max, 1000)
        plt.plot(z_fine, self.bestFit(z_fine), color="k", label="KDE Fit")

        plt.title(f"Observed Supernovae with $m < {self.mag_cutoff}$")
        plt.legend()
        plt.show()


class SnRateAbsmag(rv_continuous):
    """A host-galaxy absolute magnitude likelihood distribution,
       where supernova rate scales as ~0.1*L_host in units of 10^10 Lsol.
       Based on Li, Chornock et al. 2011.

    Parameters
    ----------
    a : float
        The minimum absolute magnitude of a host galaxy.
    b : float
        The maximum absolute magnitude of a host galaxy.

    Attributes
    ----------
    normalization : float
        The calculated normalization constant for the distribution.
    _calculate_normalization : function
        Calculates the normalization constant for the distribution.

    """

    def __init__(self, a, b, **kwargs):
        super().__init__(a=a, b=b, **kwargs)
        self.normalization = self._calculate_normalization(a, b)

    def _calculate_normalization(self, a, b):
        """Calculates the normalization constant for the distribution.

        Parameters
        ----------
        a : float
            The minimum absolute magnitude of a host galaxy.
        b : float
            The maximum absolute magnitude of a host galaxy.

        Returns
        -------
        result : float
            The calculated normalization constant for the distribution.

        """
        result, _ = quad(self._unnormalized_pdf, a, b)
        return result

    def _unnormalized_pdf(self, abs_mag_samples):
        """Calculates the unnormalized PDF from the supernova rate.

        Parameters
        ----------
        abs_mag_samples : np.ndarray
            Array of galaxy absolute magnitudes.

        Returns
        -------
        snrate : np.ndarray
            Supernovae rate for corresponding galaxies.

        """
        msol = 4.74
        lgal = 10 ** (-0.4 * (abs_mag_samples - msol))  # in units of Lsol
        lgal /= 1.0e10  # in units of 10^10 Lsol
        snrate = 0.1 * lgal
        return snrate

    def _pdf(self, m_abs_samples):
        """The PDF of galaxies with m_abs_samples, after normalization.

        Parameters
        ----------
        m_abs_samples : np.ndarray
            Absolute magnitudes of galaxies.

        Returns
        -------
        normalized_pdf : np.ndarray
            Normalized PDF values for m_abs_samples.

        """
        normalized_pdf = self._unnormalized_pdf(m_abs_samples) / self.normalization
        return normalized_pdf

def panstarrs_cone(
    metadata,
    ra,
    dec,
    radius,
    table="stack",
    release="dr2",
    format="csv",
    columns=None,
    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs",
    verbose=False,
    **kw,
):
    """Conducts a cone search of the Pan-STARRS 3PI catalog tables.

    Parameters
    ----------
    metadata : dict
        Dictionary of astropy tables.
    ra : float
        Right ascension of search center, in decimal degrees.
    dec : float
        Declination of search center, in decimal degrees.
    radius : float
        Radius of search cone, in degrees.
    table : str
        The table to query.
    release : str
        The pan-starrs data release. Can be "dr1" or "dr2".
    format : str
        The format for the retrieved data.
    columns : np.ndarray
        A list of columns to retrieve from 'table'.
    baseurl : str
        The api endpoint to query.
    verbose : boolean
        If True, prints details about the query.
    **kw : dict
        Any additional search parameters.

    Returns
    -------
    result : str
        String containing retrieved data (empty if none found).

    """
    data = kw.copy()
    data["ra"] = ra
    data["dec"] = dec
    data["radius"] = radius
    result = panstarrs_search(
        metadata=metadata, table=table, release=release,
        format=format, columns=columns, baseurl=baseurl, verbose=verbose, **data
    )
    return result


def panstarrs_search(
    metadata,
    table="mean",
    release="dr1",
    format="csv",
    columns=None,
    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs",
    verbose=False,
    **kw,
):
    """Queries the Pan-STARRS catalog API.

    Parameters
    ----------
    metadata : dictionary
        A dictionary containing the tables to query.
    table : str
        The table to query.
    release : str
        The pan-starrs data release. Can be "dr1" or "dr2".
    format : str
        The format for the retrieved data.
    columns : np.ndarray
        A list of columns to retrieve from 'table'.
    baseurl : str
        The api endpoint to query.
    verbose : boolean
        If True, prints details about the query.
    **kw : dict
        Any additional search parameters.

    Returns
    -------
    result : str
        String containing retrieved data (empty if none found).

    """
    data = kw.copy()  # Copy the keyword arguments to modify them

    # Construct the API URL
    url = f"{baseurl}/{release}/{table}.{format}"

    # Check and validate columns
    if columns:
        # Get all available columns from the metadata
        valid_columns = {col.lower().strip() for col in metadata[release][table]["name"]}
        badcols = set(col.lower().strip() for col in columns) - valid_columns

        if badcols:
            raise ValueError(f"Some columns not found in table: {', '.join(badcols)}")

        data["columns"] = f"[{','.join(columns)}]"

    r = requests.get(url, params=data, timeout=10)

    r.raise_for_status()

    if format == "json":
        return r.json()
    else:
        return r.text



def build_glade_candidates(
    transient,
    glade_catalog,
    cosmo,
    logger,
    search_rad=None,
    n_samples=1000,
    cat_cols=False,
    calc_host_props=['redshift', 'absmag', 'offset'],
    shred_cut=False,
    release=None,
):
    """Populates a GalaxyCatalog object with candidates from a cone search of the
       GLADE catalog (See https://glade.elte.hu/ for details). Reported luminosity
       distances have been converted to redshifts with a 5% uncertainty floor for
       faster processing.

    Parameters
    ----------
    transient : Transient
        Custom object for queried transient containing name,
        position, and positional uncertainty.
    glade_catalog : pandas.DataFrame
        The locally-packaged GLADE catalog (to avoid querying).
    search_rad : astropy.units.Angle
        Radius for cone search.
    cosmo : astropy.cosmology
        Assumed cosmology for conversions.
    n_samples : int
        Number of samples for Monte Carlo association.
    cat_cols : boolean
        If True, concatenates catalog fields for best host to final catalog.
    shred_cut : boolean
        If True, removes likely source shreds associated with the same candidate galaxy.
    calc_host_props : list
        Properties to calculate internally for each host ('offset', 'redshift', 'absmag').
    Returns
    -------
    galaxies : structured numpy array
        Array of properties for candidate sources needed
        for host association.

    cat_col_fields : list
        List of columns retrieved from the galaxy catalog
        (rather than calculated internally).

    """
    transient_name = transient.name
    transient_pos = transient.position
    transient_pos_samples = transient.position_samples

    if release != "latest":
        logger.warning("Only GLADE+ is supported at this time. Please open a pull request to expand Prost to alternative versions or other catalogs!")
    elif shred_cut:
        logger.warning("shred_cut is not implemented for GLADE+ galaxies at this time. Running with shred_cut = False.")

    if search_rad is None:
        search_rad = Angle(60 * u.arcsec)

    # start with a 1 degree cut on GLADE around the transient position to speed up the cross-match
    ra_min = transient_pos.ra.deg - 1
    ra_max = transient_pos.ra.deg + 1
    dec_min = transient_pos.dec.deg - 1
    dec_max = transient_pos.dec.deg + 1

    glade_catalog.rename(columns={'GLADE+':'objID','RAJ2000':'ra', 'DEJ2000':'dec', 'z_best':'redshift', 'z_best_std':'redshift_std'}, inplace=True)

    filtered_glade = glade_catalog[(glade_catalog["ra"] > ra_min) & (glade_catalog["ra"] < ra_max) &
                    (glade_catalog["dec"] > dec_min) & (glade_catalog["dec"] < dec_max)]
    glade_catalog = filtered_glade

    candidate_hosts = glade_catalog[
        SkyCoord(glade_catalog["ra"].values * u.deg, glade_catalog["dec"].values * u.deg)
        .separation(transient_pos)
        .arcsec
        < search_rad.arcsec
    ]

    galaxies_pos = SkyCoord(candidate_hosts["ra"].values * u.deg, candidate_hosts["dec"].values * u.deg)

    if len(candidate_hosts) < 1:
        return None, []

    galaxies, cat_col_fields = build_galaxy_array(candidate_hosts, cat_cols, transient_name, "GLADE+", release, logger)

    if galaxies is None:
        return None, []

    n_galaxies = len(galaxies)

    galaxies['objID_info'] = ['GLADE+']*n_galaxies

    # get alternate name for candidate (e.g., NGC)
    name_columns = ['HyperLEDA', 'GWGC', '2MASS', 'WISExSCOS']
    galaxies['name'] = candidate_hosts[name_columns] \
        .apply(lambda row: row.dropna().iloc[0].strip() if row.notna().any() else None, axis=1)

    if 'offset' in calc_host_props:
        temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std = calc_shape_props_glade(candidate_hosts)

        dlr_samples = calc_dlr(
            transient_pos_samples,
            galaxies_pos,
            temp_sizes,
            temp_sizes_std,
            a_over_b,
            a_over_b_std,
            phi,
            phi_std,
            n_samples=n_samples,
        )

        offset_samples = np.array([(SkyCoord(galaxies["ra"][i] * u.deg, galaxies["dec"][i] * u.deg).separation(transient_pos_samples).arcsec) for i in np.arange(n_galaxies)])

        # Calculate angular separation between SN and all galaxies (in arcseconds)
        for i in range(n_galaxies):
            galaxies["offset_samples"][i] = offset_samples[i, :]
            galaxies['offset_mean'][i] = np.nanmean(offset_samples[i, :])
            galaxies['offset_std'][i] = np.nanstd(offset_samples[i, :])
            galaxies["dlr_samples"][i] = dlr_samples[i, :]

    if ('redshift' in calc_host_props) or ('absmag' in calc_host_props):
        redshift_mean = candidate_hosts["redshift"].values
        redshift_std = candidate_hosts["redshift_std"].values

        redshift_samples = np.maximum(
            REDSHIFT_FLOOR,
            norm.rvs(
                loc=redshift_mean[:, np.newaxis], scale=redshift_std[:, np.newaxis], size=(n_galaxies, n_samples)
            ),
        )

        galaxies["redshift_mean"] = candidate_hosts["redshift"].values
        galaxies["redshift_std"] = candidate_hosts["redshift_std"].values
        galaxies['redshift_info'] = ['photo-z']

        # TODO find spec-z info for GLADE -- for now, just assume all with <10% error are spec-zs.
        good_specz = galaxies["redshift_std"]/(1+galaxies["redshift_mean"]) < 0.1
        galaxies['redshift_info'][good_specz] = ['spec-z']*np.nansum(good_specz)

        # set redshift floor
        redshift_samples[redshift_samples < REDSHIFT_FLOOR] = REDSHIFT_FLOOR

        for i in range(n_galaxies):
            galaxies["redshift_samples"][i] = redshift_samples[i, :]

    if 'absmag' in calc_host_props:
        temp_mag_r = candidate_hosts["Bmag"].values
        temp_mag_r_std = np.abs(candidate_hosts["e_Bmag"].values)

        # set a floor of 5%
        temp_mag_r_std[temp_mag_r_std < SIGMA_ABSMAG_FLOOR * temp_mag_r] = SIGMA_ABSMAG_FLOOR * temp_mag_r[temp_mag_r_std < SIGMA_ABSMAG_FLOOR * temp_mag_r]

        absmag_samples = (
            norm.rvs(
                loc=temp_mag_r[:, np.newaxis],
                scale=temp_mag_r_std[:, np.newaxis],
                size=(n_galaxies, n_samples),
            )
            - cosmo.distmod(redshift_samples).value
        )

        galaxies["absmag_mean"] = temp_mag_r
        galaxies["absmag_std"] = temp_mag_r_std
        galaxies["absmag_info"] = ["B"]*n_galaxies

        for i in range(n_galaxies):
            galaxies["absmag_samples"][i] = absmag_samples[i, :]

    return galaxies, cat_col_fields

def build_decals_candidates(transient,
                            cosmo,
                            logger,
                            search_rad=None,
                            n_samples=1000,
                            calc_host_props=['redshift', 'absmag', 'offset'],
                            cat_cols=False,
                            shred_cut=False,
                            release="dr9"):
    """Populates a GalaxyCatalog object with candidates from a cone search of the
       DECaLS catalog (See https://www.legacysurvey.org/decamls/ for details).

    Parameters
    ----------
    transient : str
        Transient object with name, position, and positional uncertainties defined.
    glade_catalog : pandas.DataFrame
        The locally-packaged GLADE catalog (to avoid querying).
    search_rad : astropy Angle
        Radius for cone search.
    cosmo : astropy cosmology
        Assumed cosmology for conversions.
    n_samples : int
        Number of samples for Monte Carlo association.
    calc_host_props : list
        Properties to calculate internally for each host ('offset', 'redshift', 'absmag').
    cat_cols : boolean
        If True, concatenates catalog fields for best host to final catalog.
    shred_cut : boolean
        If True, removes likely source shreds associated with the same candidate galaxy.
    release : str
        Can be 'dr9' or 'dr10'.

    Returns
    -------
    galaxies : structured numpy array
        Array of properties for candidate sources needed
        for host association.

    cat_col_fields : list
        List of columns retrieved from the galaxy catalog
        (rather than calculated internally).

    """
    if shred_cut:
        logger.warning("shred_cut is not implemented for decals galaxies at this time. Running with shred_cut = False.")

    transient_name = transient.name
    transient_pos = transient.position
    transient_pos_samples = transient.position_samples

    candidate_hosts = fetch_decals_sources(transient_pos, search_rad, cat_cols, calc_host_props, release)

    if candidate_hosts is None:
        return None, []

    galaxies_pos = SkyCoord(candidate_hosts["ra"].values * u.deg, candidate_hosts["dec"].values * u.deg)

    galaxies, cat_col_fields = build_galaxy_array(candidate_hosts, cat_cols, transient_name, "decals", release, logger)

    if galaxies is None:
        return None, []

    n_galaxies = len(galaxies)

    galaxies["objID_info"] = [f'decals {release}']*n_galaxies

    if 'offset' in calc_host_props:
        temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std = calc_shape_props_decals(candidate_hosts)

        dlr_samples = calc_dlr(
            transient_pos_samples, galaxies_pos, temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std
        )

        # Calculate angular separation between SN and all galaxies (in arcseconds)
        offset_samples = galaxies_pos[:, None].separation(transient_pos_samples).arcsec

        for i in range(n_galaxies):
            galaxies['offset_samples'][i] = offset_samples[i, :]
            galaxies["dlr_samples"][i] = dlr_samples[i, :]

        galaxies['offset_mean'] = np.nanmean(offset_samples, axis=1)
        galaxies['offset_std'] = np.nanstd(offset_samples, axis=1)

    if ('redshift' in calc_host_props) or ('absmag' in calc_host_props):
        galaxy_photoz_mean = candidate_hosts["z_phot_mean"].values
        galaxy_photoz_std = candidate_hosts["z_phot_std"].values
        galaxy_specz = candidate_hosts["z_spec"].values

        galaxies["redshift_mean"] = galaxy_photoz_mean
        galaxies["redshift_std"] = np.abs(galaxy_photoz_std)
        galaxies["redshift_info"] = ['photo-z']*n_galaxies

        #if we have spec-zs, replace those as the best redshift
        good_specz = galaxy_specz > REDSHIFT_FLOOR
        galaxies["redshift_mean"][good_specz] = galaxy_specz[good_specz]
        galaxies["redshift_std"][good_specz] = SIGMA_REDSHIFT_FLOOR * galaxy_specz[good_specz]  # floor of 5% for spec-zs
        galaxies["redshift_info"][good_specz] = 'spec-z'
        galaxies["redshift_std"][galaxy_photoz_std > (SIGMA_REDSHIFT_CEIL * galaxy_photoz_mean)] = (
            SIGMA_REDSHIFT_CEIL * galaxy_photoz_mean[galaxy_photoz_std > (SIGMA_REDSHIFT_CEIL * galaxy_photoz_mean)]
        )  # ceiling of 50%

        redshift_samples = norm.rvs(
            galaxies["redshift_mean"][:, np.newaxis],
            galaxies["redshift_std"][:, np.newaxis],
            size=(n_galaxies, n_samples),
        )

        # set photometric redshift floor
        redshift_samples[redshift_samples < REDSHIFT_FLOOR] = REDSHIFT_FLOOR

        for i in range(n_galaxies):
            galaxies["redshift_samples"][i] = redshift_samples[i, :]

    if 'absmag' in calc_host_props:
        temp_mag_r = candidate_hosts["dered_mag_r"].values

        temp_mag_r_std = np.abs(
            2.5
            / np.log(10)
            * np.sqrt(1 / np.maximum(PROB_FLOOR, candidate_hosts["flux_ivar_r"].values))
            / candidate_hosts["flux_r"].values
        )

        # cap at 50% the mag
        temp_mag_r_std[temp_mag_r_std > (SIGMA_ABSMAG_CEIL * temp_mag_r)] = SIGMA_ABSMAG_CEIL * temp_mag_r[temp_mag_r_std > (SIGMA_ABSMAG_CEIL * temp_mag_r)]

        # set a floor of 5%
        temp_mag_r_std[temp_mag_r_std < (SIGMA_ABSMAG_FLOOR * temp_mag_r)] = SIGMA_ABSMAG_FLOOR * temp_mag_r[temp_mag_r_std < (SIGMA_ABSMAG_FLOOR * temp_mag_r)]

        absmag_samples = (
            norm.rvs(
                loc=temp_mag_r[:, np.newaxis],
                scale=temp_mag_r_std[:, np.newaxis],
                size=(n_galaxies, n_samples),
            )
            - cosmo.distmod(redshift_samples).value
        )

        galaxies['absmag_mean'] = temp_mag_r - cosmo.distmod(galaxies["redshift_mean"]).value
        galaxies['absmag_std'] = temp_mag_r_std
        galaxies["absmag_info"] = ["r"]*n_galaxies

        for i in range(n_galaxies):
            galaxies["absmag_samples"][i] = absmag_samples[i, :]

    return galaxies, cat_col_fields


def build_panstarrs_candidates(
    transient,
    cosmo,
    logger,
    glade_catalog=None,
    search_rad=None,
    n_samples=1000,
    calc_host_props=['redshift', 'absmag', 'offset'],
    cat_cols=False,
    release='dr2',
    shred_cut=True,
    dust_path=DEFAULT_DUST_PATH,
):
    """Populates a GalaxyCatalog object with candidates from a cone search of the
       panstarrs catalog (See https://outerspace.stsci.edu/display/PANSTARRS/ for details).

    Parameters
    ----------
    transient : Transient
        Custom Transient object with name, position, and positional uncertainties defined.
    transient_pos : astropy.coord.SkyCoord
        Position of transient to associate.
    cosmo : astropy cosmology
        Assumed cosmology for conversions.
    logger : logging.Logger
        Logger instance for messages to console or output file.
    search_rad : astropy Angle
        Radius for cone search.
    n_samples : int
        Number of samples for Monte Carlo association.
    glade_catalog : pandas.DataFrame
        The locally-packaged GLADE catalog (to avoid querying).
    calc_host_props : list
        Properties to calculate internally for each host ('offset', 'redshift', 'absmag').
    cat_cols : boolean
        If True, concatenates catalog fields for best host to final catalog.
    shred_cut : boolean
        If True, removes likely source shreds associated with the same candidate galaxy.
    dust_path : str
        Path to the dust map data files.

    Returns
    -------
    galaxies : structured numpy array
        Array of properties for candidate sources needed
        for host association.

    cat_col_fields : list
        List of columns retrieved from the galaxy catalog
        (rather than calculated internally).

    """
    transient_name = transient.name
    transient_pos = transient.position
    transient_pos_samples = transient.position_samples

    if shred_cut:
        logger.info("Running with shred_cut = True...")

    candidate_hosts = fetch_panstarrs_sources(transient_pos, search_rad, cat_cols, calc_host_props, logger, release)

    if candidate_hosts is None:
        return None, []

    if 'offset' in calc_host_props:
        temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std = calc_shape_props_panstarrs(candidate_hosts)

        galaxies_pos = SkyCoord(
            candidate_hosts["ra"].values * u.deg, candidate_hosts["dec"].values * u.deg
        )

        dlr_samples = calc_dlr(
            transient_pos_samples,
            galaxies_pos,
            temp_sizes,
            temp_sizes_std,
            a_over_b,
            a_over_b_std,
            phi,
            phi_std,
            n_samples=n_samples,
        )

    temp_mag_r = candidate_hosts["KronMag"].values
    temp_mag_r_std = candidate_hosts["KronMagErr"].values

    # cap at 50% the mag
    # set a floor of 5%
    temp_mag_r_std[temp_mag_r_std > (SIGMA_ABSMAG_CEIL*temp_mag_r)] = SIGMA_ABSMAG_CEIL * temp_mag_r[temp_mag_r_std > (SIGMA_ABSMAG_CEIL*temp_mag_r)]
    temp_mag_r_std[temp_mag_r_std < (SIGMA_ABSMAG_FLOOR * temp_mag_r)] = SIGMA_ABSMAG_FLOOR * temp_mag_r[temp_mag_r_std < (SIGMA_ABSMAG_FLOOR * temp_mag_r)]

    # shred logic
    if (shred_cut) and (len(candidate_hosts) > 1):
        logger.info(f"Removing panstarrs {release} shreds.")
        shred_idxs = find_panstarrs_shreds(
            candidate_hosts["objID"].values,
            galaxies_pos,
            temp_sizes,
            temp_sizes_std,
            a_over_b,
            a_over_b_std,
            phi,
            phi_std,
            temp_mag_r,
            logger,
        )

        if len(shred_idxs) > 0:
            left_idxs = ~candidate_hosts.index.isin(shred_idxs)
            candidate_hosts = candidate_hosts[left_idxs]
            temp_mag_r = temp_mag_r[left_idxs]
            temp_mag_r_std = temp_mag_r_std[left_idxs]
            galaxies_pos = galaxies_pos[left_idxs]
            dlr_samples = dlr_samples[left_idxs]

            logger.info(f"Removed {len(shred_idxs)} flagged panstarrs sources.")
        else:
            logger.info("No panstarrs shreds found.")

    galaxies, cat_col_fields = build_galaxy_array(candidate_hosts, cat_cols, transient_name, "panstarrs", release, logger)
    if galaxies is None:
        return None, []
    n_galaxies = len(galaxies)

    galaxies['objID_info'] = [f'pan-starrs {release}']*n_galaxies

    if ('redshift' in calc_host_props) or ('absmag' in calc_host_props):
        # get photozs from Andrew Engel's code!
        pkg = pkg_resources.files("astro_prost")
        pkg_data_file = pkg / "data" / "MLP_lupton.hdf5"

        with pkg_resources.as_file(pkg_data_file) as model_path:
            model, range_z = load_lupton_model(logger=logger, model_path=model_path, dust_path=dust_path)

        x = preprocess(candidate_hosts, path=os.path.join(dust_path, "sfddata-master"))
        posteriors, point_estimates, errors = evaluate(x, model, range_z)
        point_estimates[point_estimates < REDSHIFT_FLOOR] = REDSHIFT_FLOOR  # set photometric redshift floor
        #point_estimates[point_estimates != point_estimates] = REDSHIFT_FLOOR  # set photometric redshift floor

        # inflated sigma floor for this particular photoz model
        err_bool = errors < (5*SIGMA_REDSHIFT_FLOOR * point_estimates)
        errors[err_bool] = (5*SIGMA_REDSHIFT_FLOOR * point_estimates[err_bool])

        # not QUITE the mean of the posterior, but we're assuming it's gaussian :/
        # TODO -- sample from the full posterior!
        galaxies["redshift_mean"] = point_estimates
        galaxies["redshift_std"] = np.abs(errors)

        # if the source is within 1arcsec of a GLADE host, take that spec-z.
        if glade_catalog is not None:
            logger.debug("Cross-matching with GLADE for redshifts...")

            ra_min = transient_pos.ra.deg - 1
            ra_max = transient_pos.ra.deg + 1
            dec_min = transient_pos.dec.deg - 1
            dec_max = transient_pos.dec.deg + 1

            filtered_glade = glade_catalog[(glade_catalog["RAJ2000"] > ra_min) & (glade_catalog["RAJ2000"] < ra_max) &
                            (glade_catalog["DEJ2000"] > dec_min) & (glade_catalog["DEJ2000"] < dec_max)]

            glade_catalog = filtered_glade

            if len(glade_catalog) >= 1:
                glade_coords = SkyCoord(glade_catalog["RAJ2000"], glade_catalog["DEJ2000"], unit=(u.deg, u.deg))
                idx, seps, _ = match_coordinates_sky(galaxies_pos, glade_coords)
                mask_within_1arcsec = seps.arcsec < 1

                galaxies["redshift_mean"][mask_within_1arcsec] = glade_catalog["redshift"].values[
                    idx[mask_within_1arcsec]
                ]
                galaxies["redshift_std"][mask_within_1arcsec] = glade_catalog["redshift_std"].values[
                    idx[mask_within_1arcsec]
                ]

        redshift_samples = norm.rvs(
            galaxies["redshift_mean"][:, np.newaxis],
            galaxies["redshift_std"][:, np.newaxis],
            size=(n_galaxies, n_samples),
        )

        # set photometric redshift floor
        redshift_samples[redshift_samples < REDSHIFT_FLOOR] = REDSHIFT_FLOOR

        for i in range(n_galaxies):
            galaxies["redshift_samples"][i] = redshift_samples[i, :]

    if 'absmag' in calc_host_props:
        absmag_samples = (
            norm.rvs(
                loc=temp_mag_r[:, np.newaxis],
                scale=temp_mag_r_std[:, np.newaxis],
                size=(n_galaxies, n_samples),
            )
            - cosmo.distmod(redshift_samples).value
        )

        galaxies["absmag_mean"] = temp_mag_r - cosmo.distmod(galaxies["redshift_mean"]).value
        galaxies['absmag_std'] = temp_mag_r_std
        galaxies['absmag_info'] = ["r"]*n_galaxies

        for i in range(n_galaxies):
            galaxies["absmag_samples"][i] = absmag_samples[i, :]

    if 'offset' in calc_host_props:
        # Calculate angular separation between SN and all galaxies (in arcseconds)
        offset_samples = galaxies_pos[:, None].separation(transient_pos_samples).arcsec

        for i in range(n_galaxies):
            galaxies['offset_samples'][i] = offset_samples[i, :]
            galaxies["dlr_samples"][i] = dlr_samples[i, :]

        galaxies['offset_mean'] = np.nanmean(offset_samples, axis=1)
        galaxies['offset_std'] = np.nanstd(offset_samples, axis=1)

    return galaxies, cat_col_fields


def calc_dlr(transient_pos_samples, galaxies_pos, a, a_std, a_over_b, a_over_b_std, phi, phi_std, n_samples=1000):
    """Calculates the directional light radius (DLR) between candidate host and transient, following the
       general framework in Gupta et al. (2016).

    Parameters
    ----------
    transient_pos_samples : array of astropy.coord.SkyCoord objects
        Array of transient positions with positional uncertainties
    galaxies_pos : array of astropy.coord.SkyCoord objects
        Positions of candidate host galaxies.
    a : np.ndarray
        Semi-major axes of candidates, in arcsec.
    a_std : np.ndarray
        Error in semi-major axes of candidates, in arcsec.
    a_over_b : np.ndarray
        Axis ratio (major/minor) of candidates.
    a_over_b_std : np.ndarray
        Error in axis ratio.
    phi : np.ndarray
        Position angles of sources, in radians.
    phi_std : np.ndarray
        Error in position angle.
    n_samples : type
        Number of DLR samples to draw for Monte Carlo association.

    Returns
    -------
    dlr : np.ndarray
        2D array of DLR samples of dimensionality (n_galaxies, n_samples).

    """
    n_gals = len(galaxies_pos)

    transient_ra = transient_pos_samples.ra.deg
    transient_dec = transient_pos_samples.dec.deg

    if n_samples > 1:
        hosts_ra = galaxies_pos.ra.deg[:, np.newaxis]
        hosts_dec = galaxies_pos.dec.deg[:, np.newaxis]

        a = norm.rvs(loc=a[:, np.newaxis], scale=a_std[:, np.newaxis], size=(n_gals, n_samples))
        a_over_b = norm.rvs(
            loc=a_over_b[:, np.newaxis], scale=a_over_b_std[:, np.newaxis], size=(n_gals, n_samples)
        )
        phi = norm.rvs(loc=phi[:, np.newaxis], scale=phi_std[:, np.newaxis], size=(n_gals, n_samples))
    else:
        hosts_ra = galaxies_pos.ra.deg
        hosts_dec = galaxies_pos.dec.deg

    xr = (transient_ra - hosts_ra) * 3600
    yr = (transient_dec - hosts_dec) * 3600

    gam = np.arctan2(xr, yr)
    theta = phi - gam

    dlr = a / np.sqrt(((a_over_b) * np.sin(theta)) ** 2 + (np.cos(theta)) ** 2)

    return dlr


def find_panstarrs_shreds(objids, coords, a, a_std, a_over_b, a_over_b_std, phi, phi_std, appmag, logger):
    """
    Finds and removes potentially shredded sources in Pan-STARRS.
    Drops the dimmer source in any pair where the separation is less than
    either galaxy's DLR (sep/DLR < 1). Now vectorized!

    Parameters
    ----------
    objids : np.ndarray
        Catalog IDs of sources.
    coords : np.ndarray
        Astropy SkyCoord objects.
    a : np.ndarray
        Semi-major axes (arcsec).
    a_std : np.ndarray
        Errors in semi-major axes (arcsec).
    a_over_b : np.ndarray
        Axis ratios (major/minor).
    a_over_b_std : np.ndarray
        Errors in axis ratio.
    phi : np.ndarray
        Position angles (radians).
    phi_std : np.ndarray
        Errors in position angle.
    appmag : np.ndarray
        Apparent magnitude, in r.

    Returns
    -------
    dropidxs : np.ndarray
        Indices of sources flagged as shreds.
    """

    n = len(coords)
    if n < 2:
        return np.array([])

    # Compute pairwise separations (n x n)
    seps = coords[:, None].separation(coords[None, :]).arcsec
    np.fill_diagonal(seps, np.inf)  # Ignore self-comparisons

    #assume no positional uncertainty
    ra_err = np.finfo(float).eps * u.arcsec
    dec_err = np.finfo(float).eps * u.arcsec

    coord_samples = SkyCoord(
        ra=np.random.normal(coords.ra.deg, ra_err.to(u.deg).value, size=n)*u.deg,
        dec=np.random.normal(coords.dec.deg, dec_err.to(u.deg).value, size=n)*u.deg
    )

    # Compute DLR for each source; assume calc_dlr returns a 1D array of length n
    dlr = calc_dlr(coord_samples, coords, a, a_std, a_over_b, a_over_b_std, phi, phi_std, n_samples=1)

    # Create the shred condition: compare each pair’s separation to both sources’ DLRs.
    # Broadcasting: for each (i,j), compare seps[i,j] < dlr[i] or seps[i,j] < dlr[j]
    shred_mask = (seps < dlr[:, None]) | (seps < dlr[None, :])

    # Use upper-triangle indices to avoid duplicate comparisons.
    upper_i, upper_j = np.triu_indices(n, k=1)
    valid = shred_mask[upper_i, upper_j]
    pairs = np.column_stack((upper_i[valid], upper_j[valid]))

    # For each pair, choose the index of the dimmer source.
    i_vals, j_vals = pairs[:, 0], pairs[:, 1]
    dropidxs = np.where(appmag[i_vals] > appmag[j_vals], i_vals, j_vals)
    dropidxs = np.unique(dropidxs)

    # (Optional) Log remaining sources
    keepidxs = np.setdiff1d(np.arange(n), dropidxs)
    logger.debug(f"{len(keepidxs)} sources remaining after shredding:")
    for idx in keepidxs:
        logger.debug(
            f"  objID {objids[idx]} | ra, dec: {coords[idx].ra.deg:.6f}, {coords[idx].dec.deg:.6f} | "
            f"App. Mag (r): {appmag[idx]:.2f} | DLR: {dlr[idx]:.2f}"
        )

    return dropidxs


def is_service_available(url, timeout=5):
    try:
        requests.head(url, timeout=timeout)
        return True
    except Exception:
        return False
