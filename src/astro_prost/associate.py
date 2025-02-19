import os
import pathlib
from mpire import WorkerPool
import time
from urllib.error import HTTPError
import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM
import importlib.resources as pkg_resources
import importlib
import logging
from .diagnose import plot_match
from .helpers import GalaxyCatalog, Transient
from collections import OrderedDict
import warnings
import re

# Parallel processing settings
NPROCESS_MAX = os.cpu_count() - 4

# Default survey releases
DEFAULT_RELEASES = {
    "glade": "latest",
    "decals": "dr9",
    "panstarrs": "dr2"
}

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

# Define a TRACE level below DEBUG -- for lengthy printouts
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")
logging.Logger.trace = trace

# Filter unnecessary warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

def log_host_properties(logger, transient_name, cat, host_idx, title, print_props, calc_host_props):
    """Log selected host galaxy properties for a transient.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to output messages.
    transient_name : str
        Name of the transient.
    cat : GalaxyCatalog
        Catalog containing candidate host galaxies.
    host_idx : int
        Index of the host galaxy in the catalog.
    title : str
        Header text for the log output.
    print_props : list of str
        List of property names to log directly (e.g., 'objID', 'ra', 'dec').
    calc_host_props : list of str
        List of properties (e.g., 'redshift', 'absmag', 'offset') for which mean, std, and posterior values are logged.

    Returns
    -------
    None
        Logs the formatted host properties.
    """

    prop_lines = [f"\n    {title} for {transient_name}:"]

    # Define all possible properties with labels and formats
    prop_format = {
        "objID": ("objID", "{:d}"),
        "ra": ("R.A. (deg)", "{:.6f}"),
        "dec": ("Dec. (deg)", "{:.6f}"),
        "redshift": ("Redshift", "{:.4f}"),
        "absmag": ("Absolute Magnitude", "{:.1f}"),
        "offset": (r"Transient Offset", "{:.1f}"),
        "posterior": ("Posterior", "{:.4e}"),
    }

    # Iterate through selected properties
    for prop in print_props:
        if prop in cat.galaxies.dtype.names:  # Only include if property exists
            label, fmt = prop_format.get(prop.split("_")[-1], (prop, "{:.4f}"))  # Default fmt if missing
            value = fmt.format(cat.galaxies[prop][host_idx])
            print_str = f"    {label}: {value}"
            prop_lines.append(print_str)

    # get mean, std, and posterior for specific properties
    for prop in calc_host_props:
        if f"{prop}_mean" in cat.galaxies.dtype.names:
            label, fmt = prop_format.get(prop, (prop, "{:.4f}"))  # Get format or default

            mean_value = fmt.format(cat.galaxies[f"{prop}_mean"][host_idx])
            std_value = fmt.format(cat.galaxies[f"{prop}_std"][host_idx])
            _, fmt = prop_format.get("posterior")
            posterior = fmt.format(cat.galaxies[f"{prop}_posterior"][host_idx])

            info = cat.galaxies[f"{prop}_info"][host_idx]

            print_str = f"    {label}: {mean_value} ± {std_value}"
            if prop == 'offset':
                print_str += " arcsec"
            if len(info) > 0:
                print_str += f" ({info})"
            prop_lines.append(print_str)

            prop_lines.append(f"    {label} Posterior: {posterior}")

    logger.info("\n".join(prop_lines))

def get_catalogs(user_input):
    """Convert user input into a dictionary mapping catalog names to release versions.

    Parameters
    ----------
    user_input : iterable
        An iterable of catalog entries, where each entry is either a string (catalog name)
        or a tuple (catalog name, release version).

    Returns
    -------
    dict
        A dictionary with keys as sanitized catalog names and values as the corresponding release version.
    """

    return {
        sanitize_input(cat) if isinstance(cat, str) else sanitize_input(cat[0]):
        cat[1] if isinstance(cat, tuple) else DEFAULT_RELEASES[sanitize_input(cat)]
        for cat in user_input
    }

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

def setup_logger(log_file=None, verbose=2, is_main=False):
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
        return logger

    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG, 3: TRACE_LEVEL}
    logger.setLevel(log_levels.get(verbose, logging.INFO)) #default to info (debug = 1)

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

def associate_transient(
    idx,
    row,
    glade_catalog,
    n_samples,
    priors,
    likes,
    cosmo,
    catalogs,
    cat_priority,
    cat_cols,
    log_fn,
    calc_host_props=False
):
    """Associates a transient with its most likely host galaxy.

    Parameters
    ----------
    idx : int
        Index of the transient from a larger catalog (used to cross-match properties after association).
    row : pandas Series
        Full row of transient properties.
    glade_catalog : pandas.DataFrame
        GLADE catalog of galaxies, with sizes and photo-zs.
    n_samples : int
        Number of samples for the monte-carlo sampling of associations.
    priors : dict
        Dictionary of priors for the run (at least one of redshift, offset, absolute magnitude).!
    likes : dict
        Dictionary of likelihoods for the run (at least one of offset, absolute magnitude).
    cosmo : astropy.cosmology
        Assumed cosmology for the run (defaults to LambdaCDM if unspecified).
    catalogs : dict
        Dict of source catalogs to query, with required key "name" and optional key "release".
    cat_priorities : dict
        The priority order to run the associations (with value 1 will run first, 2nd will run 2nd, etc). If None, defaults to the order
        the catalogs are provided in.
    cat_cols : boolean
        If true, concatenates the source catalog fields to the returned dataframe.
    log_fn : str, optional
        The fn associated with the logger.Logger object.
    calc_host_props : boolean
        If true, calculates host galaxy properties even if not needed for association
    Returns
    -------
    tuple
        Properties of the first and second-best host galaxy matches, and
        a dictionary of catalog columns (empty if cat_cols=False)

    """

    logger = setup_logger(log_fn, is_main=False)

    # TODO change overloaded variable here
    if calc_host_props:
        required = {'redshift', 'absmag', 'offset'}
        calc_host_props = list(required)
        missing = required - set(priors.keys())
        if missing:
            raise ValueError(f"To calculate all properties, priors must be defined for: {', '.join(missing)}.")
    else:
        calc_host_props = list(priors.keys())

    try:
        transient = Transient(
            name=row["name"],
            position=SkyCoord(row.transient_ra_deg * u.deg, row.transient_dec_deg * u.deg),
            redshift=float(row.redshift),
            n_samples=n_samples,
            logger=logger,
        )
    except (KeyError, AttributeError):
        transient = Transient(
            name=row["name"], position=SkyCoord(row.transient_ra_deg * u.deg, row.transient_dec_deg * u.deg), n_samples=n_samples, logger=logger
        )

    logger.info(
        f"\n\nAssociating {transient.name} at RA, DEC = "
        f"{transient.position.ra.deg:.6f}, {transient.position.dec.deg:.6f}"
    )

    for key, val in priors.items():
        transient.set_prior(key, val)

    for key, val in likes.items():
        transient.set_likelihood(key, val)

    if 'redshift' in priors.keys():
        transient.gen_z_samples(n_samples=n_samples)

    # Define result fields and initialize all values
    result = {
        "idx": idx,
        "best_cat": None,
        "best_cat_release": None,
        "query_time": np.nan,
        "smallcone_prob": np.nan,
        "missedcat_prob": np.nan,
        "extra_cat_cols": {}
    }

    # Define the fields that we extract for best and second-best hosts
    fields = ["objID", "total_posterior", "ra", "dec", "redshift_mean", "redshift_std"]

    # Initialize best and second-best fields with NaNs
    for key in ["best", "second_best"]:
        for field in fields:
            result[f"{key}_{field}"] = np.nan

    if cat_priority is not None:
        catalogs = sorted(
            catalogs,
            key=lambda cat: (cat_priority.get(cat[0], float("inf")), cat[1])  # Prioritize by catalog, then release
        )
        logger.info("Running association with the following catalog priorities:", catalogs)

    catalog_dict = OrderedDict(get_catalogs(catalogs))

    for cat_name, cat_release in catalog_dict.items():
        cat_release = catalog_dict[cat_name]
        cat = GalaxyCatalog(name=cat_name, n_samples=n_samples, data=glade_catalog, release=cat_release)

        try:
            cat.get_candidates(transient, time_query=True, logger=logger, cosmo=cosmo, calc_host_props=calc_host_props, cat_cols=cat_cols)
        except requests.exceptions.HTTPError:
            logger.warning(f"Candidate retrieval failed for {transient.name} in catalog {cat_name} due to an HTTPError.")
            continue

        if cat.ngals > 0:
            cat = transient.associate(cat, cosmo, calc_host_props=calc_host_props)

            if transient.best_host != -1:
                best_idx = transient.best_host
                second_best_idx = transient.second_best_host

                print_props = ['objID', 'ra', 'dec', 'total_posterior']
                log_host_properties(logger, transient.name, cat, best_idx, f"\nProperties of best host (in {cat_name} {cat_release})", print_props, calc_host_props)
                log_host_properties(logger, transient.name, cat, second_best_idx, f"\nProperties of 2nd best host (in {cat_name} {cat_release})", print_props, calc_host_props)

                # Populate results using a loop instead of manual assignments
                for key, idx in {"host": best_idx, "host_2": second_best_idx}.items():
                    for field in fields:
                        result[f"{key}_{field}"] = np.int64(cat.galaxies[field][idx]) if field == "objID" else cat.galaxies[field][idx]

                # Set additional metadata
                result.update({
                    "best_cat": cat_name,
                    "best_cat_release": cat_release,
                    "query_time": cat.query_time,
                    "smallcone_prob": transient.smallcone_prob,
                    "missedcat_prob": transient.missedcat_prob
                })

                # Collect extra catalog columns if needed
                if cat_cols:
                    result["extra_cat_cols"] = {field: cat.galaxies[field][best_idx] for field in cat.cat_col_fields}

                logger.info(
                    f"Chosen galaxy has catalog ID of {result['host_objID']} "
                    f"and RA, DEC = {result['host_ra']:.6f}, {result['host_dec']:.6f}"
                )

                if logger.getEffectiveLevel() == logging.DEBUG:
                    try:
                        plot_match(
                            [result["host_ra"]],
                            [result["host_dec"]],
                            result["host_redshift_mean"],
                            result["host_redshift_std"],
                            transient.position.ra.deg,
                            transient.position.dec.deg,
                            transient.name,
                            transient.redshift,
                            0,
                            f"{transient.name}_{cat_name}_{cat_release}",
                            logger
                        )
                    except HTTPError:
                        logger.warning("Couldn't get an image. Waiting 60s before moving on.")
                        time.sleep(60)
                        continue
                # Stop searching after first valid match
                break

    if transient.best_host == -1:
        logger.info("No good host found!")

    return result

def prepare_catalog(
    transient_catalog,
    transient_name_col="name",
    transient_coord_cols=("ra", "dec"),
    transient_redshift_col='redshift',
    debug=False,
    debug_names=None,
):
    """Preprocesses the transient catalog for fields needed by association function.

    Parameters
    ----------
    transient_catalog : pandas.DataFrame
        Contains the details of the transients to be associated.
    transient_name_col : str
        Column corresponding to transient name.
    transient_coord_cols : tuple
        Columns corresponding to transient coordinates (converted to decimal degrees internally).
    transient_redshift_col : string
        Column corresponding to transient redshift.
    debug : boolean
        If true, associates only transients in debug_names.
    debug_names : list
        List of specific transients to associate when debug=True.

    Returns
    -------
    pandas.DataFrame
        The transformed dataframe with standardized columns.

    """

    # debugging with just the ones we got wrong
    if debug and debug_names is not None:
        transient_catalog = transient_catalog[transient_catalog[transient_name_col].isin(debug_names)]

    # convert coords if needed
    if ":" in str(transient_catalog[transient_coord_cols[0]].values[0]):
        ra = transient_catalog[transient_coord_cols[0]].values
        dec = transient_catalog[transient_coord_cols[1]].values
        transient_coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    else:
        # try parsing as a float
        try:
            ra = transient_catalog[transient_coord_cols[0]].astype("float").values
            dec = transient_catalog[transient_coord_cols[1]].astype("float").values
            transient_coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        except KeyError as err:
            raise ValueError("ERROR: I could not understand your provided coordinates.") from err

    transient_catalog["transient_ra_deg"] = transient_coords.ra.deg
    transient_catalog["transient_dec_deg"] = transient_coords.dec.deg

    transient_catalog.rename(columns={transient_name_col: "name", transient_redshift_col: "redshift"}, inplace=True)

    # randomly shuffle
    transient_catalog = transient_catalog.sample(frac=1).reset_index(drop=True)
    transient_catalog.reset_index(inplace=True, drop=True)

    return transient_catalog

def associate_sample(
    transient_catalog,
    catalogs,
    cat_priority=None,
    run_name=None,
    priors=None,
    likes=None,
    n_samples=1000,
    verbose=1,
    parallel=True,
    save=True,
    save_path="./",
    log_path=None,
    cat_cols=False,
    progress_bar=False,
    cosmology=None,
    n_processes=None,
    calc_host_props=True
):
    """Wrapper function for associating sample of transients.

    Parameters
    ----------
    transient_catalog : pandas.DataFrame
        Dataframe containing transient name and coordinates.
    priors : dict
        Dictionary of prior distributions on redshift, fractional offset, and/or absolute magnitude
    likes : dict
        Dictionary of likelihood distributions on redshift, fractional offset, absolute magnitude
    catalogs : list
        List of catalogs to query (can include 'glade', 'decals', 'panstarrs')
    cat_priority : dict
        Dict of catalog priority (determines what gets run first)
    n_samples : int
        List of samples to draw for monte-carlo association.
    verbose : int
        Verbosity level for logging; can be 0 - 3.
    parallel : boolean
        If True, runs in parallel with multiprocessing via mpire. Cannot be set with ipython!
    save : boolean
        If True, saves resulting association table to save_path.
    save_path : str
        Path where the association table should be saved (when save=True).
    log_path : str
        Path where the logfile should be saved. If none, log everything to screen
    cat_cols : boolean
        If True, contatenates catalog columns to resulting DataFrame.
    progress_bar : boolean
        If True, prints a loading bar for each association (when parallel=True).
    cosmology : astropy.cosmology
        Assumed cosmology for the run (defaults to LambdaCDM if unspecified).
    n_processes : int
        Number of parallel processes to run when parallel=True (defaults to n_cores-4 if unspecified).
    calc_host_props : boolean
        If True, calculates all host properties (redshift, absmag, and fractional offset) regardless of whether or not
        they're needed for association.

    Returns
    -------
    pandas.DataFrame
        The transient dataframe with columns corresponding to the associated transient.

    """
    ts = int(time.time())

    envkey = 'PYSPAWN_' + os.path.basename(__file__)
    is_main = not os.environ.get(envkey, False)

    if log_path is not None:
        if run_name is not None:
            log_fn = f"{log_path}/Prost_log_{run_name}_{ts}.txt"
        else:
            log_fn = f"{log_path}/Prost_log_{ts}.txt"

        if is_main:
            logger = setup_logger(log_file=log_fn, verbose=verbose, is_main=is_main)
            logger.info(f"Created log file at {log_fn}.")
            os.environ['LOG_PATH_ENV'] = log_fn
        else:
            log_fn = os.environ.get("LOG_PATH_ENV", None)
            logger = setup_logger(log_file=log_fn, verbose=verbose, is_main=False)
    else:
        log_fn = None
        logger = setup_logger(verbose=verbose, is_main=is_main)

    if not cosmology:
        cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    possible_keys = ["offset", "absmag", "redshift"]

    priors = {k: v for k, v in priors.items() if k in possible_keys}
    likes = {k: v for k, v in likes.items() if k in possible_keys}

    # Validate that at least one prior remains
    if not priors:
        raise ValueError(f"ERROR: Please set a prior function for at least one of {possible_keys}.")
    if is_main:
        logger.info(f"Conditioning association on the following properties: {list(priors.keys())}")
    for key in priors:
        if key != 'redshift' and key not in likes:
            raise ValueError(f"ERROR: Please set a likelihood function for {key}.")

    # always load GLADE -- we now use it for spec-zs.
    pkg = pkg_resources.files("astro_prost")
    pkg_data_file = pkg / "data" / "GLADE+_HyperLedaSizes_mod_withz.csv"

    try:
        with pkg_resources.as_file(pkg_data_file) as csvfile:
            glade_catalog = pd.read_csv(csvfile)
    except FileNotFoundError:
        glade_catalog = None

    results = []

    events = [
        (
            idx,
            row,
            glade_catalog,
            n_samples,
            priors,
            likes,
            cosmo,
            catalogs,
            cat_priority,
            cat_cols,
            log_fn
        )
        for idx, row in transient_catalog.iterrows()
    ]

    if (parallel) and (is_main):
            # Set the environment variable in the parent process only
            os.environ[envkey] = str(os.getpid())  # Store the PID in the env var

            if (n_processes is None) or (n_processes > NPROCESS_MAX):
                logger.info("WARNING! Set n_processes to greater than the number of cpu cores on this machine."+
                       f" Falling back to n_processes = {NPROCESS_MAX}.")
                n_processes = NPROCESS_MAX

            # Create a list of tasks
            logger.info(f"Parallelizing {len(transient_catalog)} associations across {n_processes} processes.")

            with WorkerPool(n_jobs=n_processes, start_method='spawn') as pool:
                results = pool.map(associate_transient, events, progress_bar=progress_bar)
                pool.stop_and_join()
    else:
        results = [associate_transient(*event) for event in events]

    if not parallel or os.environ.get(envkey) == str(os.getpid()):

        # Convert results to a DataFrame
        results_df = pd.DataFrame.from_records(results)

        if "idx" not in results_df.columns:
            raise ValueError("No 'idx' column found in results, cannot update transient_catalog!")

        transient_catalog = transient_catalog.merge(
            results_df, left_index=True, right_on="idx", how="left"
        )

        # Collect extra catalog columns dynamically
        extra_cat_cols_list = [res["extra_cat_cols"] for res in results if "extra_cat_cols" in res]

        if extra_cat_cols_list:  # Ensure there's extra data before proceeding
            extra_cat_cols_df = pd.DataFrame.from_records(extra_cat_cols_list)

            if not extra_cat_cols_df.empty:
                # Merge extra catalog columns using transient_catalog's index
                transient_catalog = transient_catalog.merge(
                    extra_cat_cols_df, left_index=True, right_on="idx", how="left"
                )

        # Convert all ID columns to integers
        id_cols = [col for col in transient_catalog.columns if col.endswith("id")]

        for col in id_cols:
            transient_catalog[col] = pd.to_numeric(transient_catalog[col], errors="coerce").astype("Int64")

        # final log message -- we're done!
        logger.info("\n\nAssociation of all transients is complete.")

        # Save the updated catalog
        if save:
            save_name = pathlib.Path(save_path, f"associated_transient_catalog_{ts}.csv")
            transient_catalog.to_csv(save_name, index=False)
        else:
            return transient_catalog
    else:
        return transient_catalog
