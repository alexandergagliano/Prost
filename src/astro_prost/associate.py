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

NPROCESS_MAX = os.cpu_count() - 4

def setup_logger(log_file=None, verbose=2):
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

    if logger.hasHandlers():  # 🔥 Prevents duplicate handlers
        logger.handlers.clear()  # Remove existing handlers before adding new ones

    log_levels = {0:logging.WARNING, 1:logging.INFO, 2:logging.DEBUG}
    logger.setLevel(log_levels[verbose])  # Set the logging level

    # Define log format (adds timestamps, log level, and message)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler (prints to stdout)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_path = os.path.dirname(log_file)
        os.makedirs(log_path, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def associate_transient(
    idx,
    row,
    glade_catalog,
    n_samples,
    verbose,
    priors,
    likes,
    cosmo,
    catalogs,
    cat_cols,
    logger
):
    """Associates a transient with its most likely host galaxy.

    Parameters
    ----------
    idx : int
        Index of the transient from a larger catalog (used to cross-match properties after association).
    row : pandas Series
        Full row of transient properties.
    glade_catalog : Pandas DataFrame
        GLADE catalog of galaxies, with sizes and photo-zs.
    n_samples : int
        Number of samples for the monte-carlo sampling of associations.
    verbose : int
        Level of logging during run (can be 0, 1, or 2).
    priors : dict
        Dictionary of priors for the run (at least one of redshift, offset, absolute magnitude).!
    likes : dict
        Dictionary of likelihoods for the run (at least one of offset, absolute magnitude).
    cosmo : astropy cosmology
        Assumed cosmology for the run (defaults to LambdaCDM if unspecified).
    catalogs : list
        List of source catalogs to query (can include 'glade', 'decals', or 'panstarrs').
    cat_cols : boolean
        If true, concatenates the source catalog fields to the returned dataframe.
    Returns
    -------
    tuple
        Properties of the first and second-best host galaxy matches, and
        a dictionary of catalog columns (empty if cat_cols=False)

    """
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

    if verbose > 0:
        logger.info(
            f"Associating {transient.name} at RA, DEC = "
            f"{transient.position.ra.deg:.6f}, {transient.position.dec.deg:.6f}"
        )

    included_properties = list(set(priors.keys()).intersection(set(['redshift', 'absmag', 'offset'])))

    for key, val in priors.items():
        if key in included_properties:
            transient.set_prior(key, val)
            logger.info(f"Setting prior for {key}.")

    for key, val in likes.items():
        if key in included_properties:
            logger.info(f"Setting likelihood for {key}.")
            transient.set_likelihood(key, val)

    if 'redshift' in priors.keys():
        transient.gen_z_samples(n_samples=n_samples)

    (
        best_objid, best_prob, best_ra, best_dec, best_z_mean,
        best_z_std, second_best_objid, second_best_prob, second_best_ra,
        second_best_dec, second_best_z_mean, second_best_z_std, query_time, smallcone_prob, missedcat_prob
    ) = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan
    )

    extra_cat_cols = {}
    best_cat = ""

    for cat_name in catalogs:
        cat = GalaxyCatalog(name=cat_name, n_samples=n_samples, data=glade_catalog)

        try:
            cat.get_candidates(transient, time_query=True, logger=logger, verbose=verbose, cosmo=cosmo, cat_cols=cat_cols)
        except requests.exceptions.HTTPError:
            logger.info(f"Candidate retrieval failed for {transient.name} in catalog {cat_name}.")
            continue

        if cat.ngals > 0:
            cat = transient.associate(cat, cosmo, logger, verbose=verbose)

            if transient.best_host != -1:
                best_idx = transient.best_host
                second_best_idx = transient.second_best_host

                if verbose >= 2:
                    print_cols = [
                        "objID",
                        "z_prob",
                        "offset_prob",
                        "absmag_prob",
                        "total_prob",
                        "ra",
                        "dec",
                        "offset_arcsec",
                        "z_best_mean",
                        "z_best_std",
                    ]
                    logger.info("Properties of best host:")
                    for key in print_cols:
                        if "prob" in key:
                            logger.info(f"\t{key}: {cat.galaxies[key][best_idx]:.4e}")
                        else:
                            logger.info(f"\t{key}: {cat.galaxies[key][best_idx]:.4f}")
                    logger.info("")
                    logger.info("Properties of second best host:")
                    for key in print_cols:
                        if "prob" in key:
                            logger.info(f"\t{key}: {cat.galaxies[key][second_best_idx]:.4e}")
                        else:
                            logger.info(f"\t{key}: {cat.galaxies[key][second_best_idx]:.4f}")
                logger.info("")

                best_objid = np.int64(cat.galaxies["objID"][best_idx])
                best_prob = cat.galaxies["total_prob"][best_idx]
                best_ra = cat.galaxies["ra"][best_idx]
                best_dec = cat.galaxies["dec"][best_idx]
                best_z_mean = cat.galaxies['z_best_mean'][best_idx]
                best_z_std = cat.galaxies['z_best_std'][best_idx]

                second_best_objid = np.int64(cat.galaxies["objID"][second_best_idx])
                second_best_prob = cat.galaxies["total_prob"][second_best_idx]
                second_best_ra = cat.galaxies["ra"][second_best_idx]
                second_best_dec = cat.galaxies["dec"][second_best_idx]
                second_best_z_mean = cat.galaxies['z_best_mean'][second_best_idx]
                second_best_z_std = cat.galaxies['z_best_std'][second_best_idx]

                best_cat = cat_name
                query_time = cat.query_time
                smallcone_prob = transient.smallcone_prob
                missedcat_prob = transient.missedcat_prob

                if cat_cols:
                    for field in cat.cat_col_fields:
                        extra_cat_cols[field] = cat.galaxies[field][best_idx]

                if verbose > 0:
                    logger.info(
                        f"Chosen {cat_name} galaxy has catalog ID of {best_objid}"
                        f" and RA, DEC = {best_ra:.6f}, {best_dec:.6f}"
                    )
                if verbose > 1:
                    try:
                        plot_match(
                            [best_ra],
                            [best_dec],
                            best_z_mean,
                            best_z_std,
                            transient.position.ra.deg,
                            transient.position.dec.deg,
                            transient.name,
                            transient.redshift,
                            0,
                            f"{transient.name}_{cat_name}",
                            logger
                        )
                    except HTTPError:
                        logger.info("Couldn't get an image. Waiting 60s before moving on.")
                        time.sleep(60)
                        continue

    if (transient.best_host == -1) and (verbose > 0):
        logger.info("No good host found!")
    return (
        idx,
        best_objid,
        best_prob,
        best_ra,
        best_dec,
        best_z_mean,
        best_z_std,
        second_best_objid,
        second_best_prob,
        second_best_ra,
        second_best_dec,
        second_best_z_mean,
        second_best_z_std,
        query_time,
        best_cat,
        smallcone_prob,
        missedcat_prob,
        extra_cat_cols
    )


def prepare_catalog(
    transient_catalog,
    transient_name_col="name",
    transient_coord_cols=("ra", "dec"),
    debug=False,
    debug_names=None,
):
    """Preprocesses the transient catalog for fields needed by association function.

    Parameters
    ----------
    transient_catalog : Pandas DataFrame
        Contains the details of the transients to be associated.
    transient_name_col : str
        Column corresponding to transient name.
    transient_coord_cols : tuple
        Columns corresponding to transient coordinates (converted to decimal degrees internally).
    debug : boolean
        If true, associates only transients in debug_names.
    debug_names : list
        List of specific transients to associate when debug=True.

    Returns
    -------
    Pandas DataFrame
        The transformed dataframe with standardized columns.

    """
    association_fields = [
        "host_id",
        "host_ra",
        "host_dec",
        "host_prob",
        "host_z_best",
        "host_z_std",
        "host_2_id",
        "host_2_ra",
        "host_2_dec",
        "host_2_prob",
        "host_2_z_best",
        "host_2_z_std",
        "smallcone_prob",
        "missedcat_prob",
        "association_time",
    ]

    for field in association_fields:
        transient_catalog[field] = np.nan

    transient_catalog["prob_host_flag"] = 0

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

    transient_catalog.rename(columns={transient_name_col: "name"}, inplace=True)

    # randomly shuffle
    transient_catalog = transient_catalog.sample(frac=1).reset_index(drop=True)
    transient_catalog.reset_index(inplace=True, drop=True)

    return transient_catalog

def associate_sample(
    transient_catalog,
    catalogs,
    run_name=None,
    priors=None,
    likes=None,
    n_samples=1000,
    verbose=False,
    parallel=True,
    save=True,
    save_path="./",
    log_path=None,
    cat_cols=False,
    progress_bar=False,
    cosmology=None,
    n_processes=None,
):
    """Wrapper function for associating sample of transients.

    Parameters
    ----------
    transient_catalog : Pandas DataFrame
        Dataframe containing transient name and coordinates.
    priors : dict
        Dictionary of prior distributions on redshift, fractional offset, and/or absolute magnitude
    likes : dict
        Dictionary of likelihood distributions on redshift, fractional offset, absolute magnitude
    catalogs : list
        List of catalogs to query (can include 'glade', 'decals', 'panstarrs')
    n_samples : int
        List of samples to draw for monte-carlo association.
    verbose : int
        Verbosity level; can be 0, 1, or 2.
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
    cosmology : astropy cosmology
        Assumed cosmology for the run (defaults to LambdaCDM if unspecified).
    n_processes : int
        Number of parallel processes to run when parallel=True (defaults to n_cores-4 if unspecified).

    Returns
    -------
    Pandas DataFrame
        The transient dataframe with columns corresponding to the associated transient.

    """
    ts = int(time.time())
    if log_path is not None:
        if run_name is not None:
            log_fn = f"{log_path}/Prost_log_{run_name}_{ts}.txt"
        else:
            log_fn = f"{log_path}/Prost_log_{ts}.txt"
        logger = setup_logger(log_fn, verbose)
        logger.info(f"Created log file at {log_fn}.")
    else:
        logger = setup_logger(verbose=verbose)
    if not cosmology:
        cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    possible_keys = ["offset", "absmag", "redshift"]
    if not any(key in priors for key in possible_keys):
        raise ValueError(f"ERROR: Please set a prior function for at least one of {possible_keys}.")

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
            verbose,
            priors,
            likes,
            cosmo,
            catalogs,
            cat_cols,
            logger
        )
        for idx, row in transient_catalog.iterrows()
    ]


    if parallel:
        envkey = 'PYSPAWN_' + os.path.basename(__file__)

        if not os.environ.get(envkey, False):
            # Set the environment variable in the parent process only
            os.environ[envkey] = str(os.getpid())  # Store the PID in the env var

            if (n_processes is None) or (n_processes > NPROCESS_MAX):
                logger.info("WARNING! Set n_processes to greater than the number of cpu cores on this machine."+
                       f" Falling back to n_processes = {NPROCESS_MAX}.")
                n_processes = NPROCESS_MAX

            # Create a list of tasks
            if verbose > 0:
                logger.info(f"Parallelizing {len(transient_catalog)} associations across {n_processes} processes.")

            with WorkerPool(n_jobs=n_processes, start_method='spawn') as pool:
                results = pool.map(associate_transient, events, progress_bar=progress_bar)
                pool.stop_and_join()
    else:
        results = [associate_transient(*event) for event in events]

    if not parallel or os.environ.get(envkey) == str(os.getpid()):
        # Update transient_catalog with results

        main_results = [res[:-1] for res in results]

        results_df = pd.DataFrame.from_records(
            main_results,
            columns=[
                "idx", "host_id", "host_prob", "host_ra", "host_dec",
                "host_z_best", "host_z_std",
                "host_2_objid", "host_2_prob", "host_2_ra", "host_2_dec",
                "host_2_z_best", "host_2_z_std",
                "query_time", "best_cat", "smallcone_prob", "missedcat_prob"
            ]
        )

        transient_catalog.update(results_df.set_index("idx"))

        if cat_cols:
            extra_cat_cols_list = [res[-1] for res in results]
            extra_cat_cols_df = pd.DataFrame.from_records(extra_cat_cols_list)
            extra_cols = extra_cat_cols_df.columns
            extra_cat_cols_df['idx'] = results_df['idx']
            transient_catalog = pd.concat([transient_catalog, extra_cat_cols_df.set_index("idx")], axis=1)

        id_cols = [col for col in transient_catalog.columns if col.endswith('id')]

        for col in id_cols:
            transient_catalog[col] = pd.to_numeric(transient_catalog[col], errors='coerce').astype('Int64')

        logger.info("Association of all transients is complete.")

        # Save the updated catalog
        if save:
            save_name = pathlib.Path(save_path, f"associated_transient_catalog_{ts}.csv")
            transient_catalog.to_csv(save_name, index=False)
        else:
            return transient_catalog
    else:
        return transient_catalog
