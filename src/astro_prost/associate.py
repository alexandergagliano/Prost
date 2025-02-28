import os
import pathlib
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
from .diagnose import plot_match
from .helpers import GalaxyCatalog, Transient, setup_logger, sanitize_input
import logging
from collections import OrderedDict
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Parallel processing settings
NPROCESS_MAX = os.cpu_count() - 4

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# Default survey releases
DEFAULT_RELEASES = {
    "glade": "latest",
    "decals": "dr9",
    "panstarrs": "dr2"
}

# Filter unnecessary warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

def save_results(results, transient_catalog, run_name=None, save_path='./'):
    ts = int(time.time())
    valid_results = [r for r in results if r is not None]

    results_df = pd.DataFrame.from_records(valid_results)

    extra_cat_cols_list = [res["extra_cat_cols"] for res in valid_results if isinstance(res, dict) and "extra_cat_cols" in res]

    if extra_cat_cols_list:
        extra_cat_cols_DF = pd.DataFrame.from_records(extra_cat_cols_list)
        results_df = results_df.join(extra_cat_cols_DF)

    if "idx" not in results_df.columns:
        raise ValueError("No 'idx' column found in results, cannot update transient_catalog!")

    transient_catalog = transient_catalog.merge(
        results_df, left_index=True, right_on="idx", how="left"
    )

    # Convert all ID columns to integers
    id_cols = [col for col in transient_catalog.columns if col.endswith("id")]

    for col in id_cols:
        transient_catalog[col] = pd.to_numeric(transient_catalog[col], errors="coerce").astype("Int64")

    # Save the updated catalog
    save_name = pathlib.Path(save_path, f"associated_transient_catalog_{run_name or ''}_{ts}.csv")
    transient_catalog.to_csv(save_name, index=False)

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
    calc_host_props=False,
    verbose=0
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
        Number of samples for the Monte Carlo sampling of associations.
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

    logger = setup_logger(log_fn, verbose=verbose, is_main=False)

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
        "smallcone_posterior": np.nan,
        "missedcat_posterior": np.nan,
        "extra_cat_cols": {}
    }

    # Define the fields that we extract for best and second-best hosts
    fields = ["objID", "total_posterior", "ra", "dec", "redshift_mean", "redshift_std"]

    for prop in calc_host_props:
        fields.append(f"{prop}_mean")
        fields.append(f"{prop}_std")
        fields.append(f"{prop}_posterior")

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
                    "smallcone_posterior": transient.smallcone_posterior,
                    "missedcat_posterior": transient.missedcat_posterior,
                    "any_posterior": transient.any_posterior,
                    "none_posterior": transient.none_posterior,
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
    run_name : str or None
        Optional name for the run -- used to name logfiles
    n_samples : int
        List of samples to draw for monte-carlo association.
    verbose : int
        Verbosity level for logging; can be 0 - 3.
    parallel : boolean
        If True, runs in parallel with multiprocessing. Cannot be used with ipython!
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
    pkg_data_file = pkg / "data" / "GLADE+_HyperLedaSizes_mod_withz.csv.gz"
    try:
        with pkg_resources.as_file(pkg_data_file) as csvfile:
            glade_catalog = pd.read_csv(csvfile, compression="gzip", low_memory=False)
        if glade_catalog is not None:
            logger.info("Loaded GLADE+ catalog.")
    except FileNotFoundError:
        logger.warning("Could not find GLADE+ catalog.")
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
            log_fn,
            verbose
        )
        for idx, row in transient_catalog.iterrows()
    ]

    if (parallel) and (is_main):
        # Set the environment variable in the parent process only
        os.environ[envkey] = str(os.getpid())  # Store the PID in the env var

        if (n_processes is None) or (n_processes > NPROCESS_MAX):
            logger.info("WARNING! Set n_processes to greater than the number of cpu cores on this machine." +
                        f" Falling back to n_processes = {NPROCESS_MAX}.")
            n_processes = NPROCESS_MAX

        logger.info(f"Parallelizing {len(transient_catalog)} associations across {n_processes} processes in batches.")

        # Define your batch size, e.g., 250 events per batch
        batch_size = 250

        results = {}  # accumulate results here
        total_batches = int(np.ceil(len(events) / batch_size))
        for batch_num, batch in enumerate(chunks(events, batch_size), start=1):
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} events.")
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                futures = {executor.submit(safe_associate_transient, *event): event[0] for event in batch}
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"Batch {batch_num}"):
                    try:
                        results[futures[future]] = future.result()
                    except Exception as e:
                        logger.error(f"Unhandled error for event {futures[future]}: {e}", exc_info=True)
                        results[futures[future]] = None

            # Save intermediate results and clear any large temporary data if needed
            print("Saving intermediate batch results...")
            save_results(results, transient_catalog, run_name, save_path)

            # Optionally, force garbage collection
            gc.collect()
            max_retries = 3
            retry = 0
            while retry < max_retries:
                failed_ids = [event_id for event_id, res in results.items() if res is None]
                if not failed_ids:
                    logger.info("All associations succeeded; no more retries needed.")
                    break

                logger.info(f"Retry attempt {retry+1}: Rerunning {len(failed_ids)} failed associations.")

                # Select events corresponding to failed_ids. (Recall: event[0] is the transient index.)
                failed_events = [event for event in events if event[0] in failed_ids]

                with ProcessPoolExecutor(max_workers=n_processes) as executor:
                    new_futures = {executor.submit(safe_associate_transient, *event): event[0] for event in failed_events}
                    for future in tqdm(as_completed(new_futures),
                                       total=len(new_futures),
                                       desc="Retrying events"):
                        try:
                            results[new_futures[future]] = future.result()
                        except Exception as e:
                            logger.error(f"Retry failed for event {new_futures[future]}: {e}", exc_info=True)
                            results[new_futures[future]] = None

                retry += 1

                if retry == max_retries:
                    logger.warning("Some associations still failed after maximum retries.")

    else:
        results = [associate_transient(*event) for event in events]

    if not parallel or os.environ.get(envkey) == str(os.getpid()):
        # Convert results to a DataFrame
        save_results(results, transient_catalog, run_name, save_path)
    else:
        return transient_catalog

def safe_associate_transient(idx,
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
verbose):
    try:
        return associate_transient(idx,
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
        verbose)
    except Exception as e:
        logger.exception(f"Error processing event {idx}: {e}")
        return None
