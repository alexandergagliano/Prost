import os
import pathlib
from mpire import WorkerPool
from time import time
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
from .helpers import GalaxyCatalog, Transient

def associate_transient(
    idx,
    row,
    glade_catalog,
    n_samples,
    verbose,
    priorfunc_z,
    priorfunc_offset,
    priorfunc_absmag,
    likefunc_offset,
    likefunc_absmag,
    cosmo,
    catalogs,
    cat_cols,
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
    priorfunc_z : scipy stats continuous distribution
        Prior distribution on redshift. This class can be user-defined
        but needs .sample(size=n) and .pdf(x) functions.
    priorfunc_offset : scipy stats continuous distribution
        Prior distribution on fractional offset.
    priorfunc_absmag : scipy stats continuous distribution
        Prior distribution on host absolute magnitude.
    likefunc_offset : scipy stats continuous distribution
        Likelihood distribution on fractional offset.
    likefunc_absmag : scipy stats continuous distribution.
        Likelihood distribution on host absolute magnitude.
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
        )
    except (KeyError, AttributeError):
        transient = Transient(
            name=row["name"], position=SkyCoord(row.transient_ra_deg * u.deg, row.transient_dec_deg * u.deg), n_samples=n_samples
        )

    if verbose > 0:
        print(
            f"Associating {transient.name} at RA, DEC = "
            f"{transient.position.ra.deg:.6f}, {transient.position.dec.deg:.6f}"
        )

    transient.set_prior("redshift", priorfunc_z)
    transient.set_prior("offset", priorfunc_offset)
    transient.set_prior("absmag", priorfunc_absmag)

    transient.set_likelihood("offset", likefunc_offset)
    transient.set_likelihood("absmag", likefunc_absmag)

    transient.gen_z_samples(n_samples=n_samples)

    (
        best_objid, best_prob, best_ra, best_dec,
        second_best_objid, second_best_prob, second_best_ra,
        second_best_dec, query_time, smallcone_prob, missedcat_prob
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
        np.nan
    )

    extra_cat_cols = {}
    best_cat = ""

    for cat_name in catalogs:
        cat = GalaxyCatalog(name=cat_name, n_samples=n_samples, data=glade_catalog)

        try:
            cat.get_candidates(transient, time_query=True, verbose=verbose, cosmo=cosmo, cat_cols=cat_cols)
        except requests.exceptions.HTTPError:
            print(f"Candidate retrieval failed for {transient.name} in catalog {cat_name}.")
            continue

        if cat.ngals > 0:
            cat = transient.associate(cat, cosmo, verbose=verbose)

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
                    print("Properties of best host:")
                    for key in print_cols:
                        print(key)
                        print(cat.galaxies[key][best_idx])

                    print("Properties of second best host:")
                    for key in print_cols:
                        print(key)
                        print(cat.galaxies[key][second_best_idx])

                best_objid = np.int64(cat.galaxies["objID"][best_idx])
                best_prob = cat.galaxies["total_prob"][best_idx]
                best_ra = cat.galaxies["ra"][best_idx]
                best_dec = cat.galaxies["dec"][best_idx]

                second_best_objid = np.int64(cat.galaxies["objID"][second_best_idx])
                second_best_prob = cat.galaxies["total_prob"][second_best_idx]
                second_best_ra = cat.galaxies["ra"][second_best_idx]
                second_best_dec = cat.galaxies["dec"][second_best_idx]

                best_cat = cat_name
                query_time = cat.query_time
                smallcone_prob = transient.smallcone_prob
                missedcat_prob = transient.missedcat_prob

                if cat_cols:
                    for field in cat.cat_col_fields:
                        extra_cat_cols[field] = cat.galaxies[field][best_idx]

                if verbose > 0:
                    print(
                        f"Chosen {cat_name} galaxy has catalog ID of {best_objid}"
                        f" and RA, DEC = {best_ra:.6f}, {best_dec:.6f}"
                    )
                if verbose > 1:
                    try:
                        plot_match(
                            [best_ra],
                            [best_dec],
                            None,
                            None,
                            cat.galaxies["z_best_mean"][best_idx],
                            cat.galaxies["z_best_std"][best_idx],
                            transient.position.ra.deg,
                            transient.position.dec.deg,
                            transient.name,
                            transient.redshift,
                            0,
                            f"{transient.name}_{cat_name}",
                        )
                    except HTTPError:
                        print("Couldn't get an image. Waiting 60s before moving on.")
                        time.sleep(60)
                        continue

    if (transient.best_host == -1) and (verbose > 0):
        print("No good host found!")
    return (
        idx,
        best_objid,
        best_prob,
        best_ra,
        best_dec,
        second_best_objid,
        second_best_prob,
        second_best_ra,
        second_best_dec,
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
        "host_2_id",
        "host_2_ra",
        "host_2_dec",
        "host_2_prob",
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
    priors=None,
    likes=None,
    n_samples=1000,
    verbose=False,
    parallel=True,
    save=True,
    save_path="./",
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
        Dictionary of prior distributions on redshift, fractional offset, absolute magnitude
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
    if not cosmology:
        cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    for key in ["offset", "absmag", "z"]:
        if key not in priors:
            raise ValueError(f"ERROR: Please set a prior function for {key}.")
        elif (key not in likes) and (key != "z"):
            raise ValueError(f"ERROR: Please set a likelihood function for {key}.")

    # always load GLADE -- we now use it for spec-zs.
    pkg = pkg_resources.files("astro_prost")
    pkg_data_file = pkg / "data" / "GLADE+_HyperLedaSizes_mod_withz.csv"

    try:
        with pkg_resources.as_file(pkg_data_file) as csvfile:
            glade_catalog = pd.read_csv(csvfile)
    except FileNotFoundError:
        glade_catalog = None

    # unpack priors and likelihoods
    priorfunc_z = priors["z"]
    priorfunc_offset = priors["offset"]
    priorfunc_absmag = priors["absmag"]

    likefunc_offset = likes["offset"]
    likefunc_absmag = likes["absmag"]

    results = []

    events = [
        (
            idx,
            row,
            glade_catalog,
            n_samples,
            verbose,
            priorfunc_z,
            priorfunc_offset,
            priorfunc_absmag,
            likefunc_offset,
            likefunc_absmag,
            cosmo,
            catalogs,
            cat_cols,
        )
        for idx, row in transient_catalog.iterrows()
    ]


    if parallel:
        envkey = 'PYSPAWN_' + os.path.basename(__file__)

        if not os.environ.get(envkey, False):
            # Set the environment variable in the parent process only
            os.environ[envkey] = str(os.getpid())  # Store the PID in the env var

            if n_processes is None:
                n_processes = os.cpu_count() - 4
            elif n_processes > os.cpu_count():
                print("WARNING! Set n_processes to greater than the number of cpu cores on this machine."+
                       f" Falling back to n_processes = {os.cpu_count() - 4}.")
                n_processes = os.cpu_count() - 4

            # Create a list of tasks
            if verbose > 0:
                print(f"Parallelizing {len(transient_catalog)} associations across {n_processes} processes.")

            with WorkerPool(n_jobs=n_processes, start_method='spawn') as pool:
                #jobs = [associate_transient(*event) for event in events]
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
                "host_2_objid", "host_2_prob", "host_2_ra", "host_2_dec",
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

        print("Association of all transients is complete.")

        # Save the updated catalog
        if save:
            ts = int(time())
            save_name = pathlib.Path(save_path, f"associated_transient_catalog_{ts}.csv")
            transient_catalog.to_csv(save_name, index=False)
        else:
            return transient_catalog
    else:
        return transient_catalog
