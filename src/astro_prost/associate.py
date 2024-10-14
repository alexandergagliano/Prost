import os
import pathlib
import dask
from time import time
from urllib.error import HTTPError
from dask import delayed, compute
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

@delayed
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
    """Short summary.

    Parameters
    ----------
    idx : type
        Description of parameter `idx`.
    row : type
        Description of parameter `row`.
    glade_catalog : type
        Description of parameter `glade_catalog`.
    n_samples : type
        Description of parameter `n_samples`.
    verbose : type
        Description of parameter `verbose`.
    priorfunc_z : type
        Description of parameter `priorfunc_z`.
    priorfunc_offset : type
        Description of parameter `priorfunc_offset`.
    priorfunc_absmag : type
        Description of parameter `priorfunc_absmag`.
    likefunc_offset : type
        Description of parameter `likefunc_offset`.
    likefunc_absmag : type
        Description of parameter `likefunc_absmag`.
    cosmo : type
        Description of parameter `cosmo`.
    catalogs : type
        Description of parameter `catalogs`.
    cat_cols : type
        Description of parameter `cat_cols`.
    Returns
    -------
    type
        Description of returned object.

    """
    try:
        transient = Transient(
            name=row["name"],
            position=SkyCoord(row.ra * u.deg, row.dec * u.deg),
            redshift=float(row.redshift),
            n_samples=n_samples,
        )
    except KeyError:
        transient = Transient(
            name=row["name"], position=SkyCoord(row.ra * u.deg, row.dec * u.deg), n_samples=n_samples
        )

    if verbose > 0:
        print(
            f"Associating for {transient.name} at RA, DEC = "
            "{transient.position.ra.deg:.6f}, {transient.position.dec.deg:.6f}"
        )

    transient.set_prior("redshift", priorfunc_z)
    transient.set_prior("offset", priorfunc_offset)
    transient.set_prior("absmag", priorfunc_absmag)

    transient.set_likelihood("offset", likefunc_offset)
    transient.set_likelihood("absmag", likefunc_absmag)

    best_prob, best_ra, best_dec, second_best_prob, second_best_ra, second_best_dec, query_time = (
        np.nan,
        np.nan,
        np.nan,
        np.nan, 
        np.nan,
        np.nan,
        np.nan
    ) 
    best_cat = ""

    for cat_name in catalogs:
        cat = GalaxyCatalog(name=cat_name, n_samples=n_samples, data=glade_catalog)

        try:
            cat.get_candidates(transient, timequery=True, verbose=verbose, cosmo=cosmo)
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

                best_objid = cat.galaxies["objID"][best_idx]
                best_prob = cat.galaxies["total_prob"][best_idx]
                best_ra = cat.galaxies["ra"][best_idx]
                best_dec = cat.galaxies["dec"][best_idx]

                second_best_objid = cat.galaxies["objID"][second_best_idx]
                second_best_prob = cat.galaxies["total_prob"][second_best_idx]
                second_best_ra = cat.galaxies["ra"][second_best_idx]
                second_best_dec = cat.galaxies["dec"][second_best_idx]

                best_cat = cat_name
                query_time = cat.query_time

                if cat_cols:
                    print("WARNING! cat_cols not implemented yet.")

                if verbose > 0:
                    print(f"Found a good host in {cat_name}!")
                    print(
                        f"Chosen galaxy has catalog ID of {best_objid}"
                        "and RA, DEC = {best_ra:.6f}, {best_dec:.6f}"
                    )
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
    )


def prepare_catalog(
    transient_catalog,
    debug_names=None,
    transient_name_col="name",
    transient_coord_cols=("ra", "dec"),
    debug=False,
):
    """Short summary.

    Parameters
    ----------
    transient_catalog : type
        Description of parameter `transient_catalog`.
    transient_name_col : type
        Description of parameter `transient_name_col`.
    transient_coord_cols : type
        Description of parameter `transient_coord_cols`.
    debug_names : type
        Description of parameter `debug_names`.
    debug : type
        Description of parameter `debug`.

    Returns
    -------
    type
        Description of returned object.

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
        "sn_ra_deg",
        "sn_dec_deg",
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

    transient_catalog["ra"] = transient_coords.ra.deg
    transient_catalog["dec"] = transient_coords.dec.deg

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
    cosmology=None,
):
    """Short summary.

    Parameters
    ----------
    transient_catalog : type
        Description of parameter `transient_catalog`.
    priors : type
        Description of parameter `priors`.
    likes : type
        Description of parameter `likes`.
    catalogs : type
        Description of parameter `catalogs`.
    n_samples : type
        Description of parameter `n_samples`.
    verbose : type
        Description of parameter `verbose`.
    parallel : type
        Description of parameter `parallel`.
    save : type
        Description of parameter `save`.
    save_path : type
        Description of parameter `save_path`.
    cat_cols : type
        Description of parameter `cat_cols`.
    cosmology : type
        Description of parameter `cosmology`.

    Returns
    -------
    type
        Description of returned object.

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

    if parallel:
        n_processes = os.cpu_count() - 5

        # Create a list of tasks (one per transient)
        print("parallelizing...")
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

        jobs = [associate_transient(*event) for event in events]
        results = compute(*jobs, scheduler='processes') 
    else:
        results = []
        for idx, row in transient_catalog.iterrows():
            event = (
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
            results.append(associate_transient(*event))
    # Update transient_catalog with results
    for result in results:

        (idx, best_objid, best_prob, best_ra, best_dec, 
           second_best_objid, second_best_prob, second_best_ra, second_best_dec, 
           query_time, best_cat) = result

        transient_catalog.at[idx, "host_id"] = best_objid
        transient_catalog.at[idx, "host_ra"] = best_ra
        transient_catalog.at[idx, "host_dec"] = best_dec
        transient_catalog.at[idx, "host_prob"] = best_prob
        transient_catalog.at[idx, "host_2_id"] = second_best_objid
        transient_catalog.at[idx, "host_2_ra"] = second_best_ra
        transient_catalog.at[idx, "host_2_dec"] = second_best_dec
        transient_catalog.at[idx, "host_2_prob"] = second_best_prob 
        transient_catalog.at[idx, "association_time"] = query_time
        transient_catalog.at[idx, "best_cat"] = best_cat


    # TODO: add in logic to return smallcone_prob and missedcat_prob
        "smallcone_prob",
        "missedcat_prob",

    print("Association of all transients is complete.")

    # Save the updated catalog
    if save:
        ts = int(time.time())
        save_name = pathlib.Path(save_path, f"associated_transient_catalog_{ts}.csv")
        transient_catalog.to_csv(save_name, index=False)
    else:
        return transient_catalog
