import os
import time
import importlib
import pickle
import requests

import importlib.resources as pkg_resources
import matplotlib.pyplot as plt
import numpy as np
import importlib.resources as pkg_resources
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord, match_coordinates_sky
from astropy.io import ascii
from dl import queryClient as qC
from scipy import stats as st
from scipy.integrate import quad
from scipy.stats import halfnorm, norm
from io import StringIO
import pandas as pd

from .photoz_helpers import evaluate, load_lupton_model, preprocess

PROB_FLOOR = np.finfo(float).eps
REDSHIFT_FLOOR = 0.001 # minimum redshift of z=0.001
SIZE_FLOOR = 0.25 # 1 pixel, for panstarrs
ABSMAG_FLOOR = -10 # guess at a minimum absmag for a galaxy?
SHAPE_FLOOR = 1.e-10
DUMMY_FILL_VAL = -999
RAD_TO_ARCSEC = 206265
SIGMA_ABSMAG_FLOOR = SIGMA_SIZE_FLOOR = SIGMA_REDSHIFT_FLOOR = 0.05 # 5% minimum uncertainty
SIGMA_ABSMAG_CEIL = SIGMA_SIZE_CEIL = SIGMA_REDSHIFT_CEIL = 0.5 # 50% maximum uncertainty
CATALOG_FUNCTIONS = {"panstarrs": None, "decals": None, "glade": None}
DEFAULT_LIMITING_MAG = {"panstarrs": 22, "decals": 26, "glade": 17}
CATALOG_SHRED_SETTINGS = {
    "panstarrs": True,  # Only enable for Pan-STARRS
    "decals": False,
    "glade": False,
}
PROP_DTYPES = [
        ("objID", np.int64),
        ("objID_info", "U20"),
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


def build_galaxy_array(candidate_hosts, cat_cols, transient_name, catalog, release, logger, verbose=0):
    n_galaxies = len(candidate_hosts)
    base_fields = ['objID', 'ra','dec']
    calc_fields = [x[0] for x in PROP_DTYPES]

    if n_galaxies < 1:
        if verbose > 0:
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
        galaxies = np.zeros(n_galaxies, dtype=PROP_DTYPES)

        # Populate galaxies array with data from candidate_hosts
        for col in candidate_hosts.columns:
            galaxies[col] = candidate_hosts[col].values
    else:
        galaxies = np.zeros(n_galaxies, dtype=PROP_DTYPES)
        cat_col_fields = []

    # Populate galaxies array with data from candidate_hosts
    for col in base_fields:
        galaxies[col] = candidate_hosts[col].values
    return galaxies, cat_col_fields

def fetch_catalog_data(self, transient, search_rad, cosmo, logger, cat_cols, calc_host_props, verbose):
    """Generalized function to fetch catalog data.

    Parameters
    ----------
    transient : type
        Description of parameter `transient`.
    search_rad : type
        Description of parameter `search_rad`.
    cosmo : type
        Description of parameter `cosmo`.
    logger : type
        Description of parameter `logger`.
    cat_cols : type
        Description of parameter `cat_cols`.
    calc_host_props : list
        Description of parameter `calc_host_props`.
    verbose : type
        Description of parameter `verbose`.

    Returns
    -------
    type
        Description of returned object.

    """
    global CATALOG_FUNCTIONS  # Declare it as global so we can modify it

    # Now that functions are available, update the global dictionary
    CATALOG_FUNCTIONS.update({
        "panstarrs": build_panstarrs_candidates,
        "decals": build_decals_candidates,
        "glade": build_glade_candidates,
    })

    if self.name not in CATALOG_FUNCTIONS:
        raise ValueError(f"Unknown catalog: {self.name}. Open a pull request to add functionality for other catalogs!")

    self.limiting_mag = DEFAULT_LIMITING_MAG.get(self.name, None)

    # Fetch the function dynamically
    catalog_func = CATALOG_FUNCTIONS[self.name]

    # Common parameters for all catalogs
    init_params = {
        "transient_name": transient.name,
        "transient_pos": transient.position,
        "search_rad": search_rad,
        "cosmo": cosmo,
        "logger": logger,
        "calc_host_props": calc_host_props,
        "n_samples": self.n_samples,
        "verbose": verbose,
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
    """Class for source catalog containing candidate transient host galaxies.

    Parameters
    ----------
    name : str
        Name of the transient catalog. Currently 'glade', 'decals', and
        'panstarrs' are supported.
    data : Pandas DataFrame
        Locally-saved GLADE catalog for redshifts at low-z.
    n_samples : int
        Number of samples to draw in monte-carlo association.

    Attributes
    ----------
    name
    data
    n_samples
    release
    shred_cut

    """

    def __init__(self, name, release=None, data=None, n_samples=1000):
        self.name = name
        self.data = data
        self.n_samples = n_samples
        self.release = release
        self.shred_cut = CATALOG_SHRED_SETTINGS[self.name]

    def get_candidates(self, transient, cosmo, logger, time_query=False, verbose=False, calc_host_props=['offset' ,'absmag', 'redshift'], cat_cols=False):
        """Hydrates the catalog attribute catalog.galaxies with a list of candidates.

        Parameters
        ----------
        transient : Transient
            Source to associate, of custom class Transient.
        cosmo : astropy cosmology
            Assumed cosmology for conversions (defaults to LambdaCDM if not set).
        time_query : boolean
            If True, times the catalog query and stores the result in self.query_time.
        verbose : int
            Verbosity level; can be 0, 1, or 2.
        calc_host_props : list
            The list of host properties to calculate (may or may not be required for association)
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

        self.galaxies, self.cat_col_fields = fetch_catalog_data(self, transient, search_rad, cosmo, logger, cat_cols, calc_host_props, verbose)

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
    position : astropy.coord SkyCoord object
        Position of transient.
    redshift : float
        Photometric or spectroscopic redshift of transient.
    spec_class : str
        Spectroscopic class of transient, if available.
    phot_class : str
        Photometric class of transient, if available.
    n_samples : int
        Number of iterations for monte-carlo association.

    Attributes
    ----------
    best_host : int
        Catalog index of highest-probability host galaxy.
    second_best_host : int
        Catalog index of second-highest-probability host galaxy.
    redshift : type
        Description of attribute `redshift`.
    redshift_std : type
        Description of attribute `redshift_std`.
    gen_z_samples : function
        Draws samples from transient redshift for monte-carlo association run.
    priors : dict
        Prior distributions on host fractional offset, absolute brightness, and redshift.
    likes : dict
        Likelihood distributions on host fractional offset, and absolute brightness.
    name
    position
    redshift
    spec_class
    phot_class
    n_samples

    """

    def __init__(self, name, position,  logger, redshift=np.nan, redshift_std=np.nan, spec_class="", phot_class="", n_samples=1000):
        self.name = name
        self.position = position
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
            logger.info(f"Setting redshift uncertainty for {name} to floor of {redshift_std:.5f}.")

        self.priors = {}
        self.likes = {}

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
            Type of prior to retrieve (can be redshift, offset, absmag).

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
        """Generates transient redshift samples for monte-carlo association.
            If redshift is not measured, samples are drawn from the prior.

        Parameters
        ----------
        ret : boolean
            If true, returns the samples.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        samples : array-like
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
        redshift_samples : array-like
            Array of transient redshift samples.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        pdf : float or array-like
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
        fractional_offset_samples : array-like
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
        """Short summary.

        Parameters
        ----------
        absmag_samples : type
            Description of parameter `absmag_samples`.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        type
            Description of returned object.

        """
        pdf = self.get_prior("absmag").pdf(absmag_samples)
        if reduce == "mean":
            return np.nanmean(pdf, axis=1)
        elif reduce == "median":
            return np.nanmedian(pdf, axis=1)
        else:
            return pdf

    def calc_like_redshift(self, redshift_mean, redshift_std, reduce="mean"):
        """Short summary.

        Parameters
        ----------
        redshift_mean : type
            Description of parameter `redshift_mean`.
        redshift_std : type
            Description of parameter `redshift_std`.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        type
            Description of returned object.

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
        """Short summary.

        Parameters
        ----------
        fractional_offset_samples : type
            Description of parameter `fractional_offset_samples`.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        type
            Description of returned object.

        """
        likelihoods = self.get_likelihood("offset").pdf(fractional_offset_samples)
        if reduce == "mean":
            return np.nanmean(likelihoods, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == "median":
            return np.nanmedian(likelihoods, axis=1)
        else:
            return likelihoods

    def calc_like_absmag(self, absmag_samples, reduce="mean"):
        """Short summary.

        Parameters
        ----------
        absmag_samples : type
            Description of parameter `absmag_samples`.
        reduce : str
            How to collapse the samples into a prior point-estimate.
            Defaults to calculating the mean across samples.

        Returns
        -------
        type
            Description of returned object.

        """
        # assuming a typical 0.1 SN/century/10^10 Lsol (in K-band)
        # TODO -- convert to K-band luminosity of the host!
        # https://www.aanda.org/articles/aa/pdf/2005/15/aa1411.pdf
        likelihoods = self.get_likelihood("absmag").pdf(absmag_samples)
        if reduce == "mean":
            return np.nanmean(likelihoods, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == "median":
            return np.nanmedian(likelihoods, axis=1)
        else:
            return likelihoods

    def associate(self, galaxy_catalog, cosmo, logger, calc_host_props=['offset'], verbose=False):
        """Runs the main transient association module.

        Parameters
        ----------
        galaxy_catalog : GalaxyCatalog object
            The catalog populated with candidate hosts and their attributes.
        cosmo : astropy cosmology
            Assumed cosmology.
        verbose : boolean
            If True, prints detailed information about the association.

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
        n_samples = galaxy_catalog.n_samples

        post_set = []

        if 'redshift' in calc_host_props:
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

        if 'absmag' in calc_host_props:
            absmag_mean = np.array(galaxies["absmag_mean"])
            absmag_std = np.array(galaxies["absmag_std"])
            absmag_samples = np.vstack(galaxies["absmag_samples"])

            prior_absmag = self.calc_prior_absmag(absmag_samples, reduce=None)
            like_absmag = self.calc_like_absmag(absmag_samples, reduce=None)
            post_absmag = prior_absmag * like_absmag
            post_set.append(post_absmag)

        if 'offset' in calc_host_props:
            offset_mean = np.array(galaxies["offset_mean"])
            offset_std = np.array(galaxies["offset_std"])
            # just copied now assuming 0 positional uncertainty -- this can be updated later (TODO!)
            offset_samples = np.repeat(offset_mean[:, np.newaxis], n_samples, axis=1)

            galaxy_dlr_samples = np.vstack(galaxies["dlr_samples"])

            # Calculate angular diameter distances for all samples
            fractional_offset_samples = offset_samples / galaxy_dlr_samples

            prior_offset = self.calc_prior_offset(fractional_offset_samples, reduce=None)
            like_offset = self.calc_like_offset(fractional_offset_samples, reduce=None)  # Shape (N,)

            post_offset = prior_offset * like_offset
            post_set.append(post_offset)

        # Compute the posterior probabilities for all galaxies
        post_gals_stacked = np.stack(post_set, axis=0)       # shape -> (n_properties, n_gals, n_samples)
        post_gals = np.prod(post_gals_stacked, axis=0)          # shape -> (n_gals, n_samples)

        # some very low value that the SN is actually hostless, across all samples.
        post_hostless = np.ones(n_samples)*PROB_FLOOR

        if self.redshift == self.redshift:
            post_outside = self.probability_host_outside_cone(search_rad=search_rad, verbose=verbose, n_samples=n_samples, cosmo=cosmo, calc_host_props=calc_host_props)
        else:
            post_outside = PROB_FLOOR

        post_unobs = self.probability_of_unobserved_host(search_rad=search_rad, limiting_mag=limiting_mag, verbose=verbose, n_samples=n_samples, cosmo=cosmo, calc_host_props=calc_host_props)

        # sum across all galaxies
        post_tot = np.nansum(post_gals, axis=0) + post_hostless + post_outside + post_unobs

        # floor to machine precision
        post_tot[post_tot < PROB_FLOOR] = PROB_FLOOR

        p_none_norm = (post_outside + post_hostless + post_unobs) / post_tot
        p_any_norm = np.nansum(post_gals, axis=0) / post_tot
        post_gals_norm = post_gals / post_tot
        post_outside_norm = post_outside / post_tot
        post_unobs_norm = post_unobs / post_tot

        if 'offset' in calc_host_props:
            post_offset_norm = post_offset / post_tot
        if 'redshift' in calc_host_props:
            post_redshift_norm = post_redshift / post_tot
        if 'absmag' in calc_host_props:
            post_absmag_norm = post_absmag / post_tot

        # Sort and get last 2 indices in descending order
        top_idxs = np.argsort(np.nanmedian(post_gals_norm, axis=1))[::-1]

        self.associated_catalog = galaxy_catalog.name
        self.any_prob = np.nanmedian(p_any_norm)
        self.none_prob = np.nanmedian(p_none_norm)
        self.smallcone_prob = np.nanmedian(post_outside_norm)
        self.missedcat_prob = np.nanmedian(post_unobs_norm)

        if self.any_prob > self.none_prob:
            if verbose > 0:
                logger.info("Association successful!")
                logger.info("")
            # get best and second-best matches
            self.best_host = top_idxs[0]
            if ngals > 1:
                self.second_best_host = top_idxs[1]
        else:
            self.best_host = -1
            self.second_best_host = top_idxs[0]  # set best host as second-best -- best is no host
            if verbose > 0:
                if np.nanmedian(post_outside_norm) > np.nanmedian(post_unobs_norm):
                    logger.info("Association failed. Host is likely outside search cone.\n")
                else:
                    logger.info("Association failed. Host is likely missing from the catalog.\n")

        # consolidate across samples
        galaxy_catalog.galaxies["total_posterior"] = np.nanmedian(post_gals_norm, axis=1)

        if 'offset' in calc_host_props:
            galaxy_catalog.galaxies["offset_posterior"] = np.nanmedian(post_offset_norm, axis=1)
        if 'redshift' in calc_host_props:
            galaxy_catalog.galaxies["redshift_posterior"] = np.nanmedian(post_redshift_norm, axis=1)
        if 'absmag' in calc_host_props:
            absmag_best_post = np.nanmedian(post_absmag_norm, axis=1)[self.best_host]
            galaxy_catalog.galaxies["absmag_posterior"] = np.nanmedian(post_absmag_norm, axis=1)

        return galaxy_catalog

    def probability_of_unobserved_host(self, search_rad, cosmo, limiting_mag=30, verbose=False, n_samples=1000, calc_host_props=['offset']):
        """Calculates the posterior probability of the host being either dimmer than the
           limiting magnitude of the catalog or not in the catalog at all.

        Parameters
        ----------
        search_rad : float
            Cone search radius, in arcsec.
        limiting_mag : float
            Limiting magnitude of the survey, in AB mag.
        verbose : boolean
            If true, prints stats about association.
        n_samples : int
            Number of samples for monte-carlo association.
        cosmo : astropy cosmology
            Assumed cosmology for the run.

        Returns
        -------
        post_unobserved : array-like
            n_samples of posterior probabilities of the host not being in the catalog.

        """
        # only set if we have absmag and redshift priors -- otherwise set to 0!
        if ('absmag' not in calc_host_props) or ('redshift' not in calc_host_props):
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
            post_z = post_z[:, np.newaxis]  # Shape: (n_galaxies, 1)
        else:
            # Use the known supernova redshift
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

        if 'offset' in calc_host_props:
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

            fractional_offset_samples = (
                physical_offset_samples / galaxy_physical_radius_prior_samples
            )  # Shape: (n_samples, n_samples)

            prior_offset_unobs = self.calc_prior_offset(fractional_offset_samples, reduce=None)
            l_offset_unobs = self.calc_like_offset(fractional_offset_samples, reduce=None)

            post_offset = prior_offset_unobs * l_offset_unobs
            post_set.append(post_offset)

        # Compute the posterior probabilities for all galaxies
        prob_unobs_stacked = np.stack(post_set, axis=0)       # shape -> (n_properties, n_gals, n_samples)
        post_unobs = np.prod(prob_unobs_stacked, axis=0)          # shape -> (n_gals, n_samples)
        post_unobs = np.nanmean(post_unobs, axis=0)  # average over all galaxies -- keep n_samples

        return post_unobs

    def probability_host_outside_cone(self, cosmo, search_rad=60, verbose=False, n_samples=1000, calc_host_props=['offset']):
        """Calculates the posterior probability of the host being outside the cone search chosen
           for the catalog query. Primarily set by the fractional offset and redshift prior.

        Parameters
        ----------
        search_rad : float
            Cone search radius, in arcsec.
        verbose : boolean
            If True, prints stats about the probability calculation.
        n_samples : int
            Number of samples to draw for monte-carlo association.
        cosmo : astropy cosmology
            Assumed cosmology.

        Returns
        -------
        post_outside : array-like
            An array of n_samples posterior probabilities of the host being outside the search cone.

        """

        # only calculate if we have redshift and offset priors -- otherwise fix to PROB_FLOOR
        if 'redshift' not in calc_host_props:
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

        if 'absmag' in calc_host_props:
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
        prob_outside_stacked = np.stack(post_set, axis=0)       # shape -> (n_properties, n_gals, n_samples)
        post_outside = np.prod(prob_outside_stacked, axis=0)          # shape -> (n_gals, n_samples)

        # average over all simulated galaxies -- keep the samples
        post_outside = np.nanmean(post_outside, axis=0)

        return post_outside

class PriorzObservedTransients(st.rv_continuous):
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
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        # Create redshift bins
        z_bins = np.linspace(self.z_min, self.z_max, self.n_bins + 1)
        z_centers = (z_bins[:-1] + z_bins[1:])/2  # Centers of redshift bins

        # Calculate the comoving volume element dV/dz for each redshift bin
        dv_dz = self.cosmo.differential_comoving_volume(z_centers).value  # in Mpc^3 per steradian per dz

        # Full sky solid angle (4 pi steradians)
        solid_angle = 4 * np.pi  # full sky in steradians

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
        self.bestFit = st.gaussian_kde(self.observed_redshifts)

    def pdf(self, z):
        """
        Return the PDF (KDE) based on observed redshifts.
        Handles 1D and 2D arrays.

        Parameters
        ----------
        z : array-like
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
        samples : array-like
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


class SnRateAbsmag(st.rv_continuous):
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
        abs_mag_samples : array-like
            Array of galaxy absolute magnitudes.

        Returns
        -------
        snrate : array-like
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
        m_abs_samples : array-like
            Absolute magnitudes of galaxies.

        Returns
        -------
        normalized_pdf : array-like
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
    columns : array-like
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
    """Short summary.

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
    columns : array-like
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

    r = requests.get(url, params=data)

    if verbose > 2:
        logger.info(r.url)

    r.raise_for_status()

    if format == "json":
        return r.json()
    else:
        return r.text


def build_glade_candidates(
    transient_name,
    transient_pos,
    glade_catalog,
    cosmo,
    logger,
    search_rad=None,
    n_samples=1000,
    verbose=False,
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
    transient_name : str
        Name of transient to associate.
    transient_pos : astropy.coord SkyCoord
        Position of transient to associate.
    glade_catalog : Pandas DataFrame
        The locally-packaged GLADE catalog (to avoid querying).
    search_rad : astropy Angle
        Radius for cone search.
    cosmo : astropy cosmology
        Assumed cosmology for conversions.
    n_samples : int
        Number of samples for monte-carlo association.
    verbose : int
        Level of logging verbosity; can be 0, 1, or 2.
    cat_cols : boolean
        If True, concatenates catalog fields for best host to final catalog.
    shred_cut : boolean
        If True, removes likely source shreds associated with the same candidate galaxy.
    calc_host_props : list
        List of host properties to calculate (whether used for association or not)
    Returns
    -------
    galaxies : structured numpy array
        Array of properties for candidate sources needed
        for host association.

    cat_col_fields : str
        List of columns retrieved from the galaxy catalog
        (rather than calculated internally).

    """
    if release != "latest":
        logger.warning("Only GLADE+ is supported at this time. Please open a pull request to expand Prost to alternative versions or other catalogs!")
    elif shred_cut:
        logger.warning("shred_cut is not implemented for GLADE+ galaxies at this time. Running with shred_cut = False.")

    if search_rad is None:
        search_rad = Angle(60 * u.arcsec)

    ra_min = transient_pos.ra.deg - 1
    ra_max = transient_pos.ra.deg + 1
    dec_min = transient_pos.dec.deg - 1
    dec_max = transient_pos.dec.deg + 1

    glade_catalog.rename(columns={'RAJ2000':'ra', 'DEJ2000':'dec', 'z_best':'redshift', 'z_best_std':'redshift_std'}, inplace=True)

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

    # define plceholder IDs for GLADE
    candidate_hosts = candidate_hosts.assign(objID=candidate_hosts.index)
    if len(candidate_hosts) < 1:
        return None, []

    galaxies, cat_col_fields = build_galaxy_array(candidate_hosts, cat_cols, transient_name, "GLADE+", release, logger, verbose)
    if galaxies is None:
        return None, []
    n_galaxies = len(galaxies)


    galaxies['objID_info'] = ['tbl index']*n_galaxies

    if 'offset' in calc_host_props:
        temp_pa = candidate_hosts["PAHyp"].values
        temp_pa[temp_pa != temp_pa] = 0  # assume no position angle for unmeasured gals

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

        galaxy_a_over_b = 10 ** (candidate_hosts["logr25Hyp"].values)
        galaxy_a_over_b_std = galaxy_a_over_b * np.log(10) * candidate_hosts["e_logr25Hyp"].values

        # set uncertainty floor
        nanbool = galaxy_a_over_b_std != galaxy_a_over_b_std
        galaxy_a_over_b_std[nanbool] = SIGMA_SIZE_FLOOR * galaxy_a_over_b[nanbool]

        temp_pa = candidate_hosts["PAHyp"].values

        # assume no position angle for unmeasured gals (round is a decent assumption for the most distant ones)
        temp_pa[temp_pa != temp_pa] = SHAPE_FLOOR

        phi = np.radians(temp_pa)
        phi_std = SIGMA_SIZE_FLOOR * phi  # uncertainty floor

        dlr_samples = calc_dlr(
            transient_pos,
            galaxies_pos,
            temp_sizes,
            temp_sizes_std,
            galaxy_a_over_b,
            galaxy_a_over_b_std,
            phi,
            phi_std,
            n_samples=n_samples,
        )

        # Calculate angular separation between SN and all galaxies (in arcseconds)
        galaxies["offset_mean"] = (
            SkyCoord(galaxies["ra"] * u.deg, galaxies["dec"] * u.deg).separation(transient_pos).arcsec
        )

        for i in range(n_galaxies):
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

        # TODO find spec-z info for GLADE
        good_specz = galaxies["redshift_std"]/(1+galaxies["redshift_mean"]) < 0.1
        galaxies['redshift_info'][good_specz] = ['spec-z']*len(good_specz)

        redshift_samples[redshift_samples < REDSHIFT_FLOOR] = REDSHIFT_FLOOR  # set redshift floor
        #redshift_samples[redshift_samples != redshift_samples] = REDSHIFT_FLOOR # set redshift floor

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


def build_decals_candidates(transient_name,
                            transient_pos,
                            cosmo,
                            logger,
                            search_rad=None,
                            n_samples=1000,
                            verbose=False,
                            calc_host_props=['redshift', 'absmag', 'offset'],
                            cat_cols=False,
                            shred_cut=False,
                            release="dr9"):
    """Populates a GalaxyCatalog object with candidates from a cone search of the
       DECaLS catalog (See https://www.legacysurvey.org/decamls/ for details).

    Parameters
    ----------
    transient_name : str
        Name of transient to associate.
    transient_pos : astropy.coord SkyCoord
        Position of transient to associate.
    glade_catalog : Pandas DataFrame
        The locally-packaged GLADE catalog (to avoid querying).
    search_rad : astropy Angle
        Radius for cone search.
    cosmo : astropy cosmology
        Assumed cosmology for conversions.
    n_samples : int
        Number of samples for monte-carlo association.
    verbose : int
        Level of logging verbosity; can be 0, 1, or 2.
    calc_host_props : list
        TODO fill in here!
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

    cat_col_fields : str
        List of columns retrieved from the galaxy catalog
        (rather than calculated internally).

    """
    if release not in ["dr9", "dr10"]:
        raise ValueError(f"Invalid DECaLS version '{release}'. Please choose 'dr9' or 'dr10'.")
    elif shred_cut:
        logger.warning("shred_cut is not implemented for decals galaxies at this time. Running with shred_cut = False.")

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
        q3c_radial_query(t.ra, t.dec, {transient_pos.ra.deg:.5f}, {transient_pos.dec.deg:.5f}, {rad_deg})
    AND (t.nobs_r > 0) AND (t.dered_flux_r > 0) AND (t.snr_r > 0)
    AND nullif(t.dered_mag_r, 'NaN') is not null AND (t.fitbits != 8192)
    AND ((pz.z_spec > 0) OR (pz.z_phot_mean > 0))"""
    )

    candidate_hosts = pd.read_csv(StringIO(result))
    candidate_hosts.rename(columns={'ls_id':'objID'}, inplace=True)
    if len(candidate_hosts) < 1:
        return None, []

    galaxies_pos = SkyCoord(candidate_hosts["ra"].values * u.deg, candidate_hosts["dec"].values * u.deg)

    galaxies, cat_col_fields = build_galaxy_array(candidate_hosts, cat_cols, transient_name, "decals", release, logger, verbose)
    if galaxies is None:
        return None, []
    n_galaxies = len(galaxies)

    galaxies["objID_info"] = [f'decals {release}']*n_galaxies

    if 'offset' in calc_host_props:
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

        denom = temp_e1**2 + temp_e2**2
        denom = np.maximum(denom, SHAPE_FLOOR)

        partial_phi_e2 = -temp_e1 / (2.0 * denom)   # d(phi)/d(e2)
        partial_phi_e1 =  temp_e2 / (2.0 * denom)   # d(phi)/d(e1)

        # Now propagate uncertainties from e1 and e2
        phi_std = np.sqrt((partial_phi_e1**2) * (temp_e1_std**2) + (partial_phi_e2**2) * (temp_e2_std**2))

        # First clamp phi_std to be at least SIGMA_SIZE_FLOOR * |phi|
        phi_std = np.maximum(phi_std, SIGMA_SIZE_FLOOR * np.abs(phi))
        phi_std = np.maximum(phi_std, SHAPE_FLOOR)

        dlr_samples = calc_dlr(
            transient_pos, galaxies_pos, temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std
        )

        # Calculate angular separation between SN and all galaxies (in arcseconds)
        galaxies["offset_mean"] = (
            SkyCoord(galaxies["ra"] * u.deg, galaxies["dec"] * u.deg).separation(transient_pos).arcsec
        )
        galaxies['offset_std'] = PROB_FLOOR # TODO

        for i in range(n_galaxies):
            galaxies["dlr_samples"][i] = dlr_samples[i, :]
            #galaxies["offset_samples"][i] = offset_samples[i, :] #TODO

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
            galaxies["redshift_mean"][:, np.newaxis],  # Shape (N, 1) to allow broadcasting
            galaxies["redshift_std"][:, np.newaxis],  # Shape (N, 1)
            size=(n_galaxies, n_samples),  # Shape (N, M)
        )
        redshift_samples[redshift_samples < REDSHIFT_FLOOR] = REDSHIFT_FLOOR  # set photometric redshift floor
        #redshift_samples[redshift_samples != redshift_samples] = REDSHIFT_FLOOR  # set photometric redshift floor

        for i in range(n_galaxies):
            galaxies["redshift_samples"][i] = redshift_samples[i, :]

    if 'absmag' in calc_host_props:
        temp_mag_r = candidate_hosts["dered_mag_r"].values

        temp_mag_r_std = np.abs(
            2.5
            / np.log(10)
            * np.sqrt(1 / np.maximum(0, candidate_hosts["flux_ivar_r"].values))
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
    transient_name,
    transient_pos,
    cosmo,
    logger,
    glade_catalog=None,
    search_rad=None,
    n_samples=1000,
    verbose=False,
    calc_host_props=['redshift', 'absmag', 'offset'],
    cat_cols=False,
    release='dr2',
    shred_cut=True,
):
    """Populates a GalaxyCatalog object with candidates from a cone search of the
       panstarrs catalog (See https://outerspace.stsci.edu/display/PANSTARRS/ for details).

    Parameters
    ----------
    transient_name : str
        Name of transient to associate.
    transient_pos : astropy.coord SkyCoord
        Position of transient to associate.
    search_rad : astropy Angle
        Radius for cone search.
    cosmo : astropy cosmology
        Assumed cosmology for conversions.
    n_samples : int
        Number of samples for monte-carlo association.
    glade_catalog : Pandas DataFrame
        The locally-packaged GLADE catalog (to avoid querying).
    verbose : int
        Level of logging verbosity; can be 0, 1, or 2.
    cat_cols : boolean
        If True, concatenates catalog fields for best host to final catalog.
    shred_cut : boolean
        If True, removes likely source shreds associated with the same candidate galaxy.

    Returns
    -------
    galaxies : structured numpy array
        Array of properties for candidate sources needed
        for host association.

    cat_col_fields : str
        List of columns retrieved from the galaxy catalog
        (rather than calculated internally).

    """

    if release not in ["dr1", "dr2"]:
        raise ValueError(f"Invalid Pan-STARRS version '{release}'. Please choose 'dr1' or 'dr2'.")
    elif (release == 'dr1') and (('redshift' in calc_host_props) or ('absmag' in calc_host_props)):
        raise ValueError("Redshift estimation with Pan-STARRS data can only be done with release 'dr2'.")
    elif shred_cut:
        logger.info("Running with shred_cut = True...")

    # load table metadata to avoid a query
    pkg_data_file = pkg_resources.files('astro_prost') / 'data' / 'panstarrs_metadata.pkl'

    with pkg_resources.as_file(pkg_data_file) as metadata_path:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

    if search_rad is None:
        search_rad = Angle(60 * u.arcsec)

    rad_deg = search_rad.deg

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

        result = panstarrs_cone(metadata, transient_pos.ra.deg, transient_pos.dec.deg, rad_deg, columns=source_cols,
        release=release)
    else:
        result = panstarrs_cone(metadata, transient_pos.ra.deg, transient_pos.dec.deg, rad_deg, release=release)

    if (not result) and (verbose > 1):
        logging.warning(f"Found no pan-starrs {release} sources near {transient_name}.")
        return None

    candidate_hosts = pd.read_csv(StringIO(result))
    if len(candidate_hosts) < 1:
        return None, []

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
            transient_pos.ra.deg,
            transient_pos.dec.deg,
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
    #candidate_hosts.sort_values(by=["distance"], inplace=True)
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

    if 'offset' in calc_host_props:
        temp_sizes = candidate_hosts["KronRad"].values

        # assume some fiducial shape floor
        temp_sizes_std = SIGMA_SIZE_FLOOR * candidate_hosts["KronRad"].values
        temp_sizes_std = np.maximum(temp_sizes_std, SHAPE_FLOOR)  # Prevent division by zero

        # temp_sizes_std[temp_sizes_std != temp_sizes_std] = 0.05*temp_sizes[temp_sizes_std != temp_sizes_std]
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

        galaxies_pos = SkyCoord(
            candidate_hosts["ra"].values * u.deg, candidate_hosts["dec"].values * u.deg
        )

        temp_mag_r = candidate_hosts["KronMag"].values
        temp_mag_r_std = candidate_hosts["KronMagErr"].values

        # cap at 50% the mag
        # set a floor of 5%
        temp_mag_r_std[temp_mag_r_std > (SIGMA_ABSMAG_CEIL*temp_mag_r)] = SIGMA_ABSMAG_CEIL * temp_mag_r[temp_mag_r_std > (SIGMA_ABSMAG_CEIL*temp_mag_r)]
        temp_mag_r_std[temp_mag_r_std < (SIGMA_ABSMAG_FLOOR * temp_mag_r)] = SIGMA_ABSMAG_FLOOR * temp_mag_r[temp_mag_r_std < (SIGMA_ABSMAG_FLOOR * temp_mag_r)]

        dlr_samples = calc_dlr(
            transient_pos,
            galaxies_pos,
            temp_sizes,
            temp_sizes_std,
            a_over_b,
            a_over_b_std,
            phi,
            phi_std,
            n_samples=n_samples,
        )

    # shred logic
    if (shred_cut) and (len(candidate_hosts) > 1):
        if verbose > 0:
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
            verbose=verbose,
        )

        if len(shred_idxs) > 0:
            left_idxs = ~candidate_hosts.index.isin(shred_idxs)
            candidate_hosts = candidate_hosts[left_idxs]
            temp_mag_r = temp_mag_r[left_idxs]
            temp_mag_r_std = temp_mag_r_std[left_idxs]
            galaxies_pos = galaxies_pos[left_idxs]
            dlr_samples = dlr_samples[left_idxs]

            if verbose > 0:
                logger.info(f"Removed {len(shred_idxs)} flagged panstarrs sources.")
        else:
            if verbose > 0:
                logger.info("No panstarrs shreds found.")

    galaxies, cat_col_fields = build_galaxy_array(candidate_hosts, cat_cols, transient_name, "panstarrs", release, logger, verbose)
    if galaxies is None:
        return None, []
    n_galaxies = len(galaxies)

    galaxies['objID_info'] = [f'pan-starrs {release}']*n_galaxies

    if ('redshift' in calc_host_props) or ('absmag' in calc_host_props):
        # get photozs from Andrew Engel's code!
        default_dust_path = "."

        pkg = pkg_resources.files("astro_prost")
        pkg_data_file = pkg / "data" / "MLP_lupton.hdf5"

        with pkg_resources.as_file(pkg_data_file) as model_path:
            model, range_z = load_lupton_model(model_path=model_path, dust_path=default_dust_path)

        x = preprocess(candidate_hosts, path=os.path.join(default_dust_path, "sfddata-master"))
        posteriors, point_estimates, errors = evaluate(x, model, range_z)
        point_estimates[point_estimates < REDSHIFT_FLOOR] = REDSHIFT_FLOOR  # set photometric redshift floor
        #point_estimates[point_estimates != point_estimates] = 0.001  # set photometric redshift floor

        # inflated sigma floor for this particular photoz model
        err_bool = errors < (5*SIGMA_REDSHIFT_FLOOR * point_estimates)
        errors[err_bool] = (5*SIGMA_REDSHIFT_FLOOR * point_estimates[err_bool])

        # not QUITE the mean of the posterior, but we're assuming it's gaussian :/
        # TODO -- sample from the full posterior!
        galaxies["redshift_mean"] = point_estimates
        galaxies["redshift_std"] = np.abs(errors)

        # if the source is within 1arcsec of a GLADE host, take that spec-z.
        if glade_catalog is not None:
            if verbose > 1:
                logger.info("Cross-matching with GLADE for redshifts...")

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
            galaxies["redshift_mean"][:, np.newaxis],  # Shape (N, 1) to allow broadcasting
            galaxies["redshift_std"][:, np.newaxis],  # Shape (N, 1)
            size=(n_galaxies, n_samples),  # Shape (N, M)
        )

        redshift_samples[redshift_samples < REDSHIFT_FLOOR] = REDSHIFT_FLOOR  # set photometric redshift floor

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
        galaxies["offset_mean"] = galaxies_pos.separation(transient_pos).arcsec

        for i in np.arange(n_galaxies):
            galaxies["dlr_samples"][i] = dlr_samples[i, :]

    return galaxies, cat_col_fields


def calc_dlr(transient_pos, galaxies_pos, a, a_std, a_over_b, a_over_b_std, phi, phi_std, n_samples=1000):
    """Calculates the directional light radius (DLR) between candidate host and transient, following the
       general framework in Gupta et al. (2016).

    Parameters
    ----------
    transient_pos : astropy.coord SkyCoord
        Position of the transient.
    galaxies_pos : array of astropy.coord SkyCoord objects
        Positions of candidate host galaxies.
    a : array-like
        Semi-major axes of candidates, in arcsec.
    a_std : array-like
        Error in semi-major axes of candidates, in arcsec.
    a_over_b : array-like
        Axis ratio (major/minor) of candidates.
    a_over_b_std : array-like
        Error in axis ratio.
    phi : array-like
        Position angles of sources, in radians.
    phi_std : array-like
        Error in position angle.
    n_samples : type
        Number of DLR samples to draw for monte-carlo association.

    Returns
    -------
    dlr : array-like
        2D array of DLR samples of dimensionality (n_galaxies, n_samples).

    """
    n_gals = len(galaxies_pos)

    transient_ra = transient_pos.ra.deg
    transient_dec = transient_pos.dec.deg

    if n_samples > 1:
        # TODO -- incorporate uncertainty in galaxy and transient position
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


def find_panstarrs_shreds(
    objids,
    coords,
    a,
    a_std,
    a_over_b,
    a_over_b_std,
    phi,
    phi_std,
    appmag,
    logger,
    verbose=False
):
    """
    Finds potentially shredded sources in panstarrs and removes the shreds.
    If any source is in another sources light radius, the dimmer source is dropped.

    Parameters
    ----------
    objids : array-like
        catalog IDs in panstarrs of all sources.
    coord_allgals : array-like
        Astropy coords of catalog sources.
    a : array-like
        Semi-major axes of candidates, in arcsec.
    a_std : array-like
        Error in semi-major axes of candidates, in arcsec.
    a_over_b : array-like
        Axis ratio (major/minor) of candidates.
    a_over_b_std : array-like
        Error in axis ratio.
    phi : array-like
        Position angles of sources, in radians.
    phi_std : array-like
        Error in position angle.
    appmag : array-like
        Apparent magnitudes of candidate sources.
    verbose : int
        Level of logging; can be 0, 1, or 2.

    Returns
    -------
    dropidxs: array
        The indices of the candidates flagged as shreds.
    """

    dropidxs = []

    for i in np.arange(len(coords)):
        onegal_objid = objids[i]
        onegal_coord = coords[i]

        restgal_coord = np.delete(coords, i)
        restgal_ab = np.delete(a_over_b, i)
        restgal_ab_std = np.delete(a_over_b_std, i)
        restgal_phi = np.delete(phi, i)
        restgal_phi_std = np.delete(phi_std, i)
        restgal_a = np.delete(a, i)
        restgal_a_std = np.delete(a_std, i)
        restgal_appmag = np.delete(appmag, i)

        dlr = calc_dlr(
            onegal_coord,
            restgal_coord,
            restgal_a,
            restgal_a_std,
            restgal_ab,
            restgal_ab_std,
            restgal_phi,
            restgal_phi_std,
            n_samples=1,
        )
        seps = onegal_coord.separation(restgal_coord).arcsec
        min_idx = np.nanargmin(seps / dlr)

        # if within the dlr of another galaxy, remove dimmer galaxy
        original_min_idx = min_idx if min_idx < i else min_idx + 1

        if verbose == 3:
            logger.info(f"\n\nConsidering source at: {onegal_coord.ra.deg:.6f}, {onegal_coord.dec.deg:.6f}:")
            logger.info("Next-closest galaxy (by fractional offset): {restgal_coord[min_idx].ra.deg:.6f}, {restgal_coord[min_idx].dec.deg:.6f}")
            logger.info(f"Size of this nearby galaxy: {restgal_a[min_idx]:.4f}")
            logger.info(f"Size uncertainty of this nearby galaxy: {restgal_a_std[min_idx]:.4f}")
            logger.info(f"Axis ratio: {restgal_ab[min_idx]:.4f}")
            logger.info(f"Axis ratio uncertainty: {restgal_ab_std[min_idx]:.4f}")
            logger.info(f"Phi: {restgal_phi[min_idx]:.4f}")
            logger.info(f"Phi uncertainty: {restgal_phi_std[min_idx]:.4f}")
            logger.info(f"Angular offset (arcsec): {seps[min_idx]:.2f}")
            logger.info(f"Fractional separation: {np.nanmin(seps / dlr):.4f}")

        # If within the dlr of another galaxy, remove the dimmer galaxy
        if (seps / dlr)[min_idx] < 1:
            if restgal_appmag[min_idx] < appmag[i]:
                # The current galaxy is dimmer
                # (meaning the other galaxy is brighter AND
                # has a dlr that contains this galaxy), drop it
                dropidxs.append(i)
                if verbose == 3:
                    logger.info(f"Dropping objID {onegal_objid} with ra, dec= {onegal_coord.ra.deg:.6f},{onegal_coord.dec.deg:.6f}")
            else:
                dropidxs.append(original_min_idx)
                # The other galaxy is dimmer, drop it
                if verbose == 3:
                    logger.info(
                        f"Dropping objID {objids[original_min_idx]} with"
                        f"ra, dec = {restgal_coord[original_min_idx].ra.deg:.6f}, {restgal_coord[original_min_idx].dec.deg:.6f}"
                    )
    return np.array(dropidxs)
