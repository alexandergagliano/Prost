from astropy.io import ascii
from astropy.table import Table
import numpy as np
from astropy.cosmology import LambdaCDM
import astropy.cosmology.units as cu
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import ascii
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from dl import queryClient as qc
from scipy import stats as st
import time
from astropy.coordinates import Angle
from scipy.stats import norm, halfnorm, truncnorm, uniform, expon, truncexpon
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from astropy.modeling.powerlaws import Schechter1D
from astropy.cosmology import z_at_value
import os
import pickle
from multiprocessing import Pool, cpu_count
import sys
import re
import json
import requests
os.chdir("/Users/alexgagliano/Documents/Research/prob_association/")
from photoz_helper import *

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib

class GalaxyCatalog:
    def __init__(self, name, data=None, n_samples=1000):
        self.name = name
        self.data = data
        self.n_samples = n_samples

    def get_candidates(self, transient, timequery=False):
        search_rad = Angle(200*u.arcsec)
        if transient.redshift == transient.redshift:
            search_rad = Angle(np.nanmax([100 / cosmo.angular_diameter_distance(row.redshift).to(u.kpc).value * 206265, base_radius.arcsec])*u.arcsec)

        self.search_rad = search_rad
        self.search_pos = transient.position
        if timequery:
            start_time = time.time()
        if self.name == 'panstarrs':
            self.mlim = 26
            self.galaxies = build_panstarrs_candidates(transient.name, transient.position, search_rad, n_samples=self.n_samples)
        elif self.name == 'decals':
            self.mlim = 26
            self.galaxies = build_decals_candidates(transient.name, transient.position, search_rad, n_samples=self.n_samples)
        elif self.name == 'glade':
            self.mlim = 17
            if self.data is not None:
                self.galaxies = build_glade_candidates(transient.name, transient.position, search_rad=search_rad, GLADE_catalog=self.data, n_samples=self.n_samples)
            else:
                print("ERROR: Please provide GLADE catalog as 'data' when initializing GalaxyCatalog object.")
                self.galaxies = np.array([])
        if self.galaxies is None:
            self.ngals = 0
        else:
            self.ngals = len(self.galaxies)

        if timequery:
            end_time = time.time()
            elapsed = end_time - start_time
            self.query_time = elapsed

            print(f"{self.ngals} gals retrieved in {elapsed:.2f}s.")

class Transient:
    def __init__(self, name, position, redshift=np.nan, spec_class='', phot_class='', n_samples=1000):
        self.name = name
        self.position = position
        self.redshift = redshift
        self.spec_class = spec_class
        self.phot_class = phot_class
        self.n_samples = n_samples
        self.best_host = -1
        self.second_best_host = -1

        if redshift == redshift:
            self.redshift_std = 0.05*self.redshift
            self.gen_z_samples()
        else:
            self.redshift_std = np.nan
        self.priors = {}
        self.likes = {}

    def __str__(self):
        # Define what should be shown when printing the Transient object
        redshift_str = f"redshift={self.redshift:.4f}" if (self.redshift == self.redshift) else "redshift=N/A"
        class_str = (f"spec. class = {self.spec_class}" if len(self.spec_class) > 0
                     else f"phot. class = {self.phot_class}" if len(self.phot_class) > 0
                     else "unclassified")
        return f"Transient(name={self.name}, position={self.position}, {redshift_str}, {class_str}"

    def get_prior(self, type):
        try:
            prior = self.priors[type]
        except:
            print(f"ERROR: No prior set for {type}.")
            prior = None
        return prior

    def get_likelihood(self, type):
        try:
            like = self.likes[type]
        except:
            print(f"ERROR: No likelihood set for {type}.")
            like = None
        return like

    def set_likelihood(self, type, func):
        self.likes[type] = func

    def set_prior(self, type, func):
        self.priors[type] = func
        if (type == 'redshift') and (self.redshift != self.redshift):
            self.gen_z_samples()

    def set_likelihood(self, type, func):
        self.likes[type] = func

    def gen_z_samples(self, ret=False, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        if self.redshift != self.redshift:
            samples = np.maximum(0.001, self.get_prior('redshift').rvs(size=n_samples))
        else:
            samples = np.maximum(0.001, norm.rvs(self.redshift, self.redshift_std, size=n_samples))
        #now generate distances
        if ret:
            return samples
        else:
            self.redshift_samples = samples
            self.luminosity_distance = cosmo.luminosity_distance(self.redshift_samples)

    def calc_prior_redshift(self, z_best_samples, reduce='mean'):
        pdf = self.get_prior('redshift').pdf(z_best_samples)
        if reduce == 'mean':
            return np.nanmean(pdf, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == 'median':
            return np.nanmedian(pdf, axis=1)
        else:
            return pdf

    def calc_prior_offset(self, fractional_offset_samples, reduce='mean'):
        pdf = self.get_prior('offset').pdf(fractional_offset_samples)
        if reduce == 'mean':
            return np.nanmean(pdf, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == 'median':
            return np.nanmedian(pdf, axis=1)
        else:
            return pdf

    def calc_prior_absmag(self, absmag_samples, reduce='mean'):
        pdf = self.get_prior('absmag').pdf(absmag_samples)
        if reduce == 'mean':
            return np.nanmean(pdf, axis=1)
        elif reduce == 'median':
            return np.nanmedian(pdf, axis=1)
        else:
            return pdf

    def calc_like_redshift(self, z_best_mean, z_best_std, reduce='mean'):
        z_sn_samples = self.redshift_samples[np.newaxis, :]  # Shape: (n_sn_samples, 1)
        z_gal_mean = z_best_mean[:, np.newaxis]  # Shape: (1, n_galaxies)
        z_gal_std = z_best_std[:, np.newaxis]    # Shape: (1, n_galaxies)

        # Calculate the likelihood of each SN redshift sample across each galaxy
        likelihoods = norm.pdf(z_sn_samples, loc=z_gal_mean, scale=z_gal_std)  # Shape: (n_sn_samples, n_galaxies)

        if reduce == 'mean':
            return np.nanmean(likelihoods, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == 'median':
            return np.nanmedian(likelihoods, axis=1)
        else:
            return likelihoods

    def calc_like_offset(self, fractional_offset_samples, reduce='mean'):
        likelihoods = self.get_likelihood('offset').pdf(fractional_offset_samples)
        if reduce == 'mean':
            return np.nanmean(likelihoods, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == 'median':
            return np.nanmedian(likelihoods, axis=1)
        else:
            return likelihoods

    def calc_like_absmag(self, absmag_samples, reduce='mean'):
        #assuming a typical 0.1 SN/century/10^10 Lsol (in K-band)
        #TODO -- convert to K-band luminosity of the host!
        #https://www.aanda.org/articles/aa/pdf/2005/15/aa1411.pdf
        likelihoods = self.get_likelihood('absmag').pdf(absmag_samples)
        if reduce == 'mean':
            return np.nanmean(likelihoods, axis=1)  # Resulting shape: (n_galaxies,)
        elif reduce == 'median':
            return np.nanmedian(likelihoods, axis=1)
        else:
            return likelihoods

    def associate(self, galaxy_catalog, verbose=False):
        ngals = galaxy_catalog.ngals
        search_rad = galaxy_catalog.search_rad
        mlim = galaxy_catalog.mlim
        galaxies = galaxy_catalog.galaxies
        n_samples = galaxy_catalog.n_samples

        # Extract arrays for all galaxies from the catalog
        z_gal_mean = np.array(galaxies['z_best_mean'])
        z_gal_std = np.array(galaxies['z_best_std'])
        offset_arcsec = np.array(galaxies['offset_arcsec'])

        z_gal_samples = np.vstack(galaxies['z_best_samples'])
        galaxy_DLR_samples = np.vstack(galaxies['DLR_samples'])
        absmag_samples = np.vstack(galaxies['absmag_samples'])
        #galaxy_ras = galaxies['ra']
        #galaxy_decs = galaxies['dec']

        #TODO -- store just fractional offset samples
        #just copied now assuming 0 positional uncertainty -- this can be updated later (TODO!)
        offset_arcsec_samples = np.repeat(offset_arcsec[:, np.newaxis], n_samples, axis=1)
        #z_gal_samples = norm.rvs(z_gal_mean[:, np.newaxis], z_gal_std[:, np.newaxis], size=(ngals, n_samples))

        Prior_z = self.calc_prior_redshift(z_gal_samples, reduce=None)
        L_z = self.calc_like_redshift(z_gal_mean, z_gal_std, reduce=None)

        if np.isnan(self.redshift):
            #print("WARNING: No SN redshift; marginalizing to infer probability for the likelihood of observed hosts.")
            # Marginalize over the sampled supernova redshifts by integrating the likelihood over the redshift prior
            sorted_indices = np.argsort(z_gal_samples, axis=1)
            sorted_z_gal_samples = np.take_along_axis(z_gal_samples, sorted_indices, axis=1)
            sorted_integrand = np.take_along_axis(Prior_z * L_z, sorted_indices, axis=1)

            # Perform integration using simps or trapz
            P_z = np.trapz(sorted_integrand, sorted_z_gal_samples, axis=1)
            P_z = P_z[:, np.newaxis]
        else:
            P_z = Prior_z*L_z

        #depends on zgal, NOT zSN
        Prior_absmag = self.calc_prior_absmag(absmag_samples, reduce=None)
        L_absmag = self.calc_like_absmag(absmag_samples, reduce=None)

        # Calculate angular diameter distances for all samples
        galaxy_distances = cosmo.angular_diameter_distance(z_gal_samples).to(u.kpc).value  # Shape (N, M)
        fractional_offset_samples = offset_arcsec_samples/galaxy_DLR_samples

        Prior_offsets = self.calc_prior_offset(fractional_offset_samples, reduce=None)
        L_offsets = self.calc_like_offset(fractional_offset_samples, reduce=None)  # Shape (N,)

        # Compute the posterior probabilities for all galaxies
        P_gals = (P_z) * (Prior_offsets*L_offsets) * (Prior_absmag*L_absmag)

        #other probabilities
        P_hostless = np.array([0]*n_samples) #some very low value that the SN is actually hostless, across all samples.
        P_outside = self.probability_host_outside_cone(search_rad, verbose, n_samples)
        P_unobs = self.probability_of_unobserved_host(search_rad, mlim, verbose, n_samples)

        #sum across all galaxies for these overall metrics
        P_tot = np.nansum(P_gals, axis=0) + P_hostless + P_outside + P_unobs

        #floor to machine precision
        P_tot[P_tot < np.finfo(float).eps] = np.finfo(float).eps

        P_none_norm = (P_outside + P_hostless + P_unobs) / P_tot
        P_any_norm =  np.nansum(P_gals, axis=0) / P_tot

        P_gals_norm = P_gals / P_tot
        P_offsets_norm = Prior_offsets * L_offsets/ P_tot
        P_z_norm = P_z / P_tot
        P_absmag_norm = Prior_absmag * L_absmag / P_tot
        P_outside_norm = P_outside / P_tot
        P_unobs_norm = P_unobs / P_tot

        # Sort and get last 2 indices in descending order
        top_idxs = np.argsort(np.nanmedian(P_gals_norm, axis=1))[::-1]

        self.associated_catalog = galaxy_catalog.name
        self.any_prob = np.nanmedian(P_any_norm)
        self.none_prob = np.nanmedian(P_none_norm)

        #POSTERIORS ARE OFF
        #print("any_prob:")
        #print(np.nanmedian(P_any_norm))
        #print("unobserved:")
        #print(np.nanmedian(P_unobs_norm))
        #print("highest z posterior:")
        #print(np.nanmax(np.nanmedian(P_z, axis=1)))
        #print("highest offset prior:")
        #print(np.nanmax((np.nanmedian(Prior_offsets, axis=1))))
        #print("highest absmag prior:")
        #print(np.nanmax((np.nanmedian(Prior_absmag, axis=1))))

        if np.nanmedian(P_any_norm) > np.nanmedian(P_none_norm):
            print("Association successful!")
            #get best and second-best matches
            self.best_host = top_idxs[0]
            if ngals > 1:
                self.second_best_host = top_idxs[1]
        else:
            self.best_host = -1
            self.second_best_host = top_idxs[0] #set best host as second-best -- best is no host
            if np.nanmedian(P_outside_norm) > np.nanmedian(P_unobs_norm):
                print("Association failed. Host is likely outside search cone.")
            else:
                print("Association failed. Host is likely missing from the catalog.")

        galaxy_catalog.galaxies['z_prob'] = np.nanmedian(P_z_norm)
        galaxy_catalog.galaxies['offset_prob'] = np.nanmedian(P_offsets_norm)
        galaxy_catalog.galaxies['absmag_prob'] = np.nanmedian(P_absmag_norm)
        galaxy_catalog.galaxies['total_prob'] = np.nanmedian(P_gals_norm)

        return galaxy_catalog

    def probability_of_unobserved_host(self, search_rad, m_lim=30, verbose=False, n_samples=1000):
        n_gals = int(0.5*n_samples)
        z_sn = self.redshift
        z_sn_std = self.redshift_std
        z_sn_samples = self.redshift_samples

        if np.isnan(z_sn):
            # draw galaxies from the same distribution
            z_gal_mean = self.gen_z_samples(n_samples=n_gals, ret=True)
            z_gal_std = 0.05 * z_gal_mean
            z_gal_samples = np.maximum(0.001, norm.rvs(loc=z_gal_mean[:, np.newaxis],
                                                       scale=z_gal_std[:, np.newaxis], size=(n_gals, n_samples)))
            Prior_z = self.calc_prior_redshift(z_gal_samples, reduce=None)
            L_z = self.calc_like_redshift(z_gal_mean, z_gal_std, reduce=None)

            sorted_indices = np.argsort(z_gal_samples, axis=1)
            sorted_z_gal_samples = np.take_along_axis(z_gal_samples, sorted_indices, axis=1)
            sorted_integrand = np.take_along_axis(Prior_z * L_z, sorted_indices, axis=1)

            # Perform integration using simps or trapz
            P_z = np.trapz(sorted_integrand, sorted_z_gal_samples, axis=1)
            P_z = P_z[:, np.newaxis]  # Shape: (n_galaxies, 1)
        else:
           # Use the known supernova redshift
            z_best_mean = np.maximum(0.001, norm.rvs(loc=z_sn, scale=0.5*z_sn_std, size=(n_gals)))
            z_best_std = 0.05*z_best_mean #assume all well-constrained redshifts
            z_best_samples = np.maximum(0.001, norm.rvs(loc=z_best_mean[:, np.newaxis], scale=z_best_std[:, np.newaxis], size=(n_gals, n_samples)))

            Prior_z = self.calc_prior_redshift(z_best_samples, reduce=None)
            L_z = self.calc_like_redshift(z_best_mean, z_best_std, reduce=None)
            P_z = Prior_z * L_z

        galaxy_physical_radius_prior_means = halfnorm.rvs(size=n_gals, loc=0, scale=10)  # in kpc
        galaxy_physical_radius_prior_std = 0.05*galaxy_physical_radius_prior_means
        galaxy_physical_radius_prior_samples = norm.rvs(loc=galaxy_physical_radius_prior_means[:, np.newaxis],
                                         scale=galaxy_physical_radius_prior_std[:, np.newaxis],
                                         size=(n_gals, n_samples))

        sn_distance = self.luminosity_distance.to(u.pc).value  # in pc

        min_phys_rad = 0
        max_phys_rad = (search_rad.arcsec / 206265) * sn_distance/1.e3  # in kpc

        physical_offset_mean = np.linspace(min_phys_rad, max_phys_rad, n_gals)
        physical_offset_std = 0.05*physical_offset_mean

        physical_offset_samples = norm.rvs(physical_offset_mean,
                                         physical_offset_std,
                                         size=(n_gals, n_samples))

        fractional_offset_samples =  physical_offset_samples / galaxy_physical_radius_prior_samples  # Shape: (n_samples, n_samples)


        Prior_offset_unobs = self.calc_prior_offset(fractional_offset_samples, reduce=None)
        L_offset_unobs = self.calc_like_offset(fractional_offset_samples, reduce=None)

        absmag_lim_samples = m_lim - 5 * (np.log10(sn_distance / 10))

        absmag_samples = np.linspace(absmag_lim_samples, 0, n_gals)

        Prior_absmag = self.calc_prior_absmag(absmag_samples, reduce=None)
        L_absmag = self.calc_like_absmag(absmag_samples, reduce=None)

        prob_unobs = Prior_absmag * L_absmag * P_z * Prior_offset_unobs * L_offset_unobs

        P_unobserved = np.nanmean(prob_unobs, axis=0)  # Sum over all galaxies

        return P_unobserved

    def probability_host_outside_cone(self, cutout_rad=60, verbose=False, n_samples=1000):
        n_gals = int(n_samples/2)
        z_sn = self.redshift
        z_sn_std = self.redshift_std
        z_sn_samples = self.redshift_samples

        if np.isnan(z_sn):
            # draw galaxies from the same distribution
            z_best_mean = self.gen_z_samples(n_samples=n_gals, ret=True) #draw from prior if redshift is missing
            z_best_std = 0.05 * z_best_mean

            #scatter around some nominal uncertainty
            z_best_samples = np.maximum(0.001, norm.rvs(loc=z_best_mean[:, np.newaxis],
                                                       scale=z_best_std[:, np.newaxis], size=(n_gals, n_samples)))

            Prior_z = self.calc_prior_redshift(z_best_samples, reduce=None)
            L_z = self.calc_like_redshift(z_best_mean, z_best_std, reduce=None)

            sorted_indices = np.argsort(z_best_samples, axis=1)
            sorted_z_best_samples = np.take_along_axis(z_best_samples, sorted_indices, axis=1)
            sorted_integrand = np.take_along_axis(Prior_z * L_z, sorted_indices, axis=1)

            # Perform integration using trapezoidal integration
            P_z = np.trapz(sorted_integrand, sorted_z_best_samples, axis=1)
            P_z = P_z[:, np.newaxis]
        else:
            # Use the known supernova redshift
            #z_sn_std = 0.05 * z_sn
            #z_sn_samples = np.maximum(0.001, norm.rvs(z_sn, z_sn_std, size=n_samples))
            #some higher spread for host redshift photo-zs
            z_best_mean = np.maximum(0.001, norm.rvs(loc=z_sn, scale=0.5*z_sn_std, size=(n_gals)))
            z_best_std = 0.05*z_best_mean #assume all well-constrained redshifts
            z_best_samples = np.maximum(0.001, norm.rvs(loc=z_best_mean[:, np.newaxis], scale=z_best_std[:, np.newaxis], size=(n_gals, n_samples)))

            Prior_z = self.calc_prior_redshift(z_best_samples, reduce=None)
            L_z = self.calc_like_redshift(z_best_mean, z_best_std, reduce=None)

            P_z = Prior_z * L_z

        # sample brightnesses
        absmag_mean = self.get_prior("absmag").rvs(size=n_gals)
        absmag_std = 0.05 * np.abs(absmag_mean)

        absmag_samples = np.maximum(0.001, norm.rvs(loc=absmag_mean[:, np.newaxis],
                                                    scale=absmag_std[:, np.newaxis], size=(n_gals, n_samples)))

        Prior_absmag = self.calc_prior_absmag(absmag_samples, reduce=None)

        # Calculate the distance to the supernova for each sampled redshift
        sn_distances = cosmo.comoving_distance(self.redshift_samples).value  # in Mpc

        # Convert angular cutout radius to physical offset at each sampled redshift
        min_phys_rad = (cutout_rad.arcsec / 206265) * sn_distances * 1e3  # in kpc
        max_phys_rad = 5 * min_phys_rad

        galaxy_physical_radius_prior_means = halfnorm.rvs(size=n_gals, loc=0, scale=10)  # in kpc
        galaxy_physical_radius_prior_std = 0.05*galaxy_physical_radius_prior_means
        galaxy_physical_radius_prior_samples = norm.rvs(loc=galaxy_physical_radius_prior_means[:, np.newaxis],
                                         scale=galaxy_physical_radius_prior_std[:, np.newaxis],
                                         size=(n_gals, n_samples))

        physical_offset_samples = np.linspace(min_phys_rad, max_phys_rad, n_gals)
        fractional_offset_samples = physical_offset_samples / galaxy_physical_radius_prior_samples

        Prior_offset = self.calc_prior_offset(fractional_offset_samples, reduce=None)
        L_offset = self.calc_like_offset(fractional_offset_samples, reduce=None)
        L_absmag = self.calc_like_absmag(absmag_samples, reduce=None)

        prob_outside = Prior_absmag * L_absmag * P_z * Prior_offset * L_offset

        # average over all simulated galaxies -- keep the sample
        P_outside = np.nanmean(prob_outside, axis=0)

        return P_outside


class prior_z_observed_transients(st.rv_continuous):
    def __init__(self, z_min=0, z_max=1, n_bins=100, mag_cutoff=22, Mmean=-19, Mmin=-24, Mmax=-17, R_SN=1e-5, t_obs=1.0, **kwargs):
        super().__init__(**kwargs)

        # Assign the parameters
        self.z_min = z_min
        self.z_max = z_max
        self.n_bins = n_bins
        self.mag_cutoff = mag_cutoff
        self.Mmean = Mmean
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.R_SN = R_SN
        self.t_obs = t_obs

        # Automatically run the internal function to generate the distribution
        self._generate_distribution()

    def _generate_distribution(self):
        # Create redshift bins
        z_bins = np.linspace(self.z_min, self.z_max, self.n_bins + 1)
        z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])  # Centers of redshift bins

        # Calculate the comoving volume element dV/dz for each redshift bin
        dV_dz = cosmo.differential_comoving_volume(z_centers).value  # in Mpc^3 per steradian per dz

        # Full sky solid angle (4 pi steradians)
        solid_angle = 4 * np.pi  # full sky in steradians

        # Supernovae per redshift bin (for full sky)
        supernovae_per_bin = (self.R_SN * dV_dz * solid_angle * np.diff(z_bins)).astype(int)

        # Generate random redshifts for all supernovae
        z_scattered = np.hstack([
            np.random.uniform(z_bins[i], z_bins[i+1], size=n)
            for i, n in enumerate(supernovae_per_bin) if n > 0
        ])

        # Calculate luminosity distance (in parsecs) for all supernovae
        d_L = cosmo.luminosity_distance(z_scattered).to(u.pc).value

        # Sample absolute magnitudes from a Gaussian and clip the values to the range [Mmin, Mmax]
        absolute_magnitudes = np.random.normal(loc=self.Mmean, scale=1.5, size=len(z_scattered))
        absolute_magnitudes = np.clip(absolute_magnitudes, self.Mmin, self.Mmax)

        # Calculate apparent magnitudes using distance modulus for all supernovae
        m_apparent = absolute_magnitudes + 5 * np.log10(d_L / 10)

        # Filter supernovae based on apparent magnitude cutoff
        observed_indices = m_apparent <= self.mag_cutoff
        self.observed_redshifts = z_scattered[observed_indices]

        # Calculate the best fit KDE for observed redshifts
        self.bestFit = st.gaussian_kde(self.observed_redshifts)

    def pdf(self, z):
        """
        Return the PDF (KDE) based on observed redshifts.
        Handles 1D and 2D arrays.
        """
        if z.ndim == 2:
            flat_z = z.flatten()
            flat_pdf = self.bestFit(flat_z)
            return flat_pdf.reshape(z.shape)
        else:
            return self.bestFit(z)

    def rvs(self, size=None, random_state=None):
        """
        Resample the pdf based on observed redshifts.
        """
        return self.bestFit.resample(size=size).reshape(-1)


    def plot(self):
        """
        Plot histogram of observed supernovae and KDE fit.
        """
        z_bins = np.linspace(self.z_min, self.z_max, self.n_bins + 1)

        plt.figure(figsize=(8, 6))
        plt.hist(self.observed_redshifts, bins=z_bins, edgecolor='k', alpha=0.7, density=True, label="Observed Histogram")
        plt.xlabel("Redshift (z)")
        plt.ylabel("Density")

        # Plot the KDE over a fine grid of redshift values
        z_fine = np.linspace(self.z_min, self.z_max, 1000)
        plt.plot(z_fine, self.bestFit(z_fine), color='k', label="KDE Fit")

        plt.title(f"Observed Supernovae with $m < {self.mag_cutoff}$")
        plt.legend()
        plt.show()

class SNRate_absmag(st.rv_continuous):
    def __init__(self, a, b, **kwargs):
        super().__init__(a=a, b=b, **kwargs)
        self.normalization = self._calculate_normalization(a, b)

    def _calculate_normalization(self, a, b):
        """
        Calculate the normalization constant for the Schechter function over the range [a, b].
        """
        result, _ = quad(self._unnormalized_pdf, a, b)
        return result

    def _unnormalized_pdf(self, M_abs_samples):
        """
        Unnormalized function.
        """
        M_sol = 4.74
        Lgal = 10**(-0.4 * (M_abs_samples - M_sol)) # in units of Lsol
        Lgal /= 1.e10 #in units of 10^10 Lsol
        SNRate = 0.1*Lgal
        return SNRate

    def _pdf(self, M_abs_samples):
        """
        Normalized function.
        """
        return self._unnormalized_pdf(M_abs_samples) / self.normalization

#from https://ps1images.stsci.edu/ps1_dr2_api.html
def ps1cone(ra,dec,radius,table="stack",release="dr2",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a cone search of the PS1 catalog

    Parameters
    ----------
    ra (float): (degrees) J2000 Right Ascension
    dec (float): (degrees) J2000 Declination
    radius (float): (degrees) Search radius (<= 0.5 degrees)
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2)
    """

    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    return ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data)

def ps1search(table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    data = kw.copy()
    url = f"{baseurl}/{release}/{table}.{format}"
    if columns:
        dcols = {}
        for col in ps1metadata(table,release)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(badcols)))
        # two different ways to specify a list of column values in the API
        data['columns'] = '[{}]'.format(','.join(columns))

    r = requests.get(url, params=data)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        return r.json()
    else:
        return r.text

def monte_carlo_sample(sn_redshift, transient_pos, galaxies, m_lim, cutout_rad, verbose, n_samples_per_process, seed):
    np.random.seed(seed)
    any_prob, none_prob, post_probs, post_offset, post_z, post_absmag = posterior_samples(
        sn_redshift, transient_pos, galaxies, m_lim=m_lim, cutout_rad=cutout_rad, verbose=verbose, n_samples=n_samples_per_process
    )
    return any_prob, none_prob, post_probs, post_offset, post_z, post_absmag

def parallel_monte_carlo(sn_redshift, transient_pos, galaxies, m_lim, cutout_rad, verbose, n_samples, n_processes):
    n_samples_per_process = n_samples // n_processes
    seeds = np.random.randint(0, 2**32 - 1, size=n_processes)

    start_time = time.time()

    any_prob_list = []
    none_prob_list = []
    post_probs_list = []
    post_offset_list = []
    post_z_list = []
    post_absmag_list = []

    for seed in seeds:
        any_prob, none_prob, post_probs, post_offset, post_z, post_absmag = monte_carlo_sample(
            sn_redshift, transient_pos, galaxies, m_lim, cutout_rad, verbose, n_samples_per_process, seed
        )
        any_prob_list.append(any_prob)
        none_prob_list.append(none_prob)
        post_probs_list.append(post_probs)
        post_offset_list.append(post_offset)
        post_z_list.append(post_z)
        post_absmag_list.append(post_absmag)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"took a total of {elapsed:.2f}s internally to map everything.")

    any_prob = np.concatenate(any_prob_list)
    none_prob = np.concatenate(none_prob_list)
    post_probs = np.concatenate(post_probs_list, axis=1)
    post_offset = np.concatenate(post_offset_list, axis=1)
    post_z = np.concatenate(post_z_list, axis=1)
    post_absmag = np.concatenate(post_absmag_list, axis=1)

    # Concatenate the results
    any_prob_med = np.nanmedian(any_prob)
    none_prob_med = np.nanmedian(none_prob)

    post_probs_med = np.nanmedian(post_probs, axis=1)
    post_offset_med = np.nanmedian(post_offset, axis=1)
    post_z_med = np.nanmedian(post_z, axis=1)
    post_absmag_med = np.nanmedian(post_absmag, axis=1)

    if verbose:
        print(f"Probability your true host is not in this catalog/cone: {none_prob_med:.4e}")
        print(f"Probability your true host is in this catalog: {any_prob_med:.4e}")

    return any_prob_med, none_prob_med, post_probs_med, post_offset_med, post_z_med, post_absmag_med

def build_glade_candidates(transient_name, transient_pos, GLADE_catalog, search_rad=Angle(60*u.arcsec), n_samples=1000):
    candidate_hosts = GLADE_catalog[SkyCoord(GLADE_catalog['RAJ2000'].values*u.deg, GLADE_catalog['DEJ2000'].values*u.deg).separation(transient_pos).arcsec < search_rad.arcsec]
    n_galaxies = len(candidate_hosts)

    if n_galaxies < 1:
        #print(f"No sources found around {transient_name} in GLADE!")
        return None

    temp_PA = candidate_hosts['PAHyp'].values
    temp_PA[temp_PA != temp_PA] = 0 #assume no position angle for unmeasured gals

    # (n) HyperLEDA decimal logarithm of the length of the projected major axis of a galaxy at the isophotal level 25mag/arcsec2 in the B-band, to semi-major half-axis (half-light radius) in arcsec
    temp_sizes = 0.5*3*10**(candidate_hosts['logd25Hyp'].values)

    temp_sizes[temp_sizes < 0.25] = 0.25 #1 pixel, at least for PS1
    temp_sizes_std = np.minimum(temp_sizes, np.abs(temp_sizes) * np.log(10) * candidate_hosts['e_logd25Hyp'].values)

    #if len(candidate_hosts) >1:
    #    print("Removing GLADE shreds...")
    #    shred_idxs = find_glade_shreds(candidate_hosts['RAJ2000'].values, candidate_hosts['DEJ2000'].values, 10**(candidate_hosts['logr25Hyp'].values), temp_PA, temp_sizes, candidate_hosts['Bmag'].values)
    #    if len(shred_idxs) > 0:
    #        print(f"Removing {len(shred_idxs)} indices from tentative matches in GLADE!")
    #    else:
    #        print("No GLADE shreds found.")
    #    candidate_hosts = candidate_hosts[~candidate_hosts.index.isin(shred_idxs)]

    dtype = [('ra', float), ('dec', float), ('z_best_samples', object),('z_best_mean', float),('z_best_std', float),
            ('DLR_samples', object), ('absmag_samples', object), ('offset_arcsec', float),
            ('z_prob', float), ('offset_prob', float), ('absmag_prob', float), ('total_prob', float)]

    galaxies = np.zeros(len(candidate_hosts), dtype=dtype)

    galaxies_pos = SkyCoord(candidate_hosts['RAJ2000'].values*u.deg, candidate_hosts['DEJ2000'].values*u.deg)

    galaxies['ra'] = galaxies_pos.ra.deg
    galaxies['dec'] = galaxies_pos.dec.deg

    # (n) HyperLEDA decimal logarithm of the length of the projected major axis of a galaxy at the isophotal level 25mag/arcsec2 in the B-band, to semi-major half-axis (half-light radius) in arcsec
    temp_sizes = 0.5*3*10**(candidate_hosts['logd25Hyp'].values)
    temp_sizes[temp_sizes < 0.25] = 0.25 #1 pixel, at least for PS1
    temp_sizes_std = np.minimum(temp_sizes, np.abs(temp_sizes) * np.log(10) * candidate_hosts['e_logd25Hyp'].values)


    temp_sizes_std[temp_sizes_std != temp_sizes_std] = 0.05*temp_sizes[temp_sizes_std != temp_sizes_std]
    temp_sizes_std[temp_sizes_std < 0.05*temp_sizes] = 0.05*temp_sizes[temp_sizes_std < 0.05*temp_sizes]

    #dist_samples = np.maximum(5, norm.rvs(loc=candidate_hosts['Dist'].values[:, np.newaxis],
    #                        scale=0.3*candidate_hosts['Dist'].values[:, np.newaxis], #assuming, conservatively, a 30% distance uncertainty
    #                        size=(len(candidate_hosts), n_samples)))

    # Convert distances to corresponding redshift (no sampling yet, just conversion)
    z_best_mean = z_at_value(cosmo.luminosity_distance, candidate_hosts['Dist'].values * u.Mpc, zmin=1.e-15)
    z_std = 0.3 * z_best_mean  # 30% uncertainty on redshift

    z_best_samples = np.maximum(0.001, norm.rvs(loc=z_best_mean[:, np.newaxis],
                                                scale=z_std[:, np.newaxis],
                                                size=(len(candidate_hosts), n_samples)))

    #z_best_samples = z_at_value(cosmo.luminosity_distance, dist_samples*u.Mpc, zmin=1.e-15)

    galaxy_a_over_b = 10**(candidate_hosts['logr25Hyp'].values)
    galaxy_a_over_b_std = galaxy_a_over_b * np.log(10) * candidate_hosts['e_logr25Hyp'].values

    #set uncertainty floor
    nanbool = galaxy_a_over_b_std != galaxy_a_over_b_std
    galaxy_a_over_b_std[nanbool] = 0.05*galaxy_a_over_b[nanbool]

    temp_PA = candidate_hosts['PAHyp'].values
    temp_PA[temp_PA != temp_PA] = 0 #assume no position angle for unmeasured gals

    phi = np.radians(temp_PA)
    phi_std = 0.05*phi #uncertainty floor

    DLR_samples = calc_DLR(transient_pos, galaxies_pos,
        temp_sizes, temp_sizes_std, galaxy_a_over_b, galaxy_a_over_b_std, phi, phi_std, n_samples=n_samples)

    galaxies['z_best_mean'] = np.nanmean(z_best_samples, axis=1)
    galaxies['z_best_std'] = np.nanstd(z_best_samples, axis=1)

    z_best_samples[z_best_samples < 0] = 0.001 #set photometric redshift floor
    z_best_samples[z_best_samples != z_best_samples] = 0.001 #set photometric redshift floor

    temp_mag_r = candidate_hosts['Bmag'].values
    temp_mag_r_std = np.abs(candidate_hosts['e_Bmag'].values)

    absmag_samples = norm.rvs(loc=temp_mag_r[:, np.newaxis], scale=temp_mag_r_std[:, np.newaxis], size=(len(temp_mag_r), n_samples)) - cosmo.distmod(z_best_samples).value

    # Calculate angular separation between SN and all galaxies (in arcseconds)
    transient_ra, transient_dec = transient_pos.ra.degree, transient_pos.dec.degree
    galaxies['offset_arcsec'] = SkyCoord(galaxies['ra']*u.deg, galaxies['dec']*u.deg).separation(transient_pos).arcsec

    for i in range(n_galaxies):
        galaxies['z_best_samples'][i] = z_best_samples[i, :]
        galaxies['DLR_samples'][i] = DLR_samples[i, :]
        galaxies['absmag_samples'][i] = absmag_samples[i, :]

    return galaxies


def build_decals_candidates(transient_name, transient_pos, search_rad=60, n_samples=1000):
    rad_deg = search_rad.deg

    result = qc.query(sql=f"""SELECT
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
        ls_dr9.tractor t
    INNER JOIN
        ls_dr9.photo_z pz
    ON
        t.ls_id= pz.ls_id
    WHERE
        q3c_radial_query(t.ra, t.dec, {transient_pos.ra.deg:.5f}, {transient_pos.dec.deg:.5f}, {rad_deg})
    AND (t.nobs_r > 0) AND (t.dered_flux_r > 0) AND (t.snr_r > 0) AND nullif(t.dered_mag_r, 'NaN') is not null AND (t.fitbits != 8192) AND ((pz.z_spec > 0) OR (pz.z_phot_mean > 0))""")

    candidate_hosts = ascii.read(result).to_pandas()

    n_galaxies = len(candidate_hosts)

    if n_galaxies < 1:
        #print(f"No sources found around {transient_name} in DECaLS! Double-check that the SN coords overlap the survey footprint.")
        return None

    dtype = [('ra', float), ('dec', float), ('physical_size_kpc', float),
            ('z_best_samples', object),('z_best_mean', float),('z_best_std', float),('ls_id', int),
            ('DLR_samples', object), ('absmag_samples', object), ('offset_arcsec', float),
            ('z_prob', float), ('offset_prob', float), ('absmag_prob', float), ('total_prob', float)]

    galaxies = np.zeros(len(candidate_hosts), dtype=dtype)
    galaxies_pos = SkyCoord(candidate_hosts['ra'].values*u.deg, candidate_hosts['dec'].values*u.deg)

    galaxies['ra'] = galaxies_pos.ra.deg
    galaxies['dec'] = galaxies_pos.dec.deg

    temp_sizes = candidate_hosts['shape_r'].values
    temp_sizes[temp_sizes < 0.25] = 0.25 #1 pixel, at least for PS1
    temp_sizes_std = candidate_hosts['shape_r_ivar'].values
    temp_sizes_std = np.maximum(temp_sizes_std, 1.e-10)
    temp_sizes_std  = np.minimum(np.sqrt(1/temp_sizes_std), temp_sizes)

    temp_sizes_std[temp_sizes_std != temp_sizes_std] = 0.05*temp_sizes[temp_sizes_std != temp_sizes_std]
    temp_sizes_std[temp_sizes_std < 0.05*temp_sizes] = 0.05*temp_sizes[temp_sizes_std < 0.05*temp_sizes]

    galaxy_photoz_median = candidate_hosts['z_phot_median'].values
    galaxy_photoz_mean = candidate_hosts['z_phot_mean'].values
    galaxy_photoz_std = candidate_hosts['z_phot_std'].values
    galaxy_specz = candidate_hosts['z_spec'].values

    temp_e1 = candidate_hosts['shape_e1'].astype("float").values
    temp_e1_std = candidate_hosts['shape_e1_ivar'].values
    temp_e1_std[temp_e1_std < 0.05*temp_e1] = 0.05*temp_e1[temp_e1_std < 0.05*temp_e1]
    temp_e1_std[temp_e1_std > 1.e-10]  = np.sqrt(1/temp_e1_std[temp_e1_std > 1.e-10])

    temp_e2 = candidate_hosts['shape_e2'].astype("float").values
    temp_e2_std = candidate_hosts['shape_e2_ivar'].values
    temp_e2_std[temp_e2_std < 0.05*temp_e2] = 0.05*temp_e2[temp_e2_std < 0.05*temp_e2]
    temp_e2_std[temp_e1_std > 1.e-10]  = np.sqrt(1/temp_e2_std[temp_e2_std > 1.e-10])

    # Calculate ellipticity and axis ratio for all samples
    e = np.sqrt(temp_e1**2 + temp_e2**2)
    e = np.maximum(e, 1.e-10)

    a_over_b = (1 + e) / (1 - e)

    #propagated uncertainties
    # Compute uncertainty in e (sigma_e)
    e_std = (1 / e) * np.sqrt(temp_e1**2 * temp_e1_std**2 + temp_e2**2 * temp_e2_std**2)

    # Compute uncertainty in a_over_b (sigma_a_over_b)
    a_over_b_std = (2 / (1 - e)**2) * e_std

    # Position angle and angle calculations for all samples
    phi = -np.arctan2(temp_e2, temp_e1) / 2

    #DOUBLE-CHECK THESE
    denom = temp_e1**2 + temp_e2**2
    denom = np.maximum(1.e-10, denom)

    partial_phi_e1 = (temp_e2 / denom)
    partial_phi_e2 = (-temp_e1 / denom)

    # Propagate uncertainties
    phi_std = 0.5 * np.sqrt((partial_phi_e1**2 * temp_e1_std**2) + (partial_phi_e2**2 * temp_e2_std**2))
    phi_std = np.abs(phi_std)
    phi_std[phi_std != phi_std] = 0.05*phi[phi_std != phi_std]
    phi_std = np.maximum(1.e-10, np.maximum(0.05*np.abs(phi), phi_std)) #5% uncertainty floor

    DLR_samples = calc_DLR(transient_pos, galaxies_pos, temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std)

    galaxies['z_best_mean'] = galaxy_photoz_mean
    galaxies['z_best_std'] = np.abs(galaxy_photoz_std)
    galaxies['z_best_std'][galaxy_specz > 0] = 0.05*galaxy_specz[galaxy_specz > 0] #floor of 5%
    galaxies['z_best_mean'][galaxy_specz > 0] = galaxy_specz[galaxy_specz > 0]
    galaxies['z_best_std'][np.abs(galaxy_photoz_std) > 0.5*galaxy_photoz_mean] = 0.5*galaxy_photoz_mean[np.abs(galaxy_photoz_std) > 0.5*galaxy_photoz_mean]#ceiling of 50%
    galaxies['ls_id'] = candidate_hosts['ls_id'].values

    # Calculate angular separation between SN and all galaxies (in arcseconds)
    transient_ra, transient_dec = transient_pos.ra.degree, transient_pos.dec.degree
    galaxies['offset_arcsec'] = SkyCoord(galaxies['ra']*u.deg, galaxies['dec']*u.deg).separation(transient_pos).arcsec

    z_best_samples = norm.rvs(
        galaxies['z_best_mean'][:, np.newaxis],  # Shape (N, 1) to allow broadcasting
        galaxies['z_best_std'][:, np.newaxis],  # Shape (N, 1)
        size=(n_galaxies, n_samples)  # Shape (N, M)
    )
    z_best_samples[z_best_samples < 0] = 0.001 #set photometric redshift floor
    z_best_samples[z_best_samples != z_best_samples] = 0.001 #set photometric redshift floor

    temp_mag_r = candidate_hosts['dered_mag_r'].values

    temp_mag_r_std = np.abs(2.5/np.log(10)*np.sqrt(1/np.maximum(0, candidate_hosts['flux_ivar_r'].values))/candidate_hosts['flux_r'].values)

    #cap at 50% the mag
    temp_mag_r_std[temp_mag_r_std > temp_mag_r] = 0.5*temp_mag_r[temp_mag_r_std > temp_mag_r]

    absmag_samples = norm.rvs(loc=temp_mag_r[:, np.newaxis], scale=temp_mag_r_std[:, np.newaxis], size=(len(temp_mag_r), n_samples)) - cosmo.distmod(z_best_samples).value

    for i in range(n_galaxies):
        galaxies['z_best_samples'][i] = z_best_samples[i, :]
        galaxies['DLR_samples'][i] = DLR_samples[i, :]
        galaxies['absmag_samples'][i] = absmag_samples[i, :]

    return galaxies

#galaxies = build_panstarrs_candidates('testName', SkyCoord(229.18614421, 72.4478135, unit=(u.deg, u.deg)), search_rad=Angle(10*u.arcsec))

#the right host
#result = ps1cone(229.18372841, 72.44804326, 200/3600, release='dr2')
#result_photoz = ps1cone(229.18372841, 72.44804326, 200/3600, columns=photoz_cols, table='forced_mean',  release='dr2')

def build_panstarrs_candidates(transient_name, transient_pos, search_rad=Angle(60*u.arcsec), n_samples=1000):
    rad_deg = search_rad.deg
    start_query = time.time()

    result = ps1cone(transient_pos.ra.deg, transient_pos.dec.deg, rad_deg, release='dr2')

    photoz_cols = ['objID', 'raMean', 'decMean', 'gFKronFlux', 'rFKronFlux', 'iFKronFlux', 'zFKronFlux', 'yFKronFlux',
                        'gFPSFFlux', 'rFPSFFlux', 'iFPSFFlux', 'zFPSFFlux', 'yFPSFFlux',
                        'gFApFlux', 'rFApFlux', 'iFApFlux', 'zFApFlux', 'yFApFlux',
                        'gFmeanflxR5', 'rFmeanflxR5', 'iFmeanflxR5', 'zFmeanflxR5', 'yFmeanflxR5',
                        'gFmeanflxR6', 'rFmeanflxR6', 'iFmeanflxR6', 'zFmeanflxR6', 'yFmeanflxR6',
                        'gFmeanflxR7', 'rFmeanflxR7', 'iFmeanflxR7', 'zFmeanflxR7', 'yFmeanflxR7']

    result_photoz = ps1cone(transient_pos.ra.deg, transient_pos.dec.deg, rad_deg, columns=photoz_cols, table='forced_mean', release='dr2')

    end_query = time.time()
    elapsed = end_query - start_query

    print(f"Took {elapsed:.2f}s to query for the data (detection & photo-z cols).")

    candidate_hosts = ascii.read(result).to_pandas()
    candidate_hosts_pzcols = ascii.read(result_photoz).to_pandas()
    candidate_hosts = candidate_hosts.set_index('objID')
    candidate_hosts_pzcols = candidate_hosts_pzcols.set_index('objID')
    candidate_hosts = candidate_hosts.join(candidate_hosts_pzcols, lsuffix="_DROP").filter(regex="^(?!.*DROP)").reset_index()

    candidate_hosts.replace(-999, np.nan, inplace=True)
    candidate_hosts.sort_values(by=['distance'], inplace=True)
    candidate_hosts.reset_index(inplace=True, drop=True)
    candidate_hosts.loc[candidate_hosts['objID'] == 194932291837398481, 'nDetections']

    #no good shape info?
    candidate_hosts['momentXX'] = candidate_hosts[['gmomentXX', 'rmomentXX', 'imomentXX', 'zmomentXX', 'ymomentXX']].median(axis=1)
    candidate_hosts['momentYY'] = candidate_hosts[['gmomentYY', 'rmomentYY', 'imomentYY', 'zmomentYY', 'ymomentYY']].median(axis=1)
    candidate_hosts['momentXY'] = candidate_hosts[['gmomentXY', 'rmomentXY', 'imomentXY', 'zmomentXY', 'ymomentXY']].median(axis=1)

    #candidate_hosts = candidate_hosts[candidate_hosts['nDetections'] > 1] #some VERY basic filtering
    candidate_hosts['KronRad'] = candidate_hosts[['gKronRad','rKronRad','iKronRad','zKronRad','yKronRad']].median(axis=1)
    candidate_hosts['KronMag'] = candidate_hosts[['gKronMag','rKronMag','iKronMag','zKronMag','yKronMag']].median(axis=1)
    candidate_hosts['KronMagErr'] = candidate_hosts[['gKronMagErr','rKronMagErr','iKronMagErr','zKronMagErr','yKronMagErr']].median(axis=1)

    candidate_hosts.dropna(subset=['raMean', 'decMean', 'momentXX', 'momentYY', 'momentYY','KronRad', 'KronMag'], inplace=True)

    n_galaxies = len(candidate_hosts)

    if n_galaxies < 1:
        #print(f"No sources found around {transient_name} in Panstarrs DR2! Double-check that the SN coords overlap the survey footprint.")
        return None

    dtype = [('ra', float), ('dec', float), ('physical_size_kpc', float),
            ('z_best_samples', object),('z_best_mean', float),('z_best_std', float),('ls_id', int),
            ('DLR_samples', object), ('absmag_samples', object), ('offset_arcsec', float),
            ('z_prob', float), ('offset_prob', float), ('absmag_prob', float), ('total_prob', float)]

    galaxies = np.zeros(len(candidate_hosts), dtype=dtype)

    #get photozs from Andrew Engel's code!
    # TODO -- speedup this line!
    DEFAULT_MODEL_PATH = './MLP_lupton.hdf5'
    DEFAULT_DUST_PATH = '.'

    model, range_z = load_lupton_model(model_path=DEFAULT_MODEL_PATH, dust_path=DEFAULT_DUST_PATH)

    X = preprocess(candidate_hosts, PATH=os.path.join(DEFAULT_DUST_PATH, 'sfddata-master'))
    posteriors, point_estimates, errors = evaluate(X, model, range_z)
    point_estimates[point_estimates < 0.001] = 0.001 #set photometric redshift floor
    point_estimates[point_estimates != point_estimates] = 0.001 #set photometric redshift floor

    #5% uncertainty floor
    err_bool = errors < 0.05*point_estimates
    errors[err_bool] = 0.05*point_estimates[err_bool]

    # not QUITE the mean of the posterior, but we're assuming it's gaussian :/
    #TODO -- sample from the full posterior!
    galaxies['z_best_mean'] = point_estimates
    galaxies['z_best_std'] = np.abs(errors)

    z_best_samples = norm.rvs(
        galaxies['z_best_mean'][:, np.newaxis],  # Shape (N, 1) to allow broadcasting
        galaxies['z_best_std'][:, np.newaxis],  # Shape (N, 1)
        size=(n_galaxies, n_samples)  # Shape (N, M)
    )

    z_best_samples[z_best_samples < 0.001] = 0.001 #set photometric redshift floor
    #z_best_samples[z_best_samples != z_best_samples] = 0.001 #set photometric redshift floor

    galaxies_pos = SkyCoord(candidate_hosts['raMean'].values*u.deg, candidate_hosts['decMean'].values*u.deg)

    galaxies['ra'] = galaxies_pos.ra.deg
    galaxies['dec'] = galaxies_pos.dec.deg

    temp_sizes = candidate_hosts['KronRad'].values
    #temp_sizes[temp_sizes < 0.25] = 0.25 #1 pixel, at least for PS1

    temp_sizes_std = 0.05* candidate_hosts['KronRad'].values
    temp_sizes_std = np.maximum(temp_sizes_std, 1e-10)  # Prevent division by zero

    #temp_sizes_std[temp_sizes_std != temp_sizes_std] = 0.05*temp_sizes[temp_sizes_std != temp_sizes_std]
    temp_sizes_std[temp_sizes_std < 0.05*temp_sizes] = 0.05*temp_sizes[temp_sizes_std < 0.05*temp_sizes]

    U = candidate_hosts['momentXY'].values
    Q = candidate_hosts['momentXX'].values - candidate_hosts['momentYY'].values

    phi = 0.5*np.arctan(U/Q)
    phi_std = 0.05*np.abs(phi)
    phi_std = np.maximum(1.e-10, phi_std)
    kappa = Q**2 + U**2
    a_over_b = (1 + kappa + 2*np.sqrt(kappa))/(1 - kappa)
    a_over_b_std = 0.05*np.abs(a_over_b) #uncertainty floor

    DLR_samples = calc_DLR(transient_pos, galaxies_pos, temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std, n_samples=n_samples)
    plt.hist(DLR_samples.ravel())

    temp_mag_r = candidate_hosts['KronMag'].values
    temp_mag_r_std = candidate_hosts['KronMagErr'].values

    #cap at 50% the mag
    temp_mag_r_std[temp_mag_r_std > temp_mag_r] = 0.5*temp_mag_r[temp_mag_r_std > temp_mag_r]

    absmag_samples = norm.rvs(loc=temp_mag_r[:, np.newaxis], scale=np.abs(temp_mag_r_std[:, np.newaxis]), size=(len(temp_mag_r), n_samples)) - cosmo.distmod(z_best_samples).value

    # Calculate angular separation between SN and all galaxies (in arcseconds)
    transient_ra, transient_dec = transient_pos.ra.degree, transient_pos.dec.degree
    galaxies['offset_arcsec'] = SkyCoord(galaxies['ra']*u.deg, galaxies['dec']*u.deg).separation(transient_pos).arcsec

    for i in range(n_galaxies):
        galaxies['z_best_samples'][i] = z_best_samples[i, :]
        galaxies['DLR_samples'][i] = DLR_samples[i, :]
        galaxies['absmag_samples'][i] = absmag_samples[i, :]

    return galaxies

def calc_DLR(transient_pos, galaxies_pos, a, a_std,
             a_over_b, a_over_b_std, phi, phi_std, n_samples=1000):
    n_gals = len(galaxies_pos)

    transient_ra = transient_pos.ra.deg
    transient_dec = transient_pos.dec.deg

    if n_samples > 1:
        #TODO -- incorporate uncertainty in galaxy and transient position
        hosts_ra = galaxies_pos.ra.deg[:, np.newaxis]
        hosts_dec = galaxies_pos.dec.deg[:, np.newaxis]

        a = norm.rvs(loc=a[:, np.newaxis], scale=a_std[:, np.newaxis], size=(n_gals, n_samples))
        a_over_b = norm.rvs(loc=a_over_b[:, np.newaxis], scale=a_over_b_std[:, np.newaxis], size=(n_gals, n_samples))
        phi = norm.rvs(loc=phi[:, np.newaxis], scale=phi_std[:, np.newaxis], size=(n_gals, n_samples))
    else:
        hosts_ra =galaxies_pos.ra.deg
        hosts_dec = galaxies_pos.dec.deg

    xr = (transient_ra - hosts_ra)*3600
    yr = (transient_dec - hosts_dec)*3600

    gam = np.arctan2(xr, yr)
    theta = phi - gam

    DLR = a/np.sqrt(((a_over_b)*np.sin(theta))**2 + (np.cos(theta))**2)

    return DLR
