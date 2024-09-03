import numpy as np
from astropy.cosmology import LambdaCDM
import astropy.cosmology.units as cu
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import ascii
import pandas as pd
from dl import queryClient as qc
import time
from astropy.coordinates import Angle
from scipy.stats import norm, halfnorm, truncnorm, uniform, expon, truncexpon
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from astropy.modeling.powerlaws import Schechter1D
from astropy.cosmology import z_at_value
import os
from astro_ghost.PS1QueryFunctions import getcolorim
from astro_ghost.PS1QueryFunctions import get_PS1_Pic, find_all
from astropy.io import fits
from astropy.visualization import SqrtStretch
from astropy.visualization import ZScaleInterval
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
import pickle
os.chdir("/Users/alexgagliano/Documents/Research/prob_association/")
from diagnose import *
import seaborn as sns
import scipy.stats as st

sns.set_context("talk")

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

class schechter(st.rv_continuous):
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
        Unnormalized Schechter function.
        """
        h = 0.7  # Flat cosmology
        phi_star = 5.37e-5 * h**3  # h^3 Mpc^-3 mag^-1 yr^-1
        M_star = -20.60 - 5 * np.log10(h / 1.0)
        alpha = -0.62

        L_samples = 10**(-0.4 * (M_abs_samples - M_star))
        schechter_values = 0.4 * np.log(10) * phi_star * L_samples**(alpha + 1) * np.exp(-L_samples)
        return schechter_values

    def _pdf(self, M_abs_samples):
        """
        Normalized Schechter function.
        """
        return self._unnormalized_pdf(M_abs_samples) / self.normalization

# define priors for properties
priorfunc_z      = halfnorm(loc=0, scale=0.3)
priorfunc_offset = uniform(loc=0, scale=5)
priorfunc_absmag = schechter(a=-25, b=-10, name='Hosts_Schechter')

def build_decals_candidates(sn_position, search_rad=60, n_samples=1000):
    rad_deg = search_rad/3600.

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
        q3c_radial_query(t.ra, t.dec, {sn_position.ra.deg:.5f}, {sn_position.dec.deg:.5f}, {rad_deg})
    AND (t.nobs_r > 0) AND (t.dered_flux_r > 0) AND (t.snr_r > 0) AND nullif(t.dered_mag_r, 'NaN') is not null AND ((pz.z_spec > 0) OR (pz.z_phot_mean > 0))""")

    candidate_hosts = ascii.read(result).to_pandas()

    n_galaxies = len(candidate_hosts)

    if n_galaxies < 1:
        print(f"No sources found around {sn_name} in DECaLS! Double-check that the SN coords overlap the survey footprint.")
        return None

    dtype = [('ra', float), ('dec', float), ('redshift', float), ('physical_size_kpc', float), ('angular_size_arcsec', float),
            ('angular_size_arcsec_std', float),('z_best_samples', object),('z_best_mean', float),('z_best_std', float),('ls_id', int),
            ('DLR_samples', object), ('absmag_samples', object)]

    galaxies = np.zeros(len(candidate_hosts), dtype=dtype)
    galaxies['ra'] = candidate_hosts['ra'].values
    galaxies['dec'] = candidate_hosts['dec'].values

    temp_sizes = candidate_hosts['shape_r'].values
    temp_sizes[temp_sizes < 0.25] = 0.25 #1 pixel, at least for PS1
    temp_sizes_std = candidate_hosts['shape_r_ivar'].values
    temp_sizes_std  = np.minimum(np.sqrt(1/temp_sizes_std), temp_sizes)

    temp_sizes_std[temp_sizes_std != temp_sizes_std] = 0.05*temp_sizes[temp_sizes_std != temp_sizes_std]
    temp_sizes_std[temp_sizes_std < 0.05*temp_sizes] = 0.05*temp_sizes[temp_sizes_std < 0.05*temp_sizes]
    galaxies['angular_size_arcsec'] = temp_sizes
    galaxies['angular_size_arcsec_std'] = temp_sizes_std

    galaxy_photoz_median = candidate_hosts['z_phot_median'].values
    galaxy_photoz_mean = candidate_hosts['z_phot_mean'].values
    galaxy_photoz_std = candidate_hosts['z_phot_std'].values
    galaxy_specz = candidate_hosts['z_spec'].values

    temp_e1 = candidate_hosts['shape_e1'].astype("float")
    temp_e1_std = candidate_hosts['shape_e1_ivar'].values
    temp_e1_std[temp_e1_std < 0.05*temp_e1] = 0.05*temp_e1[temp_e1_std < 0.05*temp_e1]
    temp_e1_std[temp_e1_std > 1.e-10]  = np.sqrt(1/temp_e1_std[temp_e1_std > 1.e-10])

    temp_e2 = candidate_hosts['shape_e2'].astype("float")
    temp_e2_std = candidate_hosts['shape_e2_ivar'].values
    temp_e2_std[temp_e2_std < 0.05*temp_e2] = 0.05*temp_e2[temp_e2_std < 0.05*temp_e2]
    temp_e2_std[temp_e1_std > 1.e-10]  = np.sqrt(1/temp_e2_std[temp_e2_std > 1.e-10])

    DLR_samples = calc_decals_DLR_with_err(sn_position.ra.deg, sn_position.dec.deg, candidate_hosts['ra'],
        candidate_hosts['dec'], candidate_hosts['shape_e1'], candidate_hosts['shape_e2'],
        candidate_hosts['shape_r'], temp_e1_std, temp_e2_std, temp_sizes_std, search_rad=search_rad)

    galaxies['z_best_mean'] = galaxy_photoz_mean
    galaxies['z_best_std'] = np.abs(galaxy_photoz_std)
    galaxies['z_best_std'][galaxy_specz > 0] = 0.05*galaxy_specz[galaxy_specz > 0] #ffloor
    galaxies['z_best_mean'][galaxy_specz > 0] = galaxy_specz[galaxy_specz > 0]
    galaxies['ls_id'] = candidate_hosts['ls_id'].values

    z_best_samples = norm.rvs(
        galaxies['z_best_mean'][:, np.newaxis],  # Shape (N, 1) to allow broadcasting
        galaxies['z_best_std'][:, np.newaxis],  # Shape (N, 1)
        size=(n_galaxies, n_samples)  # Shape (N, M)
    )
    z_best_samples[z_best_samples < 0] = 0.001 #set photometric redshift floor
    z_best_samples[z_best_samples != z_best_samples] = 0.001 #set photometric redshift floor

    temp_mag_r = candidate_hosts['dered_mag_r'].values

    temp_mag_r_std = np.abs(2.5/np.log(10)*np.sqrt(1/np.maximum(0, candidate_hosts['flux_ivar_r'].values))/candidate_hosts['flux_r'].values)

    absmag_samples = norm.rvs(loc=temp_mag_r[:, np.newaxis], scale=temp_mag_r_std[:, np.newaxis], size=(len(temp_mag_r), n_samples)) - cosmo.distmod(z_best_samples).value

    for i in range(n_galaxies):
        galaxies['z_best_samples'][i] = z_best_samples[i, :]
        galaxies['DLR_samples'][i] = DLR_samples[i, :]
        galaxies['absmag_samples'][i] = absmag_samples[i, :]

    return galaxies

def build_glade_candidates(sn_position, search_rad, GLADE_catalog, n_samples):

    candidate_hosts = GLADE_catalog[SkyCoord(GLADE_catalog['RAJ2000'].values*u.deg, GLADE_catalog['DEJ2000'].values*u.deg).separation(sn_position).arcsec < search_rad]
    n_galaxies = len(candidate_hosts)

    if n_galaxies < 1:
        print(f"No sources found around {sn_name} in GLADE!")
        return None

    dtype = [('ra', float), ('dec', float), ('redshift', float), ('angular_size_arcsec', float),
            ('angular_size_arcsec_std', float),('z_best_samples', object),('z_best_mean', float),('z_best_std', float),
            ('DLR_samples', object), ('absmag_samples', object)]

    galaxies = np.zeros(len(candidate_hosts), dtype=dtype)
    galaxies['ra'] = candidate_hosts['RAJ2000'].values
    galaxies['dec'] = candidate_hosts['DEJ2000'].values

    # (n) HyperLEDA decimal logarithm of the length of the projected major axis of a galaxy at the isophotal level 25mag/arcsec2 in the B-band, to semi-major half-axis (half-light radius) in arcsec
    temp_sizes = 0.5*3*10**(candidate_hosts['logd25Hyp'].values)
    temp_sizes[temp_sizes < 0.25] = 0.25 #1 pixel, at least for PS1
    temp_sizes_std = np.minimum(temp_sizes, np.abs(temp_sizes) * np.log(10) * candidate_hosts['e_logd25Hyp'].values)

    temp_sizes_std[temp_sizes_std != temp_sizes_std] = 0.05*temp_sizes[temp_sizes_std != temp_sizes_std]
    temp_sizes_std[temp_sizes_std < 0.05*temp_sizes] = 0.05*temp_sizes[temp_sizes_std < 0.05*temp_sizes]

    galaxies['angular_size_arcsec'] = temp_sizes
    galaxies['angular_size_arcsec_std'] = temp_sizes_std

    dist_samples = norm.rvs(loc=candidate_hosts['Dist'].values[:, np.newaxis],
                            scale=0.05*candidate_hosts['Dist'].values[:, np.newaxis], #assuming, conservatively, a 5% distance uncertainty
                            size=(len(candidate_hosts), n_samples))

    z_best_samples = z_at_value(cosmo.luminosity_distance, dist_samples*u.Mpc)
    #galaxy_specz =  z_at_value(cosmo.luminosity_distance, candidate_hosts['Dist'].values*u.Mpc)
    galaxy_a_over_b = 10**(candidate_hosts['logr25Hyp'])
    galaxy_a_over_b_std = galaxy_a_over_b * np.log(10) * candidate_hosts['e_logr25Hyp'].values

    DLR_samples = calc_GLADE_DLR_with_err(sn_position.ra.deg, sn_position.dec.deg, candidate_hosts['RAJ2000'],
        candidate_hosts['DEJ2000'], candidate_hosts['PAHyp'],
        temp_sizes, galaxy_a_over_b, temp_sizes_std, galaxy_a_over_b_std, n_samples=n_samples, search_rad=search_rad)

    galaxies['z_best_mean'] = np.nanmean(photo_z_samples, axis=1)
    galaxies['z_best_std'] = np.nanstd(photo_z_samples, axis=1)

    z_best_samples[z_best_samples < 0] = 0.001 #set photometric redshift floor
    z_best_samples[z_best_samples != z_best_samples] = 0.001 #set photometric redshift floor

    temp_mag_r = candidate_hosts['BmagHyp'].values
    temp_mag_r_std = np.abs(candidate_hosts['e_BmagHyp'].values)

    absmag_samples = norm.rvs(loc=temp_mag_r[:, np.newaxis], scale=temp_mag_r_std[:, np.newaxis], size=(len(temp_mag_r), n_samples)) - cosmo.distmod(photo_z_samples).value

    for i in range(n_galaxies):
        galaxies['z_best_samples'][i] = photo_z_samples[i, :]
        galaxies['DLR_samples'][i] = DLR_samples[i, :]
        galaxies['absmag_samples'][i] = absmag_samples[i, :]

    return galaxies

def calc_DLR_decals(ra_SN, dec_SN, ra_hosts, dec_hosts, e1, e2, size):
    #definitions buried deep in the tractor documentation i.e. https://github.com/dstndstn/tractor/blob/main/tractor/ellipses.py

    # Calculate angular offsets for all galaxies
    xr = (ra_SN - ra_hosts.values[:, np.newaxis]) * 3600
    yr = (dec_SN - dec_hosts.values[:, np.newaxis]) * 3600

    # Calculate ellipticity and axis ratio for all samples
    e = np.sqrt(e1**2 + e2**2)
    a_over_b = (1 + e) / (1 - e)

    # Position angle and angle calculations for all samples
    phi = -np.arctan2(e2, e1) / 2
    gam = np.arctan2(xr, yr)
    theta = phi - gam

    # Vectorized calculation of DLR for all samples
    DLR = size / np.sqrt(((a_over_b) * np.sin(theta))**2 + (np.cos(theta))**2)

    #if no shape information, radius in direction of the SN is radius of the galaxy.
    mask = (e1 < 1.e-5) & (e2 < 1.e-5) & (size < 1.e-5)
    DLR[mask] = size[mask]
    return DLR

def calc_decals_DLR_with_err(ra_SN, dec_SN, ra_hosts, dec_hosts, e1_hosts, e2_hosts, angular_size_hosts, sigma_e1, sigma_e2, sigma_size, n_samples=1000, search_rad=60):
    # Generate 2D samples for e1, e2, and angular_size for each galaxy
    e1_samples = norm.rvs(loc=e1_hosts.values[:, np.newaxis], scale=sigma_e1[:, np.newaxis], size=(len(e1_hosts), n_samples))
    e2_samples = norm.rvs(loc=e2_hosts.values[:, np.newaxis], scale=sigma_e2[:, np.newaxis], size=(len(e2_hosts), n_samples))

    #size_samples = np.maximum(0.25, norm.rvs(loc=angular_size_hosts.values[:, np.newaxis], scale=sigma_size[:, np.newaxis], size=(len(angular_size_hosts), n_samples)))
    size_samples = norm.rvs(angular_size_hosts.values[:, np.newaxis], sigma_size[:, np.newaxis], (len(angular_size_hosts), n_samples))

    # Calculate DLR for all galaxies and all samples in a vectorized way -- can't be negative!
    DLR_samples = calc_DLR_decals(ra_SN, dec_SN, ra_hosts, dec_hosts, e1_samples, e2_samples, size_samples)

    return DLR_samples

def calc_DLR_GLADE(ra_SN, dec_SN, ra_hosts, dec_hosts, a_over_b, pa, size):
    #definitions buried deep in the tractor documentation i.e. https://github.com/dstndstn/tractor/blob/main/tractor/ellipses.py

    # Calculate angular offsets for all galaxies
    xr = (ra_SN - ra_hosts.values[:, np.newaxis]) * 3600
    yr = (dec_SN - dec_hosts.values[:, np.newaxis]) * 3600

    # Position angle and angle calculations for all samples
    phi = np.radians(pa)
    gam = np.arctan2(xr, yr)

    theta = phi - gam

    # Vectorized calculation of DLR for all samples
    DLR = size / np.sqrt(((a_over_b) * np.sin(theta))**2 + (np.cos(theta))**2)
    return DLR

def calc_GLADE_DLR_with_err(ra_SN, dec_SN, ra_hosts, dec_hosts, pa_hosts, angular_size_hosts, a_over_b_hosts, sigma_size, sigma_a_over_b, n_samples=10000, search_rad=60):
    #LATER -- consider uncertainty in position angle
    pa_hosts = np.repeat(pa_hosts.values[:, np.newaxis], n_samples, axis=1)
    # Generate 2D samples for e1, e2, and angular_size for each galaxy
    size_samples = norm.rvs(angular_size_hosts[:, np.newaxis], sigma_size[:, np.newaxis], (len(angular_size_hosts), n_samples))
    a_over_b_samples = norm.rvs(a_over_b_hosts.values[:, np.newaxis], sigma_a_over_b.values[:, np.newaxis], (len(a_over_b_hosts), n_samples))

    # Calculate DLR for all galaxies and all samples in a vectorized way -- can't be negative!
    DLR_samples = calc_DLR_GLADE(ra_SN, dec_SN, ra_hosts, dec_hosts, a_over_b_samples, pa_hosts, size_samples)

    return DLR_samples

def prior_redshifts(z_gal_samples, reduce='mean'):
    pdf = priorfunc_z.pdf(z_gal_samples)
    if reduce == 'mean':
        return np.nanmean(pdf, axis=1)  # Resulting shape: (n_galaxies,)
    else:
        return pdf

def prior_offsets(fractional_offsets, reduce='mean'):
    pdf = priorfunc_offset.pdf(fractional_offsets)
    if reduce == 'mean':
        return np.nanmean(pdf, axis=1)  # Resulting shape: (n_galaxies,)
    else:
        return pdf

def prior_absolute_magnitude(M_abs_samples, reduce='mean'):
    pdf = priorfunc_absmag.pdf(M_abs_samples)
    if reduce == 'mean':
        return np.nanmean(pdf, axis=1)
    else:
        return pdf

def likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std, reduce='mean'):
    z_sn_samples = z_sn_samples[np.newaxis, :]  # Shape: (n_sn_samples, 1)
    z_gal_mean = z_gal_mean[:, np.newaxis]  # Shape: (1, n_galaxies)
    z_gal_std = z_gal_std[:, np.newaxis]    # Shape: (1, n_galaxies)

    # Calculate the likelihood of each SN redshift sample across each galaxy
    likelihoods = norm.pdf(z_sn_samples, loc=z_gal_mean, scale=z_gal_std)  # Shape: (n_sn_samples, n_galaxies)

    if reduce == 'mean':
        return np.nanmean(likelihoods, axis=1)  # Resulting shape: (n_galaxies,)
    else:
        return likelihoods


def likelihood_offsets(fractional_offsets):
    #set a DLR cutoff of 100 -- ridiculously high
    return np.nanmean(truncexpon.pdf(fractional_offsets, loc=0, scale=1, b=100), axis=1)


def likelihood_absolute_magnitude(M_abs_samples, reduce='mean'):
    #assuming a typical 0.1 SN/century/10^10 Lsol (in K-band)
    #TODO -- convert to K-band luminosity of the host!
    #https://www.aanda.org/articles/aa/pdf/2005/15/aa1411.pdf
    M_sol = 4.74
    Lgal = 10**(-0.4 * (M_abs_samples - M_sol)) # in units of Lsol
    Lgal /= 1.e10 #in units of 10^10 Lsol
    if reduce == 'mean':
        return  0.1*np.nanmean(Lgal, axis=1)
    else:
        return 0.1*Lgal

def probability_of_unobserved_host(z_sn, z_sn_std, z_sn_samples, cutout_rad=60, m_lim=30, verbose=False, n_samples=1000):
    n_gals = n_samples

    if np.isnan(z_sn):
        z_sn_std = np.nan
        z_sn_samples = np.maximum(0.001, priorfunc_z.rvs(size=n_samples))

        #draw galaxies from the same distribution
        z_gal_mean = np.maximum(0.001, priorfunc_z.rvs(size=n_samples))
        z_gal_std = 0.05*z_gal_mean
        z_gal_samples = np.maximum(0.001, norm.rvs(loc=z_gal_mean[:, np.newaxis],
                                                scale=z_gal_std[:, np.newaxis], size=(n_gals, n_samples)))
        Prior_z = prior_redshifts(z_gal_samples, reduce=None)
        L_z  = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std, reduce=None)

        sorted_indices = np.argsort(z_gal_samples, axis=1)
        sorted_z_gal_samples = np.take_along_axis(z_gal_samples, sorted_indices, axis=1)
        sorted_integrand = np.take_along_axis(Prior_z * L_z, sorted_indices, axis=1)

        # Perform integration using simps or trapz
        P_z = np.trapz(sorted_integrand, sorted_z_gal_samples, axis=1)
    else:
       # Use the known supernova redshift
        z_sn_std = 0.05 * z_sn
        z_gal_mean = z_sn  # Assume galaxy redshift is close to SN redshift
        z_gal_std = z_sn_std
        z_sn_samples = np.maximum(0.001, norm.rvs(z_sn, z_sn_std, size=n_samples))
        z_gal_samples = np.maximum(0.001, norm.rvs(loc=z_gal_mean, scale=z_gal_std, size=n_samples))

        Prior_z = prior_redshifts(z_gal_samples)
        L_z = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std)
        P_z = Prior_z * L_z

    galaxy_physical_radius_prior_means = halfnorm.rvs(size=n_samples, loc=0, scale=10)  # in kpc
    galaxy_physical_radius_prior_std = 0.05*galaxy_physical_radius_prior_means
    galaxy_physical_radius_prior_samples = norm.rvs(loc=galaxy_physical_radius_prior_means[:, np.newaxis],
                                     scale=galaxy_physical_radius_prior_std[:, np.newaxis],
                                     size=(n_gals, n_samples))

    sn_distance = cosmo.comoving_distance(np.nanmean(z_sn_samples)).value  # in Mpc

    min_phys_rad = 0
    max_phys_rad = (cutout_rad / 206265) * sn_distance * 1e3  # in kpc

    physical_offset_mean = np.linspace(min_phys_rad, max_phys_rad, n_gals)
    physical_offset_std = 0.05*physical_offset_mean
    physical_offset_samples = norm.rvs(physical_offset_mean[:, np.newaxis],
                                     physical_offset_std[:, np.newaxis],
                                     size=(n_gals, n_samples))

    fractional_offset_samples =  physical_offset_samples / galaxy_physical_radius_prior_samples  # Shape: (n_samples, n_samples)

    Prior_offset_unobs = prior_offsets(fractional_offset_samples)
    L_offset_unobs = likelihood_offsets(fractional_offset_samples)

    d_l_samples = cosmo.luminosity_distance(z_sn_samples).value * 1e6  # Convert to pc
    absmag_lim_samples = m_lim - 5 * (np.log10(d_l_samples / 10))

    absmag_samples = np.linspace(absmag_lim_samples, 0, n_gals)

    Prior_absmag = prior_absolute_magnitude(absmag_samples)
    L_absmag = likelihood_absolute_magnitude(absmag_samples)

    print("prior_absmag:")
    print(np.nansum(Prior_absmag < 0))
    print("like absmag")
    print(np.nansum(L_absmag < 0))
    print("prior offset")
    print(np.nansum(Prior_offset_unobs < 0))
    print("like offset")
    print(np.nansum(L_offset_unobs < 0))

    prob_unobs = Prior_absmag * L_absmag * P_z * Prior_offset_unobs * L_offset_unobs

    P_unobserved = np.nanmean(prob_unobs)  # Sum over all galaxies

    if verbose:
        print(f"Unobserved host probability: {P_unobserved}")

    return P_unobserved


def probability_host_outside_cone(z_sn, cutout_rad=60, verbose=False, n_samples=1000):
    n_gals = n_samples

    if np.isnan(z_sn):
        z_sn_samples = np.maximum(0.001, priorfunc_z.rvs(size=n_samples))

        # draw galaxies from the same distribution
        z_gal_mean = np.maximum(0.001, priorfunc_z.rvs(size=n_samples))
        z_gal_std = 0.05 * z_gal_mean
        z_gal_samples = np.maximum(0.001, norm.rvs(loc=z_gal_mean[:, np.newaxis],
                                                   scale=z_gal_std[:, np.newaxis], size=(n_gals, n_samples)))
        Prior_z = prior_redshifts(z_gal_samples, reduce=None)
        L_z = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std, reduce=None)

        sorted_indices = np.argsort(z_gal_samples, axis=1)
        sorted_z_gal_samples = np.take_along_axis(z_gal_samples, sorted_indices, axis=1)
        sorted_integrand = np.take_along_axis(Prior_z * L_z, sorted_indices, axis=1)

        # Perform integration using simps or trapz
        P_z = np.trapz(sorted_integrand, sorted_z_gal_samples, axis=1)
    else:
        # Use the known supernova redshift
        z_sn_std = 0.05 * z_sn
        z_sn_samples = np.maximum(0.001, norm.rvs(z_sn, z_sn_std, size=n_samples))
        z_gal_samples = np.maximum(0.001, norm.rvs(loc=z_sn, scale=z_sn_std, size=(n_gals, n_samples)))

        Prior_z = prior_redshifts(z_gal_samples)
        L_z = likelihood_redshifts(z_sn_samples, z_sn, z_sn_std)
        P_z = Prior_z * L_z

    # sample brightnesses
    absmag_mean = priorfunc_absmag.rvs(size=n_gals)
    absmag_std = 0.05 * np.abs(absmag_mean)
    absmag_samples = np.maximum(0.001, norm.rvs(loc=absmag_mean[:, np.newaxis],
                                                scale=absmag_std[:, np.newaxis], size=(n_gals, n_samples)))

    Prior_absmag = prior_absolute_magnitude(absmag_samples)

    # Calculate the distance to the supernova for each sampled redshift
    sn_distances = cosmo.comoving_distance(np.mean(z_gal_samples, axis=1)).value  # in Mpc
    # Convert angular cutout radius to physical offset at each sampled redshift
    min_phys_rad = (cutout_rad / 206265) * sn_distances * 1e3  # in kpc
    max_phys_rad = 5 * min_phys_rad
    fractional_offset_samples = np.linspace(min_phys_rad[:, np.newaxis], max_phys_rad[:, np.newaxis], n_samples)
    fractional_offset_samples = fractional_offset_samples / np.maximum(0.001, halfnorm.rvs(size=n_samples))[:, np.newaxis]

    Prior_offset = prior_offsets(fractional_offset_samples)
    L_offset = likelihood_offsets(fractional_offset_samples)
    L_absmag = likelihood_absolute_magnitude(absmag_samples)

    prob_outside = Prior_absmag * L_absmag * P_z * Prior_offset * L_offset

    # Sum over the samples and galaxies to get the total probability
    P_outside = np.nanmean(prob_outside, axis=1)  # Average over the samples for each galaxy
    P_outside = np.nanmean(P_outside)  # average over all simulated galaxies

    return P_outside

def posterior_samples(z_sn, sn_position, galaxy_catalog, cutout_rad=60, n_samples=1000, m_lim=24.5, verbose=False):
    # Extract arrays for all galaxies from the catalog
    #z_gal_samples = np.array([gal['photo_z_samples'] for gal in galaxy_catalog])  # Shape (N, M)
    z_gal_mean = np.array([gal['z_best_mean'] for gal in galaxy_catalog])
    z_gal_std = np.array([gal['z_best_std'] for gal in galaxy_catalog])  # Shape (N, M)
    galaxy_ras = np.array([gal['ra'] for gal in galaxy_catalog])
    galaxy_decs = np.array([gal['dec'] for gal in galaxy_catalog])
    galaxy_DLR_samples = np.array([gal['DLR_samples'] for gal in galaxy_catalog])
    absmag_samples = np.array([gal['absmag_samples'] for gal in galaxy_catalog])

    # Calculate angular separation between SN and all galaxies (in arcseconds)
    sn_ra, sn_dec = sn_position.ra.degree, sn_position.dec.degree
    offset_arcsec = SkyCoord(galaxy_ras*u.deg, galaxy_decs*u.deg).separation(sn_position).arcsec

    #just copied now assuming 0 positional uncertainty -- this can be updated later (TODO!)
    offset_arcsec_samples = np.repeat(offset_arcsec[:, np.newaxis], n_samples, axis=1)

    #samples from the redshift prior for sources without reported redshifts
    z_gal_samples = norm.rvs(z_gal_mean[:, np.newaxis], z_gal_std[:, np.newaxis], size=(len(z_gal_mean), n_samples))

    # Sample z values
    if np.isnan(z_sn):
        print("WARNING: No SN redshift; marginalizing to infer probability for the likelihood of observed hosts.")
        # Marginalize over the sampled supernova redshifts by integrating the likelihood over the redshift prior
        z_sn_std = np.nan
        z_sn_samples = np.maximum(0.001, priorfunc_z.rvs(size=n_samples))
        Prior_z = prior_redshifts(z_gal_samples, reduce=None)
        L_z = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std, reduce=None)

        sorted_indices = np.argsort(z_gal_samples, axis=1)
        sorted_z_gal_samples = np.take_along_axis(z_gal_samples, sorted_indices, axis=1)
        sorted_integrand = np.take_along_axis(Prior_z * L_z, sorted_indices, axis=1)

        # Perform integration using simps or trapz
        P_z = np.trapz(sorted_integrand, sorted_z_gal_samples, axis=1)
    else:
        z_sn_std = 0.05*z_sn
        z_sn_samples = np.maximum(0.001, norm.rvs(z_sn, z_sn_std, size=n_samples))  # Ensure non-negative redshifts
        L_z = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std)  # Shape (N,)
        Prior_z = prior_redshifts(z_gal_samples)
        P_z = Prior_z*L_z

    #depends on zgal, NOT zSN
    Prior_absmag = prior_absolute_magnitude(absmag_samples)
    L_absmag = likelihood_absolute_magnitude(absmag_samples)

    # Calculate angular diameter distances for all samples
    galaxy_distances = cosmo.angular_diameter_distance(z_gal_samples).to(u.kpc).value  # Shape (N, M)

    fractional_offset_samples = offset_arcsec_samples/galaxy_DLR_samples

    Prior_offsets = prior_offsets(fractional_offset_samples)
    L_offsets = likelihood_offsets(fractional_offset_samples)  # Shape (N,)

    # Compute the posterior probabilities for all galaxies
    P_gals = (P_z) * (Prior_offsets*L_offsets) * (Prior_absmag*L_absmag)

    #other probabilities
    P_hostless = 0. #some very low value that the SN is actually hostless.
    P_outside = probability_host_outside_cone(z_sn, cutout_rad=cutout_rad, verbose=False, n_samples=1000)
    P_unobs = probability_of_unobserved_host(z_sn, z_sn_std, z_sn_samples, cutout_rad=cutout_rad, m_lim=m_lim, verbose=verbose)

    if False:
        print("Unnormalized posteriors of true galaxy:")
        print("AbsMag prior:")
        print(np.nanmax(Prior_absmag))
        print("Redshift posterior:")
        print(np.nanmax(P_z))
        print("Offset prior:")
        print(np.nanmax(Prior_offsets))
        print("AbsMag likelihood:")
        print(np.nanmax(L_absmag))
        print("Offset likelihood:")
        print(np.nanmax(L_offsets))
        print("Full posterior for best galaxy:", np.nanmax(post_probs))
        print("Probability of being outside the region:", P_outside)
        print("Probability of being unobserved:", P_unobs)
        print("\n\n")

    P_tot = np.nansum(P_gals) + P_hostless + P_outside + P_unobs

    P_none_norm = (P_outside + P_hostless + P_unobs) / P_tot
    P_any_norm =  np.nansum(P_gals) / P_tot

    P_gals_norm = P_gals / P_tot
    P_offsets_norm = Prior_offsets * L_offsets/ P_tot
    P_z_norm = P_z / P_tot
    P_absmag_norm = Prior_absmag*L_absmag / P_tot
    P_outside_norm = P_outside / P_tot
    P_unobs_norm = P_unobs / P_tot

    print(f"Probability your true host was outside the search cone: {P_outside_norm:.4e}")
    print(f"Probability your true host is not in this catalog: {P_unobs_norm:.4e}")
    print(f"Probability your true host is hostless: {P_hostless:.4e}")
    print(f"Probability your true host IS in this catalog: {P_any_norm:.4e}")

    return P_any_norm, P_none_norm, P_gals_norm, P_offsets_norm, P_z_norm, P_absmag_norm

# Convert physical sizes to angular sizes (arcseconds)
def physical_to_angular_size(physical_size, redshift):
    return (physical_size/(cosmo.angular_diameter_distance(redshift)).to(u.kpc).value)*206265

#sn_catalog = pd.read_csv("/Users/alexgagliano/Documents/Conferences/FreedomTrail_Jan24/sn_catalog/ZTFBTS_TransientTable.csv")
#sn_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/slsne_hosts.txt", delim_whitespace=True)
#sn_catalog_names = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/slsn_sne.txt", delim_whitespace=True)
#sn_catalog_names.rename(columns={'Name':'object_name'}, inplace=True)
#sn_catalog = sn_catalog.merge(sn_catalog_names)
#sn_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/ZTFBTS_TransientTable.csv")
#sn_catalog = pd.read_csv("/Users/alexgagliano/Documents/Research/prob_association/localsn_public_cuts.txt", delim_whitespace=True)
#sn_catalog = sn_catalog[1:]

source = "DELIGHT"
#source = "Jones+18"

with open('/Users/alexgagliano/Documents/Research/prob_association/all.pkl', 'rb') as f:
    data = pickle.load(f)
data.reset_index(inplace=True)
sn_catalog = data

if source == 'FLEET':
    sn_catalog['host_ra']
elif source == 'ZTF':
    for col in ['host_ra', 'host_dec', 'host_Pcc']:
        sn_catalog[col] = np.nan
    sn_coords = SkyCoord(sn_catalog['RA'].values, sn_catalog['Dec'].values, unit=(u.hourangle, u.deg))
    sn_catalog['RA_deg'] = sn_coords.ra.deg
    sn_catalog['DEC_deg']= sn_coords.dec.deg
    sn_catalog.rename(columns={'redshift':'Redshift', 'IAUID':'object_name'}, inplace=True)
    sn_catalog['Redshift'] = sn_catalog['Redshift'].replace('-',np.nan)
    sn_catalog['Redshift'] = sn_catalog['Redshift'].astype("float")
elif source == 'Jones+18':
    sn_catalog['host_Pcc'] = 1.0 #assume all are chosen
    sn_coords = SkyCoord(sn_catalog['RA'].values, sn_catalog['DEC'].values, unit=(u.hourangle, u.deg))
    sn_catalog['RA_deg'] = sn_coords.ra.deg
    sn_catalog['DEC_deg']= sn_coords.dec.deg
    host_coords = SkyCoord(sn_catalog['hostRA'].values, sn_catalog['hostDec'].values, unit=(u.hourangle, u.deg))
    sn_catalog['host_ra']  = host_coords.ra.deg
    sn_catalog['host_dec'] = host_coords.dec.deg
    sn_catalog.rename(columns={'z':'Redshift', 'ID':'object_name'}, inplace=True)
    sn_catalog['Redshift'] = sn_catalog['Redshift'].replace('-',np.nan)
    sn_catalog['Redshift'] = sn_catalog['Redshift'].astype("float")
elif source == 'DELIGHT':
    sn_catalog['Redshift'] = np.nan # :(
    sn_catalog['host_Pcc'] = 1.0
    sn_catalog.rename(columns={'meanra':'RA_deg', 'meandec':'DEC_deg', 'oid':'object_name'}, inplace=True)

sn_catalog['prob_host_ra'] = np.nan
sn_catalog['prob_host_dec'] = np.nan
sn_catalog['prob_host_score'] = np.nan
sn_catalog['prob_host_2_ra'] = np.nan
sn_catalog['prob_host_2_dec'] = np.nan
sn_catalog['prob_host_2_score'] = np.nan
sn_catalog['prob_host_flag'] = 0
sn_catalog['sn_ra_deg'] = np.nan
sn_catalog['sn_dec_deg'] = np.nan
sn_catalog['prob_association_time'] = np.nan

sn_catalog['agreement_w_Pcc'] = np.nan

#randomly shuffle
Nassociated = 0
verbose = True
Ntot = 5

sn_catalog = sn_catalog.sample(frac=1)

#debugging with specific objects
sn_catalog = sn_catalog[sn_catalog['object_name'] == 'ZTF20achplfl']

angular_scale = 0.258 #arcsec/pixel

for idx, row in sn_catalog.iterrows():
    decals_match = False
    glade_match = False

    start_time = time.time()
    sn_name = row.object_name#row.IAUID
    sn_position = SkyCoord(Angle(row.RA_deg, unit=u.deg), Angle(row.DEC_deg, unit=u.deg)) #SkyCoord(Angle(row.RA, unit=u.hourangle), Angle(row.Dec, unit=u.deg))
    base_radius = 200/3600
    sn_redshift = row.Redshift
    if not np.isnan(row.Redshift):
        sn_redshift = float(sn_redshift) #0.003
        # Convert 100 kpc to angular offset in arcseconds at the supernova redshift
        rad = np.nanmin([100 / cosmo.angular_diameter_distance(sn_redshift).to(u.kpc).value * 206265/3600, base_radius])
    else:
        # Fallback search radius (200 arcseconds in degrees)
        rad = base_radius

    rad_arcsec = rad*3600
    stamp_size = int(rad_arcsec/angular_scale)

    print(f"\n\nSN Redshift: {sn_redshift:.4f}")
    print(f"SN Name: {sn_name}")

    sn_catalog.at[idx, 'sn_ra_deg'] = sn_position.ra.deg
    sn_catalog.at[idx, 'sn_dec_deg'] = sn_position.dec.deg

    #glade logic
    n_samples = 1000
    GLADE_catalog = pd.read_csv("/Users/alexgagliano/Documents/Research/prob_association/GLADE+_HyperLedaSizes.csv", delim_whitespace=True,na_values=['', np.nan],keep_default_na=True)
    GLADE_catalog.dropna(subset=['logd25Hyp', 'logr25Hyp', 'e_logr25Hyp', 'PAHyp'], inplace=True)
    galaxies = build_glade_candidates(sn_position, rad_arcsec, GLADE_catalog, n_samples=n_samples)

    if galaxies is None:
        print("Nothing found in GLADE.")
        glade_match = False
    else:
        any_prob, none_prob, post_probs, post_offset, post_z, post_absmag = posterior_samples(sn_redshift, sn_position, galaxies, m_lim=16, cutout_rad=rad_arcsec,verbose=False)

        if (any_prob > none_prob):
            print("Found match by GLADE!")
            glade_match = True
        else:
            print(f"P(no host observed in GLADE cone): {none_prob:.4e}, P(host observed):{any_prob:.4e}")
            glade_match = False
    if not glade_match:
        print("Moving on to decals...")
        galaxies = build_decals_candidates(sn_position, rad_arcsec, n_samples=n_samples)

        if galaxies is None:
            continue
        #setting the limiting magnitude to something exceptionally low
        any_prob, none_prob, post_probs, post_offset, post_z, post_absmag = posterior_samples(sn_redshift, sn_position, galaxies, m_lim=27, cutout_rad=rad_arcsec,verbose=verbose)
        if any_prob > none_prob:
            decals_match = True
    best_gal = np.argsort(post_probs)[::-1][0]
    top_prob = post_probs[best_gal]

    if verbose:
        if glade_match:
            candidate_ids = np.arange(len(galaxies))
        else:
            candidate_ids = galaxies['ls_id']
        diagnose_ranking(-1, post_probs, galaxies, post_offset, post_z, post_absmag, candidate_ids, sn_redshift, sn_position, verbose=True)

    if any_prob > none_prob:
        print("Probability of observing the host is higher than missing it.")
        print(f"P(host observed):{any_prob:.4e}")
        if not glade_match:
            sn_catalog.at[idx, 'prob_host_ls_id'] = galaxies['ls_id'][best_gal]
            print(f"DECALS ID of matched host:{galaxies['ls_id'][best_gal]}")
        sn_catalog.at[idx, 'prob_host_z_best_mean'] = galaxies['z_best_mean'][best_gal]
        sn_catalog.at[idx, 'prob_host_z_best_std'] = galaxies['z_best_std'][best_gal]
        sn_catalog.at[idx, 'prob_host_ra'] = galaxies['ra'][best_gal]
        sn_catalog.at[idx, 'prob_host_dec'] = galaxies['dec'][best_gal]
        sn_catalog.at[idx, 'prob_host_score'] = post_probs[best_gal]
        sn_catalog.at[idx, 'prob_host_xr'] = (sn_position.ra.deg - galaxies['ra'][best_gal])*3600
        sn_catalog.at[idx, 'prob_host_yr'] = (sn_position.dec.deg - galaxies['dec'][best_gal])*3600

        second_best_gal = None
        if len(post_probs) > 1:
            second_best_gal = np.argsort(post_probs)[::-1][1]

            second_prob = post_probs[second_best_gal]
            bayes_factor = top_prob / second_prob

            if bayes_factor < 3:
                print(f"Warning: The evidence for the top candidate over the second-best is weak. Bayes Factor = {bayes_factor:.2f}. Check for a shred!")
                sn_catalog.at[idx, 'prob_host_flag'] = 1
            elif bayes_factor > 100:
                print(f"The top candidate has very strong evidence over the second-best. Bayes Factor = {bayes_factor:.2f}")
                sn_catalog.at[idx, 'prob_host_flag'] = 2

            if not glade_match:
                sn_catalog.at[idx, 'prob_host_2_ls_id'] = galaxies['ls_id'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_z_best_mean'] = galaxies['z_best_mean'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_z_best_std'] = galaxies['z_best_std'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_ra'] = galaxies['ra'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_dec'] = galaxies['dec'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_score'] = post_probs[second_best_gal]

        #compare to probability of chance coincidence
        sep_Pcc = SkyCoord(row['host_ra']*u.deg, row['host_dec']*u.deg).separation(SkyCoord(galaxies['ra'][best_gal]*u.deg, galaxies['dec'][best_gal]*u.deg)).arcsec
        print(f"Separation from Pcc host: {sep_Pcc:.2f}\".")
        if sep_Pcc < 2: #give a little leeway; these are big galaxies now!
            sn_catalog.at[idx, 'agreement_w_Pcc'] = True
        else:
            sn_catalog.at[idx, 'agreement_w_Pcc'] = False
    else:
        print("WARNING: unlikely that any of these galaxies hosted this SN! Setting best galaxy as second-best.")
        print(f"P(no host observed in DECALS cone): {none_prob:.4e}, P(host observed):{any_prob:.4e}")
        sn_catalog.at[idx, 'prob_host_flag'] = 5
        sn_catalog.at[idx, 'prob_host_ls_id'] = np.nan
        sn_catalog.at[idx, 'prob_host_z_best_mean'] = np.nan
        sn_catalog.at[idx, 'prob_host_z_best_std'] = np.nan
        sn_catalog.at[idx, 'prob_host_ra'] = np.nan
        sn_catalog.at[idx, 'prob_host_dec'] = np.nan
        sn_catalog.at[idx, 'prob_host_score'] = np.nan
        sn_catalog.at[idx, 'prob_host_2_z_best_mean'] = galaxies['z_best_mean'][best_gal]
        sn_catalog.at[idx, 'prob_host_2_z_best_std'] = galaxies['z_best_std'][best_gal]
        sn_catalog.at[idx, 'prob_host_2_ra'] = galaxies['ra'][best_gal]
        sn_catalog.at[idx, 'prob_host_2_dec'] = galaxies['dec'][best_gal]
        sn_catalog.at[idx, 'prob_host_2_score'] = post_probs[best_gal]
        #best_gal = None

        #comparison to Pcc
        if (row['host_Pcc'] > 0.1):
            print("Oh no! Pcc gets a host here.")
            sn_catalog.at[idx, 'agreement_w_Pcc'] = False
        else:
            print("AGREEMENT: No host found by Pcc.")
            sn_catalog.at[idx, 'agreement_w_Pcc'] = True

    end_time = time.time()
    match_time = end_time - start_time
    print(f"Completed in {match_time:.2f} seconds.")
    Nassociated +=1

    #######################
    #######################
    #get PS1 postage stamp
    #######################
    #######################
    #im = getcolorim(sn_position.ra.deg, sn_position.dec.deg, size=stamp_size, filters="grizy", format="png")
    #plt.imshow(im)
    #sn_pos = (stamp_size/2, stamp_size/2)

    if row.host_Pcc > 0.1:
        Pcc_host_ra = float(row.host_ra)
        Pcc_host_dec = float(row.host_dec)
    else:
        Pcc_host_ra = Pcc_host_dec = None
    if (glade_match or decals_match):
        chosen_gal_ra = [galaxies['ra'][best_gal]]
        chosen_gal_dec = [galaxies['dec'][best_gal]]
        if (second_best_gal is not None) and (second_best_gal < len(galaxies)):
            chosen_gal_ra.append(galaxies['ra'][second_best_gal])
            chosen_gal_dec.append(galaxies['dec'][second_best_gal])
    else:
        chosen_gal_ra = chosen_gal_dec = []

    #try:
    if True:
        plotSNhost(chosen_gal_ra, chosen_gal_dec, Pcc_host_ra, Pcc_host_dec,
            galaxies['z_best_mean'][best_gal], galaxies['z_best_std'][best_gal],
            sn_position.ra.deg, sn_position.dec.deg, row.object_name, row.Redshift, sn_catalog.at[idx, 'prob_host_flag'], f"./plots_likelihoodoffsetscale1_wGLADE/DELIGHT/{row.object_name}")
    #except:
    if False:
        print("Failed to plot match. Trying again after 60s...")
        time.sleep(60)
        plotSNhost(chosen_gal_ra, chosen_gal_dec, Pcc_host_ra, Pcc_host_dec,
        galaxies['z_best_mean'][best_gal], galaxies['z_best_std'][best_gal],
        sn_position.ra.deg, sn_position.dec.deg, row.object_name, row.Redshift, sn_catalog.at[idx, 'prob_host_flag'],
        f"./plots_likelihoodoffsetscale1_wGLADE/DELIGHT/{row.object_name}")

    stats_done = sn_catalog['agreement_w_Pcc'].dropna()
    agree_frac = np.nansum(stats_done)/len(stats_done)
    print(f"Current agreement fraction: {agree_frac:.2f}")

    sn_catalog.at[idx, 'prob_association_time'] = match_time

#sn_catalog.to_csv("/Users/alexgagliano/Documents/Research/prob_association/slsn_catalog_Alexprob_PccCompare.csv",index=False)
#sn_catalog.to_csv("/Users/alexgagliano/Desktop/prob_association/ZTFBTS_Alexprob.csv",index=False)
sn_catalog.to_csv("/Users/alexgagliano/Desktop/prob_association/DELIGHT_Alexprob.csv",index=False)

#sky_sep_measured = SkyCoord(sn_catalog['sn_ra_deg'].values*u.deg, sn_catalog['sn_dec_deg'].values*u.deg).separation(SkyCoord(sn_catalog['prob_host_ra'].values*u.deg, sn_catalog['prob_host_dec'].values*u.deg)).arcsec
#ky_sep_jones = SkyCoord(sn_catalog['sn_ra_deg'].values*u.deg, sn_catalog['sn_dec_deg'].values*u.deg).separation(SkyCoord(sn_catalog['host_ra'].values*u.deg, sn_catalog['host_dec'].values*u.deg)).arcsec

#plt.plot(sky_sep_measured, sky_sep_jones, 'o', mec='k', c='tab:blue', zorder=500)
#plt.plot([0, 50], [0, 50], c='k', ls='--')
#plt.xscale("log")
#plt.yscale("log")
#plt.xlabel("Measured Offset (\")")
#plt.ylabel("Jones+18 Offset (\")")

#match_std = SkyCoord(sn_catalog['host_ra'].values*u.deg, sn_catalog['host_dec'].values*u.deg).separation(SkyCoord(sn_catalog['prob_host_ra'].values*u.deg, sn_catalog['prob_host_dec'].values*u.deg)).arcsec
#match_std = match_std[match_std == match_std]

# histogram on linear scale
#hist, bins, _ = plt.hist(match_std, bins=np.linspace(0, 2));

#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
#plt.hist(match_std, bins=logbins)
#plt.xscale('log')
#plt.yscale("log")
#plt.xlabel("Error (\")")

#plt.hist(galaxies[576]['photo_z_samples'])
#likelihood_absolute_magnitude([galaxies[576]['absmag_samples']])
#plt.hist(galaxies[576]['absmag_samples'], bins=100)
#galaxies['angular_size_arcsec'][311]
#galaxies['angular_size_arcsec_std'][311]

#galaxies['ra'][311]
#galaxies['dec'][311]
#galaxies['ls_id'][311]
#len(galaxies)
#plt.hist(galaxies['absmag_samples'][311])

#galaxies[2]['ra']
#galaxies[2]['dec']
#GLADE_catalog.iloc[[742309]]
#np.argmin(SkyCoord(GLADE_catalog['RAJ2000'].values*u.deg, GLADE_catalog['DEJ2000'].values*u.deg).separation(SkyCoord(galaxies[2]['ra']*u.deg, galaxies[2]['dec']*u.deg)))
#plt.ylabel("")

#9906627633152972
#galaxies['ra'][832]
#galaxies['dec'][832]

#plt.hist(galaxies['absmag_samples'][832])

#likelihood_absolute_magnitude([galaxies['absmag_samples'][832]])

#z_sn_std = np.nan
#z_sn_samples = np.maximum(0.001, priorfunc_z.rvs(size=n_samples))
#Prior_z = prior_redshifts([galaxies['z_best_samples'][832]])
#L_z_all = likelihood_redshifts(z_sn_samples, [galaxies['z_best_mean'][832]], [galaxies['z_best_std'][832]])
#P_z = np.sum(Prior_z[:, np.newaxis] * L_z_all, axis=0) / np.sum(Prior_z)  # Marginalization

#plt.show()
