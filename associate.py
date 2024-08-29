import numpy as np
#from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
#change cosmology to be consistent with other catalogs
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

from astro_ghost.PS1QueryFunctions import get_PS1_Pic, find_all
from astropy.io import fits
from astropy.visualization import SqrtStretch
from astropy.visualization import ZScaleInterval
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
import pickle
os.chdir("/Users/alexgagliano/Documents/Research/prob_association/")
from diagnose import *

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
        t.flux_r,
        t.flux_ivar_r,
        t.nobs_g,
        t.nobs_r,
        t.nobs_z,
        t.fitbits,
        t.ra_ivar,
        t.dec_ivar,
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
    AND (t.nobs_r > 0) AND ((pz.z_spec > 0) OR (pz.z_phot_mean > 0))""")

    candidate_hosts = ascii.read(result).to_pandas()

    n_galaxies = len(candidate_hosts)

    if n_galaxies < 1:
        print(f"No sources found around {sn_name} in DECaLS! Double-check that the SN coords overlap the survey footprint.")
        return None

    dtype = [('ra', float), ('dec', float), ('redshift', float), ('physical_size_kpc', float), ('angular_size_arcsec', float),
            ('angular_size_arcsec_err', float),('photo_z_dist', object),('z_phot_median', float),('z_phot_mean', float),('z_phot_std', float),('ls_id', int),
            ('DLR_samples', object), ('absmag_mean', float), ('absmag_std', float)]

    galaxies = np.zeros(len(candidate_hosts), dtype=dtype)
    galaxies['ra'] = candidate_hosts['ra'].values
    galaxies['dec'] = candidate_hosts['dec'].values

    temp_sizes = candidate_hosts['shape_r'].values
    temp_sizes[temp_sizes < 0.25] = 0.25 #1 pixel, at least for PS1
    temp_sizes_err = candidate_hosts['shape_r_ivar'].values
    #temp_sizes_err[temp_sizes_err < 0.05*temp_sizes] = 0.05*temp_sizes[temp_sizes_err < 0.05*temp_sizes]
    temp_sizes_err  = np.minimum(np.sqrt(1/temp_sizes_err), temp_sizes)

    temp_sizes_err[temp_sizes_err != temp_sizes_err] = 0.05*temp_sizes[temp_sizes_err != temp_sizes_err]
    temp_sizes_err[temp_sizes_err < 0.05*temp_sizes] = 0.05*temp_sizes[temp_sizes_err < 0.05*temp_sizes]
    galaxies['angular_size_arcsec'] = temp_sizes
    galaxies['angular_size_arcsec_err'] = temp_sizes_err

    galaxy_photoz_median = candidate_hosts['z_phot_median'].values
    galaxy_photoz_mean = candidate_hosts['z_phot_mean'].values
    galaxy_photoz_std = candidate_hosts['z_phot_std'].values
    galaxy_specz = candidate_hosts['z_spec'].values

    temp_e1 = candidate_hosts['shape_e1'].astype("float")
    temp_e1_err = candidate_hosts['shape_e1_ivar'].values
    temp_e1_err[temp_e1_err < 0.05*temp_e1] = 0.05*temp_e1[temp_e1_err < 0.05*temp_e1]
    temp_e1_err[temp_e1_err > 1.e-10]  = np.sqrt(1/temp_e1_err[temp_e1_err > 1.e-10])

    temp_e2 = candidate_hosts['shape_e2'].astype("float")
    temp_e2_err = candidate_hosts['shape_e2_ivar'].values
    temp_e2_err[temp_e2_err < 0.05*temp_e2] = 0.05*temp_e2[temp_e2_err < 0.05*temp_e2]
    temp_e2_err[temp_e1_err > 1.e-10]  = np.sqrt(1/temp_e2_err[temp_e2_err > 1.e-10])

    #is_largegal = np.array([(len(f'{x:0b}') > 8) and (int(f'{x:0b}'[8]) == 1) for x in candidate_hosts['fitbits'].values])
    #galaxies['fitbits_weight'] = is_largegal+1 #weighting is 1 for regular sources and 2 for largegals

    DLR_samples = calc_decals_DLR_with_err(sn_position.ra.deg, sn_position.dec.deg, candidate_hosts['ra'],
        candidate_hosts['dec'], candidate_hosts['shape_e1'], candidate_hosts['shape_e2'],
        candidate_hosts['shape_r'], temp_e1_err, temp_e2_err, temp_sizes_err, search_rad=search_rad)

    galaxies['z_phot_median'] = galaxy_photoz_median
    galaxies['z_phot_mean'] = galaxy_photoz_mean
    galaxies['z_phot_std'] = galaxy_photoz_std
    galaxies['z_phot_std'][galaxy_specz > 0] = 0.05*galaxy_specz[galaxy_specz > 0] #ffloor
    galaxies['z_phot_median'][galaxy_specz > 0] = galaxy_specz[galaxy_specz > 0]
    galaxies['z_phot_mean'][galaxy_specz > 0] = galaxy_specz[galaxy_specz > 0]
    galaxies['ls_id'] = candidate_hosts['ls_id'].values

    photo_z_samples = np.random.normal(
        loc=galaxy_photoz_mean[:, np.newaxis],  # Shape (N, 1) to allow broadcasting
        scale=galaxy_photoz_std[:, np.newaxis],  # Shape (N, 1)
        size=(n_galaxies, n_samples)  # Shape (N, M)
    )
    photo_z_samples[photo_z_samples < 0] = 0.001 #set photometric redshift floor
    photo_z_samples[photo_z_samples != photo_z_samples] = 0.001 #set photometric redshift floor
    for i in range(n_galaxies):
        galaxies['photo_z_dist'][i] = photo_z_samples[i, :]
        galaxies['DLR_samples'][i] = DLR_samples[i, :]

    temp_mag_r = candidate_hosts['dered_mag_r'].values
    temp_mag_r_err = np.abs(2.5/np.log(10)*np.sqrt(1/candidate_hosts['flux_ivar_r'].values)/candidate_hosts['flux_r'].values)

    absmag_samples = norm.rvs(loc=temp_mag_r[:, np.newaxis], scale=temp_mag_r_err[:, np.newaxis], size=(len(temp_mag_r), n_samples)) - cosmo.distmod(photo_z_samples).value

    galaxies['absmag_mean'] = np.nanmean(absmag_samples)
    galaxies['absmag_std'] = np.nanstd(absmag_samples)

    return galaxies

def build_glade_candidates(sn_position, search_rad, GLADE_catalog, n_samples):

    candidate_hosts = GLADE_catalog[SkyCoord(GLADE_catalog['RAJ2000'].values*u.deg, GLADE_catalog['DEJ2000'].values*u.deg).separation(sn_position).arcsec < search_rad]
    n_galaxies = len(candidate_hosts)

    if n_galaxies < 1:
        print(f"No sources found around {sn_name} in GLADE!")
        return None

    dtype = [('ra', float), ('dec', float), ('redshift', float), ('angular_size_arcsec', float),
            ('angular_size_arcsec_err', float),('photo_z_dist', object),('z_phot_median', float),('z_phot_mean', float),('z_phot_std', float),
            ('DLR_samples', object), ('absmag_mean', float), ('absmag_std', float)]

    galaxies = np.zeros(len(candidate_hosts), dtype=dtype)
    galaxies['ra'] = candidate_hosts['RAJ2000'].values
    galaxies['dec'] = candidate_hosts['DEJ2000'].values

    temp_sizes = 0.5*3*10**(candidate_hosts['logd25Hyp'].values) # (n) HyperLEDA decimal logarithm of the length of the projected major axis of a galaxy at the isophotal level 25mag/arcsec2 in the B-band, to semi-major half-axis (half-light radius) in arcsec
    temp_sizes[temp_sizes < 0.25] = 0.25 #1 pixel, at least for PS1
    temp_sizes_err = np.minimum(temp_sizes, np.abs(temp_sizes) * np.log(10) * candidate_hosts['e_logd25Hyp'].values)

    temp_sizes_err[temp_sizes_err != temp_sizes_err] = 0.05*temp_sizes[temp_sizes_err != temp_sizes_err]
    temp_sizes_err[temp_sizes_err < 0.05*temp_sizes] = 0.05*temp_sizes[temp_sizes_err < 0.05*temp_sizes]

    galaxies['angular_size_arcsec'] = temp_sizes
    galaxies['angular_size_arcsec_err'] = temp_sizes_err

    #20% uncertainty floor on distances (may be a terrible assumption)
    dist_samples = norm.rvs(loc=candidate_hosts['Dist'].values[:, np.newaxis],
                            scale=0.2*candidate_hosts['Dist'].values[:, np.newaxis], #assuming, conservatively, a high distance uncertainty
                            size=(len(candidate_hosts), n_samples))

    photo_z_samples = z_at_value(cosmo.luminosity_distance, dist_samples*u.Mpc)

    galaxy_specz =  z_at_value(cosmo.luminosity_distance,  candidate_hosts['Dist'].values*u.Mpc)
    galaxy_a_over_b = 10**(candidate_hosts['logr25Hyp'])
    galaxy_a_over_b_err = galaxy_a_over_b * np.log(10) * candidate_hosts['e_logr25Hyp'].values

    DLR_samples = calc_GLADE_DLR_with_err(sn_position.ra.deg, sn_position.dec.deg, candidate_hosts['RAJ2000'],
        candidate_hosts['DEJ2000'], candidate_hosts['PAHyp'],
        temp_sizes, galaxy_a_over_b, temp_sizes_err, galaxy_a_over_b_err, n_samples=n_samples, search_rad=search_rad)

    galaxies['z_phot_median'] = np.nanmedian(photo_z_samples, axis=1)
    galaxies['z_phot_mean'] = np.nanmean(photo_z_samples, axis=1)
    galaxies['z_phot_std'] = np.nanstd(photo_z_samples, axis=1)
    #galaxies['z_phot_std'][galaxy_specz > 0] = 0.05*galaxy_specz[galaxy_specz > 0] #5% uncertainty floor
    galaxies['z_phot_median'][galaxy_specz > 0] = galaxy_specz[galaxy_specz > 0]
    galaxies['z_phot_mean'][galaxy_specz > 0] = galaxy_specz[galaxy_specz > 0]

    photo_z_samples[photo_z_samples < 0] = 0.001 #set photometric redshift floor
    photo_z_samples[photo_z_samples != photo_z_samples] = 0.001 #set photometric redshift floor
    for i in range(n_galaxies):
        galaxies['photo_z_dist'][i] = photo_z_samples[i, :]
        galaxies['DLR_samples'][i] = DLR_samples[i, :]

    temp_mag_r = candidate_hosts['BmagHyp'].values
    temp_mag_r_err = np.abs(candidate_hosts['e_BmagHyp'].values)

    absmag_samples = norm.rvs(loc=temp_mag_r[:, np.newaxis], scale=temp_mag_r_err[:, np.newaxis], size=(len(temp_mag_r), n_samples)) - cosmo.distmod(photo_z_samples).value

    galaxies['absmag_mean'] = np.nanmean(absmag_samples)
    galaxies['absmag_std'] = np.nanstd(absmag_samples)

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
    size_samples = np.random.normal(angular_size_hosts.values[:, np.newaxis], sigma_size[:, np.newaxis], (len(angular_size_hosts), n_samples))

    # Calculate DLR for all galaxies and all samples in a vectorized way -- can't be negative!
    DLR_samples = calc_DLR_decals(ra_SN, dec_SN, ra_hosts, dec_hosts, e1_samples, e2_samples, size_samples)

    # Estimate the mean and standard deviation of the DLR for each galaxy
    #mean_DLR = np.mean(DLR_samples, axis=1)
    #std_DLR = np.std(DLR_samples, axis=1)

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
    size_samples = np.random.normal(angular_size_hosts[:, np.newaxis], sigma_size[:, np.newaxis], (len(angular_size_hosts), n_samples))
    a_over_b_samples = np.random.normal(a_over_b_hosts.values[:, np.newaxis], sigma_a_over_b.values[:, np.newaxis], (len(a_over_b_hosts), n_samples))

    # Calculate DLR for all galaxies and all samples in a vectorized way -- can't be negative!
    DLR_samples = calc_DLR_GLADE(ra_SN, dec_SN, ra_hosts, dec_hosts, a_over_b_samples, pa_hosts, size_samples)

    # Estimate the mean and standard deviation of the DLR for each galaxy
    #mean_DLR = np.mean(DLR_samples, axis=1)
    #std_DLR = np.std(DLR_samples, axis=1)
    return DLR_samples

def prior_redshifts(z_gal_samples):
    return np.nanmean(uniform(loc=0.0001, scale=1.0).pdf(z_gal_samples), axis=1) #WIDE

def prior_offsets(fractional_offsets):
    return np.nanmean(uniform(loc=0, scale=5).pdf(fractional_offsets), axis=1) #WIDE

def likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std):
    z_sn_samples = z_sn_samples[:, np.newaxis]  # Shape: (n_sn_samples, 1)
    z_gal_mean = z_gal_mean[np.newaxis, :]  # Shape: (1, n_galaxies)
    z_gal_std = z_gal_std[np.newaxis, :]    # Shape: (1, n_galaxies)

    # Calculate the likelihood of each SN redshift sample across each galaxy
    likelihoods = norm.pdf(z_sn_samples, loc=z_gal_mean, scale=z_gal_std)  # Shape: (n_sn_samples, n_galaxies)

    return np.nanmean(likelihoods, axis=0)  # Resulting shape: (n_galaxies,)

def integrate_likelihood_over_redshift(z_gal_mean, z_gal_std, n_samples=1000):
    # Define the integrand function
    def integrand(z, z_gal_mean, z_gal_std):
        z_samples = norm.rvs(loc=z, scale=0.05*z, size=n_samples)
        return likelihood_redshifts(z_samples, z_gal_mean, z_gal_std)

    # Perform the integration over the redshift prior (uniform from 0 to 1)
    L_z_integrated = np.array([quad(integrand, 0.001, 1, args=(z_gal_mean, z_gal_std))])

    return L_z_integrated

def prior_absolute_magnitude(M_abs_samples):
    # Define a prior over the absolute magnitude range (e.g., uniform or based on a luminosity function)
    return uniform.pdf(M_abs_samples, loc=-25, scale=15)  # Example: Uniform prior

def likelihood_absolute_magnitude(M_abs_samples, z_samples, cutout_rad=60):
    h = 1  # Flat cosmology
    phi_star = 5.37e-5 * h**3  # h^3 Mpc^-3 mag^-1 yr^-1
    M_star = -20.60 - 5 * np.log10(h / 1.0)
    alpha = -0.62
    theta = cutout_rad / 206265  # Convert arcseconds to radians

    L_samples = 10**(-0.4 * (M_abs_samples - M_star))
    schechter_values = 0.4 * np.log(10) * phi_star * L_samples**(alpha + 1) * np.exp(-L_samples)

    dV_dz_samples = (theta**2 / 4) * cosmo.differential_comoving_volume(z_samples).value

    return schechter_values * dV_dz_samples  # This is your likelihood function

def likelihood_offsets(fractional_offsets):
    #set a DLR cutof to 4
    return np.nanmean(truncexpon.pdf(fractional_offsets, loc=0, scale=1, b=4), axis=1)

def schechter_function(M, phi_star, M_star, alpha):
    L = 10**(-0.4 * (M - M_star))
    return (0.4 * np.log(10) * phi_star * L**(alpha+1) * np.exp(-L))


# include delta-mag prior & likelihood --  host is probably not more than 2mag dimmer than the supernoa at peak
# add in brightnesses of the host galaxies
# write out first host as second-most-likely host if presumed hostless
# write out probability of hostless vs first host vs second host
# add in SDSS next instead of 3Pi? P_unobserved as a threshold for moving between catalogs
# see if shreds have low photo-zs with high errors

def probability_of_unobserved_host(z_sn, z_sn_err, z_sn_samples, cutout_rad=60, m_lim=30, verbose=False, n_samples=1000):
    #first, draw a set of galaxies from the prior
    z_gal_mean = np.random.uniform(0.0001, 1.0, n_samples)  # Broad prior from z ~ 0.0001 to z ~ 1.0
    z_gal_std = 0.05*z_gal_mean

    z_gal_samples = np.random.normal(loc=z_gal_mean[:, np.newaxis],
                                     scale=z_gal_std[:, np.newaxis],
                                     size=(n_samples, n_samples))

    Prior_z_unobs = prior_redshifts(z_gal_samples)

    if np.isnan(z_sn):
        print("WARNING: No SN redshift; marginalizing to infer probability for the likelihood of observed hosts.")
        # Marginalize over the sampled supernova redshifts by integrating the likelihood over the redshift prior
        # Marginalize over the prior*likelihood
        z_sn_samples = np.maximum(0.001, uniform.rvs(loc=0, scale=1, size=n_samples))
        L_z_all = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std)
        L_z_unobs = np.sum(Prior_z_unobs[:, np.newaxis] * L_z_all, axis=0) / np.sum(Prior_z_unobs)  # Marginalization
    else:
        z_sn_samples = np.random.normal(loc=z_sn, scale=z_sn_err, size=n_samples)

        # Calculate likelihood without marginalization if `z_sn` is known
        L_z_unobs = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std)

    h =1 #flat cosmology
    phi_star = 5.37e-5*h**3  # h^3 Mpc^-3 mag^-1 yr^-1
    M_star = -20.60 - 5 * np.log10(h / 1.0)
    alpha = -0.62
    theta = cutout_rad / 206265  # Convert arcseconds to radians

    fractional_offset_mean = uniform.rvs(loc=0, scale=5, size=n_samples)
    fractional_offset_std = 0.05 * fractional_offset_mean * (1 + z_gal_mean) #scale with increasing redshift
    fractional_offset_samples = np.random.normal(loc=fractional_offset_mean[:, np.newaxis],
                                     scale=fractional_offset_std[:, np.newaxis],
                                     size=(n_samples, n_samples))

    P_offset_unobs = prior_offsets(fractional_offset_samples)
    L_offset_unobs = likelihood_offsets(fractional_offset_samples)

    # Sample over a massive range of absolute magnitudes
    absmag_samples = np.linspace(-100, 0, n_samples)

    # Likelihood and prior for all galaxies (denominator)
    L_absmag_tot = np.nanmean(likelihood_absolute_magnitude(absmag_samples[:, np.newaxis], z_gal_samples), axis=1)
    P_absmag_tot = prior_absolute_magnitude(absmag_samples)

    d_l_samples = cosmo.luminosity_distance(z_sn).value * 1e6  # Convert to pc
    absmag_lim = m_lim - 5 * (np.log10(d_l_samples / 10))

    # We are interested in galaxies with M_abs > M_lim, so integrate over this range
    sumL_absmag_tot = np.trapz(P_absmag_tot[:, np.newaxis] * L_absmag_tot, absmag_samples, axis=0)
    sumL_absmag_dim = np.trapz(P_absmag_tot[:, np.newaxis] * L_absmag_tot * (absmag_samples[:, np.newaxis] > absmag_lim), absmag_samples, axis=0)

    # Combine with other probabilities
    prob_unobs = sumL_absmag_dim * (Prior_z_unobs * L_z_unobs) * (P_offset_unobs * L_offset_unobs)

    # Sum to get the total probability
    P_unobserved = np.nansum(prob_unobs)/np.nansum(sumL_absmag_tot)
    if verbose:
        print(f"Unobserved host probability: {P_unobserved}")

    return P_unobserved

def probability_host_outside_cone(z_sn, cutout_rad=60, verbose=False, n_samples=1000):
    # Handle NaN case by drawing from a uniform(0, 1) prior
    #TODO -- marginalize, don't sample
    if np.isnan(z_sn):
        z_samples = np.random.uniform(0, 1, n_samples)
    else:
        z_err = 0.05 * z_sn  # Assume nominal uncertainty
        z_samples = np.maximum(0.001, np.random.normal(z_sn, z_err, n_samples))  # Ensure non-negative redshifts

    # Calculate the distance to the supernova for each sampled redshift
    sn_distances = cosmo.comoving_distance(z_samples).value  # in Mpc

    # Convert angular cutout radius to physical offset at each sampled redshift
    physical_offset_cutouts = (cutout_rad / 206265) * sn_distances * 1e3  # in kpc
    if verbose:
        print(f"Physical offset of cutout, in kpc (mean):{np.mean(physical_offset_cutouts):.3f}")

    # Sample a range of galaxy half-light radii
    galaxy_physical_radius_prior_samples = halfnorm.rvs(size=n_samples, loc=0, scale=10)  # in kpc
    angular_offets = physical_offset_cutouts / galaxy_physical_radius_prior_samples

    # Calculate the probability that the true host is outside the cone for each sample
    P_outside = halfnorm.cdf(10, scale=1) - np.nanmean(halfnorm.cdf(angular_offets, scale=1))

    return P_outside

def posterior_samples(z_sn, sn_position, galaxy_catalog, cutout_rad=60, n_samples=1000, m_lim=24.5, verbose=False):
    # Extract arrays for all galaxies from the catalog
    #z_gal_samples = np.array([gal['photo_z_dist'] for gal in galaxy_catalog])  # Shape (N, M)
    z_gal_mean = np.array([gal['z_phot_mean'] for gal in galaxy_catalog])
    z_gal_std = np.array([gal['z_phot_std'] for gal in galaxy_catalog])  # Shape (N, M)
    galaxy_ras = np.array([gal['ra'] for gal in galaxy_catalog])
    galaxy_decs = np.array([gal['dec'] for gal in galaxy_catalog])
    galaxy_DLR_samples = np.array([gal['DLR_samples'] for gal in galaxy_catalog])
    #galaxy_DLR = np.array([gal['DLR_mean'] for gal in galaxy_catalog])
    #galaxy_DLR_err = np.array([gal['DLR_std'] for gal in galaxy_catalog])

    # Calculate physical sizes of galaxies in kpc
    #galaxy_physical_radius_samples = (galaxy_angular_sizes[:, np.newaxis] / 206265) * galaxy_distances  # Shape (N, M)
    #don't anymore -- just use the observed offsets
    # Calculate angular separation between SN and all galaxies (in arcseconds)
    sn_ra, sn_dec = sn_position.ra.degree, sn_position.dec.degree
    offset_arcsec = SkyCoord(galaxy_ras*u.deg, galaxy_decs*u.deg).separation(sn_position).arcsec
    #just copied now assuming 0 positional uncertainty -- this can be updated later (TODO!)
    offset_arcsec_samples = np.repeat(offset_arcsec[:, np.newaxis], n_samples, axis=1)

    #samples from the redshift prior for sources without reported redshifts
    no_z_gal_idx = np.where(np.isnan(z_gal_mean))[0]
    z_gal_samples = np.random.normal(z_gal_mean[:, np.newaxis], z_gal_std[:, np.newaxis], size=(len(z_gal_mean), n_samples))

    #TODO: marginalize over missing host galaxy redshifts as well...?
    # Sample z values
    Prior_z = prior_redshifts(z_gal_samples)

    if not np.isnan(z_sn):
        z_sn_err = 0.05*z_sn
        z_sn_samples = np.maximum(0.001, norm.rvs(z_sn, z_sn_err, size=n_samples))  # Ensure non-negative redshifts
        L_z = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std)  # Shape (N,)

    else:
        print("WARNING: No SN redshift; marginalizing to infer probability for the likelihood of observed hosts.")
        # Marginalize over the sampled supernova redshifts by integrating the likelihood over the redshift prior
        # Marginalize over the prior*likelihood
        z_sn_err = np.nan
        z_sn_samples = np.maximum(0.001, uniform.rvs(loc=0, scale=1, size=n_samples))
        L_z_all = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std)
        L_z = np.sum(Prior_z[:, np.newaxis] * L_z_all, axis=0) / np.sum(Prior_z)  # Marginalization

    #galaxy_DLR_samples = truncnorm.rvs(loc=galaxy_DLR[:, np.newaxis], scale=galaxy_DLR_err[:, np.newaxis], a=-0.05, b=0.05, size=(len(galaxy_DLR), n_samples))
    # Calculate angular diameter distances for all samples
    galaxy_distances = cosmo.angular_diameter_distance(z_gal_samples).to(u.kpc).value  # Shape (N, M)
    sn_distance = cosmo.angular_diameter_distance(z_sn).to(u.kpc).value  # Shape (N, M)

    fractional_offset_samples = offset_arcsec_samples/galaxy_DLR_samples

    Prior_offsets = prior_offsets(fractional_offset_samples)
    L_offsets = likelihood_offsets(fractional_offset_samples)  # Shape (N,)

    # Compute the posterior probabilities for all galaxies
    P_gals = (Prior_z*L_z) * (Prior_offsets*L_offsets)

    #other probabilities
    P_hostless = 1.e-5 #some very low value that the SN is actually hostless.
    P_outside = probability_host_outside_cone(z_sn, cutout_rad=cutout_rad, verbose=False, n_samples=1000)
    P_unobs = probability_of_unobserved_host(z_sn, z_sn_err, z_sn_samples, cutout_rad=cutout_rad, m_lim=m_lim, verbose=verbose)

    P_tot = np.sum(P_gals) + P_hostless + P_outside + P_unobs

    #if verbose:
    #    print("Unnormalized posteriors:")
    #    print("AbsMag prior:")
    #    print(P_absmag)
    #    print("Redshift prior:")
    #    print(P_z)
    #    print("Offset prior:")
    #    print(P_offsets)
    #    print("AbsMag likelihood:")
    #    print(L_absmag)
    #    print("Redshift likelihood:")
    #    print(L_z)
    #    print("Offset likelihood:")
    #    print(L_offsets)
    #    print("Full posterior for best galaxy:", np.nanmax(post_probs))
    #    print("Probability of being outside the region:", P_outside_region)
    #    print("Probability of being unobserved:", P_unobs)
    #    print("\n\n")

    P_none_norm = (P_outside + P_hostless + P_unobs) / P_tot
    P_any_norm =  np.sum(P_gals) / P_tot

    P_gals_norm = P_gals / P_tot
    P_offsets_norm = Prior_offsets * L_offsets/ P_tot
    P_z_norm =  Prior_z*L_z / P_tot
    P_outside_norm = P_outside / P_tot
    P_unobs_norm = P_unobs / P_tot

    print(f"Probability your true host was outside the search cone: {P_outside_norm:.4e}")
    print(f"Probability your true host is not in this catalog: {P_unobs_norm:.4e}")

    return P_any_norm, P_none_norm, P_gals_norm, P_offsets_norm, P_z_norm

# Convert physical sizes to angular sizes (arcseconds)
def physical_to_angular_size(physical_size, redshift):
    # Convert kpc to arcsec using cosmology
    return (physical_size/(cosmo.angular_diameter_distance(redshift)).to(u.kpc).value)*206265

#sn_catalog = pd.read_csv("/Users/alexgagliano/Documents/Conferences/FreedomTrail_Jan24/sn_catalog/ZTFBTS_TransientTable.csv")
#sn_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/slsne_hosts.txt", delim_whitespace=True)
#sn_catalog_names = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/slsn_sne.txt", delim_whitespace=True)
#sn_catalog_names.rename(columns={'Name':'object_name'}, inplace=True)
#sn_catalog = sn_catalog.merge(sn_catalog_names)
#sn_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/ZTFBTS_TransientTable.csv")
sn_catalog = pd.read_csv("/Users/alexgagliano/Documents/Research/prob_association/localsn_public_cuts.txt", delim_whitespace=True)
sn_catalog = sn_catalog[1:]

#sn_catalog = sn_catalog[sn_catalog['ID'].isin([
#'ASASSN-15mf'])]
# also a failure but
#'ASASSN-15mf',

#understandably ambiguous
#'2000dk',
#'2001en',
#'420100',
#'PS16bnz',

# wrong in DJones, right with us
#'1997E','2002dp'

#sn_catalog = sn_catalog[sn_catalog['ID'].isin(['21502'])]

#source = "DELIGHT"
source = "Jones+18"

#with open('/Users/alexgagliano/Documents/Research/prob_association/all.pkl', 'rb') as f:
#    data = pickle.load(f)
#data.reset_index(inplace=True)
#sn_catalog = data
#sn_catalog = sn_catalog[sn_catalog['ID'] == '2007ux']

if source == 'FLEET':
    sn_catalog['host_ra']
elif source == 'ZTF':
    #or ZTF BTS?
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

#sn_catalog = sn_catalog[sn_catalog['object_name'] == '2022ued']
#sn_catalog[['object_name', 'host_ra','host_dec']]
#sn_catalog['object_name', 'ra_deg', 'dec_deg']

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

#sn_catalog = sn_catalog[sn_catalog['object_name'] == '2020jii']

#1.5% wrong, first pass. Second pass associating GLADE first and fixing marginalizing over dim galaxies (fewer? we'll see)

#for getting images!
from astro_ghost.PS1QueryFunctions import getcolorim

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
    GLADE_catalog.dropna(subset=['logd25Hyp', 'logr25Hyp', 'e_logr25Hyp'], inplace=True)
    galaxies = build_glade_candidates(sn_position, rad_arcsec, GLADE_catalog, n_samples=n_samples)

    if galaxies is None:
        glade_match = False
    else:
        any_prob, none_prob, post_probs, post_offset, post_z = posterior_samples(sn_redshift, sn_position, galaxies, m_lim=16, cutout_rad=rad_arcsec,verbose=False)

        if (any_prob > none_prob) & (none_prob < 0.01):
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
        any_prob, none_prob, post_probs, post_offset, post_z = posterior_samples(sn_redshift, sn_position, galaxies, m_lim=27, cutout_rad=rad_arcsec,verbose=verbose)
        if any_prob > none_prob:
            decals_match = True
    best_gal = np.argsort(post_probs)[::-1][0]
    top_prob = post_probs[best_gal]

    if verbose:
        if glade_match:
            candidate_ids = np.arange(len(galaxies))
        else:
            candidate_ids = galaxies['ls_id']
        diagnose_ranking(-1, post_probs, galaxies, post_offset, post_z, candidate_ids, sn_redshift, sn_position, verbose=True)


    if any_prob > none_prob:
        print("Probability of observing the host is higher than missing it.")
        print(f"P(host observed):{any_prob:.4e}")
        if not glade_match:
            sn_catalog.at[idx, 'prob_host_ls_id'] = galaxies['ls_id'][best_gal]
            print(f"DECALS ID of matched host:{galaxies['ls_id'][best_gal]}")
        sn_catalog.at[idx, 'prob_host_z_phot_median'] = galaxies['z_phot_median'][best_gal]
        sn_catalog.at[idx, 'prob_host_z_phot_std'] = galaxies['z_phot_std'][best_gal]
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
                print(f"Warning: The evidence for the top candidate over the second-best is weak. Bayes Factor = {bayes_factor:.2f}")
                sn_catalog.at[idx, 'prob_host_flag'] = 1
            elif bayes_factor > 100:
                #print(f"The top candidate has very strong evidence over the second-best. Bayes Factor = {bayes_factor:.2f}")
                sn_catalog.at[idx, 'prob_host_flag'] = 2

            if not glade_match:
                sn_catalog.at[idx, 'prob_host_2_ls_id'] = galaxies['ls_id'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_z_phot_median'] = galaxies['z_phot_median'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_z_phot_std'] = galaxies['z_phot_std'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_ra'] = galaxies['ra'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_dec'] = galaxies['dec'][second_best_gal]
            sn_catalog.at[idx, 'prob_host_2_score'] = post_probs[second_best_gal]

        #compare to probability of chance coincidence
        sep_Pcc = SkyCoord(row['host_ra']*u.deg, row['host_dec']*u.deg).separation(SkyCoord(galaxies['ra'][best_gal]*u.deg, galaxies['dec'][best_gal]*u.deg)).arcsec
        print(f"Separation from Pcc host: {sep_Pcc:.2f}\".")
        if sep_Pcc < 1:
            sn_catalog.at[idx, 'agreement_w_Pcc'] = True
        else:
            sn_catalog.at[idx, 'agreement_w_Pcc'] = False
    else:
        print("WARNING: unlikely that any of these galaxies hosted this SN!")
        second_best_gal = best_gal = None
        print(f"P(no host observed in DECALS cone): {none_prob:.4e}, P(host observed):{any_prob:.4e}")
        sn_catalog.at[idx, 'prob_host_flag'] = 5
        sn_catalog.at[idx, 'prob_host_ls_id'] = np.nan
        sn_catalog.at[idx, 'prob_host_z_phot_median'] = np.nan
        sn_catalog.at[idx, 'prob_host_z_phot_std'] = np.nan
        sn_catalog.at[idx, 'prob_host_ra'] = np.nan
        sn_catalog.at[idx, 'prob_host_dec'] = np.nan
        sn_catalog.at[idx, 'prob_host_score'] = np.nan

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

    plotSNhost(chosen_gal_ra, chosen_gal_dec, Pcc_host_ra, Pcc_host_dec,
        galaxies['z_phot_mean'][best_gal], galaxies['z_phot_std'][best_gal],
        sn_position.ra.deg, sn_position.dec.deg, row.object_name, row.Redshift, f"./plots_likelihoodoffsetscale1_wGLADE/Jones+18/Disagree/{row.object_name}")

    stats_done = sn_catalog['agreement_w_Pcc'].dropna()
    agree_frac = np.nansum(stats_done)/len(stats_done)
    print(f"Current agreement fraction: {agree_frac:.2f}")

    sn_catalog.at[idx, 'prob_association_time'] = match_time

sn_catalog.to_csv("/Users/alexgagliano/Documents/Research/prob_association/slsn_catalog_Alexprob_PccCompare.csv",index=False)
#sn_catalog.to_csv("/Users/alexgagliano/Desktop/prob_association/ZTFBTS_Alexprob.csv",index=False)
#sn_catalog.to_csv("/Users/alexgagliano/Desktop/prob_association/DELIGHT_Alexprob.csv",index=False)




#sky_sep_measured = SkyCoord(sn_catalog['sn_ra_deg'].values*u.deg, sn_catalog['sn_dec_deg'].values*u.deg).separation(SkyCoord(sn_catalog['prob_host_ra'].values*u.deg, sn_catalog['prob_host_dec'].values*u.deg)).arcsec
#sky_sep_jones = SkyCoord(sn_catalog['sn_ra_deg'].values*u.deg, sn_catalog['sn_dec_deg'].values*u.deg).separation(SkyCoord(sn_catalog['host_ra'].values*u.deg, sn_catalog['host_dec'].values*u.deg)).arcsec

#import seaborn as sns
#sns.set_context("talk")

#plt.plot(sky_sep_measured, sky_sep_jones, 'o', mec='k', zorder=500)
#plt.plot([0, 50], [0, 50], c='k', ls='--')
#plt.xscale("log")
#plt.yscale("log")
#plt.xlabel("Measured Offset (\")")
#plt.ylabel("Jones+18 Offset (\")")

#match_err = SkyCoord(sn_catalog['host_ra'].values*u.deg, sn_catalog['host_dec'].values*u.deg).separation(SkyCoord(sn_catalog['prob_host_ra'].values*u.deg, sn_catalog['prob_host_dec'].values*u.deg)).arcsec

# histogram on linear scale
#hist, bins, _ = plt.hist(match_err, bins=30);

#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
#plt.hist(match_err, bins=logbins)
#plt.xscale('log')
#plt.yscale("log")
#plt.xlabel("Error (\")")
#plt.ylabel("")
#plt.show()

#sn_catalog.loc[sn_catalog['object_name'] == '1997E', 'prob_host_ra']
#sn_catalog.loc[sn_catalog['object_name'] == '1997E', 'prob_host_dec']
#sn_catalog['object_name'][(match_err > 6)].values

#plt.hist(galaxies['DLR_samples'])
#plt.hist(truncnorm.rvs(loc=5, scale=3, size=1000, a=-2, b=2))
#sn_catalog
#sn_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/Jones+18_Alexprob.csv")

#sn_catalog.columns.values
