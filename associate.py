import numpy as np
from scipy.stats import norm, halfnorm
#from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
#change cosmology to be consistent with other catalogs
import astropy.cosmology.units as cu
from astropy.coordinates import SkyCoord
import astropy.units as u
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

os.chdir("/Users/alexgagliano/Desktop/prob_association/plots_likelihoodoffsetscale1_wGLADE/")


def plotSNhost(host_ra, host_dec, Pcc_host_ra, Pcc_host_dec, host_z_mean, host_z_std, SN_ra, SN_dec, SN_name, SN_z, fn):
    cols = np.array(['#ff9f1c', '#2cda9d', '#f15946', '#da80dd', '#f4e76e', '#b87d4b', '#ff928b', '#c73e1d', '#58b09c', '#e7e08b'])
    bands = 'zrg'
    if host_ra:
        sep = np.nanmax(SkyCoord(host_ra*u.deg, host_dec*u.deg).separation(SkyCoord(SN_ra*u.deg, SN_dec*u.deg)).arcsec)
    else:
        sep = 0
    if Pcc_host_ra:
        sep_Pcc = SkyCoord(Pcc_host_ra*u.deg, Pcc_host_dec*u.deg).separation(SkyCoord(SN_ra*u.deg, SN_dec*u.deg)).arcsec
        if (Pcc_host_ra) and (Pcc_host_dec) and (sep_Pcc > sep):
            sep = sep_Pcc
    rad = np.nanmax([60., 1.5*sep]) #arcsec to pixels, scaled by 1.5x host-SN separation
    print(f"Getting img with size len {rad:.2f}")
    pic_data = []
    for band in bands:
        get_PS1_Pic('./', None, SN_ra, SN_dec, int(rad*4), band)
        a = find_all('PS1_ra={}_dec={}_{}arcsec_{}.fits'.format(SN_ra, SN_dec, int(rad),band), ".")
        pixels = fits.open(a[0])[0].data
        pixels = pixels.astype('float32')
        # normalize to the range 0-255
        pixels *= 255/np.nanmax(pixels)
        #plt.hist(pixels)
        pic_data.append(pixels)
        hdu = fits.open(a[0])[0]
        os.remove(a[0])

    stretch = SqrtStretch() + ZScaleInterval()

    lo_val, up_val = np.percentile(np.array(pic_data).ravel(), (0.5, 99.5))  # Get the value of lower and upper 0.5% of all pixels

    stretch_val = up_val - lo_val

    # stretch of 10, stretch of x, stretch of 2
    rgb_default = make_lupton_rgb(pic_data[0], pic_data[1], pic_data[2], minimum=lo_val, stretch=stretch_val, Q=0)
    wcs = WCS(hdu.header)
    plt.figure(num=None, figsize=(12, 8), facecolor='w', edgecolor='k')
    ax = plt.subplot(projection=wcs)
    ax.set_xlabel("RA", fontsize=24)
    ax.set_ylabel("DEC", fontsize=24)

    #really emphasize the supernova location
    plt.axvline(x=int(rad*2), c='tab:red', alpha=0.5, lw=2)
    plt.axhline(y=int(rad*2), c='tab:red', alpha=0.5, lw=2)

    if (Pcc_host_ra and Pcc_host_dec):
        Pcc_str = ""
        ax.scatter(Pcc_host_ra, Pcc_host_dec, transform=ax.get_transform('fk5'), marker='+', lw=2, s=200,
            color='magenta', zorder=100)
    else:
        Pcc_str = "(no Pcc)"
    if (host_ra and host_dec):
        for i in np.arange(len(host_ra)):
            ax.scatter(host_ra[i], host_dec[i], transform=ax.get_transform('fk5'), marker='o', lw=2, s=100,
                edgecolor='k', facecolor=cols[i], zorder=100)
        plt.title(f"{SN_name}, z={SN_z:.4f}; Host Match, z={host_z_mean:.4f}+/-{host_z_std:.4f} {Pcc_str}")
    else:
        plt.title(f"{SN_name}, z={SN_z:.4f}; No host found {Pcc_str}")
    ax.imshow(rgb_default, origin='lower')
    plt.axis('off')
    plt.savefig("./%s.png"%fn, bbox_inches='tight')


def build_decals_candidates(sn_position, rad, n_samples=1000):
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
        q3c_radial_query(t.ra, t.dec, {sn_position.ra.deg:.5f}, {sn_position.dec.deg:.5f}, {rad})
    AND (t.nobs_r > 0) AND ((pz.z_spec > 0) OR (pz.z_phot_mean > 0))""")

    candidate_hosts = ascii.read(result).to_pandas()

    n_galaxies = len(candidate_hosts)

    if n_galaxies < 1:
        print(f"No sources found around {sn_name} in DECaLS! Double-check that the SN coords overlap the survey footprint.")
        return None

    dtype = [('ra', float), ('dec', float), ('redshift', float), ('physical_size_kpc', float), ('angular_size_arcsec', float),
            ('angular_size_arcsec_err', float),('photo_z_dist', object),('z_phot_median', float),('z_phot_mean', float),('z_phot_std', float),('ls_id', int),
            ('DLR_mean', float), ('DLR_std', float), ('absmag_mean', float), ('absmag_std', float)]

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

    galaxies['DLR_mean'], galaxies['DLR_std'] = calc_decals_DLR_with_err(sn_position.ra.deg, sn_position.dec.deg, candidate_hosts['ra'],
        candidate_hosts['dec'], candidate_hosts['shape_e1'], candidate_hosts['shape_e2'],
        candidate_hosts['shape_r'], temp_e1_err, temp_e2_err, temp_sizes_err)

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

    return galaxies

def build_glade_candidates(sn_position, rad, GLADE_catalog, n_samples):

    candidate_hosts = GLADE_catalog[SkyCoord(GLADE_catalog['RAJ2000'].values*u.deg, GLADE_catalog['DEJ2000'].values*u.deg).separation(sn_position).deg < rad]
    n_galaxies = len(candidate_hosts)

    if n_galaxies < 1:
        print(f"No sources found around {sn_name} in GLADE!")
        return None

    dtype = [('ra', float), ('dec', float), ('redshift', float), ('angular_size_arcsec', float),
            ('angular_size_arcsec_err', float),('photo_z_dist', object),('z_phot_median', float),('z_phot_mean', float),('z_phot_std', float),
            ('DLR_mean', float), ('DLR_std', float)]

    galaxies = np.zeros(len(candidate_hosts), dtype=dtype)
    galaxies['ra'] = candidate_hosts['RAJ2000'].values
    galaxies['dec'] = candidate_hosts['DEJ2000'].values

    temp_sizes = 3*10**(candidate_hosts['logd25Hyp'].values) # (n) HyperLEDA decimal logarithm of the length of the projected major axis of a galaxy at the isophotal level 25mag/arcsec2 in the B-band, to semi-major axis in arcsec
    temp_sizes[temp_sizes < 0.25] = 0.25 #1 pixel, at least for PS1
    temp_sizes_err = temp_sizes * np.log(10) * candidate_hosts['e_logd25Hyp'].values

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

    galaxies['DLR_mean'], galaxies['DLR_std'] = calc_GLADE_DLR_with_err(sn_position.ra.deg, sn_position.dec.deg, candidate_hosts['RAJ2000'],
        candidate_hosts['DEJ2000'], candidate_hosts['PAHyp'],
        temp_sizes, galaxy_a_over_b, temp_sizes_err, galaxy_a_over_b_err)

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

def calc_decals_DLR_with_err(ra_SN, dec_SN, ra_hosts, dec_hosts, e1_hosts, e2_hosts, angular_size_hosts, sigma_e1, sigma_e2, sigma_size, n_samples=10000):
    # Generate 2D samples for e1, e2, and angular_size for each galaxy
    e1_samples = np.random.normal(e1_hosts.values[:, np.newaxis], sigma_e1[:, np.newaxis], (len(e1_hosts), n_samples))
    e2_samples = np.random.normal(e2_hosts.values[:, np.newaxis], sigma_e2[:, np.newaxis], (len(e2_hosts), n_samples))
    size_samples = np.maximum(0.25, np.random.normal(angular_size_hosts.values[:, np.newaxis], sigma_size[:, np.newaxis], (len(angular_size_hosts), n_samples)))

    # Calculate DLR for all galaxies and all samples in a vectorized way
    DLR_samples = calc_DLR_decals(ra_SN, dec_SN, ra_hosts, dec_hosts, e1_samples, e2_samples, size_samples)

    # Estimate the mean and standard deviation of the DLR for each galaxy
    mean_DLR = np.mean(DLR_samples, axis=1)
    std_DLR = np.std(DLR_samples, axis=1)

    return mean_DLR, std_DLR

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

def calc_GLADE_DLR_with_err(ra_SN, dec_SN, ra_hosts, dec_hosts, pa_hosts, angular_size_hosts, a_over_b_hosts, sigma_size, sigma_a_over_b, n_samples=10000):
    #LATER -- consider uncertainty in position angle
    pa_hosts = np.repeat(pa_hosts.values[:, np.newaxis], n_samples, axis=1)
    # Generate 2D samples for e1, e2, and angular_size for each galaxy
    size_samples = np.random.normal(angular_size_hosts[:, np.newaxis], sigma_size[:, np.newaxis], (len(angular_size_hosts), n_samples))
    a_over_b_samples = np.random.normal(a_over_b_hosts.values[:, np.newaxis], sigma_a_over_b.values[:, np.newaxis], (len(a_over_b_hosts), n_samples))

    # Calculate DLR for all galaxies and all samples in a vectorized way
    DLR_samples = calc_DLR_GLADE(ra_SN, dec_SN, ra_hosts, dec_hosts, a_over_b_samples, pa_hosts, size_samples)

    # Estimate the mean and standard deviation of the DLR for each galaxy
    mean_DLR = np.mean(DLR_samples, axis=1)
    std_DLR = np.std(DLR_samples, axis=1)

    return mean_DLR, std_DLR

def prior_redshifts(z_gal_samples):
    return np.nanmean(uniform(loc=0.0001, scale=1.0).pdf(z_gal_samples), axis=1) #WIDE

def prior_offsets(fractional_offsets):
    return np.nanmean(uniform(loc=0, scale=5).pdf(fractional_offsets), axis=1) #WIDE

def likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std):
    # Ensure z_sn_samples is a column vector (n_sn_samples, 1)
    z_sn_samples = z_sn_samples[:, np.newaxis]  # Shape: (n_sn_samples, 1)

    # Reshape z_gal_mean and z_gal_std to enable broadcasting with z_sn_samples
    z_gal_mean = z_gal_mean[np.newaxis, :]  # Shape: (1, n_galaxies)
    z_gal_std = z_gal_std[np.newaxis, :]    # Shape: (1, n_galaxies)

    # Calculate the likelihood of each SN redshift sample across each galaxy
    likelihoods = norm.pdf(z_sn_samples, loc=z_gal_mean, scale=z_gal_std)  # Shape: (n_sn_samples, n_galaxies)

    return np.nanmean(likelihoods, axis=0)  # Resulting shape: (n_galaxies,)

def integrate_likelihood_over_redshift(z_gal_mean, z_gal_std):
    # Define the integrand function
    def integrand(z, z_gal_mean, z_gal_std):
        return likelihood_redshifts(z, z_gal_mean, z_gal_std)

    # Perform the integration over the redshift prior (uniform from 0 to 1)
    L_z_integrated = np.array([quad(integrand, 0, 1, args=(z_mean, z_std))[0] for z_mean, z_std in zip(z_gal_mean, z_gal_std)])

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
    return np.nanmean(expon.pdf(fractional_offsets, loc=0, scale=1), axis=1)

def schechter_function(M, phi_star, M_star, alpha):
    L = 10**(-0.4 * (M - M_star))
    return (0.4 * np.log(10) * phi_star * L**(alpha+1) * np.exp(-L))

def probability_of_unobserved_host(z_sn, z_sn_err, z_sn_samples, cutout_rad=60, m_lim=30, verbose=False, n_samples=1000):
    #first, draw a set of galaxies from the prior
    z_gal_mean = np.random.uniform(0.0001, 1.0, n_samples)  # Broad prior from z ~ 0.0001 to z ~ 1.0
    z_gal_std = 0.05*z_gal_mean

    z_gal_samples = np.random.normal(loc=z_gal_mean[:, np.newaxis],
                                     scale=z_gal_std[:, np.newaxis],
                                     size=(n_samples, n_samples))

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

    z_sn_samples = np.random.normal(loc=z_sn, scale=z_sn_err, size=n_samples)

    P_z_unobs = prior_redshifts(z_gal_samples)
    L_z_unobs = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std)

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
    prob = sumL_absmag_dim * (P_z_unobs * L_z_unobs) * (P_offset_unobs * L_offset_unobs)

    # Sum to get the total probability
    P_unobserved = np.nansum(prob)/np.nansum(sumL_absmag_tot)

    if verbose:
        print(f"Calculated unobserved host probability: {P_unobserved}")

    return P_unobserved

def probability_host_outside_cone(z_sn, cutout_rad=60, verbose=False, n_samples=1000):
    # Handle NaN case by drawing from a uniform(0, 1) prior
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

    # Calculate the probability that the true host is outside the cone for each sample
    P_outside = np.nanmean([
        halfnorm.cdf(1e500, scale=1) - halfnorm.cdf(physical_offset_cutout / galaxy_physical_radius_prior, scale=1)
        for physical_offset_cutout, galaxy_physical_radius_prior in zip(physical_offset_cutouts, galaxy_physical_radius_prior_samples)
    ])

    return P_outside

def posterior_samples(z_sn, sn_position, galaxy_catalog, cutout_rad=60, n_samples=1000, m_lim=24.5, verbose=False):
    # Extract arrays for all galaxies from the catalog
    #z_gal_samples = np.array([gal['photo_z_dist'] for gal in galaxy_catalog])  # Shape (N, M)
    z_gal_mean = np.array([gal['z_phot_mean'] for gal in galaxy_catalog])
    z_gal_std = np.array([gal['z_phot_std'] for gal in galaxy_catalog])  # Shape (N, M)
    galaxy_ras = np.array([gal['ra'] for gal in galaxy_catalog])
    galaxy_decs = np.array([gal['dec'] for gal in galaxy_catalog])
    galaxy_DLR = np.array([gal['DLR_mean'] for gal in galaxy_catalog])
    galaxy_DLR_err = np.array([gal['DLR_std'] for gal in galaxy_catalog])

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

    #TODO: marginalize over missing host galaxy redshifts as well

    # Sample z values
    if not np.isnan(z_sn):
        print("WARNING: No SN redshift; marginalizing to infer probability that the host is unobserved.")
        z_sn_err = 0.05*z_sn
        z_sn_samples = np.maximum(0.001, np.random.normal(z_sn, z_sn_err, n_samples))  # Ensure non-negative redshifts

    galaxy_DLR_samples = np.random.normal(galaxy_DLR[:, np.newaxis], galaxy_DLR_err[:, np.newaxis], size=(len(galaxy_DLR), n_samples))

    # Calculate angular diameter distances for all samples
    galaxy_distances = cosmo.angular_diameter_distance(z_gal_samples).to(u.kpc).value  # Shape (N, M)
    sn_distance = cosmo.angular_diameter_distance(z_sn).to(u.kpc).value  # Shape (N, M)

    # Calculate the likelihoods for all galaxies (vectorized)
    Prior_z = prior_redshifts(z_gal_samples)

    fractional_offset_samples = offset_arcsec_samples/galaxy_DLR_samples

    Prior_offsets = prior_offsets(fractional_offset_samples)

    if np.isnan(z_sn):
        print("WARNING: No SN redshift; marginalizing to infer probability that the host is unobserved.")
        # Marginalize over the sampled supernova redshifts by integrating the likelihood over the redshift prior
        L_z = integrate_likelihood_over_redshift(z_gal_mean, z_gal_std)
    else:
        # Calculate the likelihoods for all galaxies with the given z_sn (vectorized)
        L_z = likelihood_redshifts(z_sn_samples, z_gal_mean, z_gal_std)  # Shape (N,)
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
    print(f"Probability your true host was not observed in your catalog (but is within the search cone): {P_unobs_norm:.4e}")

    return P_any_norm, P_none_norm, P_gals_norm, P_offsets_norm, P_z_norm

# Generate random positions within the circle
def uniform_circle_positions(center, radius_deg, n_points):
    ras = []
    decs = []
    for _ in range(n_points):
        # Uniformly sample the square root of the distance
        r = np.sqrt(np.random.uniform(0, 1)) * radius_deg
        # Random angle
        theta = np.random.uniform(0, 2 * np.pi)
        # Convert polar to Cartesian coordinates
        delta_ra = r * np.cos(theta)
        delta_dec = r * np.sin(theta)
        # Adjust with the center
        ra = center.ra.degree + delta_ra
        dec = center.dec.degree + delta_dec
        ras.append(ra)
        decs.append(dec)
    return np.array(ras), np.array(decs)

def random_physical_sizes(n_pts, size_min=0.1, size_max=500): #some crazy high max size
    # Parameters for the "Centrals" distribution in log10 space
    central_mean_log = 0.7  # Mean of log10(Ra/kpc)
    central_std_log = 0.4   # Standard deviation of log10(Ra/kpc) -- doubled from literature

    # Truncation bounds in log10 space
    log_min = np.log10(size_min)  # Corresponds to Ra ≈ 0.3 kpc
    log_max = np.log10(size_max)   # Corresponds to Ra ≈ 100 kpc

    # Calculate the a and b parameters for truncnorm based on the truncation bounds
    a = (log_min - central_mean_log) / central_std_log
    b = (log_max - central_mean_log) / central_std_log

    # Generate log10 sizes for Centrals
    log_sizes = truncnorm.rvs(a, b, loc=central_mean_log, scale=central_std_log, size=n_pts)

    # Convert log sizes to actual sizes in kpc
    return 10**log_sizes

# Convert physical sizes to angular sizes (arcseconds)
def physical_to_angular_size(physical_size, redshift):
    # Convert kpc to arcsec using cosmology
    return (physical_size/(cosmo.angular_diameter_distance(redshift)).to(u.kpc).value)*206265

# Function to diagnose the discrepancy when the top-ranked galaxy is not the true host
def diagnose_ranking(true_index, post_probs, galaxy_catalog, post_offset, post_redshift, galaxy_ids, z_sn, sn_position, verbose=False):
    top_indices = np.argsort(post_probs)[-5:][::-1]  # Top 5 ranked galaxies

    if verbose:
        if true_index > 0:
            print(f"True Galaxy: {true_index + 1}")

            # Check if the true galaxy is in the top 5
            if (true_index not in top_indices):
                print(f"Warning: True Galaxy {true_index + 1} is not in the top 5!")

        # Print top 5 and compare with the true galaxy
        for rank, i in enumerate(top_indices, start=1):
            is_true = "(True Galaxy)" if i == true_index and true_index > 0 else ""
            print(f"Rank {rank}: DECaLS ID {galaxy_ids[top_indices[rank-1]]} has a posterior probability of being the host: {post_probs[i]:.4f} {is_true}")

    # Detailed comparison of the top-ranked and true galaxy
    top_index = top_indices[0]
    top_gal = galaxy_catalog[top_index]
    true_gal = galaxy_catalog[true_index]

    if verbose:
        print(f"Coords (SN): {sn_position.ra.deg:.4f}, {sn_position.dec.deg:.4f}")
        print(f"Redshift (SN): {z_sn:.4f}")
        print(f"Top Galaxy (Rank 1): Coords: {top_gal['ra']:.4f}, {top_gal['dec']:.4f}")
        print(f"                     Redshift = {top_gal['z_phot_median']:.4f}+/-{top_gal['z_phot_std']:.4f}, Angular Size = {top_gal['angular_size_arcsec']:.4f} arcsec")
        if true_index > 0:
            print(f"True Galaxy: Redshift = {true_gal['redshift']:.4f}, Angular Size = {true_gal['angular_size_arcsec']:.4f} arcsec")

    # Calculate angular separation for detailed comparison
    top_theta = sn_position.separation(SkyCoord(ra=top_gal['ra']*u.degree, dec=top_gal['dec']*u.degree)).arcsec
    true_theta = sn_position.separation(SkyCoord(ra=true_gal['ra']*u.degree, dec=true_gal['dec']*u.degree)).arcsec

    if verbose:
        print(f"Top Galaxy (Rank 1): Fractional Separation = {top_theta/top_gal['angular_size_arcsec']:.4f} host radii")
        print(f"                      Angular Separation (\"): {top_theta:.2f}")

        if true_index > 0:
            print(f"True Galaxy: Fractional Separation = {true_theta/true_gal['angular_size_arcsec']:.4f} host radii")

    # Retrieve precomputed priors instead of recalculating
    post_offset_top = post_offset[top_index]
    if true_index > 0:
        post_offset_true = post_offset[true_index]

    post_z_top = post_z[top_index]
    if true_index > 0:
        post_z_true = post_z[true_index]

    if verbose:
        print(f"Top Galaxy (Rank 1): Redshift posterior (unnorm) = {post_z_top:.4e}, Offset posterior (unnorm) = {post_offset_top:.4e}")
        if true_index > 0:
            print(f"True Galaxy: Redshift posterior (unnorm) = {post_z_true:.4e}, Offset posterior (unnorm) = {post_offset_true:.4e}")

    ranked_indices = np.argsort(post_probs)[::-1]

    # Find the position of the true galaxy's index in the ranked list
    if true_index > 0:
        true_rank = np.where(ranked_indices == true_index)[0][0]
    else:
        true_rank = -1

    # Return the rank (0-based index) of the true galaxy
    return true_rank, post_probs[true_index]

###################################
###### applying to real data! ######
###################################

# :) close enough to a proof of concept -- let's test it on real data!!
#get SN coordinates for 2000cn
from astropy.io import ascii
import pandas as pd
from dl import queryClient as qc
import time
from astropy.coordinates import Angle

#sn_catalog = pd.read_csv("/Users/alexgagliano/Documents/Conferences/FreedomTrail_Jan24/sn_catalog/ZTFBTS_TransientTable.csv")
#sn_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/slsne_hosts.txt", delim_whitespace=True)
#sn_catalog_names = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/slsn_sne.txt", delim_whitespace=True)
#sn_catalog_names.rename(columns={'Name':'object_name'}, inplace=True)
#sn_catalog = sn_catalog.merge(sn_catalog_names)
#sn_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/ZTFBTS_TransientTable.csv")
#sn_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/localsn_public_cuts.txt", delim_whitespace=True)
#sn_catalog = sn_catalog[1:]

import pickle

source = "DELIGHT"

with open('/Users/alexgagliano/Desktop/prob_association/all.pkl', 'rb') as f:
    data = pickle.load(f)
data.reset_index(inplace=True)

#sn_catalog = sn_catalog[sn_catalog['IAUID'] == 'AT2021blu']
#SN2022acko and AT2021blu
#are we dealing with the fleet SLSN catalog?
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
    sn_catalog = data
    sn_catalog['Redshift'] = np.nan # :(
    sn_catalog['host_Pcc'] = 1.0
    sn_catalog.rename(columns={'meanra':'RA_deg', 'meandec':'DEC_deg', 'oid':'object_name'}, inplace=True)

#sn_catalog = sn_catalog[sn_catalog['object_name'] == '2022ued']
#sn_catalog[['object_name', 'host_ra','host_dec']]
#sn_catalog['object_name', 'ra_deg', 'dec_deg']
#sn_catalog.columns.values

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
    #if Nassociated > Ntot:
    #    break
    decals_match = False
    glade_match = False

    start_time = time.time()
    sn_name = row.object_name#row.IAUID
    sn_position = SkyCoord(Angle(row.RA_deg, unit=u.deg), Angle(row.DEC_deg, unit=u.deg)) #SkyCoord(Angle(row.RA, unit=u.hourangle), Angle(row.Dec, unit=u.deg))
    base_radius = 200/3600
    if not np.isnan(row.Redshift):
        sn_redshift = float(row.Redshift) #0.003
        # Convert 100 kpc to angular offset in arcseconds at the supernova redshift
        rad = np.nanmin([100 / cosmo.angular_diameter_distance(sn_redshift).to(u.kpc).value * 206265/3600, base_radius])
    else:
        # Fallback search radius (200 arcseconds in degrees)
        rad = base_radius

    stamp_size = int(rad*3600/angular_scale)

    print(f"\n\nSN Redshift: {sn_redshift:.4f}")
    print(f"SN Name: {sn_name}")

    sn_catalog.at[idx, 'sn_ra_deg'] = sn_position.ra.deg
    sn_catalog.at[idx, 'sn_dec_deg'] = sn_position.dec.deg

    #glade logic
    n_samples = 1000
    GLADE_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/GLADE+_HyperLedaSizes.csv", delim_whitespace=True,na_values=['', np.nan],keep_default_na=True)
    GLADE_catalog.dropna(subset=['logd25Hyp', 'logr25Hyp', 'e_logr25Hyp'], inplace=True)
    galaxies = build_glade_candidates(sn_position, rad, GLADE_catalog, n_samples=n_samples)
    #turn off GLADE for now as we don't have brightnesses

    if galaxies is None:
        glade_match = False
    else:
        any_prob, none_prob, post_probs, post_offset, post_z = posterior_samples(sn_redshift, sn_position, galaxies, m_lim=16, cutout_rad=rad*3600,verbose=False)
        print("GLADE diagnostics:")
        if verbose:
            diagnose_ranking(-1, post_probs, galaxies, post_offset, post_z, np.arange(len(galaxies)), sn_redshift, sn_position, verbose=verbose)
        if any_prob > none_prob:
            print(f"P(no host observed in GLADE cone): {none_prob:.4e}, P(host observed):{any_prob:.4e}")
            glade_match = False
        else:
            print("Found match by GLADE!")
            glade_match = True
    if not glade_match:
        print("Moving on to decals...")
        galaxies = build_decals_candidates(sn_position, rad, n_samples=n_samples)

        if galaxies is None:
            continue
        #setting the limiting magnitude to something exceptionally low
        any_prob, none_prob, post_probs, post_offset, post_z = posterior_samples(sn_redshift, sn_position, galaxies, m_lim=27, cutout_rad=rad*3600,verbose=verbose)
        if any_prob > none_prob:
            decals_match = True
    best_gal = np.argsort(post_probs)[::-1][0]
    top_prob = post_probs[best_gal]

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

        if verbose:
            diagnose_ranking(-1, post_probs, galaxies, post_offset, post_z, galaxies['ls_id'], sn_redshift, sn_position, verbose=True)

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
        chosen_gal_ra = chosen_gal_dec = None
    plotSNhost(chosen_gal_ra, chosen_gal_dec, Pcc_host_ra, Pcc_host_dec,
        galaxies['z_phot_mean'][best_gal], galaxies['z_phot_std'][best_gal],
        sn_position.ra.deg, sn_position.dec.deg, row.object_name, row.Redshift, row.object_name)
    #    host_pos = (sn_pos[0]+sn_catalog.at[idx, 'prob_host_xr']/angular_scale, sn_pos[1]+sn_catalog.at[idx, 'prob_host_yr']/angular_scale)
    #    host_z_median = float(galaxies['z_phot_median'][best_gal])
    #    host_z_std = float(galaxies['z_phot_std'][best_gal])
    #    plt.title(f"{row.object_name}, zSN={row.redshift:.2f}, z={host_z_median:.2f}+/-{host_z_std:.2f}", fontsize=14)
    #    plt.scatter(*host_pos, marker='*', s=150, ec='k', fc='tab:purple', zorder=1000)
    #else:
    #    plt.title(f"{row.object_name}, zSN={row.redshift:.2f}, No host found.", fontsize=14)

    #plt.axvline(x=sn_pos[0], c='tab:red')
    #plt.axhline(y=sn_pos[1], c='tab:red')
    #not quite the SN pos (x and y don't correspond to ra and dec here), but close enough
    #plt.savefig(f"/Users/alexgagliano/Desktop/prob_association/plots_likelihoodoffsetscale1_wGLADE/{row.object_name}.png",bbox_inches='tight')
    #plt.clf()
    #######################
    #######################
    #get PS1 postage stamp
    #######################
    #######################

    stats_done = sn_catalog['agreement_w_Pcc'].dropna()
    agree_frac = np.nansum(stats_done)/len(stats_done)
    print(f"Current agreement fraction: {agree_frac:.2f}")

    sn_catalog.at[idx, 'prob_association_time'] = match_time

#sn_catalog.to_csv("/Users/alexgagliano/Desktop/prob_association/slsn_catalog_Alexprob_PccCompare.csv",index=False)
#sn_catalog.to_csv("/Users/alexgagliano/Desktop/prob_association/ZTFBTS_Alexprob.csv",index=False)
sn_catalog.to_csv("/Users/alexgagliano/Desktop/prob_association/DELIGHT_Alexprob.csv",index=False)
#sn_catalog
#sn_catalog = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/Jones+18_Alexprob.csv")
#sn_catalog.columns.values


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

###################################
###### diagnostic code below ######
###################################

#plt.hist(sn_catalog['prob_association_time'], bins=np.linspace(1, 10))
#plt.xlabel("Association Time (s)")
#plt.show()

#angular_offset_arcsec_set.append(angular_offset_arcsec)
#physical_offset_kpc_set.append(offset_kpc)
#fractional_offset_set.append(offset_kpc/true_host_radius_kpc)
#redshift_sn_set.append(sn_redshift)
#host_angular_size_set.append(true_host_radius_arcsec)

# Output the true host and SN information
#print("True Host Galaxy:")
#print(f"RA: {true_host_ra:.5f} deg, Dec: {true_host_dec:.5f} deg, Redshift: {true_host_redshift:.4f}")
#print(f"Physical Size: {true_host_radius_kpc:.1f} kpc")

#print("\nSupernova Position and Redshift:")
#print(f"RA: {sn_ra:.5f} deg, Dec: {sn_dec:.5f} deg, Redshift: {sn_redshift:.4f}")
#print(f"Offset from Host: {offset_kpc:.1f} kpc")
#print(f"Angular Offset: {angular_offset_arcsec:.2f}\"")

#plt.figure(figsize=(5,5))
#plt.xlabel("RA")
#plt.ylabel("Dec")
#plt.scatter(galaxy_ras, galaxy_decs, marker='o', s=galaxy_angular_radius**2)
#plt.scatter(true_host_ra, true_host_dec, color='red', s=galaxy_angular_radius[true_host_index]**2, label="True Host")
#plt.scatter(sn_ra, sn_dec, color='k', marker='+', s=200, label="Supernova")

#plt.xlabel("angular offset, \"")
#plt.ylabel("SN redshift")
#plt.plot(angular_offset_arcsec_set, redshift_sn_set, 'o')
#plt.xlim((0, 20))
#plt.xscale("log")

#plt.plot(redshift_sn_set, host_angular_size_set, 'o')
#plt.ylim((0, 50))

## TODO: Incorporate directional light radius calculation and random ellipticities between 0 and 1
##       Add spurious objects (HII regions, etc)
##       weight by whether the object is a
##       Test on a sample of GHOST galaxies (against GHOST)
##       Look for speedups
## NOTE -- if we have spec-z of the galaxy, set some fractional uncertainty floor and sample from that






#######################################################
############### generate random data ##################
#######################################################
def run_experiment():
    sky_location = SkyCoord(ra=150*u.degree, dec=2*u.degree)

    radius_arcsec = 60  # 60 arcseconds
    radius_deg = radius_arcsec / 3600  # Convert to degrees
    # Define volumetric density (n galaxies per Mpc^3)
    n_density = 0.01  # Example: galaxies per ''^2

    # Define the redshift range (z_min, z_max)
    z_min = 0.01
    z_max = 1.0

    # Angular radius of the cone (60 arcseconds in radians)
    size_region = np.pi*radius_arcsec**2
    n_galaxies = int(n_density * size_region)

    print(f"Number of galaxies in 30'' radius: {n_galaxies}")

    galaxy_redshifts = np.random.uniform(z_min, z_max, size=n_galaxies)

    # Generate galaxy coordinates
    galaxy_ras, galaxy_decs = uniform_circle_positions(sky_location, radius_deg, n_galaxies)
    galaxy_physical_radius = random_physical_sizes(n_galaxies)
    galaxy_angular_radius = physical_to_angular_size(galaxy_physical_radius, galaxy_redshifts)

    # Combine all into a structured array
    dtype = [('ra', float), ('dec', float), ('redshift', float), ('physical_size_kpc', float), ('angular_size_arcsec', float),('angular_size_arcsec_err', float),('photo_z_dist', object)]
    galaxies = np.zeros(n_galaxies, dtype=dtype)
    galaxies['ra'] = galaxy_ras
    galaxies['dec'] = galaxy_decs
    galaxies['redshift'] = galaxy_redshifts
    galaxies['physical_size_kpc'] = galaxy_physical_radius
    galaxies['angular_size_arcsec'] = galaxy_angular_radius
    galaxies['angular_size_arcsec_err'] = np.random.uniform(0.1, 0.3, size=len(galaxy_angular_radius))*galaxy_angular_radius #random uncertainties from 0 to 30% of reported angular size

    #assume photometric redshift uncertainties that are up to 20% the true redshift
    scale_factors = np.random.uniform(0, 0.2, size=len(galaxy_redshifts))  # Shape (N,)

    #for i in range(n_galaxies):
#        galaxies['photo_z_dist'][i] = photo_z_samples[i, :]

    # Choose a random galaxy as the true host, weighted by the size of the galaxy and only picking a galaxy if it's within z<0.3 (observing selection function)
    true_host_redshift = 100 #some made-up high number
    while true_host_redshift > 0.3:
        true_host_index = np.random.choice(len(galaxy_physical_radius),  p=(galaxy_physical_radius)/np.nansum(galaxy_physical_radius)) #np.random.randint(0, n_galaxies)
        true_host_ra = galaxy_ras[true_host_index]
        true_host_dec = galaxy_decs[true_host_index]
        true_host_redshift = galaxy_redshifts[true_host_index]
        true_host_radius_kpc = galaxy_physical_radius[true_host_index]
        true_host_radius_arcsec = galaxy_angular_radius[true_host_index]


    # Determine SN redshift from Gaussian centered at host redshift with a variance of 10% the true redshift
    sn_redshift = np.random.normal(loc=true_host_redshift, scale=0.1*true_host_redshift)

    # Determine random offset from the host size -- assume sigma of half-radius, and max of 10*R_a
    offset_kpc = truncnorm.rvs(a=0, b=10, scale=1)*true_host_radius_kpc

    # Convert physical offset to angular offset (arcseconds)
    sn_distance_mpc = cosmo.angular_diameter_distance(sn_redshift).value  # in Mpc
    angular_offset_deg = (offset_kpc / (sn_distance_mpc * 1e3)) * (180 / np.pi)  # in degrees
    angular_offset_arcsec = angular_offset_deg * 3600  # in arcseconds

    # Random direction for the offset
    theta = np.random.uniform(0, 2 * np.pi)
    delta_ra = angular_offset_arcsec * np.cos(theta) / 3600  # convert to degrees
    delta_dec = angular_offset_arcsec * np.sin(theta) / 3600  # convert to degrees

    # Calculate SN position
    sn_ra = true_host_ra + delta_ra
    sn_dec = true_host_dec + delta_dec

    # Convert back to a SkyCoord for convenience
    sn_position = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree)

    #if random bool is flipped, the SN redshift is unknown
    if np.random.choice(a=[False, True]):
        sn_redshift = np.nan
        print("No SN redshift!")

    ##################################################
    ###now run our probabilistic association code:####
    ##################################################

    # Compute posterior probabilities using the full posterior distributions
    redshift_prior = np.linspace(0.01, 0.5, 100)
    post_probs, offset_priors = posterior(sn_redshift, sn_position, galaxies, redshift_prior)
    ranked_idx_true, post_prob_true = diagnose_ranking(true_host_index, post_probs, galaxies, offset_priors, sn_redshift, sn_position, verbose=True)
    print(f"\n\nRanked idx:{ranked_idx_true}")
    return sn_redshift, ranked_idx_true, post_prob_true

run_sim_experiment = False

if run_sim_experiment:
    sn_redshifts = []
    indices = []
    post_probs = []
    for i in np.arange(100):
        print(f"\n\nTrial {i+1}:")
        sn_redshift, index, post_prob = run_experiment()
        sn_redshifts.append(sn_redshift)
        indices.append(index)
        post_probs.append(post_prob)

    sn_redshifts = np.array(sn_redshifts)
    indices = np.array(indices)
    post_probs = np.array(post_probs)

    import matplotlib.pyplot as plt

    plt.hist(post_probs);
    plt.hist(indices, bins=100);

#61% agreement between Pcc and this method.

#wrongGHOST = pd.read_csv("/Users/alexgagliano/Desktop/prob_association/Jones+18_wrongGHOST.csv")

#import shutil
#import glob

#for name in wrongGHOST['name'].values:
#    fns = glob.glob(f"/Users/alexgagliano/Desktop/prob_association/plots_likelihoodoffsetscale1_wGLADE/Jones+18/{name}.png")
#    if len(fns) > 0:
#        new_fn = fns[0].split("/")
#        new_fn = "/".join(new_fn[:-1]) + "/wrongGHOST/" + new_fn[-1]
#        shutil.copyfile(fns[0], new_fn)
