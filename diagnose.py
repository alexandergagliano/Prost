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
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
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

def plotmatch(host_ra, host_dec, Pcc_host_ra, Pcc_host_dec, host_z_mean, host_z_std, SN_ra, SN_dec, SN_name, SN_z, Bayesflag, fn):
    cols = np.array(['#ff9f1c', '#2cda9d', '#f15946', '#da80dd', '#f4e76e', '#b87d4b', '#ff928b', '#c73e1d', '#58b09c', '#e7e08b'])
    bands = 'zrg'
    if len(host_ra) > 0:
        sep = np.nanmax(SkyCoord(host_ra*u.deg, host_dec*u.deg).separation(SkyCoord(SN_ra*u.deg, SN_dec*u.deg)).arcsec)
    else:
        sep = 0
    if Pcc_host_ra:
        sep_Pcc = SkyCoord(Pcc_host_ra*u.deg, Pcc_host_dec*u.deg).separation(SkyCoord(SN_ra*u.deg, SN_dec*u.deg)).arcsec
        if (Pcc_host_ra) and (Pcc_host_dec) and (sep_Pcc > sep):
            sep = sep_Pcc
    rad = np.nanmax([30., 2*sep]) #arcsec to pixels, scaled by 1.5x host-SN separation
    print(f"Getting img with size len {rad:.2f}...")
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
    lo_val, up_val = np.nanpercentile(np.array(pic_data).ravel(), (0.5, 99.5))  # Get the value of lower and upper 0.5% of all pixels
    stretch_val = up_val - lo_val

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
        ax.scatter(Pcc_host_ra, Pcc_host_dec, transform=ax.get_transform('fk5'), marker='+', alpha=0.8, lw=2, s=200,
            color='magenta', zorder=100)
    else:
        Pcc_str = "(no Pcc)"
    Bayesstr = '. '
    if Bayesflag == 2:
        Bayesstr += 'Strong match!'
        #don't plot the second-best host
        host_ra = host_ra[:1]
        host_dec = host_dec[:1]
    elif Bayesflag == 1:
        Bayesstr +=  'Weak match.'
    if (host_ra and host_dec):
        for i in np.arange(len(host_ra)):
            #print(f"Plotting host {i}")
            ax.scatter(host_ra[i], host_dec[i], transform=ax.get_transform('fk5'), marker='o', alpha=0.8, lw=2, s=100,
                edgecolor='k', facecolor=cols[i], zorder=100)
        if SN_z == SN_z:
            plt.title(f"{SN_name}, z={SN_z:.4f}; Host Match, z={host_z_mean:.4f}+/-{host_z_std:.4f} {Pcc_str}{Bayesstr}")
        else:
            plt.title(f"{SN_name}, no z; Host Match, z={host_z_mean:.4f}+/-{host_z_std:.4f} {Pcc_str}{Bayesstr}")
    else:
        if SN_z == SN_z:
            plt.title(f"{SN_name}, z={SN_z:.4f}; No host found {Pcc_str}")
        else:
            plt.title(f"{SN_name}, no z; No host found {Pcc_str}")
    ax.imshow(rgb_default, origin='lower')
    plt.axis('off')
    plt.savefig("./%s.png"%fn, bbox_inches='tight')
    plt.close()

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
def diagnose_ranking(true_index, post_probs, galaxy_catalog, post_offset, post_z, post_absmag, galaxy_ids, z_sn, sn_position, verbose=False):
    top_indices = np.argsort(post_probs)[-3:][::-1]  # Top 3 ranked galaxies
    top_index = top_indices[0]

    if verbose:
        if true_index > 0:
            print(f"True Galaxy: {true_index + 1}")

            # Check if the true galaxy is in the top 5
            if (true_index not in top_indices):
                print(f"Warning: True Galaxy {true_index + 1} is not in the top 5!")

        # Print top 5 and compare with the true galaxy
        for rank, i in enumerate(top_indices, start=1):
            is_true = "(True Galaxy)" if i == true_index and true_index > 0 else ""
            print(f"Rank {rank}: ID {galaxy_ids[top_indices[rank-1]]} has a posterior probability of being the host: {post_probs[i]:.4f} {is_true}")

    # Detailed comparison of the top-ranked and true galaxy
    print(f"Coords (SN): {sn_position.ra.deg:.4f}, {sn_position.dec.deg:.4f}")
    for rank, i in enumerate(top_indices, start=1):
        top_gal = galaxy_catalog[i]
        top_theta = sn_position.separation(SkyCoord(ra=top_gal['ra']*u.degree, dec=top_gal['dec']*u.degree)).arcsec

        if verbose:
            print(f"Redshift (SN): {z_sn:.4f}")
            print(f"Top Galaxy (Rank {i}): Coords: {top_gal['ra']:.4f}, {top_gal['dec']:.4f}")
            print(f"                     Redshift = {top_gal['z_best_mean']:.4f}+/-{top_gal['z_best_std']:.4f}, Angular Size = {top_gal['angular_size_arcsec']:.4f} arcsec")
            print(f"                     Fractional Separation = {top_theta/top_gal['angular_size_arcsec']:.4f} host radii")
            print(f"                     Angular Separation (\"): {top_theta:.2f}")
            print(f"                     Redshift posterior = {post_z[i]:.4e}, Offset posterior = {post_offset[i]:.4e}")
            print(f"                     Absolute mag posterior = {post_absmag[i]:.4e}")

    if verbose and true_index > 0:
            true_gal = galaxy_catalog[true_index]
            true_theta = sn_position.separation(SkyCoord(ra=true_gal['ra']*u.degree, dec=true_gal['dec']*u.degree)).arcsec
            print(f"True Galaxy: Fractional Separation = {true_theta/true_gal['angular_size_arcsec']:.4f} host radii")
            print(f"             Redshift = {true_gal['redshift']:.4f}, Angular Size = {true_gal['angular_size_arcsec']:.4f} arcsec")
            print(f"             Redshift posterior = {post_z_true:.4e}, Offset posterior = {post_offset_true:.4e}")

    # Retrieve precomputed priors instead of recalculating
    post_offset_top = post_offset[top_index]
    if true_index > 0:
        post_offset_true = post_offset[true_index]

    post_z_top = post_z[top_index]
    if true_index > 0:
        post_z_true = post_z[true_index]

    ranked_indices = np.argsort(post_probs)[::-1]

    # Find the position of the true galaxy's index in the ranked list
    if true_index > 0:
        true_rank = np.where(ranked_indices == true_index)[0][0]
    else:
        true_rank = -1

    # Return the rank (0-based index) of the true galaxy
    return true_rank, post_probs[true_index]


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

def crossmatch_glade_decals(GLADE_catalog):
    shape_decals = []
    shape_glade = []
    for idx, row in GLADE_catalog.iterrows():
        result = qc.query(sql=f"""SELECT
            t.ls_id,
            t.shape_r,
            t.shape_r_ivar,
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
            q3c_radial_query(t.ra, t.dec, {row.RAJ2000:.5f}, {row.DEJ2000:.5f}, 0.002777777777777778)
        AND (t.nobs_r > 0) AND ((pz.z_spec > 0) OR (pz.z_phot_mean > 0))""")
        tbl = ascii.read(result).to_pandas()
        if len(tbl) > 0:
            shape_decals.append(np.nanmax(tbl['shape_r'].values))
            shape_glade.append(0.5*3*10**(row.logd25Hyp))
            if len(shape_decals) > 500:
                break

    plt.figure(figsize=(10, 8))
    plt.plot(shape_decals, shape_glade, 'o', mec='k')
    plt.plot([0, 60], [0, 60], '--', ls='--', c='k')
    plt.xlabel("DECALS Half-Light Radius, r (\")")
    plt.ylabel("GLADE d25 Isophotal half-semi-major axis (\")")
    plt.xlim((0, 60))
    plt.ylim((0, 60))
    plt.show()
