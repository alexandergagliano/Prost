import pandas as pd
import numpy as np
from astro_prost.associate import setup_logger
from astro_prost.helpers import fetch_panstarrs_sources, find_panstarrs_shreds, calc_shape_props_panstarrs, is_service_available
# constants
from astro_prost.helpers import SIGMA_ABSMAG_CEIL, SIGMA_ABSMAG_FLOOR
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import time
import pytest

@pytest.mark.skipif(
    not is_service_available("https://catalogs.mast.stsci.edu"),
    reason="Remote service is unavailable"
)
def test_panstarrs_shred():
    # set random seed to fix sampling in DLR
    np.random.seed(42)

    # calculate panstarrs shreds around 3 SNe (2020oi, 2023zkd, 2023ixf)
    transient_pos_set = []
    transient_pos_set.append(SkyCoord(237.19810932, 9.20003951886, unit=(u.deg, u.deg)))
    transient_pos_set.append(SkyCoord(185.728875, 15.8236, unit=(u.deg, u.deg)))
    transient_pos_set.append(SkyCoord(210.910674637, 54.3116510708, unit=(u.deg, u.deg)))

    search_rad = Angle(60 * u.arcsec)
    calc_host_props = ['redshift', 'offset', 'absmag']
    release = 'dr2'
    cat_cols = False # whether to concatenate pan-starrs columns onto the output

    logger = setup_logger(verbose=0)

    Nshred_true = [0, 165, 64] 

    for idx, transient_pos in enumerate(transient_pos_set):

        candidate_hosts = fetch_panstarrs_sources(transient_pos, search_rad, cat_cols, calc_host_props, release)

        temp_sizes, temp_sizes_std, a_over_b, a_over_b_std, phi, phi_std = calc_shape_props_panstarrs(candidate_hosts)

        temp_mag_r = candidate_hosts["KronMag"].values
        temp_mag_r_std = candidate_hosts["KronMagErr"].values
    
        # cap at 50% the mag
        # set a floor of 5%
        temp_mag_r_std[temp_mag_r_std > (SIGMA_ABSMAG_CEIL*temp_mag_r)] = SIGMA_ABSMAG_CEIL * temp_mag_r[temp_mag_r_std > (SIGMA_ABSMAG_CEIL*temp_mag_r)]
        temp_mag_r_std[temp_mag_r_std < (SIGMA_ABSMAG_FLOOR * temp_mag_r)] = SIGMA_ABSMAG_FLOOR * temp_mag_r[temp_mag_r_std < (SIGMA_ABSMAG_FLOOR * temp_mag_r)]

        galaxies_pos = SkyCoord(
             candidate_hosts["ra"].values * u.deg, candidate_hosts["dec"].values * u.deg
        )

        logger.info(f"Removing panstarrs {release} shreds.")

        start = time.perf_counter()
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
            logger)
        end = time.perf_counter()

        Nshred = len(shred_idxs)
        print("Nshred", Nshred)

        # Print execution times
        print("Execution time: {:.6f} seconds".format(end - start))

        assert Nshred == Nshred_true[idx]

