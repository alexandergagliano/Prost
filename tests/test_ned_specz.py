import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
import numpy as np
import pytest
import requests

from astro_prost.associate import associate_sample
from astro_prost.helpers import SnRateAbsmag, get_ned_specz


def test_ned_specz_2025qoz():
    """Test that NED spectroscopic redshift is retrieved for SN 2025qoz.

    This test verifies that when best_redshift=True, PROST queries NED for
    spectroscopic redshifts and updates the host redshift information
    accordingly. SN 2025qoz has a host galaxy with a known spectroscopic
    redshift of z~0.028 in NED, but GLADE+ contains only a photometric
    redshift of z~0.049 for this object.

    The test confirms that:
    1. The host galaxy is correctly identified
    2. NED is queried and finds the spectroscopic redshift
    3. The host redshift is updated from PHOT to SPEC
    4. The spectroscopic redshift value is approximately 0.028
    """
    # Check if NED is accessible before running the test
    # Expected host coordinates from GLADE
    test_ra, test_dec = 69.438805, -20.507317
    _, _, ned_available = get_ned_specz(test_ra, test_dec, search_radius=1.0, logger=None)

    if not ned_available:
        pytest.skip("NED service is not accessible (network timeout or service unavailable)")

    np.random.seed(42)

    # SN 2025qoz: RA=69.4397792, Dec=-20.5065667
    # Known spec-z of host: 0.028 (from NED)
    # GLADE+ has photo-z: 0.049
    transient_catalog = pd.DataFrame({
        'IAUID': ['2025qoz'],
        'RA': [69.4397792],
        'Dec': [-20.5065667]
    })

    # Define priors for properties
    priorfunc_offset = uniform(loc=0, scale=5)
    priorfunc_absmag = uniform(loc=-30, scale=20)

    likefunc_offset = gamma(a=0.75)
    likefunc_absmag = SnRateAbsmag(a=-25, b=20)

    priors = {
        "offset": priorfunc_offset,
        "absmag": priorfunc_absmag
    }
    likes = {
        "offset": likefunc_offset,
        "absmag": likefunc_absmag
    }

    # Set up properties of the association run
    verbose = 2
    parallel = False
    save = False
    progress_bar = False
    cat_cols = False

    # Run with best_redshift=False (default behavior)
    try:
        hostTable_no_ned = associate_sample(
            transient_catalog,
            priors=priors,
            likes=likes,
            catalogs=["glade"],
            name_col="IAUID",
            coord_cols=("RA", "Dec"),
            parallel=parallel,
            verbose=verbose,
            save=save,
            progress_bar=progress_bar,
            cat_cols=cat_cols,
            best_redshift=False
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pytest.skip("Service timeout")

    # Run with best_redshift=True (query NED for spec-z)
    try:
        hostTable_with_ned = associate_sample(
            transient_catalog,
            priors=priors,
            likes=likes,
            catalogs=["glade"],
            name_col="IAUID",
            coord_cols=("RA", "Dec"),
            parallel=parallel,
            verbose=verbose,
            save=save,
            progress_bar=progress_bar,
            cat_cols=cat_cols,
            best_redshift=True
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pytest.skip("Service timeout")

    # Verify that without NED query, we get photometric redshift from GLADE
    assert hostTable_no_ned['host_redshift_info'][0] == 'PHOT', \
        "Expected PHOT redshift from GLADE without NED query"
    assert abs(hostTable_no_ned['host_redshift_mean'][0] - 0.049) < 0.01, \
        f"Expected GLADE photo-z ~0.049, got {hostTable_no_ned['host_redshift_mean'][0]}"

    # Verify that with NED query, we get spectroscopic redshift
    assert hostTable_with_ned['host_redshift_info'][0] == 'SPEC', \
        "Expected SPEC redshift from NED query"
    assert abs(hostTable_with_ned['host_redshift_mean'][0] - 0.028) < 0.005, \
        f"Expected NED spec-z ~0.028, got {hostTable_with_ned['host_redshift_mean'][0]}"

    # Verify that the host galaxy is the same in both cases
    assert hostTable_no_ned['host_objID'][0] == hostTable_with_ned['host_objID'][0], \
        "Host galaxy should be the same with or without NED query"

    print("✓ Test passed: NED spectroscopic redshift correctly retrieved for SN 2025qoz")
    print(f"  GLADE photo-z: {hostTable_no_ned['host_redshift_mean'][0]:.4f} (PHOT)")
    print(f"  NED spec-z: {hostTable_with_ned['host_redshift_mean'][0]:.4f} (SPEC)")


if __name__ == "__main__":
    test_ned_specz_2025qoz()
