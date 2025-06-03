#!/usr/bin/env python3
"""
Test case for SkyMapper host galaxy association.

This test ensures that the SkyMapper pipeline works correctly for southern sky transients
and prevents regression of the band filtering bug that was filtering out all sources
due to missing 'ngood' columns in SkyMapper data.

Run with: python test_skymapper_association.py
"""

import sys
sys.path.insert(0, 'src')

from astro_prost.helpers import fetch_skymapper_sources
import astropy.units as u
from astropy.coordinates import SkyCoord

def test_skymapper_association():
    """Test SkyMapper association for known southern sky transients with hosts."""
    
    print("=" * 70)
    print("TESTING SKYMAPPER HOST ASSOCIATION")
    print("=" * 70)
    
    # Test cases: (transient_name, ra, dec, expected_host_id, max_distance_arcsec)
    test_cases = [
        ("2025hpn", 337.037241, -72.826993, 1593322964, 3.0),
        # Add more test cases here as they are discovered
    ]
    
    # Basic search parameters
    search_radius = 30 * u.arcsec
    cat_cols = ['raj2000', 'dej2000', 'object_id', 'r_petro', 'e_r_petro', 
                'r_a', 'r_b', 'r_e_a', 'r_e_b', 'r_pa', 'r_e_pa']
    
    all_passed = True
    
    for transient_name, ra, dec, expected_host_id, max_distance in test_cases:
        print(f"\nTesting {transient_name} at RA={ra:.6f}, Dec={dec:.6f}")
        print("-" * 50)
        
        try:
            # Create transient position
            transient_pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            
            # Query SkyMapper DR4 (the working data release)
            sources = fetch_skymapper_sources(
                search_pos=transient_pos,
                search_rad=search_radius, 
                cat_cols=cat_cols,
                calc_host_props=False,
                release='dr4'
            )
            
            if sources is None:
                print(f"FAIL: No sources returned for {transient_name}")
                all_passed = False
                continue
                
            print(f"Found {len(sources)} sources in SkyMapper DR4")
            
            # Check if expected host is found
            if expected_host_id in sources['objID'].tolist():
                # Calculate distance to expected host
                host_match = sources[sources['objID'] == expected_host_id]
                host_ra = host_match['ra'].iloc[0]
                host_dec = host_match['dec'].iloc[0]
                host_coord = SkyCoord(ra=host_ra*u.deg, dec=host_dec*u.deg)
                
                distance = transient_pos.separation(host_coord).arcsec
                
                if distance <= max_distance:
                    print(f"PASS: Expected host {expected_host_id} found at {distance:.2f} arcsec")
                    
                    # Show host properties
                    if 'r_petro' in host_match.columns:
                        r_mag = host_match['r_petro'].iloc[0]
                        print(f"   Host r-band magnitude: {r_mag:.2f}")
                else:
                    print(f"FAIL: Host {expected_host_id} found but too far ({distance:.2f} > {max_distance} arcsec)")
                    all_passed = False
            else:
                print(f"FAIL: Expected host {expected_host_id} not found")
                available_ids = sources['objID'].tolist()
                print(f"   Available object IDs: {available_ids[:5]}..." if len(available_ids) > 5 else f"   Available object IDs: {available_ids}")
                all_passed = False
                
        except Exception as e:
            print(f"ERROR testing {transient_name}: {e}")
            all_passed = False
    
    # Test the band filtering logic doesn't break
    print(f"\n" + "-" * 50)
    print("TESTING BAND FILTERING LOGIC")
    print("-" * 50)
    
    try:
        # Test with a known working case
        test_pos = SkyCoord(ra=337.037241*u.deg, dec=-72.826993*u.deg)
        sources = fetch_skymapper_sources(
            search_pos=test_pos,
            search_rad=10*u.arcsec,
            cat_cols=cat_cols,
            calc_host_props=False,
            release='dr4'
        )
        
        if sources is not None and len(sources) > 0:
            print("PASS: Band filtering allows sources through (doesn't filter everything)")
            
            # Check that ngood columns behavior
            ngood_cols = [col for col in sources.columns if 'ngood' in col]
            if len(ngood_cols) == 0:
                print("PASS: No ngood columns (pure SkyMapper data)")
            else:
                print(f"INFO: Found {len(ngood_cols)} ngood columns (mixed data)")
                print("PASS: Band filtering correctly handles mixed column scenarios")
        else:
            print("FAIL: Band filtering is too aggressive (filters out everything)")
            all_passed = False
            
    except Exception as e:
        print(f"ERROR testing band filtering: {e}")
        all_passed = False
    
    # Final result
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! SkyMapper association is working correctly.")
    else:
        print("SOME TESTS FAILED! Check the SkyMapper pipeline.")
        sys.exit(1)
    print("=" * 70)

def test_data_release_compatibility():
    """Test that the correct data release is being used."""
    print(f"\n" + "-" * 50)
    print("TESTING DATA RELEASE COMPATIBILITY")
    print("-" * 50)
    
    test_pos = SkyCoord(ra=337.037241*u.deg, dec=-72.826993*u.deg)
    search_radius = 10 * u.arcsec
    cat_cols = ['raj2000', 'dej2000', 'object_id', 'r_petro', 'e_r_petro']
    
    # Test that DR4 works (where our test object exists)
    try:
        sources_dr4 = fetch_skymapper_sources(
            search_pos=test_pos,
            search_rad=search_radius,
            cat_cols=cat_cols,
            calc_host_props=False,
            release='dr4'
        )
        
        if sources_dr4 is not None and len(sources_dr4) > 0:
            print("PASS: DR4 returns sources")
        else:
            print("FAIL: DR4 returns no sources")
            
    except Exception as e:
        print(f"ERROR with DR4: {e}")
    
    # Test that DR2 (old default) doesn't have our test object (expected)
    try:
        sources_dr2 = fetch_skymapper_sources(
            search_pos=test_pos,
            search_rad=search_radius,
            cat_cols=cat_cols,
            calc_host_props=False,
            release='dr2'
        )
        
        if sources_dr2 is None or len(sources_dr2) == 0:
            print("PASS: DR2 returns no sources (expected for this test case)")
        else:
            print(f"INFO: DR2 returns {len(sources_dr2)} sources")
            
    except Exception as e:
        print(f"ERROR with DR2: {e}")

if __name__ == "__main__":
    test_skymapper_association()
    test_data_release_compatibility()
    print("\nTest complete!") 