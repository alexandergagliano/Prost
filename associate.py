import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("/Users/alexgagliano/Documents/Research/prob_association/")
from diagnose import *
from helpers import *
import seaborn as sns
import scipy.stats as st
import glob

def associate_supernova(idx, row, GLADE_catalog, n_samples, n_processes, angular_scale, verbose, priorfunc_offset, priorfunc_absmag):
    start_time = time.time()

    decals_match = False
    glade_match = False
    sn_name = row.object_name
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

    row['sn_ra_deg'] = sn_position.ra.deg
    row['sn_dec_deg'] = sn_position.dec.deg

    n_samples_per_process = n_samples // n_processes
    galaxies = build_glade_candidates(sn_name, sn_position, rad_arcsec, GLADE_catalog, n_samples=n_samples_per_process)

    if galaxies is None:
        print("Nothing found in GLADE.")
        any_prob = 0
        none_prob = 1
        glade_match = False
    else:
        #monte_carlo_glade
        any_prob, none_prob, post_probs, post_offset, post_z, post_absmag = parallel_monte_carlo(
            sn_redshift, sn_position, galaxies, m_lim=18, cutout_rad=rad_arcsec, verbose=False, n_samples=n_samples, n_processes=n_processes
        )
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Took {elapsed:.2f}s to find a GLADE match.")
        if (any_prob > none_prob):
            print("Found match by GLADE!")
            glade_match = True
        else:
            print(f"P(no host observed in GLADE cone): {none_prob:.4e}, P(host observed):{any_prob:.4e}")
            glade_match = False

        best_gal = np.argsort(post_probs)[::-1][0]
        top_prob = post_probs[best_gal]
        diagnose_ranking(-1, post_probs, galaxies, post_offset, post_z, post_absmag, np.arange(len(galaxies)), sn_redshift, sn_position, verbose=True)

    if not glade_match:
        print("Moving on to decals...")
        galaxies = build_decals_candidates(sn_name, sn_position, rad_arcsec, n_samples=n_samples_per_process)
        if galaxies is None:
            print("Nothing found in DECaLS.")
            any_prob = 0
            none_prob = 1
            decals_match = False
        else:
            start_time = time.time()
            any_prob, none_prob, post_probs, post_offset, post_z, post_absmag = parallel_monte_carlo(
                sn_redshift, sn_position, galaxies, m_lim=27, cutout_rad=rad_arcsec, verbose=verbose, n_samples=n_samples, n_processes=n_processes
            )
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Took {elapsed:.2f}s to find a decals match.")
            if any_prob > none_prob:
                decals_match = True
            best_gal = np.argsort(post_probs)[::-1][0]
            top_prob = post_probs[best_gal]
            diagnose_ranking(-1, post_probs, galaxies, post_offset, post_z, post_absmag, galaxies['ls_id'], sn_redshift, sn_position, verbose=True)

    if (glade_match == False) and (decals_match == False):
        print("Moving on to PS2...")
        galaxies = build_panstarrs_candidates(sn_name, sn_position, rad_arcsec, n_samples=n_samples_per_process)
        if galaxies is None:
            print("Nothing found in Panstarrs2.")
            any_prob = 0
            none_prob = 1
            panstarrs_match = False
        else:
            #monte_carlo_glade
            any_prob, none_prob, post_probs, post_offset, post_z, post_absmag = parallel_monte_carlo(
                sn_redshift, sn_position, galaxies, m_lim=24, cutout_rad=rad_arcsec, verbose=False, n_samples=n_samples, n_processes=n_processes
            )
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Took {elapsed:.2f}s to find a Panstarrs match.")
            if (any_prob > none_prob):
                print("Found match by Panstarrs!")
                panstarrs_match = True
            else:
                print(f"P(no host observed in Panstarrs cone): {none_prob:.4e}, P(host observed):{any_prob:.4e}")
                panstarrs_match = False

            best_gal = np.argsort(post_probs)[::-1][0]
            top_prob = post_probs[best_gal]
            diagnose_ranking(-1, post_probs, galaxies, post_offset, post_z, post_absmag, np.arange(len(galaxies)), sn_redshift, sn_position, verbose=True)

    if (any_prob > none_prob):
        print("Probability of observing the host is higher than missing it.")
        print(f"P(host observed):{any_prob:.4e}")
        if decals_match:
            row['prob_host_ls_id'] = galaxies['ls_id'][best_gal]
            print(f"DECALS ID of matched host:{galaxies['ls_id'][best_gal]}")
        row['prob_host_z_best_mean'] = galaxies['z_best_mean'][best_gal]
        row['prob_host_z_best_std'] = galaxies['z_best_std'][best_gal]
        row['prob_host_ra'] = galaxies['ra'][best_gal]
        row['prob_host_dec'] = galaxies['dec'][best_gal]
        row['prob_host_score'] = post_probs[best_gal]
        row['prob_host_xr'] = (sn_position.ra.deg - galaxies['ra'][best_gal])*3600
        row['prob_host_yr'] = (sn_position.dec.deg - galaxies['dec'][best_gal])*3600

        second_best_gal = None
        if len(post_probs) > 1:
            second_best_gal = np.argsort(post_probs)[::-1][1]

            second_prob = post_probs[second_best_gal]
            bayes_factor = top_prob / second_prob

            if bayes_factor < 3:
                print(f"Warning: The evidence for the top candidate over the second-best is weak. Bayes Factor = {bayes_factor:.2f}. Check for a shred!")
                row['prob_host_flag'] = 1
            elif bayes_factor > 100:
                print(f"The top candidate has very strong evidence over the second-best. Bayes Factor = {bayes_factor:.2f}")
                row['prob_host_flag'] = 2

            if decals_match:
                row['prob_host_2_ls_id'] = galaxies['ls_id'][second_best_gal]
            row['prob_host_2_z_best_mean'] = galaxies['z_best_mean'][second_best_gal]
            row['prob_host_2_z_best_std'] = galaxies['z_best_std'][second_best_gal]
            row['prob_host_2_ra'] = galaxies['ra'][second_best_gal]
            row['prob_host_2_dec'] = galaxies['dec'][second_best_gal]
            row['prob_host_2_score'] = post_probs[second_best_gal]

        #compare to probability of chance coincidence
        sep_Pcc = SkyCoord(row['host_ra']*u.deg, row['host_dec']*u.deg).separation(SkyCoord(galaxies['ra'][best_gal]*u.deg, galaxies['dec'][best_gal]*u.deg)).arcsec
        print(f"Separation from Pcc host: {sep_Pcc:.2f}\".")
        row['separation_from_Pcc'] = sep_Pcc
    else:
        print("WARNING: unlikely that any of these galaxies hosted this SN! Setting best galaxy as second-best.")
        print(f"P(no host observed in DECALS cone): {none_prob:.4e}, P(host observed):{any_prob:.4e}")
        row['prob_host_flag'] = 5
        row['prob_host_ls_id'] = np.nan
        row['prob_host_z_best_mean'] = np.nan
        row['prob_host_z_best_std'] = np.nan
        row['prob_host_ra'] = np.nan
        row['prob_host_dec'] = np.nan
        row['prob_host_score'] = np.nan
        if galaxies:
            row['prob_host_2_z_best_mean'] = galaxies['z_best_mean'][best_gal]
            row['prob_host_2_z_best_std'] = galaxies['z_best_std'][best_gal]
            row['prob_host_2_ra'] = galaxies['ra'][best_gal]
            row['prob_host_2_dec'] = galaxies['dec'][best_gal]
            row['prob_host_2_score'] = post_probs[best_gal]
        #best_gal = None

        #comparison to Pcc
        if (row['host_Pcc'] > 0.1):
            print("Oh no! Pcc gets a host here.")
        else:
            print("AGREEMENT: No host found by Pcc.")

    end_time = time.time()
    match_time = end_time - start_time
    print(f"Completed in {match_time:.2f} seconds.")

    if row.host_Pcc > 0.1:
        Pcc_host_ra = float(row.host_ra)
        Pcc_host_dec = float(row.host_dec)
    else:
        Pcc_host_ra = Pcc_host_dec = None
    if (glade_match or decals_match or panstarrs_match):
        chosen_gal_ra = [galaxies['ra'][best_gal]]
        chosen_gal_dec = [galaxies['dec'][best_gal]]
        if (second_best_gal is not None) and (second_best_gal < len(galaxies)):
            chosen_gal_ra.append(galaxies['ra'][second_best_gal])
            chosen_gal_dec.append(galaxies['dec'][second_best_gal])
    else:
        chosen_gal_ra = chosen_gal_dec = []

    agreeStr = ""
    matchStr = ""
    if row['separation_from_Pcc'] < 2:
        agreeStr += "_disagree"
    if glade_match:
        matchStr += "_GLADE"
    elif decals_match:
        matchStr += "_DECaLS"
    elif panstarrs_match:
        matchStr += "_PS2"
    try:
        plotSNhost(chosen_gal_ra, chosen_gal_dec, Pcc_host_ra, Pcc_host_dec,
            galaxies['z_best_mean'][best_gal], galaxies['z_best_std'][best_gal],
            sn_position.ra.deg, sn_position.dec.deg, row.object_name, row.Redshift, row['prob_host_flag'], f"./plots_likelihoodoffsetscale1_wGLADE/DELIGHT/{row.object_name}{agreeStr}{matchStr}")
    except:
        try:
            print("Failed to plot match. Trying again after 60s...")
            time.sleep(60)
            plotSNhost(chosen_gal_ra, chosen_gal_dec, Pcc_host_ra, Pcc_host_dec,
            galaxies['z_best_mean'][best_gal], galaxies['z_best_std'][best_gal],
            sn_position.ra.deg, sn_position.dec.deg, row.object_name, row.Redshift, row['prob_host_flag'],
            f"./plots_likelihoodoffsetscale1_wGLADE/DELIGHT/{row.object_name}{agreeStr}{matchStr}")
        except:
            print("Couldn't get image....")

    #stats_done = sn_catalog['agreement_w_Pcc'].dropna()
    #agree_frac = np.nansum(stats_done)/len(stats_done)
    #print(f"Current agreement fraction: {agree_frac:.2f}")
    row['prob_association_time'] = match_time
    return idx, row

if __name__ == "__main__":
    source = "DELIGHT"
    #source = "Jones+18"

    save = False

    with open('/Users/alexgagliano/Documents/Research/prob_association/data/all.pkl', 'rb') as f:
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

    sn_catalog['separation_from_Pcc'] = np.nan

    #debugging with just the ones we got wrong
    #redoSet = [fn.split("/")[-1].split("_")[0] for fn in glob.glob("/Users/alexgagliano/Documents/Research/prob_association/plots_likelihoodoffsetscale1_wGLADE/DELIGHT/*_disagree.png")]
    redoSet = ['SN 1968V'] #only look at one
    sn_catalog = sn_catalog[sn_catalog['object_name'].isin(redoSet)]

    #randomly shuffle
    verbose = True
    save = False
    sn_catalog = sn_catalog.sample(frac=1)

    angular_scale = 0.258 #arcsec/pixel
    n_samples = 1000
    n_processes = max(1, os.cpu_count() - 1)  # Limit to one less than the number of CPUs

    GLADE_catalog = pd.read_csv("/Users/alexgagliano/Documents/Research/prob_association/data/GLADE+_HyperLedaSizes.csv", delimiter='\t',na_values=['     ', np.nan],keep_default_na=True,low_memory=False)
    GLADE_catalog.dropna(subset=['Bmag', 'logd25Hyp'], inplace=True)
    GLADE_catalog['e_Bmag'] = 0.05*GLADE_catalog['Bmag']

    cat = GalaxyCatalog(name='glade', data=GLADE_catalog)
    pos = SkyCoord(185.7288697*u.deg, 15.8235796*u.deg)
    transient = Transient(name='SN2020oi', position=pos, redshift=0.0052)
    search_rad = Angle(300*u.arcsec)
    cat.get_sources(search_rad, pos)

    # define priors for properties
    transient.set_prior('redshift', halfnorm(loc=0, scale=0.1))
    transient.set_prior('offset', uniform(loc=0, scale=5))
    transient.set_prior('absmag', uniform(loc=-25, scale=15))

    transient.set_likelihood('offset', truncexpon(loc=0, scale=1, b=100))
    transient.set_likelihood('absmag', SNRate_absmag(a=-23, b=-10))
    cat = transient.associate(cat)

    # Create a list of tasks (one per SN)
    tasks = [(idx, row, GLADE_catalog, n_samples, n_processes, angular_scale, verbose, priorfunc_z, priorfunc_offset, priorfunc_absmag) for idx, row in sn_catalog.iterrows()]

    # Run the association tasks in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(associate_supernova, tasks)
        pool.close()
        pool.join()  # Ensures that all resources are released

    # Update the sn_catalog with the results
    for idx, updated_row in results:
        # Update the row at the specific index directly
        sn_catalog.loc[idx] = updated_row

    if (np.nansum(sn_catalog['separation_from_Pcc'] == sn_catalog['separation_from_Pcc']) > 0) and (save):
        #sn_catalog.to_csv("/Users/alexgagliano/Documents/Research/prob_association/slsn_catalog_Alexprob_PccCompare.csv",index=False)
        #sn_catalog.to_csv("/Users/alexgagliano/Desktop/prob_association/ZTFBTS_Alexprob.csv",index=False)
        sn_catalog_done = pd.concat([sn_catalog_done, sn_catalog], ignore_index=True)

        sn_catalog_done.to_csv("/Users/alexgagliano/Desktop/prob_association/DELIGHT_Alexprob.csv",index=False)
    print("Complete.")
