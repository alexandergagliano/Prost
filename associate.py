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
import requests

def associate_transient(idx, row, GLADE_catalog, n_samples, verbose, priorfunc_z, priorfunc_offset, priorfunc_absmag, likefunc_offset, likefunc_absmag, catalogs=['glade', 'decals', 'panstarrs']):
    try:
        transient = Transient(name=row['name'], position=SkyCoord(row.ra*u.deg, row.dec*u.deg), redshift=float(row.redshift), n_samples=n_samples)
    except:
        transient = Transient(name=row['name'], position=SkyCoord(row.ra*u.deg, row.dec*u.deg), n_samples=n_samples)

    if verbose > 0:
        print(f"Associating for {transient.name} at RA, DEC = {transient.position.ra.deg:.6f}, {transient.position.dec.deg:.6f}")

    transient.set_prior('redshift', priorfunc_z)
    transient.set_prior('offset', priorfunc_offset)
    transient.set_prior('absmag', priorfunc_absmag)

    transient.set_likelihood('offset', likefunc_offset)
    transient.set_likelihood('absmag', likefunc_absmag)

    best_prob, best_ra, best_dec, query_time = np.nan, np.nan, np.nan, np.nan # Default values when no good host is found
    best_cat = ''

    for cat_name in catalogs:
        cat = GalaxyCatalog(name=cat_name, n_samples=n_samples, data=GLADE_catalog)

        try:
            cat.get_candidates(transient, timequery=True, verbose=verbose)
        except requests.exceptions.HTTPError:
            print(f"Candidate retrieval failed for {transient.name} in catalog {cat_name}.")
            continue

        if cat.ngals > 0:
            cat = transient.associate(cat, verbose=verbose)

            if transient.best_host != -1:
                best_idx = transient.best_host
                second_best_idx = transient.second_best_host

                if verbose >= 2:
                    print_cols = ['objID','z_prob', 'offset_prob', 'absmag_prob', 'total_prob', 'ra', 'dec', 'offset_arcsec','z_best_mean','z_best_std']
                    print("Properties of best host:")
                    for key in print_cols:
                        print(key)
                        print(cat.galaxies[key][best_idx])

                    print("Properties of second best host:")
                    for key in print_cols:
                        print(key)
                        print(cat.galaxies[key][second_best_idx])
                best_objID = cat.galaxies['objID'][best_idx]
                best_prob = cat.galaxies['total_prob'][best_idx]
                best_ra = cat.galaxies['ra'][best_idx]
                best_dec = cat.galaxies['dec'][best_idx]

                second_best_objID = cat.galaxies['objID'][second_best_idx]
                second_best_prob = cat.galaxies['total_prob'][second_best_idx]
                second_best_ra = cat.galaxies['ra'][second_best_idx]
                second_best_dec = cat.galaxies['dec'][second_best_idx]

                best_cat = cat_name
                query_time = cat.query_time

                if verbose > 0:
                    print(f"Found a good host in {cat_name}!")
                    print(f"Chosen galaxy has catalog ID of {best_objID} and RA, DEC = {best_ra:.6f}, {best_dec:.6f}")
                    try:
                        plotmatch([best_ra], [best_dec], None, None,
                            cat.galaxies['z_best_mean'][best_idx], cat.galaxies['z_best_std'][best_idx],
                            transient.position.ra.deg, transient.position.dec.deg, transient.name, transient.redshift, 0, f"{transient.name}_{cat_name}")
                    except:
                        print("Couldn't get an image. Waiting 60s before moving on.")
                        time.sleep(60)
                        continue

    if (transient.best_host == -1) and (verbose > 0):
        print("No good host found!")
    return idx, best_prob, best_ra, best_dec, query_time, best_cat


def prepare_catalog(transient_catalog, transient_name_col='name', transient_coord_cols=('ra', 'dec'), debug_names=[], debug=False):
    association_fields = ['prob_host_ra', 'prob_host_dec','prob_host_score','prob_host_2_ra', 'prob_host_2_dec', 'prob_host_2_score',
        'sn_ra_deg', 'sn_dec_deg', 'prob_association_time', 'separation_from_Pcc']

    for field in association_fields:
        transient_catalog[field] = np.nan

    transient_catalog['prob_host_flag'] = 0

    #debugging with just the ones we got wrong
    if debug and len(debug_names) > 0:
        transient_catalog = transient_catalog[transient_catalog[transient_name_col].isin(debug_names)]

    #convert coords if needed
    if ':' in transient_catalog[transient_coord_cols[0]].values[0]:
        ra = transient_catalog[transient_coord_cols[0]].values
        dec = transient_catalog[transient_coord_cols[1]].values
        transient_coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    else:
        #try parsing as a float
        try:
            ra = transient_catalog[transient_coord_cols[0]].astype("float").values
            dec = transient_catalog[transient_coord_cols[1]].astype("float").values
            transient_coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        except exception as e:
            raise ValueError("ERROR: I could not understand your provided coordinates.")

    transient_catalog['ra'] = transient_coords.ra.deg
    transient_catalog['dec'] = transient_coords.dec.deg

    transient_catalog.rename(columns={transient_name_col:'name'}, inplace=True)

    #randomly shuffle
    transient_catalog = transient_catalog.sample(frac=1).reset_index(drop=True)
    transient_catalog.reset_index(inplace=True, drop=True)

    return transient_catalog

def associate_sample(transient_catalog, priors=None, likes=None, catalogs=['glade', 'decals', 'panstarrs'], n_samples=1000, verbose=False, parallel=True, save=True):
    for key in ['offset', 'absmag', 'z']:
        if key not in priors.keys():
            raise ValueError(f"ERROR: Please set a prior function for {key}.")
        elif (key not in likes.keys()) and (key != 'z'):
            raise ValueError(f"ERROR: Please set a likelihood function for {key}.")

    #always load GLADE -- we now use it for spec-zs.
    try:
        GLADE_catalog = pd.read_csv("/Users/alexgagliano/Documents/Research/prob_association/data/GLADE+_HyperLedaSizes_mod_withz.csv")
    except:
        GLADE_catalog = None

    #foundHost = np.argmin(SkyCoord(GLADE_catalog['RAJ2000'].values, GLADE_catalog['DEJ2000'], unit=(u.deg, u.deg)).separation(SkyCoord(185.7288746, 15.8223044, unit=(u.deg, u.deg))).arcsec)
    #GLADE_catalog.to_csv("/Users/alexgagliano/Documents/Research/prob_association/data/GLADE+_HyperLedaSizes_mod_withz.csv",index=False)

    #unpack priors and likelihoods
    priorfunc_z = priors['z']
    priorfunc_offset = priors['offset']
    priorfunc_absmag = priors['absmag']

    likefunc_offset = likes['offset']
    likefunc_absmag = likes['absmag']

    if parallel:
        n_processes = os.cpu_count() - 5

        # Create a list of tasks (one per transient)
        print("parallelizing...")
        tasks = [(idx, row, GLADE_catalog, n_samples, verbose, priorfunc_z, priorfunc_offset, priorfunc_absmag, likefunc_offset, likefunc_absmag, catalogs)
                 for idx, row in transient_catalog.iterrows()]

        # Run the association tasks in parallel
        with Pool(processes=n_processes) as pool:
            results = pool.starmap(associate_transient, tasks)
            pool.close()
            pool.join()  # Ensures that all resources are released
    else:
        results = []
        for idx, row in transient_catalog.iterrows():
            event = (idx, row, GLADE_catalog, n_samples, verbose, priorfunc_z, priorfunc_offset, priorfunc_absmag, likefunc_offset, likefunc_absmag, catalogs)
            results.append(associate_transient(*event))
    # Update transient_catalog with results
    for result in results:
        idx, best_prob, best_ra, best_dec, query_time, best_cat = result
        transient_catalog.at[idx, 'prob_host_ra'] = best_ra
        transient_catalog.at[idx, 'prob_host_dec'] = best_dec
        transient_catalog.at[idx, 'prob_host_score'] = best_prob
        transient_catalog.at[idx, 'prob_query_time'] = query_time
        transient_catalog.at[idx, 'prob_best_cat'] = best_cat
    print("Association of all transients is complete.")

    # Save the updated catalog
    if save:
        ts = int(time.time())
        transient_catalog.to_csv(f"updated_transient_catalog_{ts}.csv", index=False)
    else:
        return transient_catalog

if __name__ == "__main__":
    #source = "DELIGHT"
    #source = "Jones+18"
    source = 'ZTF BTS'

    #with open('/Users/alexgagliano/Documents/Research/prob_association/data/all.pkl', 'rb') as f:
    #    transient_catalog = pickle.load(f)
    transient_catalog = pd.read_csv("/Users/alexgagliano/Documents/Research/multimodal-supernovae/data/ZTFBTS/ZTFBTS_TransientTable.csv")

    # define priors for properties
    priorfunc_z = halfnorm(loc=0.0001, scale=0.5)
    #priorfunc_z = prior_z_observed_transients(z_min=0, z_max=1, mag_cutoff=19, Mmean=-19, Mmin=-24, Mmax=-17)
    #%matplotlib inline
    #priorfunc_z.plot()
    priorfunc_offset = uniform(loc=0, scale=10)
    priorfunc_absmag = uniform(loc=-30, scale=20)

    likefunc_offset = st.gamma(a=0.75) #truncexpon(loc=0, scale=1, b=10)
    likefunc_absmag = SNRate_absmag(a=-30, b=-10)

    priors = {'offset':priorfunc_offset, 'absmag':priorfunc_absmag, 'z':priorfunc_z}
    likes = {'offset':likefunc_offset, 'absmag':likefunc_absmag}

    #set up properties of the run
    verbose = 0
    parallel = True
    save = False
    debug = False
    catalogs = ['panstarrs'] #options are (in order) GLADE, decals, panstarrs

    #alreadyMatched = pd.read_csv("/Users/alexgagliano/Documents/Research/prob_association/updated_sn_catalog_ZTFBTS_1727848419_N500.csv")
    #debug_names = alreadyMatched['name'].values
    #debug_names = ['ZTF20aaelulu']#['ZTF23aaubzzz', 'ZTF19aaklbok', 'ZTF19aapszzy', 'ZTF22aaoolua']
    transient_coord_cols = ("RA", "Dec") #the name of the coord columns in the dataframe
    transient_name_col = 'ZTFID'

    transient_catalog = prepare_catalog(transient_catalog, transient_name_col=transient_name_col, transient_coord_cols=transient_coord_cols, debug=debug)#, debug_names=debug_names)
    transient_catalog = associate_sample(transient_catalog, priors=priors, likes=likes, catalogs=catalogs, parallel=parallel, verbose=verbose, save=save)


    df1 = pd.read_csv("/Users/alexgagliano/Documents/Research/prob_association/updated_transient_catalog_1728420461.csv")
    df2 = pd.read_csv("/Users/alexgagliano/Documents/Research/prob_association/updated_sn_catalog_ZTFBTS_1727848419_N500.csv")
