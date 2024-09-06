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

def associate_sample(idx, row, GLADE_catalog, n_samples, verbose, priorfunc_z, priorfunc_offset, priorfunc_absmag, likefunc_offset, likefunc_absmag, catalogs=['glade', 'decals', 'panstarrs']):
    try:
        transient = Transient(name=row.name, position=SkyCoord(row.ra*u.deg, row.dec*u.deg), redshift=row.redshift, n_samples=n_samples)
    except:
        transient = Transient(name=row.name, position=SkyCoord(row.ra*u.deg, row.dec*u.deg), n_samples=n_samples)


    transient.set_prior('redshift', priorfunc_z)
    transient.set_prior('offset', priorfunc_offset)
    transient.set_prior('absmag', priorfunc_absmag)

    transient.set_likelihood('offset', likefunc_offset)
    transient.set_likelihood('absmag', likefunc_absmag)

    for cat_name in catalogs:
        if cat_name == 'glade':
            cat = GalaxyCatalog(name=cat_name, data=GLADE_catalog, n_samples=n_samples)
        else:
            cat = GalaxyCatalog(name=cat_name, data=GLADE_catalog, n_samples=n_samples)

        cat.get_candidates(transient)
        if cat.ngals > 0:
            cat = transient.associate(cat)
            if transient.best_host != -1:
                print(f"Found a good host in {cat_name}!")
                print(f"{transient.name}:")
                print(transient.position)
                best_prob = cat.galaxies['total_prob'][transient.best_host]
                best_ra = cat.galaxies['ra'][transient.best_host]
                best_dec = cat.galaxies['dec'][transient.best_host]
                print("Best host:")
                print(SkyCoord(best_ra*u.deg, best_dec*u.deg))
                #diagnose_ranking should go here
                #if verbose:
                #    diagnose_ranking(-1, post_probs, galaxies, post_offset, post_z, post_absmag, np.arange(len(galaxies)), sn_redshift, sn_position, verbose=True)
                return best_prob, best_ra, best_dec
    if transient.best_host != -1:
        print("No good hosts found...")
    return

if __name__ == "__main__":
    source = "DELIGHT"
    #source = "Jones+18"

    with open('/Users/alexgagliano/Documents/Research/prob_association/data/all.pkl', 'rb') as f:
        sn_catalog = pickle.load(f)
    sn_catalog.reset_index(inplace=True)

    association_fields = ['prob_host_ra', 'prob_host_dec','prob_host_score','prob_host_2_ra', 'prob_host_2_dec', 'prob_host_2_score',
        'sn_ra_deg', 'sn_dec_deg', 'prob_association_time', 'separation_from_Pcc']
    for field in association_fields:
        sn_catalog[field] = np.nan
    sn_catalog.rename(columns={'meanra':'ra', 'meandec':'dec'}, inplace=True)

    sn_catalog['prob_host_flag'] = 0

    #debugging with just the ones we got wrong
    #redoSet = [fn.split("/")[-1].split("_")[0] for fn in glob.glob("/Users/alexgagliano/Documents/Research/prob_association/plots_likelihoodoffsetscale1_wGLADE/DELIGHT/*_disagree.png")]
    #redoSet = ['SN 1968V'] #only look at one
    #sn_catalog = sn_catalog[sn_catalog['object_name'].isin(redoSet)]

    #randomly shuffle
    sn_catalog = sn_catalog.sample(frac=1)

    verbose = True
    save = False
    n_samples = 1000
    n_processes = max(1, os.cpu_count() - 1)  # Limit to one less than the number of CPUs

    GLADE_catalog = pd.read_csv("/Users/alexgagliano/Documents/Research/prob_association/data/GLADE+_HyperLedaSizes_mod.csv")

    # define priors for properties
    priorfunc_z = halfnorm(loc=0, scale=0.5)
    priorfunc_offset = uniform(loc=0, scale=5)
    priorfunc_absmag = uniform(loc=-25, scale=15)

    likefunc_offset = truncexpon(loc=0, scale=1, b=100)
    likefunc_absmag = SNRate_absmag(a=-23, b=-10)

    # Create a list of tasks (one per transient)
    tasks = [(idx, row, GLADE_catalog, n_samples, verbose, priorfunc_z, priorfunc_offset, priorfunc_absmag, likefunc_offset, likefunc_absmag)
             for idx, row in sn_catalog.iterrows()]

    # Run the association tasks in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(associate_sample, tasks)
        pool.close()
        pool.join()  # Ensures that all resources are released

    print("Association of all transients is complete.")

    #TODO -- add information back to catalog
