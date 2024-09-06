import os
import sys
from astro_ghost.PS1QueryFunctions import getAllPostageStamps
from astro_ghost.TNSQueryFunctions import getTNSSpectra
from astro_ghost.NEDQueryFunctions import getNEDSpectra
from astro_ghost.ghostHelperFunctions import getTransientHosts, getGHOST
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from datetime import datetime
from astroquery.vizier import Vizier
import glob
import astropy
import numpy as np
import pickle
from astropy.io import fits, ascii
from astro_ghost.PS1QueryFunctions import ps1cone
import time

verbose=1

getGHOST()
import astro_ghost


with open('/Users/alexgagliano/Documents/Research/prob_association/data/all.pkl', 'rb') as f:
    data = pickle.load(f)
data.reset_index(inplace=True)
sn_catalog = data


sn_catalog['Redshift'] = np.nan # :(
sn_catalog['host_Pcc'] = 1.0
sn_catalog.rename(columns={'meanra':'RA_deg', 'meandec':'DEC_deg', 'oid':'object_name'}, inplace=True)

sn_catalog['prob_host_ra'] = np.nan
sn_catalog['prob_host_dec'] = np.nan
sn_catalog['prob_association_time'] = np.nan

sn_catalog['agreement_w_Pcc'] = np.nan

sn_catalog.columns.values

for idx, row in sn_catalog.iterrows():
    try:
        start = time.time()
        host_data = getTransientHosts(
            transientCoord=[SkyCoord(row.RA_deg*u.deg, row.DEC_deg*u.deg)],
            transientName=[row.object_name],
            verbose=0,
            starcut="gentle",
            ascentMatch=False,
        )
        end = time.time()
        elapsed = end - start
    except ConnectionError:
        time.sleep(60)
        continue
    if len(host_data) < 1:
        continue
    sn_catalog.at[idx, 'prob_host_ra'] = host_data['raMean'].values
    sn_catalog.at[idx, 'prob_host_dec'] = host_data['decMean'].values
    sn_catalog.at[idx, 'prob_association_time'] = elapsed
    sep_Pcc = SkyCoord(row['host_ra']*u.deg, row['host_dec']*u.deg).separation(SkyCoord(host_data['raMean'].values*u.deg, host_data['decMean'].values*u.deg)).arcsec
    sep_Pcc = sep_Pcc[0]
    print(f"Separation from Pcc host: {sep_Pcc:.2f}\".")
    sn_catalog.at[idx, 'sep_from_Pcc'] = sep_Pcc
    if sep_Pcc < 2:
        sn_catalog.at[idx, 'agreement_w_Pcc'] = True
    print(f"Association time: {elapsed:.2f}s.")
    #save periodically
    if idx%50 == 0:
        print("Checkpoint, saving....")
        sn_catalog.to_csv("/Users/alexgagliano/Documents/Research/prob_association/DELIGHT_ghostMatch.csv",index=False)
    #cleanup
    fns = glob.glob("/Users/alexgagliano/Documents/Research/prob_association/GHOST/transient_20240904*")
    fn2 = glob.glob("/Users/alexgagliano/Documents/Research/prob_association/transient_20240904*")
    fns = np.concatenate([fns, fn2])
    for fn in fns:
        os.rm(fn)
