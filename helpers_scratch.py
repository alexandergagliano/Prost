
def associate_supernova(idx, row, GLADE_catalog, n_samples, n_processes, verbose, priorfunc_offset, priorfunc_absmag):
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

def find_decals_shreds(ra_allgals, dec_allgals, size, e1, e2, appmag):
    #deprecated for now
    dropidxs = []
    for i in np.arange(len(ra_allgals)):
        onegal_ra = ra_allgals[i]
        onegal_dec = dec_allgals[i]
        DLR = calc_DLR_decals(onegal_ra, onegal_dec, np.delete(ra_allgals, i), np.delete(dec_allgals, i), np.delete(e1, i), np.delete(e2, i), np.delete(size, i))
        seps = SkyCoord(onegal_ra*u.deg, onegal_dec*u.deg).separation(SkyCoord(np.delete(ra_allgals, i)*u.deg, np.delete(dec_allgals, i)*u.deg)).arcsec
        min_idx = np.nanargmin(DLR/seps)
        #if within the DLR of another brighter galaxy, remove source with idx i
        if (DLR[min_idx]/seps[min_idx] < 1) and (np.delete(appmag, i)[min_idx] < appmag[i]):
            dropidxs.append(i)
    return np.array(dropidxs)

def find_glade_shreds(ra_allgals, dec_allgals, a_over_b, pa, size, appmag):
    #deprecated for now!

    dropidxs = []

    for i in np.arange(len(ra_allgals)):
        onegal_ra = ra_allgals[i]
        onegal_dec = dec_allgals[i]
        onesize = size[i]
        onemag = appmag[i]

        restgal_ra = np.delete(ra_allgals, i)
        restgal_dec = np.delete(dec_allgals, i)
        restgal_ab = np.delete(a_over_b, i)
        restgal_pa = np.delete(pa, i)
        restgal_size = np.delete(size, i)
        restgal_appmag = np.delete(appmag, i)

        DLR = calc_DLR(onegal_ra, onegal_dec, restgal_ra, restgal_dec, restgal_ab, restgal_pa, restgal_size)
        seps = SkyCoord(onegal_ra*u.deg, onegal_dec*u.deg).separation(SkyCoord(restgal_ra*u.deg, restgal_dec*u.deg)).arcsec
        min_idx = np.nanargmin(seps/DLR)

        #if within the DLR of another galaxy, remove dimmer galaxy
        if min_idx < i:
            original_min_idx = min_idx
        else:
            original_min_idx = min_idx + 1  # Shift by 1 to account for the deletion at index i

        # If within the DLR of another galaxy, remove the dimmer galaxy
        if ((seps/DLR)[min_idx] < 1):
            if restgal_appmag[min_idx] < appmag[i]:
                dropidxs.append(i)  # The current galaxy is dimmer, drop it
            else:
                dropidxs.append(original_min_idx)  # The other galaxy is dimmer, drop it

    return np.array(dropidxs)
