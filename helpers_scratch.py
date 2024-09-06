
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
