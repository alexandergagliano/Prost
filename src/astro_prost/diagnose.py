import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import os
import re

from astropy.io import fits
from astropy.table import Table
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS

def get_images(ra, dec, size=240, filters="grizy", type="stack"):
    """Query ps1filenames.py service to get a list of images.

    Parameters
    ----------
    ra : float
        Right ascension of position, in decimal degrees.
    dec : float
        Declination of position, in decimal degrees.
    size : int
        The image size in pixels (0.25 arcsec/pixel).
    filters : str
        A string with the filters to include.

    Returns
    -------
    astropy.Table
        The results of the search for relevant images.
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits" "&filters={filters}&type={type}").format(
        **locals()
    )
    table = Table.read(url, format="ascii")
    return table


def get_url(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False, type="stack"):
    """Get the URL for images in the table.

    Parameters
    ----------

    ra : float
        Right ascension of position, in degrees.
    dec : float
        Declination of position, in degrees.
    size : int
        The extracted image size in pixels (0.25 arcsec/pixel)
    output_size : int
        output (display) image size in pixels (default = size).
        The output_size has no effect for fits format images.
    filters : str
        The string with filters to include.
    format : str
        The data format (options are \\"jpg\\", \\"png" or \\"fits\\").
    color : boolean
        If True, creates a color image (only for jpg or png format).
        If False, return a list of URLs for single-filter grayscale images.

    Returns
    -------
    url
        The url for the image to download.
    """

    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = get_images(ra, dec, size=size, filters=filters, type=type)
    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?" "ra={ra}&dec={dec}&size={size}&format={format}"
    ).format(**locals())
    if output_size:
        url = url + f"&output_size={output_size}"

    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table["filter"]]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table) // 2, len(table) - 1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table["filename"][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table["filename"]:
            url.append(urlbase + filename)
    return url


def get_ps1_pic(path, objid, ra, dec, size, band, safe=False, save=False):
    """Downloads PS1 picture (in fits) centered at a given location.

    Parameters
    ----------

    path : str
        The filepath where the fits file will be saved.
    objid : int
        The PS1 objid of the object of interest (to save as filename).
    ra : float
        Right ascension of position, in degrees.
    dec : float
        Declination of position, in degrees.
    size : int
        The extracted image size in pixels (0.25 arcsec/pixel)
    band : str
        The PS1 band.
    safe : boolean
        If True, include the objid of the object of interest in the filename
        (useful when saving multiple files at similar positions).

    Returns
    -------
    fn : fits file
        File retrieved from ps1
    """

    fitsurl = get_url(ra, dec, size=size, filters=f"{band}", format="fits")[0]

    with fits.open(fitsurl) as fn:
        if save:
            filename = f"/PS1_{objid}_{int(size*0.25)}arcsec_{band}.fits" if safe else \
                       f"/PS1_ra={ra}_dec={dec}_{int(size*0.25)}arcsec_{band}.fits"
            fn.writeto(path + filename, overwrite=True)
        else:
            return fn

def find_all(name, path):
    """Crawls through a directory and all its sub-directories looking for a file matching
       'name'. If found, it is returned.

    Parameters
    ----------

    name : str
        The filename for which to search.
    path : str
        the absolute path to search within.

    Returns
    -------
    result : list
        List of absolute paths to all files called 'name' in 'path'.

    """

    result = []
    for root, _, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def plot_match(
    host_ra,
    host_dec,
    host_redshift_mean,
    host_redshift_std,
    transient_ra,
    transient_dec,
    transient_name,
    transient_redshift,
    bayesflag,
    fn,
    logger,
    true_host_ra=None,
    true_host_dec=None,
):
    """Plots a host-galaxy match with panstarrs postage stamp.

    Parameters
    ----------
    host_ra : list
        List of right ascension coordinates for associated hosts, in decimal degrees.
        (Can provide up to 10 hosts)
    host_dec : list
        List of declination coordinates for associated hosts, in decimal degrees.
    host_redshift_mean : float
        Point estimate of host-galaxy redshift.
    host_redshift_std : float
        Error on host_redshift_mean.
    transient_ra : float
        Right ascension of transient, in decimal degrees.
    transient_dec : float
        Declination of transient, in decimal degrees.
    transient_name : str
        Name of transient for plot.
    transient_redshift : float
        Redshift of transient, if available.
    bayesflag : int
        Flag from association run. If 0, bayes factor is sufficient for confident association.
        If 1, bayes factor is weak. If 2, bayes factor is strong.
    fn : str
        Name of saved image.
    logger : logging.Logger
        Logger instance for messages to console or output file.
    true_host_ra : float
        Right ascension of the true galaxy in decimal degrees.
    true_host_dec : float
        Declination of the true galaxy in decimal degrees.
    """

    #sanitize transient name for fn
    fn = re.sub(r"[\s]", "", fn)

    cols = np.array(
        [
            "#ff9f1c",
            "#2cda9d",
            "#f15946",
            "#da80dd",
            "#f4e76e",
            "#b87d4b",
            "#ff928b",
            "#c73e1d",
            "#58b09c",
            "#e7e08b",
        ]
    )
    bands = "zrg"
    if len(host_ra) > 0:
        sep = np.nanmax(
            SkyCoord(host_ra * u.deg, host_dec * u.deg)
            .separation(SkyCoord(transient_ra * u.deg, transient_dec * u.deg))
            .arcsec
        )
    else:
        sep = 0
    if true_host_ra is not None:
        sep_true = (
            SkyCoord(true_host_ra * u.deg, true_host_dec * u.deg)
            .separation(SkyCoord(transient_ra * u.deg, transient_dec * u.deg))
            .arcsec
        )
        if (true_host_ra) and (true_host_dec) and (sep_true > sep):
            sep = sep_true
    rad = np.nanmax([30.0, 2 * sep])
    logger.info(f"Getting {rad*4}'' x {rad*4}'' Pan-STARRS image of the field...")
    pic_data = []
    for band in bands:
        get_ps1_pic("./", None, transient_ra, transient_dec, int(rad * 4), band, save=True)
        a = find_all(f"PS1_ra={transient_ra}_dec={transient_dec}_{int(rad)}arcsec_{band}.fits", ".")
        if not a:
            raise FileNotFoundError(f"FITS file not found for RA={transient_ra}, DEC={transient_dec}, radius={int(rad)}, band={band}")
        else:
            pixels = fits.open(a[0])[0].data
            pixels = pixels.astype("float32")
            # normalize to the range 0-255
            pixels *= 255 / np.nanmax(pixels)
            pic_data.append(pixels)
            hdu = fits.open(a[0])[0]
            os.remove(a[0])

    lo_val, up_val = np.nanpercentile(
        np.array(pic_data).ravel(), (0.5, 99.5)
    )  # Get the value of lower and upper 0.5% of all pixels
    stretch_val = up_val - lo_val

    rgb_default = make_lupton_rgb(
        pic_data[0], pic_data[1], pic_data[2], minimum=lo_val, stretch=stretch_val, Q=0
    )
    wcs = WCS(hdu.header)
    plt.figure(num=None, figsize=(12, 8), facecolor="w", edgecolor="k")
    ax = plt.subplot(projection=wcs)
    ax.set_xlabel("RA", fontsize=24)
    ax.set_ylabel("DEC", fontsize=24)

    # really emphasize the supernova location
    plt.axvline(x=int(rad * 2), c="tab:red", alpha=0.5, lw=2)
    plt.axhline(y=int(rad * 2), c="tab:red", alpha=0.5, lw=2)

    if true_host_ra and true_host_dec:
        true_str = ""
        ax.scatter(
            true_host_ra,
            true_host_dec,
            transform=ax.get_transform("fk5"),
            marker="+",
            alpha=0.8,
            lw=2,
            s=200,
            color="magenta",
            zorder=100,
        )
    else:
        true_str = "(no true)"
    bayesstr = ". "
    if bayesflag == 2:
        bayesstr += "Strong match!"
        # don't plot the second-best host
        host_ra = host_ra[:1]
        host_dec = host_dec[:1]
    elif bayesflag == 1:
        bayesstr += "Weak match."
    if host_ra and host_dec:
        for i in np.arange(len(host_ra)):
            # print(f"Plotting host {i}")
            ax.scatter(
                host_ra[i],
                host_dec[i],
                transform=ax.get_transform("fk5"),
                marker="o",
                alpha=0.8,
                lw=2,
                s=100,
                edgecolor="k",
                facecolor=cols[i],
                zorder=100,
            )
        if transient_redshift == transient_redshift:
            plt.title(
                f"{transient_name}, z={transient_redshift:.4f}; Host Match,"
                f"z={host_redshift_mean:.4f}+/-{host_redshift_std:.4f} {true_str}{bayesstr}"
            )
        else:
            plt.title(
                f"{transient_name}, no z; Host Match, "
                f"z={host_redshift_mean:.4f}+/-{host_redshift_std:.4f} {true_str}{bayesstr}"
            )
    else:
        if transient_redshift == transient_redshift:
            plt.title(f"{transient_name}, z={transient_redshift:.4f}; No host found {true_str}")
        else:
            plt.title(f"{transient_name}, no z; No host found {true_str}")
    ax.imshow(rgb_default, origin="lower")
    plt.axis("off")
    plt.savefig(f"{fn}.png", bbox_inches="tight")
    plt.close()


# Function to diagnose the discrepancy when the top-ranked galaxy is not the true host
def diagnose_ranking(
    true_index,
    post_probs,
    galaxy_catalog,
    post_offset,
    post_z,
    post_absmag,
    galaxy_ids,
    z_sn,
    transient_position,
    logger,
    post_offset_true=None,
    post_z_true=None,
    post_absmag_true=None,
):
    """Prints summary statistics from association run.

    Parameters
    ----------
    true_index : int
        Index of the true host in galaxy_ids.
    post_probs : list
        List of total posterior probabilities for candidate hosts.
    galaxy_catalog : str
        Galaxy catalog used for association.
    post_offset : list
        Posterior probabilities for fractional offset.
    post_z : list
        Posterior probabilities for redshift.
    post_absmag : list
        Posterior probabilities for absolute magnitude.
    galaxy_ids : list
        List of catalog ids for candidate hosts.
    z_sn : float
        Redshift of the transient.
    transient_position : astropy.coordinates SkyCoord
        Position of the associated transient.
    logger : logging.Logger
        Logger instance for messages to console or output file.
    post_offset_true : float
        Posterior probability for true host's fractional offset.
    post_z_true : float
        Posterior probability for true host's redshift.
    post_absmag_true : float
        Posterior probability for true host's absolute magnitude.

    Returns
    -------
    true_rank : int
        Rank of the true host galaxy in final association.
    post_probs[true_index] : float
        Total posterior probability for true host
    """
    top_indices = np.argsort(post_probs)[-3:][::-1]  # Top 3 ranked galaxies

    if true_index > 0:
        logger.debug(f"True Galaxy: {true_index + 1}")

        # Check if the true galaxy is in the top 5
        if true_index not in top_indices:
            logger.warning(f"True Galaxy {true_index + 1} is not in the top 5!")

    # Print top 5 and compare with the true galaxy
    for rank, i in enumerate(top_indices, start=1):
        is_true = "(True Galaxy)" if i == true_index and true_index > 0 else ""
        logger.debug(
            f"Rank {rank}: ID {galaxy_ids[top_indices[rank-1]]}"
            f"has a Posterior probability of being the host: {post_probs[i]:.4f} {is_true}"
        )

    # Detailed comparison of the top-ranked and true galaxy
    logger.debug(f"Coords (SN): {transient_position.ra.deg:.4f}, {transient_position.dec.deg:.4f}")
    for _, i in enumerate(top_indices, start=1):
        top_gal = galaxy_catalog[i]
        top_theta = transient_position.separation(
            SkyCoord(ra=top_gal["ra"] * u.degree, dec=top_gal["dec"] * u.degree)
        ).arcsec

        logger.debug(f"Redshift (SN): {z_sn:.4f}")
        logger.debug(f"Top Galaxy (Rank {i}): Coords: {top_gal['ra']:.4f}, {top_gal['dec']:.4f}")
        logger.debug(
            f"\t\t\tRedshift = {top_gal['z_best_mean']:.4f}+/-{top_gal['z_best_std']:.4f},"
            " Angular Size = {top_gal['angular_size_arcsec']:.4f} arcsec"
        )
        logger.debug(f"\t\t\tFractional Sep. = {top_theta/top_gal['angular_size_arcsec']:.4f} host radii")
        logger.debug(f'\t\t\tAngular Sep. ("): {top_theta:.2f}')
        logger.debug(f"\t\t\tRedshift Posterior = {post_z[i]:.4e}," " Offset Posterior = {post_offset[i]:.4e}")
        logger.debug(f"\t\t\tAbsolute mag Posterior = {post_absmag[i]:.4e}")

    if true_index > 0:
        true_gal = galaxy_catalog[true_index]
        true_theta = transient_position.separation(
            SkyCoord(ra=true_gal["ra"] * u.degree, dec=true_gal["dec"] * u.degree)
        ).arcsec

        logger.debug(f"True Galaxy: Fractional Sep. = {true_theta/true_gal['angular_size_arcsec']:.4f} host radii")
        logger.debug(
            f"\t\t\tRedshift = {true_gal['redshift']:.4f}, "
            f"Angular Size = {true_gal['angular_size_arcsec']:.4f}\""
        )
        logger.debug(f"\t\t\tRedshift Posterior = {post_z_true:.4e}, Offset Posterior = {post_offset_true:.4e}")

    if true_index > 0:
        post_offset_true = post_offset[true_index]
        post_z_true = post_z[true_index]

    ranked_indices = np.argsort(post_probs)[::-1]

    # Find the position of the true galaxy's index in the ranked list
    true_rank = np.where(ranked_indices == true_index)[0][0] if true_index > 0 else -1

    # Return the rank (0-based index) of the true galaxy
    return true_rank, post_probs[true_index]
