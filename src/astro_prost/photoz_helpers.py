import os
import tarfile
import requests
import numpy as np
#hacky monkey-patch for python 3.8
if not hasattr(np, 'int'):
    np.int = int
import pandas as pd
import requests
import sys

# compatability for python < 3.9
if sys.version_info >= (3, 9):
    from sfdmap2 import sfdmap
else:
    import sfdmap

try:
    import tensorflow as tf
except ImportError:
    tf = None
    print("Warning: Issue importing tensorflow. Please try reinstalling or associate panstarrs sources without conditioning on redshift.")
from astropy.table import Table
from filelock import FileLock
from pathlib import Path
import platform

if platform.system() == "Darwin":  # Only on macOS
    tf.config.set_visible_devices([], 'GPU')  # turn off GPU execution on Macs
tf.config.run_functions_eagerly(False)

default_model_path = "./MLP_lupton.hdf5"
default_dust_path = "."

def convert_flux_to_luptitude(f, b, f_0=3631):
    """Converts flux to luptitude, which gives reasonable magnitude conversions
       even for small and negative values (see Lupton, Gunn, & Szalay, 1999).

    Parameters
    ----------
    f : float or array-like
        Original flux values.
    b : type
        Band-specific luptitude softening parameter.
    f_0 : float
        Zero-point flux in Janskys.

    Returns
    -------
    luptitude : float or array-like
        The converted luptitude values.

    """
    luptitude = -2.5 / np.log(10) * (np.arcsinh((f / f_0) / (2 * b)) + np.log(b))
    return luptitude

def build_sfd_dir(logger, file_path="./sfddata-master.tar.gz", data_dir="./"):
    """Downloads directory of Galactic dust maps for extinction correction.

    Parameters
    ----------
    file_path : str
        Location of Galactic dust maps.
    data_dir : str
        Location to download Galactic dust maps, if file_path doesn't exist.
    logger : logger object
        Logger for printing to console or to file.


    """

    # Define the target directory
    target_dir = os.path.join(data_dir, "sfddata-master")

    # Lock path for concurrency safety
    lock_path = Path(file_path).with_suffix(".lock")

    try:
        # Use FileLock to ensure only one process downloads/extracts at a time
        with FileLock(lock_path):
            # If the dust map directory already exists, no need to proceed
            if os.path.isdir(target_dir):
                logger.info(f"{target_dir} already exists. Skipping extraction.")
                return

            # Download the archive if it doesn't exist
            if not os.path.exists(file_path):
                url = "https://github.com/kbarbary/sfddata/archive/master.tar.gz"
                logger.info(f"Downloading dust map data from {url}...")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    raise ValueError(f"Failed to download the file: {url} - Status code: {response.status_code}")

            # Extract the tarball into the data directory
            logger.info(f"Extracting {file_path}...")
            with tarfile.open(file_path) as tar:
                tar.extractall(data_dir)

            # Remove the tar file after extraction
            if os.path.exists(file_path):
                os.remove(file_path)

    finally:
        # Ensure the lock file is removed in any case (success or failure)
        if os.path.exists(lock_path):
            os.remove(lock_path)

        logger.info("Done creating dust directory.")

def get_photoz_weights(logger, file_path=default_model_path):
    """Get weights for MLP pan-starrs photo-z model.

    Parameters
    ----------
    file_path : str
        Path to MLP model (defaults to './MLP_lupton.hdf5')
    logger : logging.Logger
        Logger instance for messages to console or output file.

    """

    # File lock path
    lock_path = Path(file_path).with_suffix(".lock")

    # Use the lock to ensure only one process downloads the weights file
    with FileLock(lock_path):
        # Check if the photo-z weights file already exists
        if os.path.exists(file_path):
            # Do not print anything here
            return

        # Download the file if it does not exist
        url = "https://uofi.box.com/shared/static/n1yiy818mv5b5riy2h3dg5yk2by3swos.hdf5"
        logger.info(f"Downloading photo-z weights from {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.raw.read())
        else:
            raise ValueError(f"Failed to download the file: {url} - Status code: {response.status_code}")

        logger.info("Done getting photo-z weights.")

def checklegal(table, release):
    """Checks if this combination of table and release is acceptable.
       Raises a ValueError exception if there is problem.

    Parameters
    ----------

    table : str
        Retrieved table type; can be 'mean', 'stack', or 'detection'.
    release : str
        The Pan-STARRS data release. Can be 'dr1' or 'dr2'.

    Raises
    ----------
     Valuerror : Error if table and release combination are invalid.

    """

    releaselist = ("dr1", "dr2")
    if release not in ("dr1", "dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(", ".join(releaselist)))
    tablelist = ("mean", "stack") if release == "dr1" else ("mean", "stack", "detection", "forced_mean")
    if table not in tablelist:
        raise ValueError(
            "Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist))
        )

def preprocess(df, path="../data/sfddata-master/", ebv=True):
    """Preprocesses the data inside pandas.DataFrame object
       returned by serial_objid_search to the space of inputs of our Neural Network.


    Parameters
    ----------

    df : pandas DataFrame
        Dataframe containing the data for each matched objid.
    path : str
        string path to extinction maps data.
    ebv : boolean
        If True, extinction is queried and corrected. False, all extinctions are set to 0.


    Returns
    ----------
    x : array-like
        Preprocessed inputs ready to be used as input to NN

    """
    if ebv:
        m = sfdmap.SFDMap(path)
        assert ("ra" in df.columns.values) and ("dec" in df.columns.values), (
            "DustMap query failed because the expected coordinates didnt"
            "exist in df, likely the match of any Hosts into PanStarrs failed"
        )
        ebv = m.ebv(df["ra"].values.astype(np.float32), df["dec"].values.astype(np.float32))

        df["ebv"] = ebv
    else:
        df["ebv"] = 0.0

    b_g = 1.7058474723241624e-09
    b_r = 4.65521985283191e-09
    b_i = 1.2132217745483221e-08
    b_z = 2.013446972858555e-08
    b_y = 5.0575501316874416e-08

    means = np.array(
        [
            18.70654578,
            17.77948707,
            17.34226094,
            17.1227873,
            16.92087669,
            19.73947441,
            18.89279411,
            18.4077393,
            18.1311733,
            17.64741402,
            19.01595669,
            18.16447837,
            17.73199409,
            17.50486095,
            17.20389615,
            19.07834251,
            18.16996592,
            17.71492073,
            17.44861273,
            17.15508793,
            18.79100201,
            17.89569908,
            17.45774026,
            17.20338482,
            16.93640741,
            18.62759241,
            17.7453392,
            17.31341498,
            17.06194499,
            16.79030564,
            0.02543223,
        ]
    )

    stds = np.array(
        [
            1.7657395,
            1.24853534,
            1.08151972,
            1.03490545,
            0.87252421,
            1.32486758,
            0.9222839,
            0.73701807,
            0.65002723,
            0.41779001,
            1.51554956,
            1.05734494,
            0.89939638,
            0.82754093,
            0.63381611,
            1.48411417,
            1.05425943,
            0.89979008,
            0.83934385,
            0.64990996,
            1.54735158,
            1.10985163,
            0.96460099,
            0.90685922,
            0.74507053,
            1.57813401,
            1.14290345,
            1.00162105,
            0.94634726,
            0.80124359,
            0.01687839,
        ]
    )

    data_columns = [
        "gFKronFlux",
        "rFKronFlux",
        "iFKronFlux",
        "zFKronFlux",
        "yFKronFlux",
        "gFPSFFlux",
        "rFPSFFlux",
        "iFPSFFlux",
        "zFPSFFlux",
        "yFPSFFlux",
        "gFApFlux",
        "rFApFlux",
        "iFApFlux",
        "zFApFlux",
        "yFApFlux",
        "gFmeanflxR5",
        "rFmeanflxR5",
        "iFmeanflxR5",
        "zFmeanflxR5",
        "yFmeanflxR5",
        "gFmeanflxR6",
        "rFmeanflxR6",
        "iFmeanflxR6",
        "zFmeanflxR6",
        "yFmeanflxR6",
        "gFmeanflxR7",
        "rFmeanflxR7",
        "iFmeanflxR7",
        "zFmeanflxR7",
        "yFmeanflxR7",
        "ebv",
    ]

    x = df[data_columns].values.astype(np.float32)
    x[:, 0:30:5] = convert_flux_to_luptitude(x[:, 0:30:5], b=b_g)
    x[:, 1:30:5] = convert_flux_to_luptitude(x[:, 1:30:5], b=b_r)
    x[:, 2:30:5] = convert_flux_to_luptitude(x[:, 2:30:5], b=b_i)
    x[:, 3:30:5] = convert_flux_to_luptitude(x[:, 3:30:5], b=b_z)
    x[:, 4:30:5] = convert_flux_to_luptitude(x[:, 4:30:5], b=b_y)

    x = (x - means) / stds
    x[x > 20] = 20
    x[x < -20] = -20
    x[np.isnan(x)] = -20

    return x

def load_lupton_model(logger, model_path=default_model_path, dust_path=default_dust_path):
    """Helper function that defines and loads the weights of our NN model
        and the output space of the NN.

    Parameters
    ----------

    logger : logger object
        Logger for printing to console or to file.
    model_path : str
        Path to the model weights.
    dust_path : str
        Path to dust map data files.

    Returns
    ----------

    mymodel : tensorflow keras model
        Trained photo-z MLP.
    range_z : array-like
        Array of binned redshift space corresponding to the output space of the NN
    """

    if tf is None:
        raise RuntimeError(
        "TensorFlow is required for photo-z estimation but is corrupted or not installed. "
        "Please reinstall using `pip install tensorflow` or associate again without redshift."
    )
    build_sfd_dir(logger, data_dir=dust_path)
    get_photoz_weights(logger, file_path=model_path)

    def model():
        input = tf.keras.layers.Input(shape=(31,))

        dense1 = tf.keras.layers.Dense(
            256,
            activation=tf.keras.layers.LeakyReLU(),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        )(input)
        drop1 = tf.keras.layers.Dropout(0.05)(dense1)

        dense2 = tf.keras.layers.Dense(
            1024,
            activation=tf.keras.layers.LeakyReLU(),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        )(drop1)
        drop2 = tf.keras.layers.Dropout(0.05)(dense2)

        dense3 = tf.keras.layers.Dense(
            1024,
            activation=tf.keras.layers.LeakyReLU(),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        )(drop2)
        drop3 = tf.keras.layers.Dropout(0.05)(dense3)

        dense4 = tf.keras.layers.Dense(
            1024,
            activation=tf.keras.layers.LeakyReLU(),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        )(drop3)

        output = tf.keras.layers.Dense(360, activation=tf.keras.activations.softmax)(dense4)

        model = tf.keras.Model(input, output)

        return model

    mymodel = model()
    mymodel.load_weights(model_path)

    nb_bins = 360
    zmin = 0.0
    zmax = 1.0
    range_z = np.linspace(zmin, zmax, nb_bins + 1)[:nb_bins]

    return mymodel, range_z

def evaluate(x, mymodel, range_z, verbose=None):
    """Evaluate the MLP for a set of PS1 inputs, and return predictions.

    Parameters
    ----------

    x : array-like
        PS1 properties of associated hosts.
    mymodel : tensorflow Keras model
        MLP model for photo-z estimation.
    range_z : array-like
        Grid over which to evaluate the posterior distribution of photo-zs.

    Returns
    ----------

    posteriors : array-like
        Posterior distributions for the grid of redshifts defined as
        np.linspace(0, 1, n).
    means : array-like
        point estimates of posteriors for each source.
    errors : array-like
        standard deviations of posteriors for each source.
    """

    assert x.shape[1] == 31, f"Expected input shape (*, 31), got {x.shape}"
    posteriors = mymodel.predict(x)
    point_estimates = np.sum(posteriors * range_z, axis=1)
    for i in range(len(posteriors)):
        posteriors[i, :] /= np.sum(posteriors[i, :])
    errors = np.ones(len(posteriors))
    for i in range(len(posteriors)):
        errors[i] = np.std(np.random.choice(a=range_z, size=1000, p=posteriors[i, :], replace=True))
    return posteriors, point_estimates, errors
