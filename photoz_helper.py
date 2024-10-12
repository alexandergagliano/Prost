import pkg_resources
import numpy as np
import requests
#from astro_ghost.PS1QueryFunctions import *

import pandas as pd
import tensorflow as tf

import codecs

from sfdmap2 import sfdmap
import os
import tarfile

DEFAULT_MODEL_PATH = './MLP_lupton.hdf5'
DEFAULT_DUST_PATH = '.'


def build_sfd_dir(file_path='./sfddata-master.tar.gz', data_dir=DEFAULT_DUST_PATH):
    """Downloads directory of Galactic dust maps for extinction correction.
       [Schlegel, Finkbeiner and Davis (1998)](http://adsabs.harvard.edu/abs/1998ApJ...500..525S).

    :param fname: Filename for dustmaps archive file.
    :type fname: str
    :param data_path: Target directory in which to extract 'sfddata-master' directory from archive file.
    :type data_path: str
    """
    target_dir = os.path.join(data_dir, 'sfddata-master')
    if os.path.isdir(target_dir):
        print(f'''Dust map data directory "{target_dir}" already exists.''')
        return
    # Download the data archive file if it is not present
    if not os.path.exists(file_path):
        url = 'https://github.com/kbarbary/sfddata/archive/master.tar.gz'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.raw.read())
    # Extract the data files
    tar = tarfile.open(file_path)
    tar.extractall(data_dir)
    tar.close()
    # Delete archive file
    os.remove(file_path)
    print("Done creating dust directory.")
    return


def get_photoz_weights(file_path=DEFAULT_MODEL_PATH):
    """Get weights for MLP photo-z model.

    :param fname: Filename of saved MLP weights.
    :type fname: str
    """
    if os.path.exists(file_path):
        print(f'''photo-z weights file "{file_path}" already exists.''')
        return
    url = 'https://uofi.box.com/shared/static/n1yiy818mv5b5riy2h3dg5yk2by3swos.hdf5'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.raw.read())
    print("Done getting photo-z weights.")
    return

def ps1objIDsearch(objID,table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do an object lookup by objID.

    :param objID: list of objIDs (or dictionary?)
    :type objID: List of objIDs
    :param table: Can be \\'mean\\', \\'stack\\', or \\'detection\\'.
    :type table: str
    :param release: Can be 'dr1' or 'dr2'.
    :type release: str
    :param format: Can be 'csv', 'votable', or 'json'
    :type format: str
    :param columns: list of column names to include (None means use defaults)
    :type columns: arrray-like
    :param baseurl: base URL for the request
    :type baseurl: str
    :param verbose: print info about request
    :type verbose: bool,optional
    :param \\*\\*kw: other parameters (e.g., 'nDetections.min':2)
    :type \\*\\*kw: dictionary
    """

    #this is a dictionary... we want a list of dictionaries
    objID=list(objID)

    data_list=[kw.copy() for i in range(len(objID))]
    assert len(data_list)==len(objID)

    for i in range(len(data_list)):
        data_list[i]['objID'] = objID[i]

    urls = []
    datas = []
    for i in range(len(objID)):
        data = ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data_list[i])

        #urls.append(url)
        datas.append(data)

    return datas

def fetch_information_serially(url, data, verbose=False, format='csv'):
    """A helper function called by serial_objID_search-- Queries PanStarrs API for data.

    :param url: Remote PS1 url.
    :type url: str
    :param data: List of objIDs requesting
    :type data: list
    :param verbose: If True,
    :type verbose: bool, optional
    :param format: Can be \\'csv\\', \\'json\\', or \\'votable\\'.
    :type format: str
    :return:
    :rtype: str in format given by \\'format\\'.
    """

    results = []
    for i in range(len(url)):
        r = requests.get(url[i], params=data[i])
        if verbose:
            print(r.url)
        r.raise_for_status()
        if format == "json":
            results.append(r.json())
        else:
            results.append(r.text)

    return results

def ps1metadata(table="mean", release="dr1", baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """Return metadata for the specified catalog and table. Snagged from the
       wonderful API at https://ps1images.stsci.edu/ps1_dr2_api.html.

    :param table: Table type. Can be \\'mean\\', \\'stack\\', or \\'detection\\'
    :type table: str
    :param release: The Pan-STARRS data release. Can be \\'dr1\\' or \\'dr2\\'.
    :type release: str
    :param baseurl: Base URL for the request.
    :type baseurl: str
    :return: Table with columns name, type, description.
    :rtype: Astropy Table
    """

    checklegal(table,release)
    url = f"{baseurl}/{release}/{table}/metadata"
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()

    # convert to astropy table
    tab = Table(rows=[(x['name'],x['type'],x['description']) for x in v],
               names=('name','type','description'))
    return tab

def post_url_parallel(results,YSE_ID):
    """TODO: unused function. The querying of PS servers is the slowest part of the script. How to parallelize?

    :param results: Description of parameter.
    :type results: type
    :param YSE_ID: Description of parameter.
    :type YSE_ID: type
    :return: Description of returned object.
    :rtype: Pandas DataFrame
    """

    if type(results) != str:
        results = codecs.decode(results,'UTF-8')
    lines = results.split('\n')
    print(lines)
    if len(lines) > 2:
        values = [line.strip().split(',') for line in lines]
        DF = pd.DataFrame(values[1:-1],columns=values[0])
    else:
        print('No Matches')
        DF = pd.DataFrame()
    DF['YSE_id'] = np.ones(len(DF))*YSE_ID
    return DF

def post_url_serial(results,YSE_ID):
    """A helper function called by serial_objID_search. Post-processes the data retrieved from PS1 Servers into a pandas.DataFrame object.

    :param results: The string resulting from PS1 query.
    :type results: str
    :param YSE_ID: local integer used for as an index tracking user objects vs retrived objects.
    :type YSE_ID: int
    :return: DataFrame object of the retrieved data from PS1 servers
    :rtype: pandas.DataFrame
    """
    if type(results) != str:
        results = codecs.decode(results,'UTF-8')
    lines = results.split('\r\n')
    #print(lines)
    if len(lines) > 2:
        values = [line.strip().split(',') for line in lines]
        DF = pd.DataFrame(values[1:-1],columns=values[0])
    else:
        DF = pd.DataFrame()
    DF['id'] = np.ones(len(DF))*YSE_ID
    return DF

def serial_objID_search(objIDs,table='forced_mean',release='dr2',columns=None,verbose=False,**constraints):
    """Given a list of ObjIDs, queries the PS1 server these object's Forced Mean Photometry, then returns matches as a pandas.DataFrame.

    :param objIDs: list of PS1 objIDs for objects user would like to query
    :type objIDs: list
    :param table: Which table to perform the query on. Default 'forced_mean'
    :type table: str
    :param release: Which release to perform the query on. Default 'dr2'
    :type release: str
    :param columns: list of what data fields to include; None means use default columns. Default None
    :type columns: list or None
    :param verbose: boolean setting level of feedback user received. default False
    :type verbose: bool
    :param \\*\\*constraints: Keyword dictionary with an additional constraints for the PS1 query
    :type \\*\\*constraints: dict
    :return: list of pd.DataFrame objects. If a match was found, then the Dataframe contains data, else it only contains a local integer.
    :rtype: pd.DataFrame
    """

    constrains=constraints.copy()
    Return = ps1objIDsearch(objID=objIDs,table='forced_mean',release=release,columns=columns,verbose=verbose,**constraints)
    #Return = fetch_information_serially(URLS,DATAS)
    DFs=[]
    for i in range(len(Return)):
        DFs.append(post_url_serial(Return[i],i))

    return DFs

def post_url_parallel(results,YSE_ID):
    """TODO: unused function. The querying of PS servers is the slowest part of the script. How to parallelize?

    :param results: Description of param.
    :type results: type
    :param YSE_ID: Description of param.
    :type YSE_ID: type
    :return:
    :rtype:
    """

    results = codecs.decode(results,'UTF-8')
    lines = results.split('\n')
    if len(lines) > 2:
        values = [line.strip().split(',') for line in lines]
        DF = pd.DataFrame(values[1:-1],columns=values[0])
    else:
        DF = pd.DataFrame()
    DF['id'] = np.ones(len(DF))*YSE_ID
    return DF

def get_common_constraints_columns():
    """Helper function that returns a dictionary of constraints used for the matching objects in PS1 archive, and the columns of data we requre.

    :return: dictionary with our constaint that we must have more than one detection
    :rtype: dict
    :return: List of PS1 fields required for matching and NN inputs
    :rtype: list
    """

    constraints = {'nDetections.gt':1}

    #objects with n_detection=1 sometimes just an artifact.
    # strip blanks and weed out blank and commented-out values
    columns ="""objID, raMean, decMean, gFKronFlux, rFKronFlux, iFKronFlux, zFKronFlux, yFKronFlux,
    gFPSFFlux, rFPSFFlux, iFPSFFlux, zFPSFFlux, yFPSFFlux,
    gFApFlux, rFApFlux, iFApFlux, zFApFlux, yFApFlux,
    gFmeanflxR5, rFmeanflxR5, iFmeanflxR5, zFmeanflxR5, yFmeanflxR5,
    gFmeanflxR6, rFmeanflxR6, iFmeanflxR6, zFmeanflxR6, yFmeanflxR6,
    gFmeanflxR7, rFmeanflxR7, iFmeanflxR7, zFmeanflxR7, yFmeanflxR7""".split(',')
    columns = [x.strip() for x in columns]
    columns = [x for x in columns if x and not x.startswith('#')]

    return constraints, columns

def preprocess(DF,PATH='../data/sfddata-master/', ebv=True):
    """Preprocesses the data inside pandas.DataFrame object returned by serial_objID_search to the space of Inputs of our Neural Network.

    :param DF: Dataframe object containing the data for each matched objID
    :type DF: pandas DataFrame
    :param PATH: string path to extinction maps data
    :type PATH: str
    :param ebv: boolean for lookup of extinction data. If False, all extinctions set to 0.
    :type ebv: False
    :return: Preprocessed inputs ready to be used as input to NN
    :rtype: numpy ndarray
    """
    if ebv:
        m = sfdmap.SFDMap(PATH)
        assert ('raMean' in DF.columns.values) and ('decMean' in DF.columns.values), 'DustMap query failed because the expected coordinates didnt'\
                                                                            'exist in DF, likely the match of any Hosts into PanStarrs failed'
        EBV = m.ebv(DF['raMean'].values.astype(np.float32),DF['decMean'].values.astype(np.float32))

        DF['ebv'] = EBV
    else:
        DF['ebv'] = 0.0

    def convert_flux_to_luptitude(f,b,f_0=3631):
        return -2.5/np.log(10) * (np.arcsinh((f/f_0)/(2*b)) + np.log(b))

    b_g = 1.7058474723241624e-09
    b_r = 4.65521985283191e-09
    b_i = 1.2132217745483221e-08
    b_z = 2.013446972858555e-08
    b_y = 5.0575501316874416e-08


    MEANS = np.array([18.70654578, 17.77948707, 17.34226094, 17.1227873 , 16.92087669,
           19.73947441, 18.89279411, 18.4077393 , 18.1311733 , 17.64741402,
           19.01595669, 18.16447837, 17.73199409, 17.50486095, 17.20389615,
           19.07834251, 18.16996592, 17.71492073, 17.44861273, 17.15508793,
           18.79100201, 17.89569908, 17.45774026, 17.20338482, 16.93640741,
           18.62759241, 17.7453392 , 17.31341498, 17.06194499, 16.79030564,
            0.02543223])

    STDS = np.array([1.7657395 , 1.24853534, 1.08151972, 1.03490545, 0.87252421,
           1.32486758, 0.9222839 , 0.73701807, 0.65002723, 0.41779001,
           1.51554956, 1.05734494, 0.89939638, 0.82754093, 0.63381611,
           1.48411417, 1.05425943, 0.89979008, 0.83934385, 0.64990996,
           1.54735158, 1.10985163, 0.96460099, 0.90685922, 0.74507053,
           1.57813401, 1.14290345, 1.00162105, 0.94634726, 0.80124359,
           0.01687839])

    data_columns = ['gFKronFlux', 'rFKronFlux', 'iFKronFlux', 'zFKronFlux', 'yFKronFlux',
    'gFPSFFlux', 'rFPSFFlux', 'iFPSFFlux', 'zFPSFFlux', 'yFPSFFlux',
    'gFApFlux', 'rFApFlux', 'iFApFlux', 'zFApFlux', 'yFApFlux',
    'gFmeanflxR5', 'rFmeanflxR5', 'iFmeanflxR5', 'zFmeanflxR5', 'yFmeanflxR5',
    'gFmeanflxR6', 'rFmeanflxR6', 'iFmeanflxR6', 'zFmeanflxR6', 'yFmeanflxR6',
    'gFmeanflxR7', 'rFmeanflxR7', 'iFmeanflxR7', 'zFmeanflxR7', 'yFmeanflxR7', 'ebv']

    X = DF[data_columns].values.astype(np.float32)
    X[:,0:30:5] = convert_flux_to_luptitude(X[:,0:30:5],b=b_g)
    X[:,1:30:5] = convert_flux_to_luptitude(X[:,1:30:5],b=b_r)
    X[:,2:30:5] = convert_flux_to_luptitude(X[:,2:30:5],b=b_i)
    X[:,3:30:5] = convert_flux_to_luptitude(X[:,3:30:5],b=b_z)
    X[:,4:30:5] = convert_flux_to_luptitude(X[:,4:30:5],b=b_y)

    X = (X-MEANS)/STDS
    X[X>20] = 20
    X[X<-20] = -20
    X[np.isnan(X)] = -20

    return X


def load_lupton_model(model_path=DEFAULT_MODEL_PATH, dust_path=DEFAULT_DUST_PATH):
    """Helper function that defines and loads the weights of our NN model and the output space of the NN.

    :param model_path: path to the model weights.
    :type model_path: str
    :param dust_path: path to dust map data files.
    :type dust_path: str
    :return: Trained photo-z MLP.
    :rtype: tensorflow keras Model
    :return: Array of binned redshift space corresponding to the output space of the NN
    :rtype: numpy ndarray
    """

    build_sfd_dir(data_dir=dust_path)
    get_photoz_weights(file_path=model_path)

    def model():
        INPUT = tf.keras.layers.Input(shape=(31,))

        DENSE1 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(1e-5))(INPUT)
        DROP1 = tf.keras.layers.Dropout(0.05)(DENSE1)

        DENSE2 = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(1e-5))(DROP1)
        DROP2 = tf.keras.layers.Dropout(0.05)(DENSE2)

        DENSE3 = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(1e-5))(DROP2)
        DROP3 = tf.keras.layers.Dropout(0.05)(DENSE3)

        DENSE4 = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(1e-5))(DROP3)

        OUTPUT = tf.keras.layers.Dense(360, activation=tf.keras.activations.softmax)(DENSE4)

        model = tf.keras.Model(INPUT, OUTPUT)

        return model
    mymodel = model()
    mymodel.load_weights(model_path)


    NB_BINS = 360
    ZMIN = 0.0
    ZMAX = 1.0
    BIN_SIZE = (ZMAX - ZMIN) / NB_BINS
    range_z = np.linspace(ZMIN, ZMAX, NB_BINS + 1)[:NB_BINS]

    return mymodel, range_z

def evaluate(X,mymodel,range_z):
    """Evaluate the MLP for a set of PS1 inputs, and return predictions.

    :param X: PS1 properties of associated hosts.
    :type X: array-like
    :param mymodel: MLP model for photo-z estimation.
    :type mymodel: tensorflow keras Model
    :param range_z: Grid over which to evaluate the posterior distribution of photo-zs.
    :type range_z: array-like

    :return: Posterior distributions for the grid of redshifts defined as
        \\`np.linspace(0, 1, n)\\`
    :rtype: numpy ndarray shape of (df.shape[0], n)
    :return: Means
    :rtype: numpy ndarray shape of (df.shape[0],)
    :return: Standard deviations
    :rtype: numpy ndarray shape of (df.shape[0],)
    """

    posteriors = mymodel.predict(X)
    point_estimates = np.sum(posteriors*range_z,axis=1)
    for i in range(len(posteriors)):
        posteriors[i,:] /= np.sum(posteriors[i,:])
    errors=np.ones(len(posteriors))
    for i in range(len(posteriors)):
        errors[i] = (np.std(np.random.choice(a=range_z,size=1000,p=posteriors[i,:],replace=True)))

    return posteriors, point_estimates, errors

#'id' column in DF is the 0th ordered index of hosts. missing rows are therefore signalled
#    by skipped numbers in index
def calc_photoz(hosts, dust_path=DEFAULT_DUST_PATH, model_path=DEFAULT_MODEL_PATH):
    """PhotoZ beta: not tested for missing objids.
       photo-z uses a artificial neural network to estimate P(Z) in range Z = (0 - 1)
       range_z is the value of z
       posterior is an estimate PDF of the probability of z
       point estimate uses the mean to find a single value estimate
       error is an array that uses sampling from the posterior to estimate a std dev.
       Relies upon the sfdmap package, (which is compatible with both unix and windows),
       found at https://github.com/kbarbary/sfdmap.

    :param hosts: The matched hosts from GHOST.
    :type hosts: pandas DataFrame
    :return: The matched hosts from GHOST, with photo-z point estimates and uncertainties.
    :rtype: pandas DataFrame
    """

    if np.nansum(hosts['decMean'] < -30) > 0:
        print("ERROR! Photo-z estimator has not yet been implemented for southern-hemisphere sources."\
        "Please remove sources below dec=-30d and try again.")
        return hosts
    objIDs = hosts['objID'].values.tolist()
    constraints, columns = get_common_constraints_columns()
    DFs = serial_objID_search(objIDs, columns=columns, **constraints)
    DF = pd.concat(DFs)

    posteriors, point_estimates, errors = get_photoz(DF, dust_path=dust_path, model_path=model_path)
    successIDs = DF['objID'].values

    #for i in np.arange(len(successIDs)):
    #    objID = int(successIDs[i])
    #    hosts.loc[hosts['objID']==objID, 'z_phot_point'] = point_estimates[i]
    #    hosts.loc[hosts['objID']==objID, 'z_phot_err'] = errors[i]

    mask = hosts['objID'].isin(successIDs)

    # Map successIDs to point_estimates and errors using pandas' Series and set these values
    id_to_point = pd.Series(point_estimates, index=successIDs)
    id_to_error = pd.Series(errors, index=successIDs)

    # Use the mask to update values in 'z_phot_point' and 'z_phot_err' columns
    hosts.loc[mask, 'z_phot_point'] = hosts['objID'].map(id_to_point)
    hosts.loc[mask, 'z_phot_err'] = hosts['objID'].map(id_to_error)

    return successIDs, hosts, posteriors


def get_photoz(df, dust_path=DEFAULT_DUST_PATH, model_path=DEFAULT_MODEL_PATH):
    """Evaluate photo-z model for Pan-STARRS forced photometry.

    :param df: Pan-STARRS forced mean photometry data, you can get it using
        \\`ps1objIDsearch\\` from this module, Pan-STARRS web-portal or via
        astroquery i.e., \\`astroquery.mast.Catalogs.query_{criteria,region}(...,
        catalog=\\'Panstarrs\\',table=\\'forced_mean\\')\\`
    :type df: pandas DataFrame
    :param dust_path: Path to dust map data files
    :type dust_path: str
    :param model_path: path to the data file with weights for MLP photo-z model
    :type model_path: str
    :return: Posterior distributions for the grid of redshifts defined as
        \\`np.linspace(0, 1, n)\\`
    :rtype: numpy ndarray shape of (df.shape[0], n)
    :return: Means
    :rtype: numpy ndarray shape of (df.shape[0],)
    :return: Standard deviations
    :rtype: numpy ndarray shape of (df.shape[0],)
    """

    # The function load_lupton_model downloads the necessary dust models and
    # weights from the ghost server.

    model, range_z = load_lupton_model(model_path=model_path, dust_path=dust_path)
    X = preprocess(df, PATH=os.path.join(dust_path, 'sfddata-master'))
    return evaluate(X, model, range_z)
