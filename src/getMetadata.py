from astro_prost.photoz_helpers import *
import pickle

def preload_all_ps1metadata(tables, release="dr1"):
    """
    Preloads metadata for a list of tables and stores it in a dictionary.

    Parameters
    ----------
    tables : list of str
        List of tables for which metadata should be preloaded.
    release : str
        The data release version.

    Returns
    -------
    metadata_dict : dict
        A dictionary with table names as keys and their corresponding metadata as values.
    """
    metadata_dict = {}

    for table in tables:
        # Get the metadata for each table and store it in the dictionary
        metadata_dict[table] = ps1metadata(table, release)

    return metadata_dict


tables = ['mean', 'stack']
dr1_tables = preload_all_ps1metadata(tables, release="dr1")
tables = ['forced_mean', 'stack']
dr2_tables = preload_all_ps1metadata(tables, release="dr2")

metadata_dict = {
    "dr1": dr1_tables,
    "dr2": dr2_tables
}


#metadata_dict.keys()
# Save the metadata to a pickle file
with open("/Users/alexgagliano/Documents/Research/prost/Prost/src/astro_prost/data/ps1_metadata.pkl", "wb") as f:
    pickle.dump(metadata_dict, f)
