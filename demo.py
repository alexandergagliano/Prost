from astro_prost.helpers import prior_z_observed_transients
from astro_prost.associate import prepare_catalog, associate_sample

source = 'ZTF BTS'

transient_catalog = pd.read_csv("/Users/alexgagliano/Documents/Research/multimodal-supernovae/data/ZTFBTS/ZTFBTS_TransientTable.csv")

# define priors for properties
priorfunc_z = halfnorm(loc=0.0001, scale=0.5)

# if you want the redshift prior to be based on an observed distribution of transients within a given absmag range
# transients are uniformly distributed in cosmological volume between zmin and zmax and the subset
# peaking above mag_cutoff sets the z distribution
#priorfunc_z = prior_z_observed_transients(z_min=0, z_max=1, mag_cutoff=19, Mmean=-19, Mmin=-24, Mmax=-17)

# look at your distribution (only available for the above experiment)
#priorfunc_z.plot()

priorfunc_offset = uniform(loc=0, scale=10)
priorfunc_absmag = uniform(loc=-30, scale=20)

likefunc_offset = st.gamma(a=0.75) #truncexpon(loc=0, scale=1, b=10)
likefunc_absmag = SNRate_absmag(a=-30, b=-10)

priors = {'offset':priorfunc_offset, 'absmag':priorfunc_absmag, 'z':priorfunc_z}
likes = {'offset':likefunc_offset, 'absmag':likefunc_absmag}

#set up properties of the association run
verbose = 0
parallel = True
save = False

# list of catalogs to search -- options are (in order) glade, decals, panstarrs
catalogs = ['panstarrs'] 

# the name of the coord columns in the dataframe
transient_coord_cols = ("ra", "dec") 

# the column containing the transient names
transient_name_col = 'name'

transient_catalog = prepare_catalog(transient_catalog, transient_name_col=transient_name_col, transient_coord_cols=transient_coord_cols)
transient_catalog = associate_sample(transient_catalog, priors=priors, likes=likes, catalogs=catalogs, parallel=parallel, verbose=verbose, save=save, cat_cols=False)
