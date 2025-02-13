# %%
import copy

import numpy as np
import pandas as pd
from astropy.io import fits

# Directory containing the fits files
# Downloaded from http://cdsarc.u-strasbg.fr/ftp/VII/282/fits/
FITS_DIR = "/scratch/tripathi/TRECS/"

# Path to dump catalog in pandas format
PATH_CATALOG_PD = "/scratch/tripathi/TRECS/catalog.pkl"


def e1e2_to_g1g2(e1, e2):
    """Function to convert ellipticity into correct format"""
    e = np.sqrt(e1**2 + e2**2)

    cos_2t = e1 / e
    sin_2t = e2 / e

    r = np.sqrt((1 - e) / (1 + e))

    g_new = (1 - r) / (1 + r)

    g1 = g_new * cos_2t
    g2 = g_new * sin_2t

    return np.nan_to_num(g1), np.nan_to_num(g2)


# %% Convert the fits file to a pandas DataFrame for easy access

flux_all = []
ra_all = []
dec_all = []
size_all = []
g1_all = []
g2_all = []

TOTAL_OBJS = 0
for i in range(1, 11):
    print(i)
    catalogsfg = fits.open(FITS_DIR + f"SFGs_complete_wide{i}.fits")

    cat1 = catalogsfg[1]  # Selecting the first slice of the FITS ("Catalogue")
    catdatasfg = cat1.data  # Extract data from slice

    flux = (
        copy.deepcopy(catdatasfg["I1400"]) * 1e-3
    )  # Flux density at 1400 MHz (in Janskys)
    size = copy.deepcopy(catdatasfg["size"])  # angular size on the sky (in arcsec)
    ra = copy.deepcopy(catdatasfg["longitude"])
    dec = copy.deepcopy(catdatasfg["latitude"])

    e1 = catdatasfg["e1"]
    e2 = catdatasfg["e2"]

    TOTAL_OBJS += len(flux)

    del catdatasfg, catalogsfg

    g1, g2 = e1e2_to_g1g2(e1, e2)

    flux_all = [*flux_all, *flux]
    size_all = [*size_all, *size]
    ra_all = [*ra_all, *ra]
    dec_all = [*dec_all, *dec]

    g1_all = [*g1_all, *g1]
    g2_all = [*g2_all, *g2]


# %% Save the catalog using pandas
catalog = {
    "flux_all": np.array(flux_all),
    "size_all": np.array(size_all),
    "ra": np.array(ra_all),
    "dec": np.array(dec_all),
    "e1": copy.deepcopy(np.array(g1_all)),
    "e2": copy.deepcopy(np.array(g2_all)),
}

df = pd.DataFrame.from_dict(catalog)  # Convert for easy filtering
df.to_pickle(PATH_CATALOG_PD)
