# %%
import copy
import pickle
from functools import reduce
from pathlib import Path

import galsim
import numpy as np
import pandas as pd

# Path for T-RECS catalog in pandas format
# Check make_catalog.py for details
PATH_CATALOG_PD = "/scratch/tripathi/TRECS/catalog.pkl"


def predict_shape(image, NPIX=128):
    """
    Predict the shape of a galaxy using adaptive moments.

    Parameters:
    ----------
    image (numpy.ndarray): The input image array containing the galaxy.
    NPIX (int, optional): The size of the image in pixels. Default is 128.

    Returns:
    tuple: A tuple containing:
        - g (numpy.ndarray): An array with two elements representing the ellipticity components (g1, g2).
        - moments_status (int): The status of the moments calculation (0 if successful, non-zero otherwise).

    Notes:
    This function uses the GalSim library to estimate the adaptive moments of the input image.
    If the initial moments estimation fails, it retries with modified parameters.
    """

    im_size = NPIX
    # create a galsim version of the data
    image_galsim = galsim.Image(image)
    # estimate the moments of the observation image
    shape = galsim.hsm.FindAdaptiveMom(
        image_galsim,
        guess_centroid=galsim.PositionD(im_size // 2, im_size // 2),
        strict=False,
    )
    if shape.error_message:
        new_params = galsim.hsm.HSMParams(
            max_mom2_iter=2000, convergence_threshold=0.1, bound_correct_wt=2.0
        )
        shape = image_galsim.FindAdaptiveMom(strict=False, hsmparams=new_params)
    g = np.array([shape.observed_shape.g1, shape.observed_shape.g2])

    return g, shape.moments_status


def secondsToStr(t):
    return "%02d:%02d:%02d.%03d" % reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(round(t * 1000),), 1000, 60, 60]
    )


def load_catalog(fmin=10, fmax=None, radius=None):
    """
    Loads T_RECS catalog saved as a pandas DataFrame (check make_catalog.py for details). Filters the catalog based on flux limits and optionally by size.

    Parameters:
    ----------
    fmin (int, optional): Minimum flux limit for the catalog in µJy (1e-32 W/m^2/Hz). Defaults to 10 µJy.
    fmax (int, optional): Maximum flux limit for the catalog in µJy (1e-32 W/m^2/Hz). Defaults to None.
    radius (float, optional): Maximum size limit for the catalog in arcsec. Defaults to None.

    Returns:
    ----------
    tuple: A tuple containing the following elements:
        - NSOURCE (int): Number of sources in the filtered catalog.
        - ra (numpy.ndarray): Right ascension values of the sources.
        - dec (numpy.ndarray): Declination values of the sources.
        - flux_all (numpy.ndarray): Flux values of the sources.
        - size_all (numpy.ndarray): Size values of the sources.
        - e1 (numpy.ndarray): Ellipticity component 1 of the sources.
        - e2 (numpy.ndarray): Ellipticity component 2 of the sources.
    """

    df = pd.read_pickle(PATH_CATALOG_PD)

    # Minimum flux limit in µJy
    df_filtered = df.loc[df["flux_all"] >= (fmin * 1e-6)]

    # Maximum flux limit in µJy
    if fmax:
        df_filtered = df_filtered.loc[df_filtered["flux_all"] < (fmax * 1e-6)]

    # Maximum size limit in arcsec
    if radius:
        df_filtered = df_filtered.loc[df_filtered["size_all"] < radius]

    NSOURCE = len(df_filtered)

    ra = copy.deepcopy(df_filtered["ra"].values)
    dec = copy.deepcopy(df_filtered["dec"].values)

    flux_all = copy.deepcopy(df_filtered["flux_all"].values)
    size_all = copy.deepcopy(df_filtered["size_all"].values)

    e1 = copy.deepcopy(df_filtered["e1"].values)
    e2 = copy.deepcopy(df_filtered["e2"].values)

    return NSOURCE, ra, dec, flux_all, size_all, e1, e2


def load(filename):
    "Convenient function to load pickled data"
    with open(filename, "rb") as fh:
        data = pickle.load(fh)
    return data


def save(data, filename):
    "Convenient function to dump data in pickled format"
    parent_dir = Path(filename).parent

    try:
        parent_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        1
    else:
        print(f"New folder created {parent_dir}")

    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dumped data at {filename}")


def border_elems(a, W):
    if len(a.shape) != 3:
        a = [a]
    bord = []

    if W == 0:
        W = int(0.1 * a.shape[-1])
    for im in a:
        n = im.shape[0]
        r = np.minimum(np.arange(n)[::-1], np.arange(n))
        bord += [im[np.minimum(r[:, None], r) < W]]
    return np.array(bord)


def calculate_SNR(true, obs, border_width=-1, PSNR=False):
    """
    Calculate the Signal-to-Noise Ratio (SNR) or Peak Signal-to-Noise Ratio (PSNR) for image reconstructions.

    Parameters:
    ----------
    true (array-like): The ground truth images. Dimensions: (NIMAGE, NPIX, NPIX).
    obs (array-like): The observed images.  Dimensions: (NIMAGE, NPIX, NPIX).
    border_width (int, optional): The width of the border to consider for noise calculation. If -1, the entire image is used. Default is -1.
    PSNR (bool, optional): If True, calculate the Peak Signal-to-Noise Ratio (PSNR) instead of SNR. Default is False.

    Returns:
    ----------
    float: The calculated SNR or PSNR value.
    """

    if isinstance(true, list):
        true = np.array(true)
    if isinstance(obs, list):
        obs = np.array(obs)
    assert true.shape == obs.shape, "Observed and true images should be of same shape"

    num = np.sqrt(np.sum(true**2, axis=(-2, -1)))

    if PSNR:
        num = np.max(true)
    res = obs - true
    denom = np.std(res, axis=(-1, -2))

    if border_width >= 0:
        bord_pix = border_elems(res, W=border_width)
        denom = np.std(bord_pix, axis=1)

    return num / denom


def RMSE(inp, out, flags=None, return_value=False):
    """
    Calculate the Root Mean Square Error (RMSE) between measured and true ellipticities.

    Parameters:
    ----------
    inp (numpy.ndarray): The truth ellipticity array. Dim: (N_meas,2)
    out (numpy.ndarray): The output/measurement ellipticity array. Dim: (N_meas,2)
    flags (numpy.ndarray, optional): A boolean array indicating which elements to consider for RMSE calculation.
                                        If provided, only elements where flags == 0 are considered.
    return_value (bool, optional): If True, the function returns the RMSE values. Default is False.

    Returns:
    ----------
    list: A list containing RMSE values for each dimension if return_value is True.

    Prints:
    ----------
    The percentage of resolved objects if flags are provided.
    The RMSE values for each dimension.
    """

    if flags is not None:
        ind = flags == 0
        print(f"Resolved objects: {ind.sum() / len(flags) * 100:.02f}")

        inp = inp[ind]
        out = out[ind]

    rmse = []

    for i in [0, 1]:
        diff = np.square(out[:, i] - inp[:, i])
        rmse += [np.sqrt(np.mean(diff))]

    print(f"RMSE 1: {rmse[0]:.3f}")
    print(f"RMSE 2: {rmse[1]:.3f}")
    if return_value:
        return rmse
