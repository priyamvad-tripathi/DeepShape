"""
This script is used to estimate ellipticities using GalSim followed by a callibration using SuperCALS method.
After submitting the paper, we managed to get access to the official version: https://github.com/itrharrison/supercals/ of the SuperCALS method (through private communication).
"""

# %% Import Modules and set constants
import sys

sys.path.append("../../Functions")
import copy
import ctypes
import gc
import time

import dask
import h5py
import helpers
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from dask.distributed import Client, LocalCluster
from helpers import secondsToStr
from imager import simulate_gal
from scipy.signal import convolve2d
from ska_sdp_func_python.image import deconvolve_cube
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# import optuna

PATH_TEST = "/scratch/tripathi/Data/ts0.h5"


# %% Dask Setup
def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


c = {
    "distributed.scheduler.active-memory-manager.measure": "managed",
    "distributed.worker.memory.rebalance.measure": "managed",
    "distributed.worker.memory.spill": False,
    "distributed.worker.memory.pause": False,
    "distributed.worker.memory.terminate": False,
    "distributed.scheduler.worker-ttl": None,
    "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
    "distributed.scheduler.work-stealing": True,
}

dask.config.set(c)

# %% Ellipticities of model sourcea
# We choose smaller values since the test set does not contain such galaxies
# Using higher ellipticites makes the callibration process slower without any significant improvement
e_model = [np.array([0, 0])]
e_values = np.linspace(0, 0.55, 5)
angs = np.arange(0, 360, 45)
for g in e_values[1:]:
    for ang in angs:
        gamma = [
            g * np.cos(ang * np.pi / 180),
            g * np.sin(ang * np.pi / 180),
        ]
        e_model = [*e_model, gamma]
e_model = np.array(e_model)


# %% Define Superclass calibration function

# MS_CLEAN params (Same as used for generating the CLEAN image)
params = {
    "clean_threshold": 0.12e-07,
    "frac_threshold": 3e-05,
    "loop_gain": 0.05 * 0.1,
    "ms_scales": [0, 2, 4],
    "niter": 1000,
}


def callibration(meas, obs, flags, model=e_model):
    """
    SuperCALS source-level callibration: Calibrates the measured ellipticities using bias parameters estimated from simulated model sources.

    Parameters:
    ----------
    meas (numpy.ndarray): The measured ellipticity of the original source to be calibrated. Dimension: (2,)
    obs (numpy.ndarray): The observed ellipticities of the model sources. Dimension: (33, 2)
    flags (numpy.ndarray): Flags indicating unresolved model sources (0 for resolved, non-zero for unresolved). Dimesion: (33,)
    model (numpy.ndarray): The true ellipticities of the model sources. Default is e_model.

    Returns:
    ----------
    numpy.ndarray or int: The calibrated ellipticity measurement if successful, otherwise 0.
    """

    # Remove model sources unresolved by galsim
    mask = flags == 0

    # If many model sources unresolved then skip callibration
    if np.array(mask).sum() < 10:
        return 0

    model = model[mask]
    obs = obs[mask]

    residual = obs - model

    # Check if measurements residual are not too bad
    # Callibration cross check (see SuperCLASS-III paper)
    if np.linalg.norm(residual) > 5:
        return 0

    # Scale model inputs
    poly = PolynomialFeatures(degree=2)
    inp = poly.fit_transform(model)

    # Fit difference in both directions
    model1 = LinearRegression(fit_intercept=True)
    model1.fit(inp, residual[:, 0])

    model2 = LinearRegression(fit_intercept=True)
    model2.fit(inp, residual[:, 1])

    # Predict bias
    b1 = model1.predict(poly.transform(meas.reshape([1, 2])))[0]
    b2 = model2.predict(poly.transform(meas.reshape([1, 2])))[0]
    callib = meas - np.array([b1, b2])

    return callib


def sc_calib(res, clean_img, psf, flux, r0, fac=20, e_model=e_model):
    """
    Full function to estimate shape of galaxy and calibrate using SuperCALS method.

    Parameters:
    ----------
    res (numpy.ndarray): The MS-CLEAN residual image.
    clean_img (numpy.ndarray): The MS-CLEAN recosntructed image.
    flux (float): The flux of the model sources. Ideally should be "estimated" from the original image.
    r0 (float): The size of the model sources. Ideally should be "estimated" from the original image.
    fac (int, optional): The normalization factor for the model sources. Default is 20. Should be tuned for optimum results.
    e_model (list of tuples): List of ellipticity for the model sources.

    Returns:
    ----------
    float: The calibrated shape measurement of the galaxy (returns 0 if unresolved).
    """

    # Initial shape measurement using MS-CLEAN reconstructed image
    meas, status = helpers.predict_shape(clean_img)

    # Remove galaxy if not resolved using galsim
    if status != 0:
        return 0

    # Inject Model sources and measure ellipticity
    obs_all = []
    flags = []
    for e in e_model:
        model_source = simulate_gal(
            flux=flux,
            r0=r0,
            e1gal=e[0],
            e2gal=e[1],
        )[0].copy()
        model_source = convolve2d(model_source, psf, model="same")
        norm = (
            np.std(model_source) / fac / np.std(res)
        )  # normalisation is required otherwise the model sources would be dominated by noise
        model_source += res * norm
        model_source_obs = deconvolve_cube(
            model_source, psf, algorithm="msclean", **params
        )
        model_source_obs = (
            model_source_obs.pixels.to_numpy().astype(np.float32).squeeze()
        )
        obs, flag = helpers.predict_shape(model_source_obs)
        obs_all = [*obs_all, obs]
        flags = [*flags, flag]

    obs_all = np.array(obs_all)
    flags = np.array(flags)

    # Callibration
    sc = callibration(meas, obs_all, flags, e_model)

    return sc


# %% Tune Normalisation factor

#! Long runtime
#! Run once and save result
# * Not that important to tune
"""
hf = h5py.File(PATH_TEST, "r")
tot = len(hf["input"])

# Choose 100 random objects to run test on
rand_ind = np.sort(np.random.choice(range(tot), 100, replace=False))


def objective(trial):
    fac = trial.suggest_float("fac", 10, 50)

    sc_all = []

    for ind in rand_ind:
        res = hf["CLEAN-RES"][ind]
        clean_img = hf["CLEAN"][ind]
        flux = hf["Flux"][ind]
        r0 = hf["HLR"][ind] / np.log(2)

        sc = sc_calib(res, clean_img, psf, flux, r0, fac=fac, e_model=e_model)
        sc_all = [*sc_all, sc]

    flags = np.array([isinstance(sc, int) for sc in sc_all])
    sc_all = np.array([sc for sc in sc_all if not isinstance(sc, int)])

    inp = hf["input"][rand_ind][~flags]

    rmse = np.mean(helpers.RMSE(inp, sc_all, summary=False))

    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

best_fac = study.best_params["fac"]
print(f"Best Factor: {best_fac}")
hf.close()"""
# %% Perform Callibration for all sources
if __name__ == "__main__":
    start = time.time()

    hf = h5py.File(PATH_TEST, "r")
    tot = len(hf["input"])
    best_fac = 4.8565  # Calculated by running previous block

    with (
        LocalCluster(
            n_workers=32,
            processes=True,
            threads_per_worker=1,
            # scheduler_port=8786,
            memory_limit=0,
        ) as cluster,
        Client(cluster) as client,
    ):
        print(client.dashboard_link)
        client.run(trim_memory)
        client.run(gc.collect)
        lazy_results = []

        for ind in range(tot):
            res = hf["CLEAN-RES"][ind]  # MS-CLEAN residual
            clean_img = hf["CLEAN"][ind]  # MS-CLEAN reconstructed image
            psf = hf["PSF"][ind]
            flux = hf["Flux"][ind]  # Flux of source
            r0 = hf["HLR"][ind] / np.log(2)  # Half light radius

            lazy_result = dask.delayed(sc_calib)(
                res, clean_img, psf, flux, r0, fac=best_fac, e_model=e_model
            )
            lazy_results.append(lazy_result)

        results = dask.compute(*lazy_results)
        hf.close()

    flags = np.array([isinstance(res, int) for res in results])
    calib = [res for res in results if not isinstance(res, int)]
    del results, lazy_results

    try:
        hf.close()
    except Exception:
        print("HF already closed")
    with h5py.File(PATH_TEST, "a") as hf:
        for key in ["SC_flagged", "Superclass"]:
            if key in hf.keys():
                del hf[key]

        hf.create_dataset(
            name="SC_flagged",
            data=copy.deepcopy(flags),
        )
        hf.create_dataset(
            name="Superclass",
            data=copy.deepcopy(calib),
        )

    end = time.time()
    seconds = end - start
    print(f"Resolved {tot - flags.sum()}/{tot} objects in {secondsToStr(seconds)}")
