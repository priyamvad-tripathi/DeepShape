"""
Script to generate 250k dirty image-PSF pairs. Specify path to save results.
Usage: python dirty_images.py
"""

# %% Import Modules and set constants
import sys

sys.path.append("../Functions/")
import copy
import ctypes
import gc
import logging
import time
import warnings

import dask
import h5py
import helpers
import numpy as np
from colorist import Color
from dask.distributed import Client, LocalCluster
from helpers import secondsToStr
from imager import simulate_gal, simulate_vis

# %% Define Settings
FLUX_LIM = (50, 200)
NSOURCE, ra, dec, flux_all, size_all, e1, e2 = helpers.load_catalog(*FLUX_LIM)
warnings.warn = lambda *args, **kwargs: None
logging.getLogger().addHandler(logging.NullHandler())


# DASK Settings
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

# SIGMA_NOISE = 6.241418e-07
# %% Decide Sources
np.random.seed(2021)
PATH_TRAIN = "/scratch/tripathi/Data/dataset_dirty_train_sers.h5"
TRAIN_OBJS = int(250e3)
sources_all = np.random.choice(range(NSOURCE), TRAIN_OBJS, replace=False)
# Split dataset into chuncks for computational efficiency
source_chuncks = np.array_split(sources_all, int(np.ceil(TRAIN_OBJS / 10000)))

# %% Simulate sources

if __name__ == "__main__":
    start = time.time()

    hf = h5py.File(PATH_TRAIN, "a")
    SNR_vis = hf.create_dataset(
        name="SNR_vis",
        shape=(0,),
        maxshape=(None,),
    )
    sers_ind = hf.create_dataset(
        name="Sersic_index",
        shape=(0,),
        maxshape=(None,),
    )
    inp = hf.create_dataset(
        name="input",
        shape=(0, 2),
        maxshape=(None, 2),
    )
    peak = hf.create_dataset(
        name="Peak",
        shape=(0,),
        maxshape=(None,),
    )
    true = hf.create_dataset(
        name="true image",
        shape=(0, 128, 128),
        maxshape=(None, 128, 128),
    )
    dirty = hf.create_dataset(
        name="dirty",
        shape=(0, 128, 128),
        maxshape=(None, 128, 128),
    )
    psf = hf.create_dataset(
        name="psf",
        shape=(0, 128, 128),
        maxshape=(None, 128, 128),
    )
    dsets = [SNR_vis, sers_ind, inp, peak, true, dirty, psf]

    #! Generate the train dataset
    for NC, chunk in enumerate(source_chuncks):
        print(
            f"{Color.GREEN} Working on chunck {NC + 1}/{len(source_chuncks)}{Color.OFF}"
        )
        chunk_result = []
        with (
            LocalCluster(
                n_workers=62,
                processes=True,
                threads_per_worker=1,
                scheduler_port=8786,
                memory_limit=0,
            ) as cluster,
            Client(cluster) as client,
        ):
            print(client.dashboard_link)
            client.run(trim_memory)
            client.run(gc.collect)

            for source in chunk:
                flux = flux_all[source]
                r0 = size_all[source]
                e1gal = e1[source]
                e2gal = e2[source]

                ra_gal = ra[source]
                dec_gal = dec[source]

                gal_data = dask.delayed(simulate_gal)(
                    flux,
                    r0,
                    e1gal=e1gal,
                    e2gal=e2gal,
                )
                res_block = dask.delayed(simulate_vis)(
                    gal_data,
                    ra_gal=ra_gal,
                    dec_gal=dec_gal,
                    PSF=True,
                    robustness=-0.5,
                )

                chunk_result.append(res_block)

            results = dask.compute(*chunk_result)

        # Filter out the sources with FFT size errors
        results = [block for block in results if not isinstance(block, int)]
        N = len(results)

        # Save simulation for current block
        for dset in dsets:
            dset.resize(dset.shape[0] + N, axis=0)

        inp[-N:] = copy.deepcopy(
            np.array([[data[0][2], data[0][3]] for data in results])
        )
        SNR_vis[-N:] = copy.deepcopy(np.array([data[0][0] for data in results]))
        peak[-N:] = copy.deepcopy(np.array([data[0][1] for data in results]))
        sers_ind[-N:] = copy.deepcopy(np.array([data[0][4] for data in results]))
        true[-N:] = copy.deepcopy(np.array([data[3] for data in results]))
        dirty[-N:] = copy.deepcopy(np.array([data[4] for data in results]))
        psf[-N:] = copy.deepcopy(np.array([data[5] for data in results]))

    print(
        f"{Color.GREEN} Generated train dataset: {len(hf[list(hf.keys())[0]])} unique objects. {Color.OFF}"
    )
    hf.close()
    del results, chunk_result

    end = time.time()
    seconds = end - start
    print(f"Script took {secondsToStr(seconds)} to run.")
