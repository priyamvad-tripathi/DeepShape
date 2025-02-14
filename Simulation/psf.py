"""
Script to generate 100k PSFs. Specify path to save results.
Usage: python psf.py
"""

# %% Import Modules and set constants
import sys

sys.path.append("../Functions/")
import copy

# import copy
import ctypes
import gc
import logging

# import pickle
import time
import warnings

import dask
import h5py
import helpers
import numpy as np
from colorist import Color
from dask.distributed import Client, LocalCluster
from helpers import secondsToStr
from imager import simulate_vis

# %% Define Settings
FLUX_LIM = (50, 200)
NSOURCE, ra, dec, _, _, _, _ = helpers.load_catalog(*FLUX_LIM)
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


np.random.seed(2345)
TRAIN_OBJS = 100000
PATH_TRAIN = "/scratch/tripathi/Data/PSFs.h5"
sources_all = np.random.choice(range(NSOURCE), size=TRAIN_OBJS, replace=False)
chuncks = np.array_split(sources_all, int(len(sources_all) / 10000))

# %% Run the deconvolution for all sources

if __name__ == "__main__":
    start = time.time()

    hf = h5py.File(PATH_TRAIN, "a")
    psf = hf.create_dataset(
        name="psf",
        shape=(0, 128, 128),
        maxshape=(None, 128, 128),
    )
    for NC, chunk in enumerate(chuncks):
        print(f"{Color.GREEN} Working on chunck {NC + 1}/{len(chuncks)}{Color.OFF}")
        #! Generate the train dataset
        lazy_results = []
        with (
            LocalCluster(
                n_workers=64,
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

            for source in chunk:
                ra_gal = ra[source]
                dec_gal = dec[source]

                res_block = dask.delayed(simulate_vis)(
                    [np.zeros((128, 128)), []],
                    ra_gal=ra_gal,
                    dec_gal=dec_gal,
                    PSF=True,
                    robustness=-0.5,
                )

                lazy_results.append(res_block)

            results = dask.compute(*lazy_results)

        # Filter out the sources with FFT size errors
        results = [block for block in results if not isinstance(block, int)]
        # Save PSF in HDF5 file
        N = len(results)
        psf.resize(psf.shape[0] + N, axis=0)
        psf[-N:] = copy.deepcopy(np.array([data[5] for data in results]))

    del results, lazy_results
    end = time.time()
    seconds = end - start
    print(f"Script took {secondsToStr(seconds)} to run.")
