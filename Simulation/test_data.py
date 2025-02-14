"""
Script to create test set. Also performs deconvolution on the test set using MS-CLEAN.
Specify the save path for the test set, the number of objects to simulate and MS-CLEAN parameters.
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
from dask.distributed import Client, LocalCluster
from helpers import secondsToStr
from imager import deconvolution, simulate_gal, simulate_vis

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


np.random.seed(1212)
TRAIN_OBJS = 20000
PATH_TRAIN = "/scratch/tripathi/Data/test_set.h5"
sources_all = np.random.choice(range(NSOURCE), size=TRAIN_OBJS, replace=False)

# MS-CLEAN Pararmeters (defaults to values used in the DeepShape paper)
params = {
    "clean_threshold": 0.12e-07,
    "frac_threshold": 3e-05,
    "loop_gain": 0.05 * 0.1,
    "ms_scales": [0, 2, 4],
    "niter": 1000,
}
# %% Run the deconvolution for all sources

if __name__ == "__main__":
    start = time.time()

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

        for source in sources_all:
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
            res_block_0 = dask.delayed(simulate_vis)(
                gal_data,
                ra_gal=ra_gal,
                dec_gal=dec_gal,
                PSF=True,
                robustness=-0.5,
            )

            res_block = dask.delayed(deconvolution)(res_block_0, **params)

            lazy_results.append(res_block)

        results = dask.compute(*lazy_results)

        # Filter out the sources with FFT size errors
        results = [block for block in results if not isinstance(block, int)]

        with h5py.File(PATH_TRAIN, "w") as hf:
            hf.create_dataset(
                name="SNR_vis",
                data=copy.deepcopy(np.array([data[0][0] for data in results])),
            )
            hf.create_dataset(
                name="Sersic_index",
                data=copy.deepcopy(np.array([data[0][4] for data in results])),
            )

            hf.create_dataset(
                name="Flux",
                data=copy.deepcopy(np.array([data[0][5] for data in results])),
            )

            hf.create_dataset(
                name="HLR",
                data=copy.deepcopy(np.array([data[0][6] for data in results])),
            )

            hf.create_dataset(
                name="input",
                data=copy.deepcopy(
                    np.array([[data[0][2], data[0][3]] for data in results])
                ),
            )
            hf.create_dataset(
                name="Peak",
                data=copy.deepcopy(np.array([data[0][1] for data in results])),
            )
            hf.create_dataset(
                name="true image",
                data=copy.deepcopy(np.array([data[3] for data in results])),
            )
            hf.create_dataset(
                name="dirty",
                data=copy.deepcopy(np.array([data[4] for data in results])),
            )
            hf.create_dataset(
                name="psf", data=copy.deepcopy(np.array([data[5] for data in results]))
            )
            hf.create_dataset(
                name="CLEAN",
                data=copy.deepcopy(np.array([data[6] for data in results])),
            )
            hf.create_dataset(
                name="CLEAN-RES",
                data=copy.deepcopy(np.array([data[7] for data in results])),
            )

    del results, lazy_results
    end = time.time()
    seconds = end - start
    print(f"Script took {secondsToStr(seconds)} to run.")
