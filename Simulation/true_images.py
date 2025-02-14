"""
Script to generate 250k true images. Specify path to save results.
Usage: python true_images.py
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
from imager import simulate_gal

PATH = "/scratch/tripathi/Data/true_images.h5"
# %% Define Settings
FLUX_LIM = (50, 200)
NSOURCE, _, _, flux_all, size_all, e1, e2 = helpers.load_catalog(*FLUX_LIM)
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
np.random.seed(12323)

# %% Run the deconvolution for all sources

if __name__ == "__main__":
    start = time.time()

    hf = h5py.File(PATH, "a")
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
    true = hf.create_dataset(
        name="true image",
        shape=(0, 128, 128),
        maxshape=(None, 128, 128),
    )
    dsets = [sers_ind, inp, true]

    #! Decide sources
    TRAIN_OBJS = int(250e3)
    sources_all = np.random.choice(range(NSOURCE), TRAIN_OBJS, replace=False)
    # Split sources into chuncks for compuational efficiency
    source_chuncks = np.array_split(sources_all, int(np.ceil(TRAIN_OBJS / 10000)))

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

                gal_data = dask.delayed(simulate_gal)(
                    flux,
                    r0,
                    e1gal=e1gal,
                    e2gal=e2gal,
                )

                chunk_result.append(gal_data)

            results = dask.compute(*chunk_result)

        # Filter out the sources with FFT size errors
        results = [block for block in results if not isinstance(block, int)]
        N = len(results)

        # Save simulation results for current block
        for dset in dsets:
            dset.resize(dset.shape[0] + N, axis=0)

        inp[-N:] = copy.deepcopy(
            np.array([[data[1][0], data[1][1]] for data in results])
        )
        sers_ind[-N:] = copy.deepcopy(np.array([data[1][2] for data in results]))
        true[-N:] = copy.deepcopy(np.array([data[0] for data in results]))

    del results, chunk_result
    print(
        f"{Color.GREEN} Generated {len(hf[list(hf.keys())[0]])} unique objects. {Color.OFF}"
    )
    hf.close()
    end = time.time()
    seconds = end - start
    print(f"Script took {secondsToStr(seconds)} to run.")
