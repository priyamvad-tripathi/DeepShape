"""
This script is used to generate a test set of 2.5k sources.
"""

# %% Import Modules and set constants
import sys

sys.path.append("../../Functions")
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

# %% Decide sources and specify save paths
np.random.seed(202123)

PATH_TEST = (
    "/scratch/tripathi/Data/vis/"  # Path to save the raw visibilities of the test set
)
PATH_hf = (
    "/scratch/tripathi/Data/tss.h5"  # Path to save the imaging data of the test set
)

TRAIN_OBJS = 2500
sources_all = np.random.choice(range(NSOURCE), size=TRAIN_OBJS, replace=False)
source_chuncks = np.array_split(sources_all, int(np.ceil(TRAIN_OBJS / 100)))
# %% Run the deconvolution for all sources
if __name__ == "__main__":
    #! Generate the visibilities
    #! Uncomment the below block to generate the visibilities
    """
    start = time.time()
    nvis = 0
    for NC, chunk in enumerate(source_chuncks):
        print(
            f"{Color.GREEN} Working on chunck {NC+1}/{len(source_chuncks)}{Color.OFF}"
        )
        chunk_result = []
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
                    vis_only=True,
                )

                chunk_result.append(res_block[0])

            results = dask.compute(*chunk_result)

            # Filter out the sources with FFT size errors
            results = [block for block in results if not isinstance(block, int)]
            nvis += len(results)
            helpers.save(copy.deepcopy(results), PATH_TEST + f"vis_{NC+1}.pkl")
            del results

    end = time.time()
    seconds = end - start
    print(
        f"{Color.RED} {nvis} visibilities generated in {secondsToStr(seconds)} {Color.OFF}"
    )

    vis_list_all = []
    for NC in range(10):
        vis_list = helpers.load(PATH_TEST + f"vis_{NC+1}.pkl")
        vis_list_all = [*vis_list_all, *vis_list]
        del vis_list

    helpers.save(vis_list_all, "/scratch/tripathi/Data/vis_val_set_small.pkl")
    del vis_list_all

    print(
        f"{Color.RED} Saved all the visibilities as single list. Now generating catalog. {Color.OFF}"
    )
    """
    #! Generate the imaging data
    #! Comment the below block when generating visibility data
    start = time.time()
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
            res_block = dask.delayed(simulate_vis)(
                gal_data,
                ra_gal=ra_gal,
                dec_gal=dec_gal,
                PSF=True,
                robustness=-0.5,
            )

            lazy_results.append(res_block)

        results = dask.compute(*lazy_results)

    # Filter out the sources with FFT size errors
    results = [block for block in results if not isinstance(block, int)]

    with h5py.File(PATH_hf, "w") as hf:
        hf.create_dataset(
            name="SNR_vis",
            data=copy.deepcopy(np.array([data[0][0] for data in results])),
            maxshape=(None,),
        )
        hf.create_dataset(
            name="Sersic_index",
            data=copy.deepcopy(np.array([data[0][4] for data in results])),
            maxshape=(None,),
        )

        hf.create_dataset(
            name="Flux",
            data=copy.deepcopy(np.array([data[0][5] for data in results])),
            maxshape=(None,),
        )

        hf.create_dataset(
            name="HLR",
            data=copy.deepcopy(np.array([data[0][6] for data in results])),
            maxshape=(None,),
        )

        hf.create_dataset(
            name="input",
            data=copy.deepcopy(
                np.array([[data[0][2], data[0][3]] for data in results])
            ),
            maxshape=(None, 2),
        )
        hf.create_dataset(
            name="Peak",
            data=copy.deepcopy(np.array([data[0][1] for data in results])),
            maxshape=(None,),
        )
        hf.create_dataset(
            name="true image",
            data=copy.deepcopy(np.array([data[3] for data in results])),
            maxshape=(None, 128, 128),
        )
        hf.create_dataset(
            name="dirty",
            data=copy.deepcopy(np.array([data[4] for data in results])),
            maxshape=(None, 128, 128),
        )
        hf.create_dataset(
            name="psf",
            data=copy.deepcopy(np.array([data[5] for data in results])),
            maxshape=(None, 128, 128),
        )

    del lazy_results
    end = time.time()
    seconds = end - start
    print(
        f"{Color.RED} Generated catalog of {len(results)} sources in {secondsToStr(seconds)} {Color.OFF}"
    )
