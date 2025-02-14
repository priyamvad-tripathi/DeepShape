"""
Script to test optimum choice for robustness parameter and integration time.
"""

# %% Import Modules
import sys

sys.path.append("../Functions/")
import ctypes
import gc
import time
import warnings

import dask
import helpers
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, LocalCluster
from imager import simulate_gal, simulate_vis

# %% Define Settings
FLUX_LIM = (50, 200)
NSOURCE, ra, dec, flux_all, size_all, e1, e2 = helpers.load_catalog(*FLUX_LIM)
warnings.warn = lambda *args, **kwargs: None


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


# %% Check Noise levels for different integration time
def noise_level(int_time, robust):
    """Function to calculate image plane noise level for a given integration time and robustness
    Repeat multiple times to get the expected noise level
    """

    # Calculate dirty image for a null image
    # This should get a "noise" map in the image plane
    block = simulate_vis(
        gal_arr=np.zeros((128, 128)),
        ra_gal=56,
        dec_gal=-30,
        robustness=robust,
        PSF=False,
        integration_time=int_time,
        summary=False,
    )
    return np.std(block[-1]) * 1e6


def PSNR(source, int_time, robust):
    """Function to calculate PSNR for a source at given integration time and robustness
    Repeat multiple times to estimate variation of PSNR as func of int time and R
    """
    flux = flux_all[source]
    r0 = size_all[source]
    e1gal = e1[source]
    e2gal = e2[source]
    ra_gal = ra[source]
    dec_gal = dec[source]

    # Assume exponential profile for simplicity
    gal_true = simulate_gal(flux, r0, e1gal, e2gal, simple=True)

    noise_block = simulate_vis(
        gal_arr=np.zeros((128, 128)),
        ra_gal=ra_gal,
        dec_gal=-dec_gal,
        robustness=robust,
        PSF=False,
        integration_time=int_time,
        summary=False,
    )
    ns = np.std(noise_block[-1])

    res_block = simulate_vis(
        gal_true,
        ra_gal=ra_gal,
        dec_gal=dec_gal,
        PSF=True,
        # noise_fac=0.3,
        robustness=robust,
        integration_time=int_time,
    )
    peak = res_block[0][1]
    psnr = peak / ns
    snr = helpers.calculate_SNR(gal_true, res_block[-2], border_width=10)[0]

    return [psnr, snr]


# %% Calculate expected Noise Levels for different integration time at a given R
"""
start = time.time()
integration_time = np.arange(60, 4200, 30)
r = 0
n_try = 100
# Path to save results
fname = "/home/tripathi/Main/noise_level_rob_n00.pkl"
if __name__ == "__main__":
    with LocalCluster(
        n_workers=60,
        processes=True,
        threads_per_worker=1,
        scheduler_port=8786,
        memory_limit=0,
    ) as cluster, Client(cluster) as client:
        print(client.dashboard_link)
        client.run(trim_memory)
        client.run(gc.collect)
        lazy_results = []
        n_int_time=1
        for int_time in integration_time:
            chunk_res = []
            for i in range(n_try):
                #print(f"Working on int time {n_int_time}/{len(integration_time)}, n_try:{i+1}/{n_try}")
                ns = dask.delayed(noise_level)(int_time, robust=r)
                chunk_res.append(ns)
            lazy_results.append(chunk_res)
            n_int_time+=1
        results = dask.compute(*lazy_results)

    helpers.save(np.array(results), fname)
    end = time.time()
    seconds = end - start
    print(f"Script took {helpers.secondsToStr(seconds)} to run.")

# %% Plot variation of noise as function of int time
plot=False
if plot:
    results = helpers.load(fname)

    mean_ns = np.mean(results, axis=1)
    max_ns = np.max(results, axis=1)
    min_ns = np.min(results, axis=1)
    fig, ax = plt.subplots()
    ax.plot(integration_time, mean_ns, label="Mean")
    ax.plot(integration_time, max_ns, label="Max")
    ax.plot(integration_time, min_ns, label="Min")
    ax.scatter(integration_time[np.argmin(mean_ns)], np.min(mean_ns), color="k")
    ax.scatter(integration_time[np.argmin(max_ns)], np.min(max_ns), color="k")
    ax.scatter(integration_time[np.argmin(min_ns)], np.min(min_ns), color="k")
    ax.set(
        xlabel="int time (s)", ylabel="Noise (uJy)", title=f"Noise level for robustness {r}"
    )
    ax.grid(True)
    ax.legend()
    plt.show()

# %% PSNR for 100 random sources at R=-0.5
start = time.time()
integration_time = np.arange(60, 4200, 30)
r = -0.5
n_try = 100
fname = "/home/tripathi/Main/psnr_level_rob_n05.pkl"
sources_all=np.random.randint(low=0, high=NSOURCE, size=n_try)
if __name__ == "__main__":
    with LocalCluster(
        n_workers=60,
        processes=True,
        threads_per_worker=1,
        scheduler_port=8786,
        memory_limit=0,
    ) as cluster, Client(cluster) as client:
        print(client.dashboard_link)
        client.run(trim_memory)
        client.run(gc.collect)
        lazy_results = []

        for int_time in integration_time:
            chunk_res = []
            for source in sources_all:
                psnr = dask.delayed(PSNR)(source, int_time, robust=r)
                chunk_res.append(psnr)
            lazy_results.append(chunk_res)

        results = dask.compute(*lazy_results)

    
    helpers.save(np.array(results), fname)
    end = time.time()
    seconds = end - start
    print(f"Script took {helpers.secondsToStr(seconds)} to run.")

# %% Variation of SNR as func of integration time
plot = False
if plot:
    try:
        results
    except Exception:
        results = helpers.load(fname)

    mean_snr = np.mean(results, axis=1)
    fig, ax = plt.subplots()
    ax.plot(integration_time, mean_snr)
    ax.scatter(integration_time[np.argmin(mean_snr)], np.min(mean_snr), color="k")
    ax.scatter(integration_time[np.argmax(mean_snr)], np.max(mean_snr), color="k")
    ax.set(
        xlabel="int time (s)",
        ylabel="SNR",
        title=f"Mean SNR for robustness {r}",
    )
    ax.grid(True)
    plt.show()
"""
# %% Calculate variation of SNR/PSNR for different R values
start = time.time()
integration_time = 300
rob = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
n_try = 1000
fname = "/home/tripathi/Main/psnr_level_diff_rob.pkl"
sources_all = np.random.randint(low=0, high=NSOURCE, size=n_try)
if __name__ == "__main__":
    with (
        LocalCluster(
            n_workers=40,
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
        lazy_results = []

        for r in rob:
            chunk_res = []
            for ns, source in enumerate(sources_all):
                psnr = dask.delayed(PSNR)(source, int_time=integration_time, robust=r)
                chunk_res.append(psnr)
            lazy_results.append(chunk_res)

        results = dask.compute(*lazy_results)

    # noise_levels = np.array(np.array_split(np.array(results), n_try))
    helpers.save(np.array(results), fname)
    end = time.time()
    seconds = end - start
    print(f"Script took {helpers.secondsToStr(seconds)} to run.")

# %% Plot variation of SNR/PSNR for different R values
plot = False
if plot:
    try:
        results
    except Exception:
        results = helpers.load(fname)

    mean_snr = np.mean(results, axis=1)
    for i, title in enumerate(["PSNR", "SNR"]):
        fig, ax = plt.subplots()
        ax.plot(rob, mean_snr[:, i])
        ax.scatter(rob[np.argmin(mean_snr[:, i])], np.min(mean_snr[:, i]), color="g")
        ax.scatter(rob[np.argmax(mean_snr[:, i])], np.max(mean_snr[:, i]), color="r")
        ax.set(
            xlabel="Robustness param",
            ylabel="Stat",
            title=title,
        )
        ax.grid(True)
        # ax.legend()
        plt.show()
