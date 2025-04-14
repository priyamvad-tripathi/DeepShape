"""
This script is used to measure the ellipticity of the sources in the test set using RadioLensfit.
Ensure that RadioLensfit is installed and the script is placed in the RadioLesnfit directory.
"""

# %% Import Modules and constants
import sys

sys.path.append("../../Functions")
import copy
import logging
import subprocess
import time
import warnings
from os import system

import helpers
import numpy as np
from colorist import Color
from rascil.processing_components import export_visibility_to_ms

warnings.warn = lambda *args, **kwargs: None

DIR = "/home/tripathi/RadioLensfit/"

FLUX_LIM = (50, 200)
NSOURCE, ra, dec, flux_all, size_all, e1, e2 = helpers.load_catalog(*FLUX_LIM)
warnings.warn = lambda *args, **kwargs: None
logging.getLogger().addHandler(logging.NullHandler())

import h5py

PATH_hf = "/scratch/tripathi/Data/tss.h5"
# %% Measurement function


def measure(vt, ellipticity, flux, r0, SNR, summary=True):
    e1gal = ellipticity[0]
    e2gal = ellipticity[1]

    export_visibility_to_ms(DIR + "example.ms", [vt])

    #! Write to catalog text file
    with open(DIR + "catalog", "w") as file:
        file.write(f"{SNR} 0 0 {flux * 1e6:.02f} {r0:.04f} {e1gal:0.04f} {e2gal:.04f}")

    #! Run RadioLensit2 on generated files
    # The timeout is set to manually prune measurements taking more than 20 minutes
    try:
        subprocess.run(
            [DIR + "RadioLensfit2", "catalog", "1", "example.ms/"], timeout=1200
        )
    except Exception:
        print(f"{Color.RED}Timed Out{Color.OFF}")
        return 0

    data = np.loadtxt(
        DIR + "results.txt", delimiter="|", skiprows=1
    )  #! Load results (results.txt should be replaced with the actual output file from RadioLensfit)

    try:
        flag = data[-1]
        e1_m = data[2]
        e2_m = -data[
            5
        ]  # The negative sign is used to convert the RadioLensfit output to the ellipticity convention used by DeepShape
        e1_err = data[3]
        e2_err = data[6]
        SNR = data[8]

    except Exception:
        return 0

    #! Remove saved files as this may cause error
    system(f"rm -r {DIR}example.ms/")
    system(f"rm -r {DIR}catalog")
    system(f"rm {DIR}results.txt")

    if summary:
        print(f"Flux={flux * 1e6:.02f} uJy, SNR_vis={SNR}")
        print(
            f"Input Ellipticity: {Color.GREEN}{e1gal:0.03f}{Color.OFF}, {Color.GREEN}{e2gal:0.03f}{Color.OFF}"
        )
        print(
            f"Measured Ellipticity: {Color.GREEN}{e1_m:0.03f} \u00b1 {e1_err:0.03f}{Color.OFF}, {Color.GREEN}{e2_m:0.03f} \u00b1 {e2_err:0.03f}{Color.OFF}"
        )
        print(
            f"Absolute Error: {Color.GREEN}{np.abs(e1gal - e1_m):.04f}{Color.OFF}, {Color.GREEN}{np.abs(e2gal - e2_m):.04f}{Color.OFF}"
        )
        if flag == 1:
            print(f"{Color.RED}Flagged measurement{Color.OFF}")

    return np.array([[e1gal, e2gal], [e1_m, e2_m], [e1_err, e2_err]]), SNR, flag


# %% Run Main Script


start = time.time()


hf = h5py.File(PATH_hf, "a")

ellipticity_all = hf["input"][:]
flux_all = hf["Flux"][:]
r0_all = hf["HLR"][:] / np.log(2)
SNR_all = hf["SNR_vis"][:]
N = len(SNR_all)
chuncks = np.array_split(np.arange(N), int(N / 100))


hf_old = h5py.File("/scratch/tripathi/Data/val_set_small.h5", "r")

RLF_SNR = hf.create_dataset(
    name="RLF SNR_vis", maxshape=(None,), data=hf_old["RLF SNR_vis"][:].copy()
)
RLF_flag = hf.create_dataset(
    name="RLF flag", maxshape=(None,), data=hf_old["RLF flag"][:].copy()
)
meas = hf.create_dataset(
    name="RLF meas", maxshape=(None, 2), data=hf_old["RLF meas"][:].copy()
)
err = hf.create_dataset(
    name="RLF err", maxshape=(None, 2), data=hf_old["RLF err"][:].copy()
)

dsets = [RLF_SNR, RLF_flag, meas, err]
hf_old.close()

ns = 1
for NC, chunck in enumerate(chuncks):
    vt_list = helpers.load(f"/scratch/tripathi/Data/vis/vis_{NC + 1}.pkl")
    block_all = []
    flags = []
    SNRs = []

    for i, ind in enumerate(chunck):
        vt = vt_list[i]
        ellipticity = ellipticity_all[ind]
        flux = flux_all[ind]
        r0 = r0_all[ind]
        SNR = SNR_all[ind]

        time_check = time.time()

        print("-" * 150)
        print(f"Source: Chunck {NC + 1}/10. N_Source {ns}/{len(ellipticity_all)}")
        print(f"Time elapsed: {helpers.secondsToStr(time_check - start)}")
        print("-" * 150)

        try:
            block, SNR, flag = measure(vt, ellipticity, flux, r0, SNR, summary=True)
        except Exception:
            block = np.ones((3, 2))
            flag = -1
            SNR = 0
        block_all = [*block_all, block]
        flags = [*flags, flag]
        SNRs = [*SNRs, SNR]
        ns += 1

    del vt_list
    block_all = np.array(block_all)
    N = len(block_all)
    for dset in dsets:
        dset.resize(dset.shape[0] + N, axis=0)

    RLF_SNR[-N:] = copy.deepcopy(SNRs)
    RLF_flag[-N:] = copy.deepcopy(flags)
    meas[-N:] = copy.deepcopy(block_all[:, 1])
    err[-N:] = copy.deepcopy(block_all[:, 2])

hf.close()
end = time.time()
seconds = end - start
print(f"Script took {helpers.secondsToStr(seconds)} to run.")
