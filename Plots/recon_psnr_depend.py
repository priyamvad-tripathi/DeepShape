# %%Import Libraries
import sys

sys.path.append("../Functions/")


import h5py
import helpers
import matplotlib.pyplot as plt
import numpy as np
import plotting_func
from helpers import RMSE
from matplotlib import container

# %% Load Data

PATH_TEST = "/scratch/tripathi/Data/ts0.h5"

hf = h5py.File(PATH_TEST, "r")

inp0 = hf["input"][:]
dirty_all = hf["dirty"][:]
psf_all = hf["psf"][:]
true_all = hf["true image"][:]
recon_clean = hf["CLEAN"][:]
PSNR = hf["Peak"][:] / (0.71e-06)
size_all = hf["HLR"][:]
flux_all = hf["Flux"][:]
recon_pnp = hf["pnp"][:]
hf.close()

inp = []
for true in true_all:
    inp += [helpers.predict_shape(true)[0]]
inp = np.array(inp)


# %%


def NMSE(recon_all, true_all):
    diff_all = true_all - recon_all

    num = np.sum(np.array([np.linalg.norm(diff) ** 2 for diff in diff_all]))
    denom = np.sum(np.array([np.linalg.norm(true) ** 2 for true in true_all]))

    return num / denom


# %% Run on dirty images
epred_dirty = []
flags_dirty = []
for im in dirty_all:
    epred, flag = helpers.predict_shape(im)
    epred_dirty += [epred]
    flags_dirty += [flag]

flags_dirty = np.array(flags_dirty)
epred_dirty = np.array(epred_dirty)

print("Dirty Image")
RMSE(inp, epred_dirty, flags_dirty)

# %% Run on MS-CLEAN Images
epred_clean = []
flags_clean = []
for im in recon_clean:
    epred, flag = helpers.predict_shape(im)
    epred_clean += [epred]
    flags_clean += [flag]

flags_clean = np.array(flags_clean)
epred_clean = np.array(epred_clean)

print("CLEAN Image")
RMSE(inp, epred_clean, flags_clean)


# %% Run on PNP images
epred_pnp = []
flags_pnp = []
for im in recon_pnp:
    epred, flag = helpers.predict_shape(im)
    epred_pnp += [epred]
    flags_pnp += [flag]

flags_pnp = np.array(flags_pnp)
epred_pnp = np.array(epred_pnp)

_ = RMSE(inp, epred_pnp, flags_pnp)


# %% Binned
def plot_PSNR_depend(
    PSNR, recon_list, true_all, bins=6, legend=["MS-CLEAN", "HQS-PnP"], fname=None
):
    PSNR_max = 50

    bin_edges = np.linspace(0, PSNR_max, bins + 1)
    x = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    x = np.array([*x, 100 - (x[-1])])

    colors = ["firebrick", "blue"]
    markers = ["o", "^"]

    fig = plt.figure(figsize=(6, 4.2))
    ax = fig.add_subplot(111)
    plt.sca(ax)

    for n, recon_all in enumerate(recon_list):
        y = []
        for i in range(bins):
            good = np.where((PSNR > bin_edges[i]) & (PSNR <= bin_edges[i + 1]))
            nmse = NMSE(recon_all[good], true_all[good])
            y.append(nmse)

        good = np.where(PSNR > bin_edges[-1])
        y.append(NMSE(recon_all[good], true_all[good]))

        ax.plot(
            x,
            y,
            lw=1.5,
            marker=markers[n],
            color=colors[n],
            label=legend[n],
        )

    ax.set(
        xlabel="Peak S/N",
        ylabel="NMSE",
    )

    ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    handles = [
        h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles
    ]

    ax.legend(handles, labels)

    if not fname:
        plt.show()

    else:
        plotting_func.savefig(fname)


plot_PSNR_depend(
    PSNR,
    recon_list=[recon_clean, recon_pnp],
    true_all=true_all,
    bins=8,
    fname="/home/tripathi/Main/Images/Reconstruction/PSNR.pdf",
)
