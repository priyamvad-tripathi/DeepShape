# %%
import sys

sys.path.append("../Functions/")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotting_func
from scipy.stats import binned_statistic as bs

# %% Load file
PATH_TEST = "/scratch/tripathi/Data/tss.h5"
hf = h5py.File(PATH_TEST, "r")

flags = hf["RLF flag"][:] == 0  # RadioLensfit flags
rlf = hf["RLF meas"][flags]  # RadioLensfit measurements

ytest = hf["input"][flags]  # Ground Truth
nn = hf["SMECNET"][flags]  # DeepShape measurements


# %%
def bias_dependence(
    delta_list,
    param,
    label="Flux" + "$~[\mu$" + "Jy]",
    legend=["RadioLensfit", "DeepShape"],
    bins=6,
    fname=None,
):
    fig = plt.figure(figsize=(6, 4.2))
    ax = fig.add_subplot(111)
    plt.sca(ax)
    colors = ["firebrick", "blue"]
    markers = ["o", "^"]

    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(param, percentiles)

    for i in range(2):
        statistic, bin_edges, bin_numbers = bs(
            param, delta_list[i], statistic="mean", bins=bin_edges
        )
        x = (bin_edges[1:] + bin_edges[:-1]) * 0.5

        ax.plot(
            x,
            np.sqrt(statistic),
            lw=1.5,
            marker=markers[i],
            color=colors[i],
            label=legend[i],
        )

    ax.set(
        ylabel=r"RMSE$_1$",
        xlabel=label,
        xlim=(48, 132),
        ylim=(0.029, 0.059),
        # yscale="log"
    )

    ax.legend()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if not fname:
        plt.show()
    else:
        plotting_func.savefig(fname)


# %%

delta1 = (rlf - ytest) ** 2
delta2 = (nn - ytest) ** 2

delta = [delta1[:, 0], delta2[:, 0]]

flux = hf["Flux"][flags] * 1e6
# inp = hf["input"][flags]
# inp = np.sqrt(inp[:, 0] ** 2 + inp[:, 1] ** 2)

bias_dependence(
    delta,
    flux,
    fname="/home/tripathi/Main/Images/Shape_Measurement/flux_depend.pdf",
)
