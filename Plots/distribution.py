# %%
import sys

import h5py

sys.path.append("../Functions/")
import helpers
import matplotlib.pyplot as plt

# import seaborn as sns
import numpy as np
import plotting_func
from matplotlib.gridspec import GridSpec

plt.rcParams.update(
    {
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


# %%
ts0 = h5py.File(
    "/scratch/tripathi/Data/ts0.h5", "r"
)  # Test set containing 20k galaxies (TS0)
train_set = h5py.File(
    "/scratch/tripathi/Data/dataset_dirty_train_sers.h5", "r"
)  # Traing set (250k galaxies)


_, _, _, flux_all, size_all, _, _ = helpers.load_catalog(50, 200)


# %% 2D Plot
def plot_2D(flux, HLR, fname=None):
    ind = HLR < 2.5
    HLR = HLR[ind]
    flux = flux[ind]

    # Create the figure and grid layout
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(4, 4, figure=fig)

    # Main 2D histogram plot
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    hb = ax_main.hexbin(flux, HLR, gridsize=50, cmap="plasma")
    ax_main.set_xlabel("Flux [" + r"$\mu$" + "Jy]")
    ax_main.set_ylabel("Half light radius [arcsec]")

    ax_main.set(xlim=(50, 200), ylim=(min(HLR), 2.5))

    # Marginal histogram for x-axis
    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_histx.hist(flux, bins=50, color="grey")
    ax_histx.axis("off")

    # Marginal histogram for y-axis
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    ax_histy.hist(HLR, bins=50, color="grey", orientation="horizontal")
    ax_histy.axis("off")

    # Add a color bar
    cb = fig.colorbar(hb, ax=ax_histy, orientation="vertical", pad=0.02)
    cb.set_label("Counts")

    plotting_func.savefig(fname)


# %% 2D Plot
plot_2D(
    flux_all * 1e6,
    size_all * np.log(2),
    fname="/home/tripathi/Main/Images/Dist/TRECS.pdf",
)

# %% 2D Plot
flux1 = ts0["Flux"][:] * 1e6
HLR1 = ts0["HLR"][:]

plot_2D(flux1, HLR1, fname="/home/tripathi/Main/Images/Dist/TS0.pdf")


# %% PSNR Plot
PSNR = ts0["Peak"][:] / (0.71e-06)


bin_edges = np.linspace(0, 50, 9)
x = (bin_edges[:-1] + bin_edges[1:]) * 0.5
x = np.array([*x, 100 - (x[-1])])

# Compute histograms (density=True normalizes the histogram)
measured_hist, _ = np.histogram(PSNR, bins=[*bin_edges, 115])


# Plot the histograms
plt.figure(figsize=(6, 4.2))
plt.hist(
    measured_hist,
    bins=x,
    color="grey",
    # density=True
)

# Customize the plot
plt.xlabel("Peak S/N")
plt.show()
