# %%
import sys

import h5py

sys.path.append("../Functions")
import helpers
import matplotlib.pyplot as plt
import numpy as np
import plotting_func
from matplotlib import container
from scipy.stats import linregress as lin

# Dataset containg details about the galaxy ensemble
data = helpers.load("/home/tripathi/shear_set_data.pkl")

colors = ["g", "m"]
# Load Data
input_shear = shear = data["shear"].copy()


# %% Shear Bias plot

output = []
peak = []
inp = []

for g in range(1, 101):
    PATH_SET = f"/scratch/tripathi/Shear_sets/test_set_{g}.h5"
    with h5py.File(PATH_SET, "r") as hf:
        peak = [*peak, hf["Peak"][:]]
        output = [*output, hf["SMECNET"][:]]

        inp = [*inp, hf["input"][:]]

output = np.array(output)
output_shear = np.mean(output, axis=1)
inp = np.array(inp)

plotting_func.shear_plot(
    output_shear,
    input_shear,
    lim=0.022,
    colors=colors,
    fname="/home/tripathi/Main/Images/Shear/bias2.pdf",
)


# %% Function to calulcate slope
def return_m(inp_shear, out_shear, comp=1):
    delta = out_shear - inp_shear
    res = lin(inp_shear[:, comp - 1], delta[:, comp - 1])

    return res.slope, res.stderr


# %% Dependence on flux cuts
labels = [r"$|\hat{M}_1|$", r"$|\hat{M}_2|$"]
markers = ["d", "d"]
cuts = np.linspace(50, 150, 6)
output_all = []
for cut in cuts:
    output_shear_cut = []
    good = data["Flux"] > cut * 1e-06
    for out in output:
        output_shear_cut = [*output_shear_cut, np.average(out[good], axis=0)]
    output_all = [*output_all, output_shear_cut]


fig = plt.figure(figsize=(6, 4.2))
ax = fig.add_subplot(111)
plt.sca(ax)

for i in [1, 2]:
    m = []
    delta_m = []
    for output_shear_cut in output_all:
        res = return_m(input_shear, output_shear_cut, comp=i)
        m = [*m, res[0]]
        delta_m = [*delta_m, res[1]]

    ax.errorbar(
        cuts,
        np.abs(m),
        yerr=delta_m,
        lw=1.5,
        marker=markers[i - 1],
        color=colors[i - 1],
        label=labels[i - 1],
        capsize=3,
    )


ax.set(
    ylabel=r"$|\hat{M}|$",
    xlabel="min flux",
    # yscale="log"
)
ax.axhline(
    0.0067,
    color="black",
    linewidth=1.2,
    linestyle=":",
    label="SKA-MID",
)

handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

ax.legend(handles, labels)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plotting_func.savefig("/home/tripathi/Main/Images/Shear/flux.pdf")


# %% Dependence on ellipticity cuts
e = inp[0] - input_shear[0]
mod_e = np.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2)
# cuts=np.percentile(mod_e,np.linspace(5,100,6))
cuts = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

output_all = []
for cut in cuts:
    output_shear_cut = []
    good = mod_e < cut
    for out in output:
        output_shear_cut = [*output_shear_cut, np.average(out[good], axis=0)]
    output_all = [*output_all, output_shear_cut]


fig = plt.figure(figsize=(6, 4.2))
ax = fig.add_subplot(111)
plt.sca(ax)

labels = [r"$|\hat{M}_1|$", r"$|\hat{M}_2|$"]
markers = ["d", "d"]
for i in [1, 2]:
    m = []
    delta_m = []
    for output_shear_cut in output_all:
        res = return_m(input_shear, output_shear_cut, comp=i)
        m = [*m, res[0]]
        delta_m = [*delta_m, 0.5 * res[1]]

    ax.errorbar(
        cuts,
        np.abs(m),
        yerr=delta_m,
        lw=1.5,
        marker=markers[i - 1],
        color=colors[i - 1],
        label=labels[i - 1],
        capsize=3,
    )


ax.set(
    ylabel=r"$|\hat{M}|$",
    xlabel="max ellipticity",
    # yscale="log"
)
ax.axhline(
    0.0067,
    color="black",
    linewidth=1.2,
    linestyle=":",
    label="SKA-MID",
)

handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

ax.legend(handles, labels)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plotting_func.savefig("/home/tripathi/Main/Images/Shear/ellipticity.pdf")
# plt.show()
