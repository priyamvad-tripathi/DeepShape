# %%
import sys

sys.path.append("../Functions/")


import h5py
import helpers
import numpy as np
import plotting_func

# %% Load file
PATH_TEST = "/scratch/tripathi/Data/tss.h5"
hf = h5py.File(PATH_TEST, "r")

flags = hf["RLF flag"][:] == 0  # RadioLensfit flags
rlf = hf["RLF meas"][flags]  # RadioLensfit measurements

ytest = hf["input"][flags]  # Ground Truth
nn = hf["Net"][flags]  # DeepShape measurements

# %% Individual plots
plotting_func.plot_bias_binned(
    rlf,
    ytest,
    pow=1e4,
    lim=0.5,
    ellipticity_cutoff=1,
    hist=False,
    bias_line=True,
)

print(
    f"RLF : {np.corrcoef(ytest[:, 0], rlf[:, 0])[0, 1]:.4f}/{np.corrcoef(ytest[:, 1], rlf[:, 1])[0, 1]:.4f}"
)

helpers.RMSE(ytest, rlf)
plotting_func.plot_bias_binned(
    nn,
    ytest,
    pow=1e4,
    lim=0.5,
    ellipticity_cutoff=1,
    hist=False,
    bias_line=True,
)
print(
    f"NN : {np.corrcoef(ytest[:, 0], nn[:, 0])[0, 1]:.4f}/{np.corrcoef(ytest[:, 1], nn[:, 1])[0, 1]:.4f}"
)
helpers.RMSE(ytest, nn)

# %% Plotting
plotting_func.contour_plot(
    ytest[:, 0],
    [rlf[:, 0], nn[:, 0]],
    legends=["RadioLensfit", "DeepShape"],
    lim=0.42,
    pow=1e3,
    fname="/home/tripathi/Main/Images/Shape_Measurement/RLF.pdf",
)
