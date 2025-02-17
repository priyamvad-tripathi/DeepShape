"""
This script is used to tune the hyperparameters of the PnP algorithm using Optuna.
Extremely important to run this for a novel dataset.
Run only once since it is time comsuming and save the best parameters to be used on the full dataset.
"""

# %% Import Libraries and Modules
import sys

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.fft import fft2, ifft2, ifftshift

sys.path.append("/../Functions/")


import torch_func
from deepinv.models import DRUNet
from plotting_func import plot

device = torch.device("cuda:0")  #! Check GPU

PATH_VAL_DATA = "/scratch/tripathi/Data/val_set_1k.h5"


# %% Define Pytorch Model for HQS-PnP
class IterationStep(nn.Module):
    """
    Define a single iteration HQS iteration.
    Eq. 13 in DeepShape Paper.
    """

    def __init__(self):
        super().__init__()

    def iter(self, z, dirty, fpsf, alpha):
        """
        Parameters:
        ----------
        z (numpy.ndarray): The current estimate of the auxillary variable: z_k.
        dirty (numpy.ndarray): The observed dirty image.
        fpsf (numpy.ndarray): The Fourier transform of the point spread function (PSF).
        alpha (float): Defined as \mu_k\sigma^2.

        Returns:
        -------
        numpy.ndarray: The updated image estimate after one iteration.
        """

        numerator = fpsf.conj() * fft2(dirty) + alpha * fft2(z)
        denominator = (fpsf.conj() * fpsf).real + alpha
        x_fft = numerator / denominator
        x = ifft2(x_fft).real
        return x

    def forward(self, z, dirty, fpsf, alpha):
        x = self.iter(z, dirty, fpsf, alpha)
        return x


class HQS_PnP(nn.Module):
    def __init__(
        self, niter, f1, f2, falpha, denoiser, SIGMA=0.71e-06, iteration=IterationStep()
    ):
        """
        Overall HQS-PnP Algorithm. SIGMA is the image plane noise level calculated using simulations.
        f1, f2, falpha are the hyperparameters to be optimized.
        Denoiser is the neural network used for denoising. We use the DRUNet in this case.
        """

        super().__init__()
        self.steps = nn.ModuleList([iteration for _ in range(niter)])
        self.denoiser = denoiser
        self.sigma_k = np.geomspace(f1 * SIGMA, f2 * SIGMA, niter)[::-1]
        self.alpha_k = falpha * (SIGMA**2) / (self.sigma_k**2)

    def forward(self, im):
        dirty = im[:, 0, :, :].unsqueeze(1)
        psf = im[:, 1, :, :].unsqueeze(1)
        fpsf = fft2(ifftshift(psf, dim=(-2, -1)))

        z = dirty.clone().detach()

        for alpha, sigma, step in zip(self.alpha_k, self.sigma_k, self.steps):
            x = step(z, dirty, fpsf, alpha).clone().detach()
            z = self.denoiser(x, sigma).clone().detach()
        return z


# %% Load Denoiser
# pretained="download" downloads the pretrained model from the internet. See deepinv.models.DRUNet for more details.
# For fine-tuned weights, set pretrained to the path of the weights file.
# If pretrained is set to None, the model will be initialized with random weights.
denoiser = DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)


# %% Define Constants
niter = 30  # No of iterations
SIGMA = 0.71e-06  # Image plane noise. Calculated using simulations for the used observation parameters.

# %% Define Datasets
"""
data = np.load("/home/tripathi/Sample_Data_sers_3_5.npy")
true = data[0].copy()
psf = data[1].copy()
dirty = data[-1].copy()
peak = np.max(data[2], axis=(-2, -1))
im = torch.from_numpy(np.array([dirty, psf]).swapaxes(0, 1)).to(device)
"""
# Load Data
train_loader = torch_func.dataloader(
    path=PATH_VAL_DATA,
    x_key=["dirty", "psf"],
    y_key=["true image"],
    split=1,
    batch_size=96,
    peak=3 * SIGMA,
    scale=False,
)

# Save first batch for visualization
im, true = next(iter(train_loader))
dirty = im[:, 0].detach().cpu().numpy().squeeze()
psf = im[:, 1].detach().cpu().numpy().squeeze()
true_np = true.detach().cpu().numpy().squeeze()


# %% Optuna Optimization
def objective(trial):
    """
    Objective function for Optuna tuning.
    We tune f1, f2, falpha using the range mentioned below.
    """
    f1 = trial.suggest_float("f1", 0, 1)
    f2 = trial.suggest_float("f2", 0, 2)
    falpha = trial.suggest_float("falpha", 0, 4)

    model = HQS_PnP(
        niter=niter,
        f1=f1,
        f2=f2,
        falpha=falpha,
        denoiser=denoiser,
        SIGMA=SIGMA,
    ).to(device)

    total_loss = 0
    #  Run on Data
    for dirty, true in train_loader:
        im = dirty.to(device)
        recon = model(im)

        loss = torch.sum(
            torch.linalg.norm(recon.squeeze(1) - true.to(device), dim=(-1, -2))
            / torch.linalg.norm(true.to(device), dim=(-1, -2))
        )
        total_loss += loss.item()
    return total_loss / len(train_loader.dataset)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=2500)

# Best Params
f1 = study.best_params["f1"]
f2 = study.best_params["f2"]
falpha = study.best_params["falpha"]
# %% Plot 1
optuna.visualization.plot_parallel_coordinate(study)
# %% Plot 2
optuna.visualization.plot_optimization_history(study)
# %% Plot 3
optuna.visualization.plot_param_importances(study)

# %% Define Model
model_best = HQS_PnP(
    niter=niter,
    f1=f1,
    f2=f2,
    falpha=falpha,
    denoiser=denoiser,
    SIGMA=SIGMA,
).to(device)

#  Run on First Batch for visualization
recon = model_best(im.to(device))
recon_np = recon.detach().cpu().numpy().squeeze()

plot(
    [true, psf, dirty, recon_np],
    titles=["True", "PSF", "Obs", "Recon"],
    # same_scale=[0, 3],
    cbar=True,
)


# %% Run on common set
"""
loc_test_set = "/scratch/tripathi/Rivi/common_test_set.pkl"
dataset_test = helpers.load(loc_test_set)

for data in dataset_test.values():
    dirty = data["dirty"].copy()
    psf = data["psf"].copy()
    im = torch.from_numpy(np.array([dirty, psf]).swapaxes(0, 1))

    recon_np = []

    for im_chunk in torch.split(im, 64):
        recon = model_best(im_chunk.to(device))
        recon_np = [*recon_np, *recon.detach().cpu().numpy().squeeze()]

    data["decon"] = np.array(recon_np).copy()

helpers.save(dataset_test, loc_test_set)

# %% Run on the full dataset

# Load Training Data
PATH = "/scratch/tripathi/Recon/dataset_decon.h5"
hf = h5py.File(PATH, "a")

if "decon" in hf.keys():
    del hf["decon"]

tot = len(hf["psf"])

start = time.time()
decon = hf.create_dataset(
    name="decon",
    shape=(tot, 128, 128),
)
for nc, chunck in enumerate(np.array_split(np.arange(tot), int(tot / 64))):
    ckp = time.time()
    print(f"Working on chunck {nc+1}/{int(tot / 64)}")
    print(f"Time elapsed: {secondsToStr(ckp-start)}")

    N = len(chunck)

    psf_chunck = copy.deepcopy(hf["psf"][chunck])
    dirty_chunck = copy.deepcopy(hf["dirty"][chunck])
    im_chunck = torch.from_numpy(
        np.array([dirty_chunck, psf_chunck]).swapaxes(0, 1)
    ).to(device)

    recon = model_best(im_chunck)
    recon_np = recon.detach().cpu().numpy().squeeze()
    decon[chunck] = recon_np


hf.close()
end = time.time()
print(f"Total time taken: {secondsToStr(end-start)}")
"""
