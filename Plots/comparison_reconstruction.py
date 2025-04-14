# %% Import Libraries and Modules
import sys

sys.path.append("../Functions/")

import helpers
import numpy as np
import torch
import torch.nn as nn

# import matplotlib.pyplot as plt
from deepinv.models import DRUNet
from plotting_func import plot
from torch.fft import fft2, ifft2, ifftshift

device = torch.device("cuda:2")


# %% Define Pytorch functions
class IterationStep(nn.Module):
    def __init__(self):
        super().__init__()

    def iter(self, z, dirty, fpsf, alpha):
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


denoiser = DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)
# %% Define Test Data
# Subset of galaxies used for plotting
data = np.load("/scratch/tripathi/Data/cleaned.npy")


true = data[0].copy()
psf = data[1].copy()
dirty = data[4].copy()

im = torch.from_numpy(np.array([dirty, psf]).swapaxes(0, 1)).to(device)

# %% Define Model
niter = 50
f1 = 0.0528
f2 = 0.3391
falpha = 2.8647


model = HQS_PnP(
    niter=niter,
    f1=f1,
    f2=f2,
    falpha=falpha,
    denoiser=denoiser,
    SIGMA=0.71e-06,
).to(device)


# %%
recon = model(im)
recon = recon.detach().cpu().numpy().squeeze()


# %%
res_clean = data[6]
res_pnp = helpers.load(
    "/scratch/tripathi/Data/residual.pkl"
)  # Residuals from HQS-PnP and MS-CLEAN
plot(
    np.array([res_clean, np.array(res_pnp)]).swapaxes(0, 1),
    caption=["MS-CLEAN", "HQS-PnP"],
    cbar=True,
    same_scale=[0, 1],
    fname="/home/tripathi/Main/Images/Reconstruction/residual.pdf",
)


# %%
# PSNR calculated beforehand
PSNR = [9.64, 24.23, 27.95, 28.64]


plot(
    np.array([true, psf, dirty, data[5].copy(), recon]).swapaxes(0, 1),
    caption=["Ground Truth", "PSF", "Dirty", "MS-CLEAN", "HQS-PnP"],
    cbar=True,
    text=PSNR,
    text_row=2,
    same_scale=[3, 4],
    scale_row=4,
    fname="/home/tripathi/Main/Images/Reconstruction/recon.pdf",
)
