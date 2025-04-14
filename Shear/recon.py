# %%Import Libraries
import sys

sys.path.append("../Functions")

import copy

import h5py
import numpy as np
import torch
import torch.nn as nn
from deepinv.models import DRUNet
from torch.fft import fft2, ifft2, ifftshift
from tqdm import tqdm

device = torch.device("cuda:2")


# %% Load Model
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


class UnfoldedAlgorithm(nn.Module):
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

niter = 30
SIGMA = 0.71e-06
f1 = 0.0528
f2 = 0.3391
falpha = 2.8647

model = UnfoldedAlgorithm(
    niter=niter,
    f1=f1,
    f2=f2,
    falpha=falpha,
    denoiser=denoiser,
    SIGMA=SIGMA,
).to(device)


# %% Perform recon on each shear set

for g in [100]:
    PATH_SET = f"/scratch/tripathi/Shear_sets/test_set_{g}.h5"

    hf = h5py.File(PATH_SET, "a")

    dirty_all = hf["dirty"][:]
    psf_all = hf["psf"][:]

    im_all = np.swapaxes(np.array([dirty_all, psf_all]), 0, 1)
    recon_all = []

    batches = np.array_split(im_all, len(im_all) / 64)
    for im in tqdm(batches, unit=" batch", colour="green", desc=f"Set: {g}/80"):
        recon = model(torch.from_numpy(im).to(device))
        recon_np = recon.detach().cpu().numpy().squeeze()
        recon_all = [*recon_all, *recon_np]

    if "pnp" in hf.keys():
        del hf["pnp"]

    hf.create_dataset(name="pnp", data=copy.deepcopy(np.array(recon_all)))
