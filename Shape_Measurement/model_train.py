"""
Script to initilize and train the shape measurement network using reconstructed image-PSF pairs.
Ensure that the path to the autoencoder model and the training/test sets is coorectly speficied.
"""

# %%Import Libraries
import random
import sys

sys.path.append("../Functions")


import copy

import h5py
import helpers
import numpy as np
import plotting_func
import torch
import torch_func
from escnn import gspaces, nn

# %% Define Paths
DIR = "/scratch/tripathi/"

loc_weights = DIR + "Model_Weights/network.pt"  # Path to save model weights
loc_train_set = DIR + "Data/dataset_dirty_train_sers.h5"  # Path to training set
loc_eq = DIR + "Model_Weights/base_eq.pt"  # Path to pre-trained model weights
loc_autoencoder = (
    DIR + "Model_Weights/autoencoder_jit.pt"
)  # Path to autoencoder model (JIT)
PATH_TEST = DIR + "Data/test_set.h5"  # Path to test sets

# Load Data
train_loader, val_loader = torch_func.dataloader(
    path=loc_train_set,
    x_key=["decon", "psf"],
    y_key=["input"],
    split=[0.8, 0.2],
    batch_size=[64, 64],
    peak=3 * 0.71 * 1e-06,
)


# %% Torch Parameters
ndev = 2
torch.cuda.set_device(ndev)
torch.cuda.empty_cache()
device = torch.device(f"cuda:{ndev}" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(True)
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2004)

# %% Model Architecture


class SO2SteerableCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # the model is equivariant under all planar rotations
        self.r2_act = gspaces.flipRot2dOnR2(N=-1)
        self.G = self.r2_act.fibergroup

        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        self.mask = nn.MaskModule(in_type, 128, margin=4)

        # convolution 1
        activation1 = nn.FourierELU(self.r2_act, 32, irreps=self.G.bl_irreps(2), N=8)
        out_type = activation1.in_type
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=4, bias=False, padding=1),
            nn.IIDBatchNorm2d(out_type),
            activation1,
            nn.PointwiseAvgPool2D(out_type, kernel_size=4),
        )

        # convolution 2
        in_type = self.block1.out_type
        activation2 = nn.FourierELU(self.r2_act, 64, irreps=self.G.bl_irreps(2), N=8)
        out_type = activation2.in_type
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=4, bias=False),
            activation2,
            nn.FieldDropout(out_type, p=0.3),
            nn.PointwiseAvgPool2D(out_type, kernel_size=2),
        )

        # convolution 3
        in_type = self.block2.out_type
        activation3 = nn.FourierELU(self.r2_act, 64, irreps=self.G.bl_irreps(2), N=8)
        out_type = activation3.in_type
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, bias=False),
            activation3,
        )

        # number of output invariant channels
        c = 32

        # last 1x1 convolution layer, which maps the regular fields to c=128 invariant scalar fields
        output_invariant_type = nn.FieldType(
            self.r2_act, c * [self.r2_act.trivial_repr]
        )
        self.invariant_map = nn.R2Conv(
            out_type, output_invariant_type, kernel_size=1, bias=False
        )

    def forward(self, input: torch.Tensor):
        x = self.input_type(input)
        x = self.mask(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # extract invariant features
        x = self.invariant_map(x)

        # unwrap the output GeometricTensor
        x = x.tensor
        return x


eq_model = SO2SteerableCNN()
eq_model = eq_model.to(device)
if loc_eq:
    ckp_eq = torch.load(loc_eq, map_location=device)
    eq_model.load_state_dict(ckp_eq)

# Load encoder
autoencoder = torch.jit.load(
    loc_autoencoder,
    map_location=f"cuda:{ndev}",
)
autoencoder.eval()


# Full Model
class Ymodel(torch.nn.Module):
    def __init__(self, eq_model=eq_model, encoder=autoencoder):
        super().__init__()

        self.eq = eq_model
        self.encode = autoencoder.encoder

        c1 = 32 * 12 * 12
        c2 = 1152

        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c1 + c2),
            torch.nn.ReLU(),
            torch.nn.Linear(c1 + c2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.Tanh(),
        )

    def forward(self, input: torch.Tensor):
        im = input[:, 0, :, :].unsqueeze(1)
        psf = input[:, 1, :, :].unsqueeze(1)

        im = self.eq(im)
        psf = self.encode(psf)

        im = torch.nn.Flatten()(im)

        features = torch.cat((im, psf), dim=1)

        out = self.fully_net(features)

        return out


model = Ymodel()
model.to(device)
print(model(torch.randn(10, 2, 128, 128).to(device)).shape)

# Freeze Encoder Layer
for param in model.encode.parameters():
    param.requires_grad = False


# %% Train the model
#! Test with different paramters for best results
n_epochs = 601

scheduler_params = {"factor": 0.5, "patience": 30, "min_lr": 1e-06}

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5
)

best_weights, train_loss_list, val_loss_list = torch_func.train(
    model,
    train_loader,
    val_loader,
    epochs=n_epochs,
    device=device,
    filename=loc_weights,
    plot=True,
    optimizer=optimizer,
    scheduler_params=scheduler_params,
    save_freq=10,
)

# %%  Calculate bias on validation set

checkpoint = torch.load(loc_weights, map_location=device)
best_weights = checkpoint["best_weights"]
val_loss_list = checkpoint["val_loss_list"]
train_loss_list = checkpoint["train_loss_list"]

plotting_func.plot_loss(train_loss_list, val_loss_list, tp=True, skip=1)

ypred_all, ytest = torch_func.predict(
    model,
    weights=best_weights,
    dataloader=val_loader,
    device=device,
)

# Calculate Bias
plotting_func.plot_bias_binned(
    ypred_all,
    ytest,
    pow=1e3,
    lim=0.5,
    ellipticity_cutoff=0.6,
)
print(
    f"The pearson coefficients are: {np.corrcoef(ytest[:, 0], ypred_all[:, 0])[0, 1]:.4f}/{np.corrcoef(ytest[:, 1], ypred_all[:, 1])[0, 1]:.4f}"
)

# %% Calculate ellipticity for test set
save_results = True
test_loader = torch_func.dataloader(
    path=PATH_TEST,
    x_key=["pnp", "psf"],
    y_key=["input"],
    split=1,
    batch_size=64,
)
ypred_all, ytest = torch_func.predict(
    model,
    weights=best_weights,
    dataloader=test_loader,
    device=device,
)


plotting_func.plot_bias_binned(
    ypred_all,
    ytest,
    pow=1e3,
    lim=0.2,
    ellipticity_cutoff=0.6,
    bias_line=True,
)
print(
    f"The pearson coefficients are: {np.corrcoef(ytest[:, 0], ypred_all[:, 0])[0, 1]:.4f}/{np.corrcoef(ytest[:, 1], ypred_all[:, 1])[0, 1]:.4f}"
)

del test_loader
helpers.RMSE(ytest, ypred_all)
if save_results:
    with h5py.File(PATH_TEST, "a") as hf:
        if "Net" in hf.keys():
            del hf["Net"]

        hf.create_dataset(
            name="Net",
            data=copy.deepcopy(ypred_all),
        )
