"""
Script to initilize and pre-train the shape measurement network using true images.
"""

# %%Import Libraries
import random
import sys

sys.path.append("../Functions/")

import numpy as np
import plotting_func
import torch
import torch_func
from escnn import gspaces, nn

# %% Define Paths
DIR = "/scratch/tripathi/"

loc_weights = DIR + "Model_Weights/true_images.pt"
loc_train_set = DIR + "Data/true_sers.h5"

PATH_TEST = DIR + "Data/test_set.h5"

# Load Data
train_loader, val_loader = torch_func.dataloader(
    path=loc_train_set,
    x_key=["true image"],
    y_key=["input"],
    split=[0.8, 0.2],
    batch_size=[64, 64],
)


# %% Torch Parameters
ndev = 1
torch.cuda.set_device(ndev)
torch.cuda.empty_cache()
device = torch.device(f"cuda:{ndev}" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(True)

# Set seed for reproducibility
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


#! The autoencoder is not used for the pre-training
# Full Model
class Ymodel(torch.nn.Module):
    def __init__(self, eq_model=eq_model):
        super().__init__()

        self.eq = eq_model

        c1 = 32

        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(c1 * 12 * 12),
            torch.nn.ReLU(),
            torch.nn.Linear(c1 * 12 * 12, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.Tanh(),
        )

    def forward(self, im: torch.Tensor):
        feat = self.eq(im)
        out = self.fully_net(feat)

        return out


model = Ymodel()
model.to(device)
print(model(torch.randn(10, 1, 128, 128).to(device)).size())

# %% Train the model and use it to make predictions
#! Test with different paramters for best results
n_epochs = 601

scheduler_params = {"factor": 0.5, "patience": 40, "min_lr": 1e-06}

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5
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
try:
    best_weights
except NameError:
    print("Best weights not found. Now loading from disk")
    checkpoint = torch.load(loc_weights[:-3] + "_checkpoint.pt", map_location=device)
    best_weights = checkpoint["best_weights"]
    val_loss_list = checkpoint["val_loss_list"]
    train_loss_list = checkpoint["train_loss_list"]

plotting_func.plot_loss(train_loss_list, val_loss_list, tp=True, skip=10)

ypred_all, ytest = torch_func.predict(
    model,
    weights=best_weights,
    dataloader=val_loader,
    device=device,
)

# Calculate Bias
plotting_func.plot_bias_binned(
    ypred_all, ytest, pow=1e4, lim=0.05, ellipticity_cutoff=0.6
)
print(
    f"The pearson coefficients are: {1 - np.corrcoef(ytest[:, 0], ypred_all[:, 0])[0, 1]:.2e}/{1 - np.corrcoef(ytest[:, 1], ypred_all[:, 1])[0, 1]:.2e}"
)

# %% Save eq weights
model.eval()
model.load_state_dict(best_weights)
torch.save(model.eq.state_dict(), DIR + "Model_Weights/base_eq.pt")
# %% Calculate ellipticity for test subset
test_loader = torch_func.dataloader(
    path=PATH_TEST,
    x_key=["true image"],
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
    f"The pearson coefficients are: {1 - np.corrcoef(ytest[:, 0], ypred_all[:, 0])[0, 1]:.2e}/{1 - np.corrcoef(ytest[:, 1], ypred_all[:, 1])[0, 1]:.2e}"
)
