# %%Import Libraries
import random
import sys

sys.path.append("../Functions")

import copy

import h5py
import helpers
import numpy as np
import torch
import torch_func
from escnn import gspaces, nn

DIR = "/scratch/tripathi/"

loc_weights = DIR + "Model_Weights/network.pt"  # Model Weights for DeepShape
data = helpers.load("/home/tripathi/shear_set_data.pkl")  # Load Data about ensemble

# %% Torch Parameters
ndev = 0
torch.cuda.set_device(ndev)
torch.cuda.empty_cache()
device = torch.device(f"cuda:{ndev}" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(True)


# Set seed for reproducibility
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2004)

# %% DeepShape Model Architecture


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

# Load encoder
autoencoder = torch.jit.load(
    DIR + "Model_Weights/autoencoder_jit.pt",
    map_location=device,
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

checkpoint = torch.load(loc_weights, map_location=device)
best_weights = checkpoint["model_state_dict"]

# %%  Calculate shear from TS1 set
save_res = True
out = []
for g in range(1, 101):
    PATH_SET = f"/scratch/tripathi/Shear_sets/test_set_{g}.h5"

    print(f"Working on set {g}/100")
    test_loader = torch_func.dataloader(
        path=PATH_SET,
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

    print(
        f"The pearson coefficients are: {np.corrcoef(ytest[:, 0], ypred_all[:, 0])[0, 1]:.4f}/{np.corrcoef(ytest[:, 1], ypred_all[:, 1])[0, 1]:.4f}"
    )

    del test_loader

    out = [*out, ypred_all]
    if save_res:
        with h5py.File(PATH_SET, "a") as hf:
            if "Net" in hf.keys():
                del hf["Net"]

            hf.create_dataset(
                name="Net",
                data=copy.deepcopy(ypred_all),
            )
