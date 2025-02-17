"""
Script to initilize and train an autoencoder model on PSF data.
Save the trained model for use in the shape measurement network.
"""

# %%Import Libraries
import copy
import random
import sys

sys.path.append("../Functions")
import time

import numpy as np
import plotting_func
import torch
import torch.nn as nn
import torch.optim as optim
import torch_func
from helpers import secondsToStr
from matplotlib import pyplot as plt
from plotting_func import plot_loss
from tqdm import tqdm

# %% Define Paths
DIR = "/scratch/tripathi/"

loc_weights = DIR + "Model_Weights/autoencoder_new.pt"
loc_train_set = DIR + "Data/PSFs.h5"

# Load Data
train_loader, val_loader = torch_func.dataloader(
    path=loc_train_set,
    x_key=["psf"],
    y_key=None,
    split=[0.8, 0.2],
    batch_size=[64, 64],
)

# %% Torch Parameters

ndev = 2

torch.cuda.set_device(ndev)
torch.cuda.empty_cache()
device = torch.device(f"cuda:{ndev}" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(device)
torch.backends.cudnn.deterministic = True

# Seed for reproducibility
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2004)


# %% Define Model Architecture
class Autoender(nn.Module):
    """
    Autoencoder model for encoding PSF data
    """

    def __init__(self):
        super().__init__()

        self.expected_image_shape = (1, 128, 128)
        self.channels = 16
        self.latent_dim = 8 * 12 * 12

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.channels, 3, padding=1),  # (128, 128)
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                self.channels, 2 * self.channels, 3, padding=1, stride=2
            ),  # (64, 64)
            nn.ReLU(),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                2 * self.channels, 4 * self.channels, 3, padding=1, stride=2
            ),  # (32, 32)
            nn.ReLU(),
            nn.Conv2d(4 * self.channels, 4 * self.channels, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * self.channels * 32 * 32, self.latent_dim),
            nn.ReLU(),
        )

        # Decoder

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 4 * self.channels * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(4 * self.channels, 32, 32)),
            nn.ConvTranspose2d(
                4 * self.channels, 4 * self.channels, 3, padding=1
            ),  # (32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(
                4 * self.channels,
                2 * self.channels,
                3,
                padding=1,
                stride=2,
                output_padding=1,
            ),  # (64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(2 * self.channels, 2 * self.channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * self.channels,
                self.channels,
                3,
                padding=1,
                stride=2,
                output_padding=1,
            ),  # (128, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(self.channels, self.channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat


model = Autoender().to(device)


# %% Define Training and Testing Function
def validation_loss(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = []

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)

            x_hat = model(x)
            val_loss = criterion(x, x_hat)
            total_val_loss.append(val_loss.item())
    return np.mean(total_val_loss)


def train(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    **kwargs,
):
    start = time.time()
    best_val_mse = np.inf
    best_weights = None
    val_loss_list = []
    train_loss_list = []
    current_epoch = 0
    best_epoch = 0

    print(f"Running of device: {device}")

    filename = kwargs.get("filename", None)
    criterion = kwargs.get("criterion", torch.nn.MSELoss())
    params = kwargs.get("parameters", model.parameters())
    learning_rate = kwargs.get("learning_rate", 1e-3)
    plot = kwargs.get("plot", False)
    precision = kwargs.get("precision", 4)
    optimizer = kwargs.get("optimizer", optim.Adam(params, lr=learning_rate))
    scheduler_params = kwargs.get("scheduler_params", None)
    save_freq = kwargs.get("save_freq", 50)

    lr_init = np.inf

    if scheduler_params:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )

    print(filename)
    try:
        ckp_path = filename[:-3] + "_checkpoint" + filename[-3:]
        model, optimizer, checkpoint = torch_func.load_ckp(
            ckp_path, model, optimizer, device
        )
        current_epoch = checkpoint["epoch"]
        print(f"Loading checkpoint at epoch: {current_epoch}")

        best_val_mse = checkpoint.get("best_val_mse", np.inf)
        best_weights = checkpoint.get("best_weights", None)
        val_loss_list = checkpoint.get("val_loss_list", [])
        train_loss_list = checkpoint.get("train_loss_list", [])
        if scheduler_params:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    except (AttributeError, FileNotFoundError, TypeError):
        print("No saved checkpoints found \nStarting training from epoch=0")

    for epoch in range(epochs):
        if epoch < current_epoch:
            continue
        # Training loop
        model.train()
        total_loss = []

        with tqdm(total=len(train_loader), unit=" batch", colour="green") as pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
            # batch_no = 1
            # n_batch = len(train_loader)
            for x in train_loader:
                pbar.update(1)
                x = x.to(device)

                # Forward pass
                xhat = model(x)
                loss = criterion(x, xhat)
                total_loss.append(loss.item())

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pfix = {"Training_Loss": f"{np.mean(total_loss):.{precision}e}"}
                if scheduler_params:
                    current_lr = optimizer.param_groups[0]["lr"]
                    pfix["LR"] = f"{current_lr:.2e}"
                    if current_lr < lr_init:
                        print(f"Changing learning rate to {current_lr:.2e}")
                        lr_init = current_lr

                pbar.set_postfix(pfix)
                # batch_no += 1
            train_loss_list.append(np.mean(total_loss))

            if val_loader:
                # Validation step using the callback
                val_loss = validation_loss(model, val_loader, criterion, device=device)
                if scheduler_params:
                    scheduler.step(val_loss)
                val_loss_list.append(val_loss)
                pbar.set_postfix(
                    {
                        "Training Loss": f"{np.mean(total_loss):.{precision}e}",
                        "Validation Loss": f"{val_loss:.{precision}e}",
                    }
                )
                if (val_loss < best_val_mse) and (epoch >= 0.1 * epochs):
                    best_epoch = epoch
                    best_val_mse = val_loss
                    best_weights = copy.deepcopy(model.state_dict())

            pbar.close()

        # Save checkpoints if filname is given
        if filename:
            if (epoch + 1) == epochs:
                print("Saving final chekpoint")
                ckp_dict = dict(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    best_weights=best_weights,
                    filename=filename,
                    best_epoch=val_loss_list.index(min(val_loss_list)) + 1,
                    best_val_mse=best_val_mse,
                    val_loss_list=val_loss_list,
                    train_loss_list=train_loss_list,
                )
                if scheduler_params:
                    ckp_dict["scheduler_state_dict"] = copy.deepcopy(
                        scheduler.state_dict()
                    )
                torch_func.save_ckp(**ckp_dict)

            elif (epoch + 1) % save_freq == 0:
                check = time.time()
                print(
                    f"Saving Checkpoint at Epoch: {epoch + 1}. Time elapsed: {secondsToStr(check - start)}"
                )
                ckp_dict = dict(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    best_weights=best_weights,
                    filename=ckp_path,
                    best_val_mse=best_val_mse,
                    val_loss_list=val_loss_list,
                    train_loss_list=train_loss_list,
                )
                if scheduler_params:
                    ckp_dict["scheduler_state_dict"] = copy.deepcopy(
                        scheduler.state_dict()
                    )
                torch_func.save_ckp(**ckp_dict)

    if not val_loader:
        best_weights = copy.deepcopy(model.state_dict())
    print("-" * 50)

    end = time.time()
    seconds = end - start

    if val_loader:
        print(
            f"Training finished in {secondsToStr(seconds)} \n \
        Best Val_Loss at {val_loss_list.index(min(val_loss_list)) + 1} Epoch \n \
        Saved at {best_epoch + 1} Epoch \n \
        MSE at Saved Epoch: Training={train_loss_list[best_epoch]:.{precision}f} \
            Validation={val_loss_list[best_epoch]:.{precision}f}"
        )
    else:
        print(
            f"Training finished in {secondsToStr(seconds)}, Best Train_Loss at #{train_loss_list.index(min(train_loss_list)) + 1} Epoch"
        )
        print(f"MSE at Best Loss: Training={min(train_loss_list):.{precision}f}")
    print("-" * 50)

    if plot:
        plotting_func.plot_loss(train_loss_list, val_loss_list, tp=plot, skip=5)

    return best_weights, train_loss_list, val_loss_list


# Plotting function
def compare_output(model, weigths, val_loader, n=8):
    model.load_state_dict(weigths)
    model.eval()

    x = next(iter(val_loader))[:n]

    with torch.no_grad():
        xhat = model(x.to(device))

    inp = x.detach().cpu().numpy().squeeze()
    out = xhat.detach().cpu().numpy().squeeze()

    plotting_func.plot(
        [inp, out, out - inp],
        max_imgs=n,
        titles=["Inp", "Recon", "Diff"],
        same_scale=[0, 1],
        cbar=True,
    )
    plt.show()


# %% Train the model and plot results
n_epochs = 1201

scheduler_params = {"factor": 0.5, "patience": 40, "min_lr": 1e-06}

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5
)

best_weights, train_loss_list, val_loss_list = train(
    model,
    train_loader,
    val_loader,
    epochs=n_epochs,
    device=device,
    filename=loc_weights,
    plot=True,
    optimizer=optimizer,
    scheduler_params=scheduler_params,
    save_freq=25,
)


# %% Test
try:
    best_weights
except NameError:
    checkpoint = torch.load(loc_weights, map_location=device)
    best_weights = checkpoint["model_state_dict"]
    val_loss_list = checkpoint["val_loss_list"]
    train_loss_list = checkpoint["train_loss_list"]

plot_loss(train_loss_list, val_loss_list, tp=True, skip=1)


compare_output(model, best_weights, val_loader, n=8)

# %% Save Model

model.load_state_dict(best_weights)
model_scripted = torch.jit.script(model)
torch.jit.save(
    m=model_scripted,
    f=loc_weights[:-3] + "_jit.pt",
)
