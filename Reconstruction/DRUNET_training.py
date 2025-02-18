"""
This script is used to fine-tune the DRUNET model using the noisy galaxy images.
Specify the path of training set and where to save the trained model.
"""

# %% Import modules
import os
import sys
import time

import h5py
import numpy as np
import torch
from deepinv.models import DRUNet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

sys.path.append("/../Functions/")

import copy

import torch.optim as optim
from helpers import secondsToStr
from plotting_func import plot, plot_loss
from torch_func import load_ckp, save_ckp
from tqdm import tqdm

# PATH to save the model
filename = "/scratch/tripathi/Model_Weights/DRUNET/drunet.pt"
# PATH to training set
PATH_TRAIN_SET = "/scratch/tripathi/Data/true_sers_1M.h5"

device = torch.device("cuda:0")

# %% Load Model and Data


def normalize(X_t):
    return (X_t - X_t.min()) / (X_t.max() - X_t.min())


# Initialize model with pre-trained weights
# See deepinv.models.DRUNet for more details.
# If pretrained is set to None, the model will be initialized with random weights.
net = DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)


# %% Build torch loader
class NoisyDataset(Dataset):
    """Noisy gal dataset."""

    def __init__(self, path, transform, scaling_factor=30):
        if not os.path.exists(path):
            raise ValueError(f"filename does not exist: {path}")

        self.hf = h5py.File(path, "r")["true image"]
        self.transform = transform
        self.scaling_factor = scaling_factor

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        true = torch.from_numpy(self.hf[idx]).unsqueeze(0) * self.scaling_factor
        true = self.transform(true)
        sigma = torch.FloatTensor(1).uniform_(
            0, 0.4 * torch.max(true)
        )  # Specify maximum noise level
        noise = torch.normal(torch.zeros_like(true), sigma)
        obs = true + noise

        return normalize(true), normalize(obs), sigma


# Add transformations for data augmentation
transform = transforms.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.Resize((64, 64), antialias=True),
        # v2.RandomRotation(degrees=(0, 180)),
    ]
)

# Create training/validation dataloaders
noisy_gal_dataset = NoisyDataset(path=PATH_TRAIN_SET, transform=transform)


train_dataset, val_dataset = torch.utils.data.random_split(
    noisy_gal_dataset, [0.8, 0.2]
)

train_loader = DataLoader(noisy_gal_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(noisy_gal_dataset, batch_size=64, shuffle=False)

del train_dataset, val_dataset, noisy_gal_dataset


# %% Training/Testing func
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
        model, optimizer, checkpoint = load_ckp(ckp_path, model, optimizer, device)
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
            for x, y, sigma in train_loader:
                pbar.update(1)
                x = x.to(device)
                y = y.to(device)
                sigma = sigma.to(device)

                # Forward pass
                ypred = model(x, sigma=sigma)
                loss = criterion(y, ypred)
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

            # Validation loss
            model.eval()
            total_val_loss = []

            with torch.no_grad():
                for xval, yval, sigma in val_loader:
                    xval = xval.to(device)
                    yval = yval.to(device)
                    sigma = sigma.to(device)

                    ypred_val = model(xval, sigma=sigma)
                    val_loss = criterion(yval, ypred_val)
                    total_val_loss.append(val_loss.item())
            val_loss_final = np.mean(total_val_loss)

            if scheduler_params:
                scheduler.step(val_loss)
            val_loss_list.append(val_loss_final)
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
                save_ckp(**ckp_dict)

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
                save_ckp(**ckp_dict)

    end = time.time()
    seconds = end - start

    print(
        f"Training finished in {secondsToStr(seconds)} \n \
    Best Val_Loss at {val_loss_list.index(min(val_loss_list)) + 1} Epoch \n \
    Saved at {best_epoch + 1} Epoch \n \
    MSE at Saved Epoch: Training={train_loss_list[best_epoch]:.{precision}f} \
        Validation={val_loss_list[best_epoch]:.{precision}f}"
    )

    if plot:
        plot_loss(train_loss_list, val_loss_list, tp=plot, skip=5)

    return best_weights, train_loss_list, val_loss_list


def predict(model, weights, dataloader, device, only_plt=True):
    model.load_state_dict(weights)
    model.eval()
    ypred_all = []
    with torch.no_grad():
        if not only_plt:
            for x, y, sigma in dataloader:
                x = x.to(device)
                y = y.to(device)
                sigma = sigma.to(device)

                # Forward pass
                ypred = model(x, sigma=sigma)

                ypred_all.append(ypred.cpu().detach().numpy())
            return np.concatenate(ypred_all)
        else:
            x, y, sigma = next(iter(dataloader))
            x = x.to(device)
            y = y.to(device)
            sigma = sigma.to(device)

            # Forward pass
            ypred = net(x, sigma=sigma)
            plot(
                [
                    x.detach().cpu().numpy().squeeze(),
                    y.detach().cpu().numpy().squeeze(),
                    ypred.detach().cpu().numpy().squeeze(),
                ]
            )


# %% Train


scheduler_params = {"factor": 0.5, "patience": 25, "min_lr": 1e-07}

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), lr=1e-5, weight_decay=1e-5
)

best_weights, train_loss_list, val_loss_list = train(
    net,
    train_loader,
    val_loader,
    epochs=201,
    device=device,
    filename=filename,
    criterion=torch.nn.L1Loss(),
    optimizer=optimizer,
    scheduler_params=scheduler_params,
    save_freq=10,
)

# %% Test
try:
    best_weights
except Exception:
    checkpoint = torch.load(
        filename[:-3] + "_checkpoint" + filename[-3:],
        map_location=f"cuda:{device.index}",
    )
    best_weights = checkpoint["model_state_dict"]
    train_loss_list = checkpoint["train_loss_list"]
    val_loss_list = checkpoint["val_loss_list"]

plot_loss(train_loss_list, val_loss_list, tp=True, skip=3)
results = predict(model=net, weights=best_weights, dataloader=val_loader, device=device)
