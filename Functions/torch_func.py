"""
Function related to testing and training PyTorch models.
"""

# %% Import Libraries
import copy
import os
import random
import time
from functools import reduce
from pathlib import Path

import h5py
import numpy as np
import plotting_func
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# Seeds to ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2004)


# %% Define Functions


def secondsToStr(t):
    return "%02d:%02d:%02d.%03d" % reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(round(t * 1000),), 1000, 60, 60]
    )


def val_loss_fn(model, val_loader, criterion, device):
    """
    Computes the average validation loss for a given model.

    Args:
    ------
    model (torch.nn.Module): The torch model to evaluate.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to use for computing the loss.
    device (torch.device): Device on which to perform computations (e.g., 'cpu' or 'cuda').

    Returns:
    --------
    float: The average validation loss.
    """

    model.eval()
    total_val_loss = []

    with torch.no_grad():
        for xval, yval in val_loader:
            xval = xval.to(device)
            yval = yval.to(device)

            ypred_val = model(xval)
            val_loss = criterion(yval, ypred_val)
            total_val_loss.append(val_loss.item())

    return np.mean(total_val_loss)


def save_ckp(model, optimizer, filename, **kwargs):
    """Convenient function to save torch model and optimizer state_dict along with any other data in a dictionary."""

    data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    parent_dir = Path(filename).parent
    try:
        parent_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        1
    else:
        print(f"New folder created {parent_dir}")

    if kwargs:
        data = {**data, **kwargs}
    torch.save(
        data,
        filename,
    )


def load_ckp(filename, model, optimizer, device):
    """Convenient function to load torch model using a saved checkpoint"""

    checkpoint = torch.load(filename, map_location=f"cuda:{device.index}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint


def train(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    **kwargs,
):
    """
    Trains a PyTorch model using the provided training and validation data loaders.

    Args:
    ------
    model (torch.nn.Module): The model to be trained.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
    epochs (int): Number of epochs to train the model.
    device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
    **kwargs: Additional keyword arguments for training configuration:
        - filename (str): Path to save the model checkpoints.
        - criterion (torch.nn.Module): Loss function (default: torch.nn.MSELoss()).
        - parameters (iterable): Model parameters to optimize (default: model.parameters()).
        - learning_rate (float): Learning rate for the optimizer (default: 1e-3).
        - plot (bool): Whether to plot the training and validation loss (default: False).
        - precision (int): Precision for printing loss values (default: 4).
        - optimizer (torch.optim.Optimizer): Optimizer for training (default: optim.Adam).
        - scheduler_params (dict): Parameters for learning rate scheduler (default: None).
        - save_freq (int): Frequency of saving checkpoints (default: 50).

    Returns:
    --------
    tuple: A tuple containing:
        - best_weights (dict): The best model weights based on validation loss.
        - train_loss_list (list): List of training loss values for each epoch.
        - val_loss_list (list): List of validation loss values for each epoch.
    """

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

    # Plateau scheduler for learning rate
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
            for x, y in train_loader:
                pbar.update(1)
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                ypred = model(x)
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

            train_loss_list.append(np.mean(total_loss))

            if val_loader:
                # Validation step using the callback
                val_loss = val_loss_fn(model, val_loader, criterion, device=device)
                # Step Scheduler
                if scheduler_params:
                    scheduler.step(val_loss)
                val_loss_list.append(val_loss)
                pbar.set_postfix(
                    {
                        "Training Loss": f"{np.mean(total_loss):.{precision}e}",
                        "Validation Loss": f"{val_loss:.{precision}e}",
                    }
                )

                # Save best weights if validation loss is at minimum
                # Second condition ensures spurious saving at the start of training
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
        MSE at Saved Epoch: Training={train_loss_list[best_epoch]:.{precision}e} \
            Validation={val_loss_list[best_epoch]:.{precision}e}"
        )
    else:
        print(
            f"Training finished in {secondsToStr(seconds)}, Best Train_Loss at #{train_loss_list.index(min(train_loss_list)) + 1} Epoch"
        )
        print(f"MSE at Best Loss: Training={min(train_loss_list):.{precision}e}")
    print("-" * 50)

    if plot:
        plotting_func.plot_loss(train_loss_list, val_loss_list, tp=plot, skip=5)

    return best_weights, train_loss_list, val_loss_list


def predict(model, weights, dataloader, device):
    "Convenient function to predict using a trained model"
    model.load_state_dict(weights)
    model.eval()
    ypred_all = []
    ytrue_all = []
    with torch.no_grad():
        with tqdm(total=len(dataloader), unit=" batch", colour="green") as pbar:
            for x, y in dataloader:
                pbar.update(1)
                x = x.to(device)
                ypred = model(x)
                ypred_all.append(ypred.cpu().detach().numpy())
                ytrue_all.append(y.cpu().detach().numpy())
    return np.concatenate(ypred_all), np.concatenate(ytrue_all)


# %% Custom dataloader
class ImageDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading images and elliipticities from an HDF5 file.

    Args:
    ----------
    path (str): Path to the HDF5 file.
    x_key (list): List of key for the images (e.g., ['dirty_image','PSF']).
    y_key (list): Key for the target ellipticity.
    peak (float, optional): Peak value for cutoff based on PSNR. Defaults to None.
    transform (callable, optional): Optional transform to be applied on the images. Defaults to None.
    scale (bool, optional): Whether to scale the images or not. Defaults to True.
    """

    def __init__(self, path, x_key, y_key, peak=None, transform=None, scale=True):
        if not os.path.exists(path):
            raise ValueError(f"filename does not exist: {path}")

        self.hf = h5py.File(path, "r")
        self.hf_keys = list(self.hf.keys())
        if peak:
            self.cutoff = np.where(self.hf["Peak"][:] > peak)[
                0
            ]  # Cut-off based on PSNR
        else:
            self.cutoff = np.arange(len(self.hf[self.hf_keys[0]]))
        self.x_key = x_key
        self.y_key = y_key
        self.transform = transform
        self.scale = scale

    # Image Normalization for smooth training
    def scaler(self, X):
        assert X.ndim == 2
        return (X - np.min(X)) / (np.max(X) - np.min(X))

    def __len__(self):
        return len(self.cutoff)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_inp = []
        for k in self.x_key:
            im = self.hf[k][self.cutoff[idx]]
            if self.scale:
                im = self.scaler(im)
            x_inp += [im]
        x_inp = torch.from_numpy(np.array(x_inp))

        if x_inp.ndim != 3:
            x_inp = x_inp.unsqueze(0)

        # Apply transformation if any
        if self.transform:
            x_inp = self.transform(x_inp)

        if self.y_key:
            y_inp = torch.from_numpy(self.hf[self.y_key[0]][self.cutoff[idx]])

            return x_inp, y_inp
        else:
            return x_inp


def dataloader(path, x_key, y_key, split, batch_size, **kwargs):
    """
    Creates data loaders for training and validation datasets.

    Parameters:
    -----------
    path (str): Path to the dataset (HDF5 format).
    x_key (str): Key for the input data in the dataset.
    y_key (str): Key for the target data in the dataset.
    split (list or int): If list, it should contain the proportions for splitting the dataset into training and validation sets.
                         If int, it indicates no split and the entire dataset is used.
    batch_size (int or list): Batch size for the data loaders. If split is a list, batch_size should also be a list with batch sizes for training and validation loaders.
    **kwargs: Additional arguments to be passed to the ImageDataset.

    Returns:
    --------
    DataLoader or tuple: If split is an int, returns a single DataLoader. If split is a list, returns a tuple of DataLoaders (train_loader, val_loader).
    """

    # assert len(batch_size) == len(split), "Batch size should be same size as split"
    # assert isclose(sum(split), 1), "Splits don't add up to 1"

    gal_dataset = ImageDataset(path=path, x_key=x_key, y_key=y_key, **kwargs)

    if not isinstance(split, int):
        train_dataset, val_dataset = torch.utils.data.random_split(gal_dataset, split)

        train_loader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size[1], shuffle=False)

        return train_loader, val_loader

    else:
        loader = DataLoader(gal_dataset, batch_size=batch_size, shuffle=False)
        return loader


def create_testloader(x, transform=None, batch_size=128, shuffle=False):
    "Convenient function to create a test data loader from a numpy array"
    tensor_x = torch.from_numpy(x)
    tensor_y = torch.zeros_like(tensor_x)

    if len(tensor_x.shape) != 4:
        tensor_x = tensor_x.unsqueeze(1)

    if transform:
        tensor_x = transform(tensor_x)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
