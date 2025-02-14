# %%
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.stats import binned_statistic as bs
from scipy.stats import linregress as lin

"Matplotlib style defaults"
# Font Sizes to use
SMALLER = 9
SMALL_SIZE = 12
MEDIUM_SIZE = 14
MEDIUM_SIZE_a = 16
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("text.latex", preamble=r"\usepackage{txfonts}")


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "legend.frameon": False,
        "legend.handlelength": 2,
        # "xtick.top": True,
        # "ytick.right": True,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.05,
        "figure.constrained_layout.w_pad": 0.05,
        "figure.constrained_layout.hspace": 0.0,
        "figure.constrained_layout.wspace": 0.0,
        "axes.labelpad": 1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
    }
)

# %%


# Function to remove ticks coinciding with axis edges
def remove_edge_ticks(ax, which="major", axis="both"):
    if which == "major":
        xticks = ax.xaxis.get_major_ticks()
        yticks = ax.yaxis.get_major_ticks()

    elif which == "minor":
        xticks = ax.xaxis.get_minor_ticks()
        yticks = ax.yaxis.get_minor_ticks()

    else:
        raise ValueError("Parameter 'which' must be 'major' or 'minor'.")

    for i in [0, -1]:
        if axis in ["x", "both"]:
            xticks[i].tick1line.set_markersize(0)
            xticks[i].tick2line.set_markersize(0)

        if axis in ["y", "both"]:
            yticks[i].tick1line.set_markersize(0)
            yticks[i].tick2line.set_markersize(0)


# Function to plot the loss curve for validation and training sets
def plot_loss(train_loss_list, val_loss_list, tp=True, skip=5):
    fig, ax = plt.subplots()
    epochs = len(train_loss_list)

    loss_list = [
        np.array(val_loss_list[skip:]).reshape(-1, 1),
        np.array(train_loss_list[skip:]).reshape(-1, 1),
    ]
    label_list = ["Val Loss", "Train Loss"]
    ax.set_ylabel("Error")
    if isinstance(tp, str):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()

        for i, loss in enumerate(loss_list):
            label_list[i] += f" :[{min(loss)[0]:.2f},{max(loss)[0]:.2f}]"
            loss_list[i] = scaler.fit_transform(loss)

        ax.set_ylabel("Scaled Error")

    for i, loss in enumerate(loss_list):
        ax.plot(np.arange(skip + 1, epochs + 1), loss, label=label_list[i])

    ax.set_xlim([0, epochs + 1])

    ax.legend()
    ax.set(xlabel="No of Epoch")
    plt.show()


# %%


# Scaling Function
def scaler(X, scale=True):
    assert X.ndim == 2
    if not scale:
        return X
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def plot(images, **kwargs):
    r"""
    Plots a list of images in a grid format.
    Parameters:
    -----------
    images : list or numpy.ndarray
        List or array of images to be plotted. The images should be of shape [B, H, W],
        where B is the batch size, H is the height of every image in pixels, and W is the width.
        The length of input is N where N is differet types of images to be plotted.

    **kwargs : dict, optional
        - titles (list of str or str, optional): Titles for each column of images.
        - max_imgs (int, optional): Maximum number of images to display. Default is 8.
        - rescale (list, optional): List of bool of length N. Decides which columns should be normalized.
        - cmap (str, optional): Colormap to use for displaying images. Default is "inferno".
        - cbar (bool, optional): Whether to display colorbars. Default is False.
        - caption (list of str, optional): Captions for each row of images.
        - same_scale (int or list, optional): If int, applies the same scale to all images.
        If list, applies the same scale to specified rows.
        - fname (str, optional): Filename to save the plot. If None, the plot is shown.
        - text (list of str, optional): Text annotations for each image.
        - text_row (int, optional): Row index for text annotations.
        - scale_row (int, optional): Row index to use for scaling images.

    Example:
    --------
        img = numpy.random.rand(4, 256, 256)
        plot([img, img, img], titles=["img1", "img2", "img3"])
    """

    titles = kwargs.get("titles", None)
    max_imgs = kwargs.get("max_imgs", 8)
    rescale = kwargs.get("rescale", None)
    cmap = kwargs.get("cmap", "inferno")
    cbar = kwargs.get("cbar", False)
    caption = kwargs.get("caption", None)
    same_scale = kwargs.get("same_scale", 0)
    fname = kwargs.get("fname", None)
    text = kwargs.get("text", None)
    text_row = kwargs.get("text_row", None)
    scale_row = kwargs.get("scale_row", None)

    plt.rcParams.update(
        {
            "axes.labelpad": 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    if isinstance(images, list):
        images = np.array(images)

    if images.ndim == 2:
        images = np.expand_dims(images, (0, 1))
    if images.ndim == 3:
        images = np.expand_dims(images, 0)

    images = np.swapaxes(images, 0, 1)
    # Oth axis: n images
    # 1st axis: t type of images

    rows = min(max_imgs, images.shape[0])
    cols = images.shape[1]

    if isinstance(titles, str):
        titles = [titles]
        assert len(titles) == cols, (
            "Title list should have same length as number of cols"
        )

    if not isinstance(same_scale, int):
        vmin = np.ones(cols) * 999999999999
        vmax = np.ones(cols) * -999999999999

        for col in range(cols):
            for row in same_scale:
                if not scale_row:
                    img = images[row, col]
                else:
                    img = images[scale_row, col]
                vmin[col] = min(vmin[col], np.min(img))
                vmax[col] = max(vmax[col], np.max(img))

    if cbar:
        figsize = (2.5 * cols, rows * 2.5)
    else:
        figsize = (2 * cols, rows * 2)

    fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.reshape(rows, cols)

    if rescale:
        assert len(rescale) == cols, (
            "Rescale list should have same length as number of cols"
        )
        for row in range(rows):
            for col, rs in enumerate(rescale):
                images[row, col] = scaler(images[row, col], rs)

    for row in range(rows):
        axr = axs[row]
        for col in range(cols):
            ax = axr[col]
            if isinstance(same_scale, int) or row not in same_scale:
                im = ax.imshow(images[row, col], cmap=cmap)
            else:
                im = ax.imshow(
                    images[row, col], cmap=cmap, vmin=vmin[col], vmax=vmax[col]
                )

            if titles and row == 0:
                ax.set_title(titles[col], size=MEDIUM_SIZE)

            if cbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
                fmt.format_data(0.1)
                # fmt.set_powerlimits((0, 0))
                colbar = fig.colorbar(im, cax=cax, orientation="vertical", format=fmt)
                colbar.ax.tick_params(labelsize=SMALL_SIZE)
                colbar.ax.yaxis.get_offset_text().set(size=SMALL_SIZE)

    if text is not None:
        for ax, txt in zip(axs[text_row], text):
            ax.text(
                0.65,
                0.8,
                txt,
                size=MEDIUM_SIZE,
                color="white",
                transform=ax.transAxes,
            )

    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    if caption:
        for cap, ax in zip(caption, axs[:, 0]):
            ax.set_ylabel(cap, fontsize=MEDIUM_SIZE)

    if cbar:
        plt.subplots_adjust(hspace=0.02, wspace=0.25)
    else:
        plt.subplots_adjust(hspace=0.01, wspace=0.05)

    if fname:
        savefig(fname)
    else:
        plt.show()


# %%
# Bias plot for ellipticity measurements (scatter plot along with binned mean and std)
def plot_bias_binned(
    ypred,
    ytest,
    **kwargs,
):
    """
    Plots the ellipticity measurement residuals along with the estimated linear bias.

    Parameters:
    -----------
    ypred : array-like
        Predicted ellipticities. Dim = [N_obs, 2].
    ytest : array-like
        True ellipticities. Dim = [N_obs, 2].

    **kwargs : dict, optional
        - lim (float): Limit for y-axis. Default is 0.05.
        - colors (list): Colors for the two ellipticity components. Default is ["blue", "orange"].
        - bins (int): Number of bins for the histogram. Default is 10.
        - pow (float): Scaling factor for the estimated bias. Default is 1e4.
        - bad_index (array-like): Indices of bad data points to exclude. Default is None.
        - bias_line (bool): Whether to plot the linear bias as a line. Default is False.
        - ellipticity_cutoff (float): Cutoff for ellipticity. Default is 0.7.
        - hist (bool): Whether to plot the histogram. Default is True.
    """

    lim = kwargs.get("lim", 0.05)
    colors = kwargs.get("colors", ["blue", "orange"])
    bins = kwargs.get("bins", 10)
    power = kwargs.get("pow", 1e4)
    bad_index = kwargs.get("bad_index", None)
    bias_line = kwargs.get("bias_line", False)
    ellipticity_cutoff = kwargs.get("ellipticity_cutoff", 0.7)
    hist = kwargs.get("hist", True)

    if isinstance(ypred, list):
        ypred = np.array(ypred)
    if isinstance(ytest, list):
        ytest = np.array(ytest)
    assert ypred.shape == ytest.shape, (
        "Measurement array should be of same shape as true array"
    )
    if bad_index is not None:
        ypred = ypred[~bad_index.astype(bool)]
        ytest = ytest[~bad_index.astype(bool)]
    delta = ypred - ytest

    def get_label(delta, test, i):
        res = lin(test, delta)

        cap = r"$\hat{m}_1$,$\hat{c}_1$" if i == 0 else r"$\hat{m}_2$,$\hat{c}_2$"
        exp = int(np.log10(power))
        label = (
            # rf"$\Delta\epsilon_{i+1}$"+
            "{"
            + cap
            + "}"
            + rf"$ \times 10^{exp}$"
            + "= {"
            + rf"{res.slope * power:.1f}$\pm${res.stderr * power:.1f}"
            + rf", {res.intercept * power:.1f}$\pm${res.intercept_stderr * power:.1f}"
        )
        label += "}"

        return label, res.slope, res.intercept

    mean = []
    std = []
    points = []

    for i in [0, 1]:
        stat1 = bs(ytest[:, i], delta[:, i], "mean", bins=bins)
        stat2 = bs(ytest[:, i], delta[:, i], "std", bins=bins)
        count = bs(ytest[:, i], delta[:, i], "count", bins=bins)
        ind = count[0] > 0

        mean += [stat1[0][ind]]
        std += [stat2[0][ind]]
        points += [0.5 * (stat1[1][1:] + stat1[1][:-1])[ind]]

    fig = plt.figure()
    ax = fig.add_subplot()
    for i in [0, 1]:
        x = points[i]
        ind = np.abs(x) < ellipticity_cutoff
        x = points[i][ind]
        y = mean[i][ind]
        err = std[i][ind]

        lab, m, c = get_label(delta[:, i], ytest[:, i], i)
        if hist:
            ax.errorbar(
                x=x,
                y=y,
                yerr=err,
                color=colors[i],
                fmt="o",
                # alpha=0.7,
                capsize=2,
            )

        ax.scatter(
            ytest[:, i],
            delta[:, i],
            label=lab,
            s=0.7,
            alpha=0.5,
            color=colors[i],
        )

        if bias_line:
            ax.axline(
                xy1=(0, c),
                slope=m,
                color=colors[i],
                linewidth=0.8,
                linestyle="--",
            )

    ax.set(
        xlabel=r"$\epsilon^T$",
        ylabel=r"$\hat{\epsilon}-\epsilon^T$",
        xlim=(-1, 1),
        ylim=(-lim, lim),
    )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    lgnd = ax.legend()
    lgnd.legend_handles[0]._sizes = [30]
    lgnd.legend_handles[1]._sizes = [30]
    fig.tight_layout()
    plt.show()


# %%
def contour_plot(
    ytest,
    ypred_list,
    legends,
    **kwargs,
):
    """
    Generates a contour plot comparing true values and predicted values from two shape measurement methods. Also plots the 1D residual distribution.
    # Fig 7 in DeepShape Paper

    Parameters:
    ------------
    ytest (array-like): True values of the ellipticity. Dim = [N_obs].
    ypred_list (list of array-like): A list of ellipticity measurements using two methods. Dim=[2, N_obs].
    legends (list of str): A list of legend labels for the two methods.

    **kwargs : dict, optional
        Additional keyword arguments:
        - lim (float): The limit for the y-axis. Default is 0.3.
        - cmaps (list of str): List of colormap names for the contour plots. Default is ["Reds", "Blues"].
        - colors (list of str): List of colors for the slope lines and KDE plots. Default is ["firebrick", "blue"].
        - fname (str or None): Filename to save the plot. If None, the plot is shown. Default is None.
    """

    lim = kwargs.get("lim", 0.3)
    cmaps = kwargs.get("cmaps", ["Reds", "Blues"])
    colors = kwargs.get("colors", ["firebrick", "blue"])
    fname = kwargs.get("fname", None)

    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    fig = plt.figure(
        figsize=(6, 4.2),
    )  # Set background canvas colour to White instead of grey default
    fig.patch.set_facecolor("white")

    ax = plt.subplot(gs[0, 0])

    ax.set_xlim(-1, 1)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\epsilon_1^{\mathrm{true}}$")
    ax.set_ylabel(r"$\hat{\epsilon}_1-\epsilon^{\mathrm{true}}_1$")
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2)

    axr = plt.subplot(gs[0, 1], sharey=ax)
    axr.get_xaxis().set_visible(False)
    axr.get_yaxis().set_visible(False)
    axr.spines["right"].set_visible(False)
    axr.spines["top"].set_visible(False)
    axr.spines["bottom"].set_visible(False)
    axr.axhline(0, color="black", linestyle=":", linewidth=1.2)

    # remove_edge_ticks(ax, which="major", axis="x")
    # remove_edge_ticks(ax, which="minor")

    ax.set_xticks([-1, -0.5, 0, 0.5, 1])

    # For each measurement values
    for ny, ypred in enumerate(ypred_list):
        delta = ypred - ytest

        sns.kdeplot(
            x=ytest,
            y=delta,
            fill=True,
            ax=ax,
            cmap=cmaps[ny],
            levels=5,
            alpha=0.6,
        )

        # Calculate slope and intercept
        res = lin(ytest, delta)
        ax.axline(
            xy1=(0, res.intercept),
            slope=res.slope,
            color=colors[ny],
            linewidth=1.5,
            linestyle="--",
        )

        kde = stats.gaussian_kde(delta)
        yy = np.linspace(-lim, lim, 1000)
        axr.plot(kde(yy), yy, color=colors[ny])

    handles = [
        mpatches.Patch(facecolor=plt.cm.Reds(100), label=legends[0]),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label=legends[1]),
    ]
    ax.legend(handles=handles)

    if not fname:
        plt.tight_layout()
        plt.show()
    else:
        savefig(fname)


# %%
def shear_plot(
    gpred,
    gtest,
    **kwargs,
):
    """
    Plots the shear measurement residuals for a given measurement.
    Fig 9 in DeepShape Paper

    Parameters:
    -----------
    gpred : numpy.ndarray
        The predicted shear values. Dim = [N_field, 2].
    gtest : numpy.ndarray
        The true shear values. Dim = [N_field, 2].

    **kwargs : dict, optional
        Additional keyword arguments to customize the plot:
        - lim (float): Limit for the y-axis. Default is 0.02.
        - colors (list): List of colors for the scatter points and bias lines. Default is ["#ff7f0e", "#2ca02c"].
        - bias_line (bool): Whether to plot the bias line. Default is True.
        - title (str): Title of the plot. Default is None.
        - fname (str): Filename to save the plot. If None, the plot is shown. Default is None.

    """

    lim = kwargs.get("lim", 0.02)
    colors = kwargs.get("colors", ["#ff7f0e", "#2ca02c"])
    bias_line = kwargs.get("bias_line", True)
    title = kwargs.get("title", None)
    fname = kwargs.get("fname", None)

    assert gpred.shape == gtest.shape, (
        "Measurement array should be of same shape as true array"
    )

    delta = gpred - gtest

    def get_label(delta, test, i):
        res = lin(test, delta)

        cap = (
            r"$10^3\hat{M}_1$, $10^4\hat{C}_1$"
            if i == 0
            else r"$10^3\hat{M}_2$, $10^4\hat{C}_2$"
        )
        label = (
            "("
            + cap
            + ")"
            + "= ("
            + rf"{res.slope * 1e3:.1f}$\pm${res.stderr * 1e3:.1f}"
            + rf", {res.intercept * 1e4:.1f}$\pm${res.intercept_stderr * 1e4:.1f}"
        )
        label += ")"

        return label, res.slope, res.intercept

    fig = plt.figure(figsize=(6, 4.2))
    ax = fig.add_subplot(111)
    plt.sca(ax)
    for i in [0, 1]:
        lab, m, c = get_label(delta[:, i], gtest[:, i], i)

        ax.scatter(
            gtest[:, i],
            delta[:, i],
            # label=lab,
            label=rf"$\alpha={i + 1}$",
            color=colors[i],
            s=8,
            marker="s",
            edgecolors="none",
        )

        if bias_line:
            ax.axline(
                xy1=(0, c),
                slope=m,
                color=colors[i],
                linestyle="--",
                linewidth=1.5,
            )

    ax.set(
        xlabel=r"$\gamma^{\mathrm{true}}_\alpha$",
        ylabel=r"$\hat{\gamma_\alpha}-\gamma_\alpha^{\mathrm{true}}$",
        xlim=(-0.1, 0.1),
        ylim=(-lim, lim),
        xticks=[-0.1, -0.05, 0, 0.05, 0.1],
        # yticks=[-0.02, -0.01, 0, 0.01, 0.02],
    )

    remove_edge_ticks(ax, which="major", axis="x")
    # remove_edge_ticks(ax, which="minor")

    if title:
        ax.set_title(title)

    ax.axhline(0, color="black", linewidth=1.2, linestyle=":")
    ax.legend()
    if not fname:
        plt.tight_layout()
        plt.show()
    else:
        savefig(fname)


# Function to save images
def savefig(filename=None, dpi=600):
    if filename:
        parent_dir = Path(filename).parent

        try:
            parent_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            1
        else:
            print(f"New folder created {parent_dir}")

        plt.savefig(
            fname=filename,
            bbox_inches="tight",
            dpi=dpi,
        )
    else:
        plt.show()
