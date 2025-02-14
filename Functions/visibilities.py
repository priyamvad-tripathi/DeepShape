"""
Functions related to visibility data and generation. Calculations are made based on SKA-MID dishes.
Obseravtion parameters default to values used in DeepShape Paper.
Make sure to install RASCIL 1.1.0 before running this code. Latest versions might give errors.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.gridded_visibility import create_griddata_from_image
from ska_sdp_datamodels.image import Image
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_func_python.grid_data import (
    grid_visibility_weight_to_griddata,
    griddata_visibility_reweight,
)
from ska_sdp_func_python.imaging import (
    create_image_from_visibility,
    invert_ng,
    invert_wg,
    predict_ng,
    predict_wg,
)


def generate_visibilities(phasecentre, **kwargs):
    """
    Generate a visibility structure for observations taken from a given telescope using RASCIL functions.

    Parameters:
    -----------

    phasecentre (SkyCoord): The phase centre of the observation. Must be an instance of SkyCoord.
    tel (str, optional): The name of the telescope configuration. Default is "MID".
    rmax (float, optional): The maximum baseline length in meters. Default is None.
    frequencies (numpy.ndarray, optional): Array of frequency values in Hz. Default is an array with a single value of 1.4 GHz.
    channel_bandwidths (numpy.ndarray or float, optional): Array of channel bandwidth values in Hz. Default is 0.3 times the frequencies.
    ha_interval (tuple, optional): Hour angle interval in hours. Default is (-4, 4).
    integration_time (int, optional): Integration time in seconds. Default is 300.

    Returns:
    -----------
    Visibility (xarray.Dataset): The generated visibility structure.
    """

    tel = kwargs.get("tel", "MID")
    rmax = kwargs.get("rmax", None)
    frequencies = kwargs.get("frequencies", np.array([1.4e9]))
    channel_bandwidths = kwargs.get("channel_bandwidths", 0.3 * frequencies)
    ha_interval = kwargs.get("ha_interval", (-4, 4))
    integration_time = kwargs.get("integration_time", 300)

    if not isinstance(phasecentre, SkyCoord):
        print("phasecentre should be a SkyCoord instance")
        return
    config = create_named_configuration(tel, rmax=rmax)

    # Now compute number of integration times and corresponding HAs
    dtime_hr = integration_time / 3600.0
    ntimes = int((ha_interval[1] - ha_interval[0]) / dtime_hr)

    # Centered w.r.t. transit, in radian
    times = (
        np.linspace(
            ha_interval[0] + dtime_hr / 2.0, ha_interval[1] - dtime_hr / 2.0, ntimes
        )
        * np.pi
        / 12.0
    )

    vt = create_visibility(
        config,
        times,
        frequencies,
        channel_bandwidth=channel_bandwidths,
        weight=1.0,
        phasecentre=phasecentre,
        polarisation_frame=PolarisationFrame("stokesI"),
        elevation_limit=None,
    )

    return vt


def dirty_psf_from_visibilities(
    vt,
    cellsize,
    **kwargs,
):
    """
    This function uses visibility data to create a dirty image and its corresponding PSF.

    Parameters:
    -----------
    vt : Visibility
        Visibility data.
    cellsize : float
        Pixel size in radians.

    **kwargs : dict, optional
        Additional keyword arguments:
        - NPIX (int): Number of pixels along each axis of the output images (default is 128).
        - weighting (str): Weighting scheme to use ('natural', 'uniform', 'robust', default is 'robust').
        - robustness (float): Robustness parameter for robust weighting (default is 0).
        - asarray (bool): If True, return the dirty image and PSF as numpy arrays (default is True).
        - use_wagg (bool): If True, use WAGG gridder (GPU based gridder), otherwise use Nifty-Gridder (CPU based Gridder) (default is False).
        - override_cellsize (bool): If True, override the cellsize parameter (default is False).

    Returns:
    -----------
    tuple
        - dirty (Image): Dirty image.
        - psf (Image): Point spread function.

        If asarray is True:

            - dirty_array (numpy.ndarray): Dirty image as a numpy array.
            - psf_array (numpy.ndarray): PSF as a numpy array.

    """

    NPIX = kwargs.get("NPIX", 128)
    weighting = kwargs.get("weighting", "robust")
    robustness = kwargs.get("robustness", 0)
    asarray = kwargs.get("asarray", True)
    use_wagg = kwargs.get("use_wagg", False)
    # nthread = kwargs.get("nthread", 4)
    override_cellsize = kwargs.get("override_cellsize", False)

    #! Give same weight to every visibility
    # vt.weight.data[np.nonzero(vt.weight.data)] = 1

    # First create empty rascil Image instance from visibilities
    model = create_image_from_visibility(
        vt, cellsize=cellsize, npixel=NPIX, override_cellsize=override_cellsize
    )

    # Reweight visibilities if not natural weighting
    grid_weights = create_griddata_from_image(
        model, polarisation_frame=model.image_acc.polarisation_frame
    )
    grid_weights = grid_visibility_weight_to_griddata(vt, grid_weights)
    vt = griddata_visibility_reweight(
        vt,
        grid_weights[0],
        weighting=weighting,
        robustness=robustness,
        sumwt=grid_weights[1],
    )

    if use_wagg:
        gridder = invert_wg
    else:
        gridder = invert_ng

    dirty, _ = gridder(vt, model)
    psf, _ = gridder(vt, model, dopsf=True)

    if not asarray:
        return (dirty, psf)

    return (
        dirty,
        psf,
        dirty.pixels.to_numpy().astype(np.float32).squeeze(),
        psf.pixels.to_numpy().astype(np.float32).squeeze(),
    )


def addNoiseToVis(vis, return_SNR=False, **kwargs):
    """
    Add noise to visibility data.

    Parameters:
    -----------
    vis : xarray.Dataset
        The visibility dataset containing the visibility data and metadata.
    return_SNR : bool, optional
        If True, return the Signal-to-Noise Ratio (SNR) in the visibility domain

    **kwargs : dict, optional
        - summary : bool
            If True, print a summary of the RMS noise and SNR.
        - noise_fac : float
            A factor to scale the noise.
        - sigma : float
            A specific noise standard deviation to use (in microJy).

    Returns:
    -----------
    nvis : xarray.Dataset
        The visibility dataset with added noise.
    SNR : float
        The Signal-to-Noise Ratio of the visibility data (only if return_SNR is True).

    Notes:
    -----------
    - The paramters eta and sens are based of the SKA-MID dishes. This need to be updated for other telescopes.

    """

    eta = 0.98  # Aperture efficiency
    k_b = 1.38064852e-23  # Boltzmann constant
    bandwidth = (vis.channel_bandwidth.data,)
    int_time = (vis["integration_time"].data,)
    sens = 10.1  # A_eff/T_sys
    bt = np.outer(int_time, bandwidth)
    sigma_arr = (np.sqrt(2) * k_b) / (sens * eta * (np.sqrt(bt)))
    summary = kwargs.get("summary", False)

    sigma = sigma_arr[0, 0] * 1e26

    if "noise_fac" in kwargs:
        sigma = kwargs["noise_fac"] * sigma

    if "sigma" in kwargs:
        sigma = kwargs["sigma"] * 1e-6

    if sigma != 0:
        SNR = np.linalg.norm(vis.vis.data) / sigma
    else:
        SNR = np.inf

    # Add noise to real and imaginary parts
    noise_real = np.random.normal(loc=0, scale=sigma, size=vis.vis.shape)
    noise_imag = np.random.normal(loc=0, scale=sigma, size=vis.vis.shape)

    if summary:
        print(f"RMS Noise= {sigma * 1e6:0.2f} uJy")
        print(f"SNR_vis= {SNR:0.2f}")

    noise = np.vectorize(complex)(noise_real, noise_imag)
    vis_with_noise = vis.vis + noise

    nvis = vis.copy(deep=True)
    nvis["vis"].data = vis_with_noise

    return nvis, SNR


def visibilities_from_array(image, vt0, phasecentre, cellsize, **kwargs):
    """
    Convert an image to visibilities using the specified parameters.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image array.
    vt0 : xarray.Dataset
        The initial visibility positions.
    phasecentre : SkyCoord
        The phase center of the image.
    cellsize : float
        The cell size in radians.

    **kwargs : dict
        Additional keyword arguments:
        - frequencies : float or list of floats, optional
            The frequency or list of frequencies in Hz. Default is 1.4 GHz.
        - channel_bandwidths : float, optional
            The channel bandwidth in Hz. Default is 0.3 * frequency.
        - use_wagg : bool, optional
            Whether to use WAGG gridder. Default is False.
        - nthread : int, optional
            Number of threads to use. Default is None.

    Returns:
    --------
    Visibility : xarray.Dataset
        The visibility object after conversion.
    """

    frequency = kwargs.get("frequencies", 1.4e9)
    if not isinstance(frequency, float):
        frequency = frequency[0]
    channel_bandwidth = kwargs.get("channel_bandwidths", 0.3 * frequency)
    use_wagg = kwargs.get("use_wagg", False)
    nthread = kwargs.get("nthread", None)
    # cellsize = kwargs.get("cellsize", imager.def_cellsize)

    cellsize_deg = cellsize * 180 / np.pi

    ny, nx = image.shape
    image = image.reshape([1, 1, ny, nx])
    np.nan_to_num(image, copy=False)

    w = WCS(naxis=4)
    w.wcs.crval = np.array([phasecentre.ra.deg, phasecentre.dec.deg, 0, frequency])
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    w.wcs.cdelt = np.array([-cellsize_deg, +cellsize_deg, 1, channel_bandwidth])

    w.wcs.radesys = "ICRS"
    w.wcs.equinox = 2000.00

    w.wcs.crpix = np.array([ny // 2 + 1, nx // 2 + 1, 1, 1])

    polarisation_frame = PolarisationFrame("stokesI")

    im = Image.constructor(
        image, wcs=w, polarisation_frame=polarisation_frame, clean_beam=None
    )

    if use_wagg:
        gridder = predict_wg
    else:
        gridder = predict_ng

    if not nthread:
        ivt = gridder(vt0, im)
    else:
        ivt = gridder(vt0, im, threads=nthread)
    return ivt


def generate_visibilities_from_array(gal_arr, ra, dec, cellsize, **kwargs):
    """
    Generate visibilities from a given galaxy image.

    Parameters:
    -----------
    gal_arr (numpy.ndarray): Galaxy image expressed as 2D numpy array of size NPIX x NPIX.
    ra (float): Right ascension of the phase centre in degrees.
    dec (float): Declination of the phase centre in degrees.
    cellsize (float): The size of each cell in the image in degrees.

    **kwargs: Additional keyword arguments to be passed to the visibility generation functions.

    Returns:
    -----------
    vt (xarray.Dataset): The generated visibilities.
    """

    # Create Phasecentre
    phasecentre = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs",
        equinox="J2000",
    )

    # Generate raw visibility positions
    vt0 = generate_visibilities(phasecentre=phasecentre, **kwargs)

    # Calculate true visibilities
    vt = visibilities_from_array(
        image=gal_arr, vt0=vt0, phasecentre=phasecentre, cellsize=cellsize, **kwargs
    )

    return vt
