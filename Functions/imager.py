"""
Main module for simulating galaxies.
All images are simulated on 128x128 pixel grids with a cell size of 6.93e-07 radians by default.
"""

# %%
import copy

import galsim
import numpy as np
import visibilities
from ska_sdp_func_python.image import deconvolve_cube

# %% Deconv Setup

def_NPIX = 128  # Default size of the image
FFTBIGSIZE = 11488  # Max size of support for Galsim (Caching the FFT)
big_fft_params = galsim.GSParams(maximum_fft_size=FFTBIGSIZE)
ha_interval = (-4, 4)


def_cellsize = 6.93e-07  # Cellsize in radians (calulated using ska_sdp_func_python.imaging.advise_wide_field)

# Discretized and cache sersic index for faster galsim implementation
sersic_indexes = np.linspace(0.7, 2, 100)


def simulate_gal(
    flux,
    r0,
    e1gal,
    e2gal,
    **kwargs,
):
    """
    Simulate a galaxy image using the GalSim library.

    Parameters:
    -----------
    flux : float
        The flux of the galaxy.
    r0 : float
        The scale radius of the galaxy.
    e1gal : float
        The first component of the galaxy's ellipticity.
    e2gal : float
        The second component of the galaxy's ellipticity.

    **kwargs : dict, optional
        Additional keyword arguments:
        - flip (bool): If True, flip the galaxy image horizontally. Default is False. Important for comapatibility with the RadioLesnfit Method.
        - summary (bool): If True, print a summary of the galaxy's properties. Default is False.
        - sersic_index (float): The Sersic index of the galaxy. If None, it will be randomly chosen. Default is None.
        - cellsize (float): The size of each pixel in arcseconds. Default is def_cellsize.
        - simple (bool): If True, use a simple Sersic index of 1. Default is False.
        - NPIX (int): The number of pixels along one dimension of the image. Default is def_NPIX.

    Returns:
    --------
    list
        A list containing the simulated galaxy image array and a list of galaxy parameters
        [e1gal, e2gal, sersic_index, flux, hlr].
    """

    flip = kwargs.get("flip", False)
    summary = kwargs.get("summary", False)
    sersic_index = kwargs.get("sersic_index", None)
    cellsize = kwargs.get("cellsize", def_cellsize)
    simple = kwargs.get("simple", False)
    NPIX = kwargs.get("NPIX", def_NPIX)

    scale = cellsize * 180 / np.pi * 60 * 60

    # Galsim Parameters
    stampimage = galsim.ImageF(NPIX, NPIX, scale=scale)
    b = galsim.BoundsI(1, NPIX, 1, NPIX)
    stamp = stampimage[b]

    if summary:
        print(f"Intensity: {flux * 1e6:0.2f}uJy")

    if sersic_index is None:
        if simple:
            sersic_index = 1
        else:
            # sersic_index = -0.5 + np.random.rand()  // Continuous distribution leads to slower performance
            sersic_index = np.random.choice(sersic_indexes)

    # Convert scale radius (default provided in TRECS catalogue) to half light radius
    hlr = r0 * np.log(2)
    gal = galsim.Sersic(
        n=sersic_index,
        half_light_radius=hlr,
        gsparams=big_fft_params,
    )

    # Shear Galaxy
    e_tot = galsim.Shear(g1=e1gal, g2=e2gal)

    gal_true = gal.shear(e_tot)
    try:
        gal_true = gal_true.drawImage(stamp, scale=scale)
    except galsim.errors.GalSimFFTSizeError:
        # print("FFT size error \n skipping this galaxy")
        return 0

    # Scale pixel intensity to match flux
    gal_wo_flux = copy.deepcopy(gal_true.array)
    gal_arr = gal_wo_flux * flux / np.sum(gal_wo_flux)

    if flip:
        gal_arr = np.flip(gal_arr, axis=1)

    return [gal_arr, [e1gal, e2gal, sersic_index, flux, hlr]]


def simulate_vis(
    gal_data,
    ra_gal,
    dec_gal,
    cellsize=def_cellsize,
    **kwargs,
):
    """
    Simulate visibilities for a given galaxy image and generate dirty images and PSF.

    Parameters:
    -----------
    gal_data (list or int): List contaiing galaxy image as a numpy.ndarray and list of galaxy parameters. If an integer is provided, the function returns 0.
    ra_gal (float): Right ascension of the galaxy.
    dec_gal (float): Declination of the galaxy.
    cellsize (float, optional): Cell size for the visibility grid. Defaults to def_cellsize.

    **kwargs: Additional keyword arguments.
        - noise (str or float, optional): RMS of noise to add to visibilities. If str, calculate RMS based on dish parameters. Defaults to "fixed".
        - vis_only (bool, optional): If True, only return visibilities and SNR. Defaults to False.
        - PSF (bool, optional): If True, include PSF array in the output. Defaults to True.

    Returns:
    --------
    list: A list containing SNR, peak value, galaxy data, dirty image, PSF, galaxy array, and optionally PSF array.
    If vis_only is True, returns a tuple of visibilities and SNR.
    If gal_data is an integer, returns 0.
    """

    noise = kwargs.get("noise", "fixed")
    vis_only = kwargs.get("vis_only", False)
    PSF = kwargs.get("PSF", True)

    if isinstance(gal_data, int):
        return 0

    gal_arr = gal_data[0]

    # Generate Visibilities
    vt = visibilities.generate_visibilities_from_array(
        gal_arr=gal_arr, ra=ra_gal, dec=dec_gal, cellsize=cellsize, **kwargs
    )

    if isinstance(noise, str):
        vt_n, SNR = visibilities.addNoiseToVis(vt, **kwargs)
    elif isinstance(noise, int | float):
        vt_n, SNR = visibilities.addNoiseToVis(vt, sigma=noise)
    else:
        vt_n = vt
        SNR = np.inf

    if vis_only:
        return vt_n, SNR

    # Make dirty image wo noise
    _, _, dirty_arr0, _ = visibilities.dirty_psf_from_visibilities(
        vt, cellsize=cellsize, **kwargs
    )
    # Save Peak value to calculate PSNR
    peak = np.max(dirty_arr0)

    # Make the dirty image and point spread function
    dirty, psf, dirty_arr, psf_arr = visibilities.dirty_psf_from_visibilities(
        vt_n, cellsize=cellsize, **kwargs
    )
    block = [[SNR, peak, *gal_data[1]], dirty, psf, gal_arr, dirty_arr]

    if PSF:
        block += [psf_arr]

    return block


def deconvolution(block, **kwargs):
    """
    Perform deconvolution on the given block of data.

    Parameters:
    -----------
    block (list or int): Result of simulate_vis function. If an integer is provided, the function returns 0.
    **kwargs: Additional keyword arguments to be passed to the deconvolution algorithm.

    Returns:
    --------
    list: A new list containing the original block elements followed by the deconvolved galaxy image and the residual image.
          If the input block is an integer, returns 0.

    Notes:
    ------
    - The deconvolution is performed using the Multi-Scale CLEAN algorithm.
    - The deconvolved galaxy image and the residual image are converted to numpy arrays of type float32 for computational efficiency.
    """

    if isinstance(block, int):
        return 0

    dirty = block[1]
    psf = block[2]

    deconv_gal, res = deconvolve_cube(dirty, psf, algorithm="msclean", **kwargs)
    deconv_gal = deconv_gal.pixels.to_numpy().astype(np.float32).squeeze()
    res = res.pixels.to_numpy().astype(np.float32).squeeze()

    block_new = [*block, deconv_gal, res]

    return block_new
