# DeepShape
**Radio Weak Lensing Shear Measurements using Deep Learning**

DeepShape is a supervised deep-learning framework that predicts the shapes of isolated radio galaxies from their dirty images and associated PSFs. DeepShape is made of two modules. The first module uses a plug-and-play (PnP) algorithm based on the Half-Quadratic Splitting (HQS) method to reconstruct galaxy images. The second module is a measurement network trained to predict galaxy shapes from the reconstructed image-PSF pairs. The measurement network is divided into two branches: one is a feature extraction branch, which employs an equivariant convolutional neural network (CNN) to extract features from the reconstructed image, while the other is a pre-trained encoder block that compresses the PSF into a low-dimensional code, accounting for PSF-dependent errors. The outputs of both branches are combined and passed through a dense layer block to predict the ellipticity.

DeepShape is based on the findings presented in the following papers:
1. Shape measurement of radio galaxies using Equivariant CNNs: [Tripathi et al (2024)](https://ieeexplore.ieee.org/abstract/document/10715370)
2. DeepShape: Radio Weak Lensing Shear Measurements using Deep Learning: tbd

## Related Papers
1. Image Reconstruction: [Zhang et al (2017)](https://arxiv.org/abs/1704.03264); [Zhang et al (2021)](https://arxiv.org/abs/2008.13751)
2. E(2) Equivariant CNN: [Cohen and Welling (2016)](https://arxiv.org/abs/1612.08498); [Weiler and Cesa (2019)](https://arxiv.org/abs/1911.08251)


## Installation
 
1. Clone repository
2. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)/[Miniconda](https://docs.anaconda.com/miniconda/install/)
3. Install git and clone repository
  ````
  conda update conda
  conda install git
  git clone https://github.com/priyamvad-tripathi/DeepShape.git
  ````
4. Create a conda environment and install all the required dependencies by running the following commands:
  ````
  cd DeepShape/Requirements/
  conda env create --name <env-name> --file implementation.yml
  ````
5. Install the [_escnn_](https://github.com/QUVA-Lab/escnn/) package in the conda environment. This package is required to build the equivariant CNN.
   This can be done by using the following command:
   ````
   pip install git+https://github.com/QUVA-Lab/escnn
   ````
   > **WARNING**: Install version 1.0.13 of _escnn_. The installation might require a separate installation of the _lie_learn_ and _py3nj_ libraries.
7. **[Optional]** Create a separate conda environment for simulating datasets:
  ````
  cd DeepShape/Requirements/
  conda env create --name <env-name> --file simulation.yml
  ````
7. **[Optional]** Install the [_RASCIL_](https://gitlab.com/ska-telescope/external/rascil-main) package in the simulation environment. This can be done by using the following commands:
  ````
  git clone https://gitlab.com/ska-telescope/external/rascil-main
  cd rascil-main
  git checkout tags/1.1.0
  make install_requirements
  ````
8. **[Optional]** Install the [RadioLensfit](https://github.com/marziarivi/RadioLensfit2/) package for comparison.

## Usage
###Simulations
All the necessary scripts for simulating the training and testing datasets can be found in [Simulation](Simulation/) folder. Make sure to download all the FITS files containing the [T-RECS catalog](http://cdsarc.u-strasbg.fr/ftp/VII/282/fits/) and run the [make_catalog.py](Simulation/make_catalog.py) script to join all the FITS file into a single pandas dataframe containing only the required information. 
