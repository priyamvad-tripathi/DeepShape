# DeepShape
## Radio Weak Lensing Shear Measurements using Deep Learning

DeepShape is a supervised deep-learning framework that predicts the shapes of isolated radio galaxies from their dirty images and associated PSFs. DeepShape is made of two modules. The first module uses a plug-and-play (PnP) algorithm based on the Half-Quadratic Splitting (HQS) method to reconstruct galaxy images. The second module is a measurement network trained to predict galaxy shapes from the reconstructed image-PSF pairs. The measurement network is divided into two branches: one is a feature extraction branch, which employs an equivariant convolutional neural network (CNN) to extract features from the reconstructed image, while the other is a pre-trained encoder block that compresses the PSF into a low-dimensional code, accounting for PSF-dependent errors. The outputs of both branches are combined and passed through a dense layer block to predict the ellipticity.

DeepShape is based on the findings presented in the following papers:
1. Shape measurement of radio galaxies using Equivariant CNNs: [Tripathi et al (2024)](https://ieeexplore.ieee.org/abstract/document/10715370)
2. DeepShape: Radio Weak Lensing Shear Measurements using Deep Learning: tbd

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
### Dataset simulation
All the necessary scripts for simulating the training and testing datasets can be found in the [Simulation/](Simulation/) folder. Make sure to download all the FITS files containing the [T-RECS catalog](http://cdsarc.u-strasbg.fr/ftp/VII/282/fits/) and run the [make_catalog.py](Simulation/make_catalog.py) script to join all the FITS file into a single pandas dataframe containing only the required information. 
### Image Reconstruction
The [Reconstruction/](Reconstruction/) folder contains the scripts connected to image reconstruction using HQS-PnP algorithm. Make sure that the _DeepInverse_ library is correctly installed. By default, DRUNet is initialized using pre-trained weights from the library. This can be changed by setting the "pretrained" argument to a path containing the user-weights (see [PnP_tuning.py](Reconstruction/PnP_tuning.py) for details)
### Shape Measurement
The [Shape_Measurement/](Shape_Measurement/) folder contains the scripts connected to the shape measurement network. It also includes the scripts to perform shape measurements using [RadioLensfit](Shape_Measurement/RadioLensfit) and [SuperCALS](Shape_Measurement/SuperCALS) methods.
### Model Weights
DeepShape uses three networks: DRUNet denoiser, PSF autoencoder, and the shape measurement network. The trained weights for all three networks can be found at [OCA Cloud](https://cloud.oca.eu/index.php/s/KbMB8SbingdWibe).

## Cite
You can cite our work using the following $\BibTeX{}$ entry:
 ````
 @article{DeepShape,
	author = {{Tripathi, P.} and {Wang, S.} and {Prunet, S.} and {Ferrari, A.}},
	title = {DeepShape: Radio weak-lensing shear measurements using deep learning},
	DOI= "10.1051/0004-6361/202554072",
	journal = {A&A},
	year = 2025,
	volume = 696,
	pages = "A216",
}
````
 ````
@INPROCEEDINGS{Tripathi2024,
  author={Tripathi, Priyamvad and Wang, Sunrise and Prunet, Simon and Ferrari, André},
  booktitle={EUSIPCO}, 
  title={Shape measurement of radio galaxies using Equivariant CNNs}, 
  year={2024},
  pages={2377-2381},
  doi={10.23919/EUSIPCO63174.2024.10715370}}
 ````
Feel free to [contact us](mailto:priyamvad.tripathi@oca.eu).

## Useful Resources
1. Image Reconstruction: [Zhang et al (2017)](https://arxiv.org/abs/1704.03264); [Zhang et al (2021)](https://arxiv.org/abs/2008.13751)
2. E(2) Equivariant CNN: [Cohen and Welling (2016)](https://arxiv.org/abs/1612.08498); [Weiler and Cesa (2019)](https://arxiv.org/abs/1911.08251)
3. T-RECS catalogue: [Bonaldi et al (2018)](https://academic.oup.com/mnras/article/482/1/2/5108200) 
4. [_escnn_ Documentation](https://quva-lab.github.io/escnn/): For building $\mathrm{E}(2)$ equivariant CNNs
5. [_RASCIL_ Documentation](https://rascil-main.readthedocs.io/en/1.1.0/index.html): For visibility simulations
6. [_DeepInverse_ Library](https://github.com/deepinv/deepinv/): For building DRUNet with pre-learned weights
7. Contemporary Methods: [SuperCALS method](https://academic.oup.com/mnras/article/495/2/1737/5815095); [RadioLensfit Method](https://www.sciencedirect.com/science/article/pii/S2213133722000191)
