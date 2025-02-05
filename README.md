# DeepShape
**Radio Weak Lensing Shear Measurements using Deep Learning**

DeepShape is a supervised deep-learning framework that predicts the shapes of isolated radio galaxies from their dirty images and associated PSFs. DeepShape is made of two modules. The first module uses a plug-and-play (PnP) algorithm based on the Half-Quadratic Splitting (HQS) method to reconstruct galaxy images. The second module is a measurement network trained to predict galaxy shapes from the reconstructed image-PSF pairs. The measurement network is divided into two branches: one is a feature extraction branch, which employs an equivariant convolutional neural network (CNN) to extract features from the reconstructed image, while the other is a pre-trained encoder block that compresses the PSF into a low-dimensional code, accounting for PSF-dependent errors. The outputs of both branches are combined and passed through a dense layer block to predict the ellipticity.

## Related Papers
1. Image Reconstruction: [Zhang et al (2021)](https://arxiv.org/abs/2008.13751)
2. E(2) Equivariant CNN: Cohen and Weiling (2016) https://arxiv.org/abs/1602.07576


## Requirements
The entire system is contained within a docker image, for which the Dockerfile is within this repository. To run: 
1. Clone repository
2. Install docker
3. Build docker image using the command > docker build -t <image_name> .
4. Run the image using > docker run -p 8888:8888 -v <host_directory>:<container_directory> -it <image_name>
5. The docker container will start. You can then run RASCIL from its home directory, as all the required julia files will be copied there.
6. Example commands for execution can be found in the README.rascil file. Simulated datasets can be found in the various data directories. 

## Splitting measurement sets
Measurement sets can be split using casa. Specifically, one can use casatools.ms, for which documentation can be found: https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms
Update: The dirty.ipynb notebook has been updated to also be able to split the datasets

## Results
The results for the paper are presented in a variety of jupyter notebooks. These are:
- dirty.ipynb for presenting the original datasets and their split
- lambda.ipynb for preliminary experiments on $\lambda$
- noise.ipynb for preliminary experiments and estimation of $\sigma^2$ and $\eta^2$
- split.ipynb for results regarding partition configurations, and against the single-step LASSO reconstruction

The raw data for our results will be made available in the near future
