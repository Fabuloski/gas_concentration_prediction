# Metal−Organic Framework-Based Chemiresistive Array for the Detection and Differentiation of Toxic Gases
:rocket: This repo contains data and code to reproduce the results for:
> Georganna Benedetto, Patrick Damacet, Elissa O. Shehayeb, Gbenga Fabusola, Cory M. Simon, and Katherine A. Mirica, "Metal−Organic Framework-Based Chemiresistive Array for the Detection and Differentiation of Toxic Gases"

We describe the sequence of steps we took to make our paper reproducible. The output of each step is saved as a file, you can start at any step.

## required software
required software/packages:
* [Python 3](https://www.python.org/downloads/) version 3.8 or newer
* [Marimo Notebook](https://docs.marimo.io/)

## the sensor array response dataset
We obtained the dataset of sensors' responses to gas mixtures (H2S + SO2) from an experimental collaboration at Dartmouth College led by Dr. Katherine A. Mirica.

## analysis
We run the PCA, supervised learning, sensor importance, and uncertainty quantification on the sensor array dataset using `sensor_response.py`.

## overview of directories
- `H2S+SO2-data`: contains Microsoft Excel raw response files of sensors to different mixture concentrations [ppm]
- `responses`: contains visualization of the response of sensors to every single experiment (ppm of gas mixture (H2S + SO2)) with the features extracted
-  `doe (Design of Experiment)`: contains the code `design_of_mixture_expts.py` used to design the experiment space
