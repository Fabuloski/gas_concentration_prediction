# Metal−Organic Framework-Based Chemiresistive Array for the Detection and Differentiation of Toxic Gases
:rocket: this repo contains data and code to reproduce the results for:
> Georganna Benedetto, Patrick Damacet, Elissa O. Shehayeb, Gbenga Fabusola, Cory M. Simon, and Katherine A. Mirica, "Metal−Organic Framework-Based Chemiresistive Array for the Detection and Differentiation of Toxic Gases"

we describe the sequence of steps we took to make our paper reproducible. the output of each step is saved as a file, so you can start at any step.

## required software
required software/packages:
* [Python 3](https://www.python.org/downloads/) version 3.8 or newer
* [Marimo Notebook](https://docs.marimo.io/)

## the sensor array response dataset
we obtained the dataset of the response of sensors to mixture from experimental collaboration from Dartmouth College led by Dr. Katherine A. Mirica.

## analysis
we run the PCA, supervised learning, sensor importance and uncertainity quantification on the sensor array dataset using `sensor_response.py`.

## overview of directories
- `H2S+SO2-data`: contains Microsoft Excel raw response files of sensors to different mixture concentrations [ppm]
- `responses`: contain visualization of the response of sensors to every single experiments (ppm of gas mixture (H2S + SO2)) with the features extracted
