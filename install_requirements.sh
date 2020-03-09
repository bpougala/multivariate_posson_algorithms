#!/usr/bin/env bash

conda create -y -n MultivariatePoisson python=3.8
conda activate MultivariatePoisson
conda install -y -c numpy scipy matplotlib scikit-learn pandas anaconda jupyter ipykernel
python -m ipykernel install --user --name MultivariatePoisson --display-name "MultiPoisson (Python 3.8)"
