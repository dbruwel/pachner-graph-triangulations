# Navigating the Pachner Graph: Algorithms for Searching and Sampling Triangulations

This repository contains a script that is an almost identical copy of a file from the repository `https://github.com/jspreer/MCMCForTriangulations`. Access to this script has been granted by its owner - Jonathan Spreer. The script is included at `src/pachner_graph_triangulations/functions3d.py`.

## Overview
This project was completed in partial fulfillment of the requirements for the degree of B.Sc. (Honours) under the supervision of Jonathan Spreer.

### Project
The aim of this project is to investigate triangulations of manifolds, with a focus on sampling triangulations and optimizing objective functions defined over them. The study primarily considers single-vertex 3-spheres. Methodologies explored include Markov Chain Monte Carlo (MCMC), simulated annealing, and direct ascent, alongside machine learning techniques employing autoregressive transformers and reinforcement learning.

This repository is not intended as a Python package but serves to showcase the work conducted. However, it is structured so that it can be easily forked and modified for those interested in exploring these techniques in other directions.

The outcomes of this research are documented in a PDF available in this repository under the `thesis` section.

## Installation
This project has been written with Sage. Not all of the scripts require Sage, and many can be run in a standard Python environment. However, for working with the triangulations as actual manifold objects, Sage is required. Sage is installable via conda, and requires either Linux or macOS (if you have Windows you will have to use WSL). The installation for Sage is as follows:
1. Ensure conda is installed.
2. Create a conda environment with `conda create -n sage-env -c conda-forge sage python=3.11`
3. Activate the environment with `conda activate sage-env`
4. Test installation with `sage`; this should print the Sage version.

If you want to run the notebooks, you additionally need to:
5. Install ipykernel and notebooks with `conda install -c conda-forge jupyterlab ipykernel`
6. Add Sage as a kernel with `python -m ipykernel install --user --name sage-env --display-name "Sage Python"`

Once Sage is set up, this repository can be cloned and installed via `pip install -e .`

## Thesis
The raw tex files for the thesis are located in `\thesis`. There is a github action setup so that when this is changed, a new PDF is generated and added to the repository.
