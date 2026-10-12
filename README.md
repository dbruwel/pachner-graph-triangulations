# Navigating the Pachner Graph: Algorithms for Searching and Sampling Triangulations

> [!WARNING]
> ## This is a work-in-progress research repository. Expect mess.
>
> **This code is under active development and will be until at least November 2027. It is a working research log, not a polished codebase.**
>
> **It is not written for collaboration.** It exists to track the history of the project and to keep files in sync between the different machines that run the experiments. As a result, you should expect:
>
> - **Messy, exploratory notebooks** that were written to try ideas quickly, not to be read
> - **Undocumented or half-documented code**, dead ends, and experiments that were abandoned partway through
> - **Inconsistent structure** as the project evolves
>
> **This is intentional, and part of the plan.** Research code moves fast, and tidying every experiment would slow the actual research down. Polished, cleaned-up code is written separately when the work is ready to be shared.
>
> The repository is public because it had to be for my Honours submission, and the work has continued on from there. You're very welcome to read through it, and there are some genuinely interesting ideas in here. Just don't expect all of it to make sense, and please don't judge my general coding standards by it.

This repository contains a script that is an almost identical copy of a file from the repository `https://github.com/jspreer/MCMCForTriangulations`. Access to this script has been granted by its owner - Jonathan Spreer. The script is included at `src/pachner_graph_triangulations/functions3d.py`.

## Overview
This project was completed in partial fulfillment of the requirements for the degree of B.Sc. (Honours) under the supervision of Jonathan Spreer.

### Project
The aim of this project is to investigate triangulations of manifolds, with a focus on sampling triangulations and optimizing objective functions defined over them. The study primarily considers single-vertex 3-spheres. Methodologies explored include Markov Chain Monte Carlo (MCMC), simulated annealing, and direct ascent, alongside machine learning techniques employing autoregressive transformers and reinforcement learning.

This repository is not intended as a Python package but serves to showcase the work conducted. However, it is structured so that it can be easily forked and modified for those interested in exploring these techniques in other directions.


### Thesis
The outcomes of this research are documented in a PDF available in this repository under the `thesis` section, which also includes the raw `.tex`.

There is a github action setup so that when this is changed, a new PDF is generated and added to the repository.


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
