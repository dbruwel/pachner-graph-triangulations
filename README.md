# Navigating the Pachner Graph: Algorithms for Searching and Sampling Triangulations

This repositroy contains a script that is an almost identical copy of a file from the repository `https://github.com/jspreer/MCMCForTriangulations`. Access to this script has been granted by it's owner - Jonathan Spreer. The script is inlcuded at `src/pachner_graph_triangulations/...`.

## Instilation
This project has been written with sage. Not all of the scripts require sage, and many can be run in a standard python environment. But for working with the triangulations as actual manifold objects, sage is required. Sage is installable via conda, and does require either Linux or MacOS (if you have windows you will have to use WSL). The installation for sage follows
1. Ensure conda is installed
2. Create a conda environment with `conda create -n sage-env -c conda-forge sage python=3.11`
3. Activate the environment with `conda activate sage-env`
4. Test instillation with `sage`, this should print the sage version.

If you want to run the notebooks, you additionally need to
5. Install ipykernel and notebooks with `conda install -c conda-forge jupyterlab ipykernel`
6. Add sage as a kernel with `python -m ipykernel install --user --name sage-env --display-name "Sage Python"`

One sage is setup. This repository can be cloned and install via `pip install -e .`
