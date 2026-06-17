#!/bin/bash
#PBS -P io00
#PBS -q normal
#PBS -l walltime=00:15:00
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l jobfs=10GB
#PBS -l storage=gdata/ab12+scratch/ab12
#PBS -N test_io
#PBS -M dbru8728@uni.sydney.edu.au
#PBS -m abe

# Path to script.
cd /g/data/io00/js1886/

# Activate the venv.
source /g/data/io00/js1886/trig/bin/activate

# Run the test.
python ./pachner-graph-triangulations/experiments/scripts/deep_learning/signature_model/auto_regression_train_scaling_nci.py test
