#!/bin/bash
#PBS -P ab12
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
cd /g/data/ab12/my_ml_project/

# Activate the venv.
source /g/data/ab12/my_venv/bin/activate

# Run the test.
python auto_regression_train_scaling_nci.py test
