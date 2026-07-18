#!/bin/bash
#PBS -P io00
#PBS -q dgxa100
#PBS -l walltime=00:15:00
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=200GB
#PBS -l jobfs=10GB
#PBS -l storage=gdata/io00+scratch/io00
#PBS -N oom_test
#PBS -M dbru8728@uni.sydney.edu.au
#PBS -m abe

# Path to script.
cd /g/data/io00/js1886/

# Activate the venv.
source /g/data/io00/js1886/trig/bin/activate

# Run the test.
python ./pachner-graph-triangulations/experiments/scripts/deep_learning/signature_model/auto_regression_train_15.py run_test_nci
