#!/bin/bash

# Load conda into the environment
source /home/doptrackbox/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate venv

cd /home/doptrackbox/DTB_Software/adapted_processing/

# Run your Python script
python ./schedule_curvefitting.py

