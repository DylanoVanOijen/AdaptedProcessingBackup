#!/usr/bin/env python3

import os
import sys
import yaml
import numpy as np

LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/dt_comparison.log'
EXTRACTION_LOC = '/home/doptrackbox/DTB_Recordings/locally_processed/Processed/'
LOCAL_LOC = '/home/doptrackbox/DTB_Recordings/data/'
REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/archive/Dylano/DopTrackBox/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
DopTrackLoc = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/'
PLOT_LOC = os.path.join(LOCi, 'Plots/')

files = ["08281604",
         "08300509",
         "08301600",
         "08310507",
         "09010505",
         "09020502",
         "09040458",
         "09050455",
         "09051546",
         "09060453",
         "09070450",
         "09080448",
         "09100443",
         "09111532",
         "09160428",
         "09161519",
         "09170425",
         "09170600",
         "09171517",
         "09171652",
         "09180423",
         "09180558",
         "09190420",
         "09200418",
         "09210550",
         "09211641",
         "09220547"]

subdirs = ["FittedBias/", "FittedBiasRecomputed/", "FittedBiasV2/", "FittedBiasRecomputedV2/"]

first = True
full = np.array([])
for file in files:
    values = []
    full_ident = f"FUNcube-1_39444_2025{file}_145935kHz"

    for subdir in subdirs:
        with open(EXTRACTION_LOC + subdir + "bias_fit_params_" + full_ident + ".yml", 'r') as f:
            biases = yaml.load(f, Loader=yaml.FullLoader)
        values.extend([biases['delta_t'], biases['delta_t_uncertainty'], biases['delta_f'], biases['delta_f_uncertainty']])

    values_array = np.array(values)
    if first == True:
        full = values_array
        first = False
    else:
        full = np.vstack((full, values_array))

print(subdirs)
print(full)

means = np.mean(full, axis = 0)
stds = np.std(full, axis = 0)

print("Means = ", means)
print("STDs = ", stds)