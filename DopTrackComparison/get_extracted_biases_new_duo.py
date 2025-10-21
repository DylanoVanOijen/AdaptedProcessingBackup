#!/usr/bin/env python3

import os
import sys
import yaml
import numpy as np
from datetime import datetime, timezone, timedelta

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/dt_comparison.log'
EXTRACTION_LOC = '/home/doptrackbox/DTB_Recordings/locally_processed/Processed/'
LOCAL_LOC = '/home/doptrackbox/DTB_Recordings/data/FUNcube-1/2025/'
REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/archive/Dylano/DopTrackBox/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
DopTrackLoc = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/'
PLOT_LOC = os.path.join(LOCi, 'Plots/')

files = ["09221504",
         "09221638",
         "09230410",
         "09230544",
         "09230720",
         "09231635",
         "09240407",
         "09240542",
         "09240717",
         "09241458",
         "09241633",
         "09250404",
         "09250539",
         "09250714",
         "09251456",
         "09251630",
         "09260402",
         "09260536",
         "09260712",
         "09261627",
         "09270533",
         "09271624",
         "09280530",
         "09281621",
         "09290527"]


print("t_start - t1:")
t_minus_t = []
for file in files:
    full_ident = f"FUNcube-1_39444_2025{file}.yml"
    with open(LOCAL_LOC + full_ident, 'r') as f:
        yml_info = yaml.load(f, Loader=yaml.FullLoader)
    t1 = yml_info["Sat"]["Record"]["time1 UTC"].replace(tzinfo=timezone.utc)
    t2 = yml_info["Sat"]["Record"]["time2 UTC"].replace(tzinfo=timezone.utc)
    samples = yml_info['Sat']['Record']['num_sample']
    sample_rate = yml_info['Sat']['Record']['sample_rate']
    duration = samples/sample_rate
    t_start = t2 - timedelta(seconds=duration)
    diff = (t_start - t1).total_seconds()
    print(f"{diff:.2f}")
    t_minus_t.append(diff)

print(f"TminusT mean = {np.mean(t_minus_t):.2f}")
print(f"TminusT std = {np.std(t_minus_t):.2f}")

# Time and frequency bias stuff
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

print(" ")
print("Delta t t1 ")
print(full[:,0:2])

print("Delta t tstart")
print(full[:,4:6])

print("Delta t V2 t1")
print(full[:,8:10])

print("Delta f V2 t1")
print(full[:,10:12])

print("Delta t V2 tstart")
print(full[:,12:14])

print("Delta f V2 tstart")
print(full[:,14:16])


means = np.mean(full, axis = 0)
stds = np.std(full, axis = 0)

print("Means = ", means)
print("STDs = ", stds)

for category in range(0,4):
    category_values = full[:,category*4:category*4+4]
    delta_t = category_values[:,0]
    delta_t_uncert = category_values[:,1]
    delta_f = category_values[:,2]
    delta_f_uncert = category_values[:,3]

    all_zeros_t = not np.any(delta_t)
    all_zeros_f = not np.any(delta_f)

    delta_t_est = np.sum(delta_t*(1/delta_t_uncert**2)) / np.sum((1/delta_t_uncert**2))
    sigma_t_est = np.sqrt(1/np.sum(1/delta_t_uncert**2))

    if all_zeros_f:
        delta_f_est = 0
        sigma_f_est = 0
    else:
        delta_f_est = np.sum(delta_f*(1/delta_f_uncert**2)) / np.sum((1/delta_f_uncert**2))    
        sigma_f_est = np.sqrt(1/np.sum(1/delta_f_uncert**2))

    
    print("")
    print(subdirs[category])
    #print(f"delta_t +- sigma = {delta_t_est:.2f} +- {sigma_t_est:.2f}" )
    print(f"delta t mean & sigma = {np.mean(delta_t):.2f} +- {np.std(delta_t):.2f}")
    #print(f"delta_f +- sigma = {delta_f_est:.2f} +- {sigma_f_est:.2f}" )
    print(f"delta f mean & sigma = {np.mean(delta_f):.2f} +- {np.std(delta_f):.2f}")



