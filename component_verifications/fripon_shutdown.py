#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt


plt.switch_backend('Agg')

# Data locations (adjust if needed)
LOC = '/home/doptrackbox/DTB_Recordings/data/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
PLOT_LOC = "./Plots/"
EXTRACTION_LOC = os.path.join(LOCi, 'Processed/')

def db_to_ratio(db):
    return 10**(db/10)

def ratio_to_db(ratio):
    return 10*np.log10(ratio)

file_names = ["FUNcube-1_39444_202509070450_145935kHz", "FUNcube-1_39444_202509170425_145935kHz"]
labels = ["Before Shutdown", "During Shutdown"]
plot_labels = ["before_shutdown", "during_shutdown"]

# Define plots
fig_1, axes_1 = plt.subplots(2,1, figsize=(6,4), sharex=True)
fig_2, axes_2 = plt.subplots(2,1, figsize=(6,4), sharex=True)

figs = [fig_1, fig_2]
axes = [axes_1, axes_2]

for file, label, plot_label, fig, ax in zip(file_names, labels, plot_labels, figs, axes):
    ax_1, ax_2 = ax
    noise_data = np.loadtxt(EXTRACTION_LOC + "ExtractedNoise/" + file + ".txt", skiprows=1)
    snr_data = np.loadtxt(EXTRACTION_LOC + "ExtractedSignal/" + file + ".txt", skiprows=1)    
    ax_1.plot(noise_data[:,0], noise_data[:,1], label=label) # Noise plot
    ax_2.plot(snr_data[:,0], ratio_to_db(snr_data[:,4]), label=label) # SNR plot

    # Makeup of before shutdown plot
    ax_1.set_ylim(0.95, 1.1)
    ax_1.set_xlim(noise_data[0,0], noise_data[-1,0])
    ax_1.set_ylabel('Noise [-]')
    ax_1.set_title(f'Typical DopTrackBox FUNcube-1 Recording {label}')
    ax_1.grid()

    # Makeup of during shutdown
    ax_2.set_xlim(snr_data[0,0], snr_data[-1,0])
    ax_2.set_xlabel('Time [s]')
    ax_2.set_ylabel('Signal-to-Noise Ratio [dB]')
    ax_2.grid()
    fig.tight_layout()
    fig.savefig(PLOT_LOC + f"fripon_{plot_label}_snr.png")

