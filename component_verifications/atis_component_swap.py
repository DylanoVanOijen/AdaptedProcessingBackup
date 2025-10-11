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

file_names = ["ATIS_99999_202501010012_128565kHz",
              "ATIS_99999_202501010011_128565kHz", 
              "ATIS_99999_202501010013_128565kHz",
              "ATIS_99999_202501010014_128565kHz",]
labels = ["Default", "DopTrack Antenna", "Different Coax Cable", "Different RSPduo & Coax Cable"]

# Define plots
fig_1, ax_1 = plt.subplots(figsize=(6,4.5))
fig_2, ax_2 = plt.subplots(figsize=(6,4.5))

for file, label in zip(file_names, labels):
    noise_data = np.loadtxt(EXTRACTION_LOC + "ExtractedNoise/" + file + ".txt", skiprows=1)
    snr_data = np.loadtxt(EXTRACTION_LOC + "MaxSigSNR/" + file + ".txt", skiprows=1)    
    ax_1.plot(noise_data[:,0], noise_data[:,1], label=label) # Noise plot
    ax_2.plot(snr_data[:,0], ratio_to_db(snr_data[:,1]), label=label) # SNR plot

# Makeup of Noise plot
ax_1.set_xlim(noise_data[0,0], noise_data[-1,0])
ax_1.set_xlabel('Time [s]')
ax_1.set_ylabel('Noise [-]')
ax_1.set_title('Extracted Noise for Different Hardware')
ax_1.grid()
ax_1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
fig_1.tight_layout()
fig_1.savefig(PLOT_LOC + "atis_hardware_swap_noise.png")

# Makeup of SNR plot
ax_2.set_xlim(snr_data[0,0], snr_data[-1,0])
ax_2.set_xlabel('Time [s]')
ax_2.set_ylabel('Signal-to-Noise Ratio [dB]')
ax_2.set_title('Extracted Noise for Different Hardware')
ax_2.grid()
ax_2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
fig_2.tight_layout()
fig_2.savefig(PLOT_LOC + "atis_hardware_swap_snr.png")