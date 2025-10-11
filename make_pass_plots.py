#!/usr/bin/env python3

import os
import sys
import yaml
import gc
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from numpy.fft import fft, fftfreq, fftshift
from datetime import datetime, timezone
from matplotlib.colors import LogNorm

plt.switch_backend('Agg')

# Data locations (adjust if needed)
LOC = '/home/doptrackbox/DTB_Recordings/data/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
PLOT_LOC = os.path.join(LOCi, 'Plots/')
EXTRACTION_LOC = os.path.join(LOCi, 'Processed/')
LOG_FILE = os.path.join(LOCi, 'plotted_files.log')

SATELLITE_FREQS = {
    "FUNcube-1" : [145935000],
    "FOX-1B" : ['TUNE'],
    "ESEO" : ['TUNE'],    
    "RIDU-Sat-1" : ['TUNE'],
    "VELOX-PII" : ['TUNE'],
    "ISS" : ['TUNE', 145.800*10**6],
    "NOAA-19" : ['TUNE'],
    "SONATE-2" : ['TUNE', 145.880*10**6, 145.825*10**6],  
    "NOAA-15" : ['TUNE'],
    "NOAA-2" : ['TUNE'],
    "NOAA-9" : ['TUNE'],
    "TRANSIT5B-5" : ['TUNE', 136.659*10**6],
    "NAYIF" : [145.939*10**6],
}

def plot_extracted_signal(file):
    if os.path.exists(EXTRACTION_LOC + "ExtractedSignal/" + file + ".txt"):
        with open(EXTRACTION_LOC + "ExtractedSignal/" + file + ".txt", 'r') as f:
            header = f.readline().strip()
        main_part = header.split("\tPower")[0]
        entries = [item for item in main_part.split("|")[1:] if ":" in item]
        parts = dict(item.split(":", 1) for item in entries)
        mask_width = float(parts["MW"])
        tune = float(parts["TUNE"])

        center_frequency = file.split('_')[-1]
        numeric_frequency = 1000*int(center_frequency.replace("kHz", ""))
        freq_offset = numeric_frequency - tune

        # Read from file
        data = np.loadtxt(EXTRACTION_LOC + "ExtractedSignal/" + file + ".txt", skiprows=1)
        time = data[:, 0]
        signal_freq = data[:, 1] - tune

        fig_1, ax_1 = plt.subplots(figsize=(10,5))    
        ax_1.scatter(time, signal_freq, s=10)
        ax_1.set_xlabel('Time [s]')
        ax_1.set_ylabel('Frequency [Hz]')
        ax_1.set_title('Extracted Signal Frequency w.r.t. tuning frequency')


        #ax_1.set_ylim(0, 40)
        ax_1.set_xlim(time[0], time[-1])
        if mask_width != 0:
            ax_1.set_ylim(freq_offset-(mask_width/2), freq_offset+(mask_width/2))
        ax_1.grid()
        #fig_1.set_size_inches(19, 10)
        fig_1.tight_layout()
        os.makedirs(PLOT_LOC+"Signal", exist_ok=True)
        fig_1.savefig(os.path.join(PLOT_LOC, f"Signal/signal_{file}.png"))
        plt.close(fig_1)
        gc.collect()

        # SNR Values
        noise_data = np.loadtxt(EXTRACTION_LOC + "ExtractedNoise/" + file + ".txt", skiprows=1)

        noise_times = noise_data[:,0]
        noise_powers = noise_data[:,1]

        idx = np.searchsorted(noise_times, data[:, 0])
        noise_at_signal_entries = noise_powers[idx]
        
        SNR = data[:, 2] / noise_at_signal_entries
        SNR_dB = 10*np.log10(SNR)

        os.makedirs(PLOT_LOC+"ExtractedSigSNR", exist_ok=True)
        fig_1, ax_1 = plt.subplots(figsize=(6,4))    
        ax_1.plot(time, SNR)
        ax_1.set_xlabel('Time [s]')
        ax_1.set_ylabel('SNR [-]')
        ax_1.set_title('SNR of extracted frequency as function of time')

        #ax_1.set_ylim(0, 40)
        ax_1.set_xlim(time[0], time[-1])
        ax_1.grid()
        #fig_1.set_size_inches(19, 10)
        fig_1.tight_layout()
        fig_1.savefig(os.path.join(PLOT_LOC, f"ExtractedSigSNR/ExtractedSigSNR_{file}.png"))
        plt.close(fig_1)
        gc.collect()

        fig_2, ax_2 = plt.subplots(figsize=(6,4))    
        ax_2.plot(time, SNR_dB)
        ax_2.set_xlabel('Time [s]')
        ax_2.set_ylabel('SNR [dB]')
        ax_2.set_title('SNR of extracted frequency as function of time')

        #ax_2.set_ylim(0, 40)
        ax_2.set_xlim(time[0], time[-1])
        ax_2.grid()
        #fig_2.set_size_inches(19, 10)
        fig_2.tight_layout()
        fig_2.savefig(os.path.join(PLOT_LOC, f"ExtractedSigSNR/ExtractedSigSNR_{file}_dB.png"))
        plt.close(fig_2)
        gc.collect()

        # HISTOGRAMS
        os.makedirs(PLOT_LOC+"ExtractedSigSNRHist", exist_ok=True)
        fig_3, ax_3 = plt.subplots(figsize=(6,4))    
        ax_3.hist(SNR, bins=25)
        ax_3.set_xlabel('SNR [-]')
        ax_3.set_ylabel('Occurance [-]')
        ax_3.set_title('Histogram of the extracted signal SNR values')

        #ax_3.set_ylim(0, 40)
        #ax_3.set_xlim(time[0], time[-1])
        ax_3.grid()
        #fig_3.set_size_inches(19, 10)
        fig_3.tight_layout()
        fig_3.savefig(os.path.join(PLOT_LOC, f"ExtractedSigSNRHist/SNR_hist_{file}.png"))
        plt.close(fig_3)
        gc.collect()

        fig_4, ax_4 = plt.subplots(figsize=(6,4))    
        ax_4.hist(SNR_dB, bins=25)
        ax_4.set_xlabel('SNR [dB]')
        ax_4.set_ylabel('Occurance [-]')
        ax_4.set_title('Histogram of the extracted signal SNR values')

        #ax_4.set_ylim(0, 40)
        #ax_4.set_xlim(time[0], time[-1])
        ax_4.grid()
        #fig_4.set_size_inches(19, 10)
        fig_4.tight_layout()
        fig_4.savefig(os.path.join(PLOT_LOC, f"ExtractedSigSNRHist/SNR_hist_{file}_dB.png"))
        plt.close(fig_4)
        gc.collect()
    else:
        print(f"No extracted signal found!")


def plot_noise(file):
    # Read from file
    data = np.loadtxt(EXTRACTION_LOC + "ExtractedNoise/" + file + ".txt", skiprows=1)
    time = data[:, 0]
    noise = data[:, 1]

    fig_1, ax_1 = plt.subplots(figsize=(6,4))    
    ax_1.plot(time, noise)
    ax_1.set_xlabel('Time [s]')
    ax_1.set_ylabel('Noise [-]')
    ax_1.set_title('Noise floor as function of time')

    #ax_1.set_ylim(0, 40)
    ax_1.set_xlim(time[0], time[-1])
    ax_1.grid()
    #fig_1.set_size_inches(19, 10)
    fig_1.tight_layout()
    os.makedirs(PLOT_LOC+"Noise", exist_ok=True)
    fig_1.savefig(os.path.join(PLOT_LOC, f"Noise/noise_{file}.png"))
    plt.close(fig_1)
    gc.collect()

def plot_max_power_SNR(file):
    # Read from file
    data = np.loadtxt(EXTRACTION_LOC + "MaxSigSNR/" + file + ".txt", skiprows=1)
    time = data[:, 0]
    SNR = data[:, 1]

    SNR_dB = []

    for SNR_val in SNR:
        if SNR_val == 0:
            SNR_dB.append(0)
        else:
            SNR_dB.append(10*np.log10(SNR_val))


    os.makedirs(PLOT_LOC+"MaxSigSNR", exist_ok=True)
    fig_1, ax_1 = plt.subplots(figsize=(6,4))    
    ax_1.plot(time, SNR)
    ax_1.set_xlabel('Time [s]')
    ax_1.set_ylabel('SNR [-]')
    ax_1.set_title('SNR of max power frequency as function of time')

    #ax_1.set_ylim(0, 40)
    ax_1.set_xlim(time[0], time[-1])
    ax_1.grid()
    #fig_1.set_size_inches(19, 10)
    fig_1.tight_layout()
    fig_1.savefig(os.path.join(PLOT_LOC, f"MaxSigSNR/MaxSigSNR_{file}.png"))
    plt.close(fig_1)
    gc.collect()

    fig_2, ax_2 = plt.subplots(figsize=(6,4))    
    ax_2.plot(time, SNR_dB)
    ax_2.set_xlabel('Time [s]')
    ax_2.set_ylabel('SNR [dB]')
    ax_2.set_title('SNR of max power frequency as function of time')

    #ax_2.set_ylim(0, 40)
    ax_2.set_xlim(time[0], time[-1])
    ax_2.grid()
    #fig_2.set_size_inches(19, 10)
    fig_2.tight_layout()
    fig_2.savefig(os.path.join(PLOT_LOC, f"MaxSigSNR/MaxSigSNR_{file}_dB.png"))
    plt.close(fig_2)
    gc.collect()

    # HISTOGRAMS
    os.makedirs(PLOT_LOC+"MaxSigSNRHist", exist_ok=True)
    fig_3, ax_3 = plt.subplots(figsize=(6,4))    
    ax_3.hist(SNR, bins=25)
    ax_3.set_xlabel('SNR [-]')
    ax_3.set_ylabel('Occurance [-]')
    ax_3.set_title('Histogram of max power frequency SNR values')

    #ax_3.set_ylim(0, 40)
    #ax_3.set_xlim(time[0], time[-1])
    ax_3.grid()
    #fig_3.set_size_inches(19, 10)
    fig_3.tight_layout()
    fig_3.savefig(os.path.join(PLOT_LOC, f"MaxSigSNRHist/SNR_hist_{file}.png"))
    plt.close(fig_3)
    gc.collect()

    fig_4, ax_4 = plt.subplots(figsize=(6,4))    
    ax_4.hist(SNR_dB, bins=25)
    ax_4.set_xlabel('SNR [dB]')
    ax_4.set_ylabel('Occurance [-]')
    ax_4.set_title('Histogram of max power frequency SNR values')

    #ax_4.set_ylim(0, 40)
    #ax_4.set_xlim(time[0], time[-1])
    ax_4.grid()
    #fig_4.set_size_inches(19, 10)
    fig_4.tight_layout()
    fig_4.savefig(os.path.join(PLOT_LOC, f"MaxSigSNRHist/SNR_hist_{file}_dB.png"))
    plt.close(fig_4)
    gc.collect()


def update_processed_log(processed_file):
    with open(LOG_FILE, 'a') as f:
        f.write(processed_file + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: make_pass_plots.py <base_filepath>")
        sys.exit(1)

    FILE = sys.argv[1]
    try:    
        plot_noise(FILE)
        plot_max_power_SNR(FILE)

        plot_extracted_signal(FILE)

        update_processed_log(FILE)

    except Exception as e:
        print(f"[ERROR] Failed to process {FILE}: {e}")
        sys.exit(1)
