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
import json

plt.switch_backend('Agg')

colours = cm.lapaz

# Data locations (adjust if needed)
LOC = '/home/doptrackbox/DTB_Recordings/data/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
PLOT_LOC = os.path.join(LOCi, 'Plots/')
EXTRACTION_LOC = os.path.join(LOCi, 'Processed/')
LOG_FILE = os.path.join(LOCi, 'processed_files.log')

# Matplotlib parameters
#params = {'axes.labelsize': 20,'axes.titlesize': 20,
#          'xtick.labelsize': 16, 'ytick.labelsize': 16,
#          'figure.titlesize': 20, 'legend.fontsize': 16}
#plt.rcParams.update(params)


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


def read_arraydata_from_fc32(path, n_bins, bin_size):
    count = bin_size
    with open(path, 'rb') as file:
        for _ in range(n_bins):
            array = np.fromfile(file, dtype=np.complex64, count=count)
            if array.size != count:
                break  # Handle short reads gracefully
            yield array

def load_data(file, dt, duration, n_samples):
    n_bins = int(duration / dt)
    bin_size = int(n_samples / n_bins)
    yield from read_arraydata_from_fc32(file + ".32fc", n_bins, bin_size)

def extract_signal(data, dt, sampling_rate, tune_freq, mask_center=None, mask_width=None):
    nfft = int(dt * sampling_rate)
    frequency = fftshift(fftfreq(nfft, 1/sampling_rate)) + tune_freq

    if mask_width:
        lower = tune_freq + mask_center - mask_width / 2
        upper = tune_freq + mask_center + mask_width / 2
        mask = np.nonzero((lower <= frequency) & (frequency < upper))[0]
        frequency = frequency[mask].copy()
    else:
        mask = None

    rows = []
    noise = []
    timekeep = -1* (dt/2) # start at half a time bin before the start, such that the first time corresponds to the time bin center
    signal_times = []
    signal_powers = []
    signal_freqs = []
    max_signal_SNR = []
    max_signal_SNR_DB = []

    for chunk in data:
        timekeep += dt
        
        # FFT and power spectrum calculation
        fft_vals = fft(chunk, nfft)
        psd = np.abs(fftshift(fft_vals))#**2 / window_norm

        # First compute median and deviation computation over full width, then apply frequency mask
        median = np.median(psd)
        mean = np.average(psd)
        deviation = np.std(psd, dtype=np.float64)
        psd = psd[mask].copy() if mask is not None else psd

        sig_mask = psd >= median + 2 * deviation
        noise_mask = psd < median + deviation

        # Separate noise and signal entries of masked area
        #noise_floor = psd[psd < median + deviation]
        noise_floor = psd[noise_mask]
        signal_lines_power = psd[sig_mask]
        signal_lines_freq = frequency[sig_mask]

        median_noise = np.median(noise_floor)
        mean_noise = np.mean(noise_floor)
        #noise.append(median_noise)
        noise.append(median_noise)

        #signal_average.append(np.average(signal_line) if signal_line.size > 0 else 0)
        #signal_maximum.append(np.max(signal_line) if signal_line.size > 0 else 0)
        #signal_median.append(np.median(signal_line) if signal_line.size > 0 else 0)
        max_signal_SNR.append(np.max(signal_lines_power) / median_noise if signal_lines_power.size > 0 else 0)
        max_signal_SNR_DB.append(10*np.log10(np.max(signal_lines_power) / median_noise) if signal_lines_power.size > 0 else 0)

        # Frequency of max power (signal peak)
        if signal_lines_power.size > 0:
            #signal_freqs.append(frequency[np.argmax(psd)])
            signal_powers.append(signal_lines_power)
            signal_freqs.append(signal_lines_freq)
            signal_times.append(timekeep)

        # SNR calculation
        if median_noise > 0:            
            row_SNR = psd / median_noise
            rows.append(row_SNR)
        else:
            print("No noise! The image will be empty!")

    image = np.array(rows)
    return image, frequency, noise, np.array(max_signal_SNR), np.array(signal_times), signal_freqs, signal_powers


def plot_spectogram(spec, dt, freqs, tuning_freq, label, in_dB):
    if freqs is None or len(freqs) == 0:
            print(f"[WARN] Empty frequency array for {label}, skipping plot")
            return
    
    df = freqs[-1] - freqs[-2] # frequency bin width
    time = np.arange(spec.shape[0]) * dt

    fig_1, ax_1 = plt.subplots()
    extent = ((freqs[0] - tuning_freq - (df/2)),
              (freqs[-1] - tuning_freq + (df/2)),
              time[-1] + dt,
              time[0])
    
    if in_dB:
        plot_data = 10*np.log10(spec)
        cbar_label = 'Signal to noise ratio [dB]'
        plot_file_label = f"Spectogram_DB/spec_{label}_spectral_dB.png"
        plot_file_label_clipped = f"Spectogram_DB_clipped/spec_{label}_spectral_dB_clipped.png"
    else:
        plot_data = spec
        cbar_label = 'Signal to noise ratio [-]'
        plot_file_label = f"Spectogram/spec_{label}_spectral.png"

    img_1 = ax_1.imshow(plot_data, aspect='auto', cmap=colours, extent=extent, interpolation='None', vmin=0, vmax=np.max(plot_data))#, norm=LogNorm(np.min(spec),np.max(spec)))
    cbar = fig_1.colorbar(img_1, ax=ax_1)
    cbar.set_label(cbar_label)

    ax_1.set_xlabel(f'Frequency offset w.r.t. tuning frequency ({(tuning_freq / 1e6):.6f} MHz) [Hz]')
    ax_1.set_ylabel('Time [s]')
    ax_1.set_title('Spectrogram of ' + label)

    scale = 2.5
    fig_1.set_size_inches(19, 10)
    fig_1.tight_layout()
    os.makedirs(PLOT_LOC, exist_ok=True)
    fig_1.savefig(os.path.join(PLOT_LOC, plot_file_label), dpi=scale * 100, bbox_inches='tight')
    
    if in_dB:
        mean = np.mean(plot_data)
        std = np.std(plot_data)

        cbar.remove()
        img_1 = ax_1.imshow(plot_data, aspect='auto', cmap=colours, extent=extent, interpolation='None', vmin=mean, vmax=mean+1.5*std)#, norm=LogNorm(np.min(spec),np.max(spec)))
        cbar = fig_1.colorbar(img_1, ax=ax_1)
        cbar.set_label(cbar_label)

        fig_1.savefig(os.path.join(PLOT_LOC, plot_file_label_clipped), dpi=scale * 100, bbox_inches='tight')
    
    plt.close(fig_1)
    gc.collect()

def store_extractions(dt, sig_time, noise, max_SNR, signal_freq, signal_powers, label, mask_width):
    # Store the extracted signal
    with open(os.path.join(EXTRACTION_LOC, f"RawSignalsFreq/{label}.txt"), "w") as f:
        # write header
        f.write(f'Time[s]\tFrequency[Hz]|MW:{mask_width}|TUNE:{TUNE}\n')
        # write data
        for ts, arr in zip(sig_time, signal_freq):
            arr_str = ",".join(map(str, arr))
            f.write(f"{ts}\t{arr_str}\n")
    #sig_data = np.column_stack((sig_time, signal_freq))
    #np.savetxt(os.path.join(EXTRACTION_LOC, f"RawSignals/{label}.txt"), sig_data, fmt='%f', delimiter='\t', header=f'Time[s]\tFrequency[Hz]|MW:{mask_width}|TUNE:{TUNE}', comments='')

    # Store the extracted signal
    with open(os.path.join(EXTRACTION_LOC, f"RawSignalsPower/{label}.txt"), "w") as f:
        # write header
        f.write(f'Time[s]\tPower[-]\n')
        # write data
        for ts, arr in zip(sig_time, signal_powers):
            arr_str = ",".join(map(str, arr))
            f.write(f"{ts}\t{arr_str}\n")

    # Store the computed Noise and max SNR
    time = (dt/2) + np.arange(len(noise))*dt
    noise_data = np.column_stack((time, noise))
    np.savetxt(os.path.join(EXTRACTION_LOC, f"ExtractedNoise/{label}.txt"), noise_data, fmt='%f', delimiter='\t', header='Time[s]\tNoise[-]', comments='')

    # Store the extracted SNR
    SNR_data = np.column_stack((time, max_SNR))
    np.savetxt(os.path.join(EXTRACTION_LOC, f"MaxSigSNR/{label}.txt"), SNR_data, fmt='%f', delimiter='\t', header='Time[s]\tSNR[-]', comments='')


def update_processed_log(processed_file):
    with open(LOG_FILE, 'a') as f:
        f.write(processed_file + '\n')


def process_dataset_ADEV(file, tune_freq, freq_offset, dt, sampling_rate, mask_width, duration, n_samples):
    spec, freqs, noise, sig_SNR, sig_time, sig_freqs, sig_powers = None, None, None, None, None, None, None
    try:
        spec, freqs, noise, sig_SNR, sig_time, sig_freqs, sig_powers = extract_signal(
            data=load_data(file, dt, duration, n_samples),
            dt=dt,
            sampling_rate=sampling_rate,
            tune_freq=tune_freq,
            mask_center=freq_offset,
            mask_width=mask_width
        )
        store_extractions(dt, sig_time, noise, sig_SNR, sig_freqs, sig_powers, f"{BASE}_{label_suffix}", mask_width)
        #plot_SNR(time, sig_SNR, f"{BASE}_{label_suffix}")
        #plot_noise(time, noise, f"{BASE}_{label_suffix}")
        #plot_and_store_signal(sig_time, sig_freqs, f"{BASE}_{label_suffix}", mask_width)
        plot_spectogram(spec, dt, freqs, tune_freq, f"{BASE}_{label_suffix}", False)
        plot_spectogram(spec, dt, freqs, tune_freq, f"{BASE}_{label_suffix}", True)
    finally:
        del spec, freqs, noise, sig_SNR, sig_time, sig_freqs 
        gc.collect()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: process_single_pass.py <base_filepath>")
        sys.exit(1)

    FILE = sys.argv[1]
    BASE = os.path.basename(FILE)

    try:
        META = FILE + '.yml'
        with open(META, 'r') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)

        TUNE = settings['Sat']['State']['Tuning Frequency']
        n_samples = settings['Sat']['Record']['num_sample']
        sample_rate = settings['Sat']['Record']['sample_rate']
        sat_name = settings['Sat']['State']['Name']
        duration = n_samples / sample_rate
        dt = 0.5

        mask_centers = SATELLITE_FREQS.get(sat_name, ['TUNE'])
        mask_width = 20000

        for freq in mask_centers:
            center_freq = TUNE if freq == 'TUNE' else freq
            freq_offset = center_freq - TUNE
            label_suffix = f"{int(center_freq/1e3)}kHz"
            process_dataset_ADEV(FILE, TUNE, freq_offset, dt, sample_rate, mask_width, duration, n_samples)

        update_processed_log(BASE)

    except Exception as e:
        print(f"[ERROR] Failed to process {BASE}: {e}")
        sys.exit(1)
