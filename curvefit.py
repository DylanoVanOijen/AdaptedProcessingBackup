#!/usr/bin/env python3

import os
import sys
import yaml
import gc
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from numpy.fft import fft, fftfreq, fftshift
from datetime import datetime, timezone
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy import constants
from datetime import datetime, timedelta

USE_FREQ_FILTER = True

plt.switch_backend('Agg')
colours = cm.lapaz

LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/fitted_files.log'
EXTRACTION_LOC = '/home/doptrackbox/DTB_Recordings/locally_processed/Processed/'
LOCAL_LOC = '/home/doptrackbox/DTB_Recordings/data/'
REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/archive/Dylano/DopTrackBox/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
DopTrackLoc = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/'
PLOT_LOC = os.path.join(LOCi, 'Plots/')

# Parameters:
# a - halve of the s-curve frequency width
# b - s-curve rate of change 
# c - TCA w.r.t. 0
# d - signal center frequency
def tanh(x, a, b, c, d):
    return -1*a*np.tanh(b*(x-c))+d

def compute_range_rate(frequencies, f0):
    return constants.c*(1-frequencies/f0)

def compute_doppler(dv, f0):
    return f0*(1-(dv/constants.c))


def find_yml(file, name, year):
    yml_filename = file.rsplit('_', 1)[0]
    used_remote = False
    try:
        yml_subdir = f"{name}/{year}/{yml_filename}.yml"
        if os.path.exists(LOCAL_LOC + yml_subdir):
            yml_path = LOCAL_LOC + yml_subdir
        else:                
            yml_path = REMOTE_LOC + yml_subdir

        with open(yml_path, 'r') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        print(f"[ERROR] Failed to retrieve YML file for file {FILE}: {e}")
        sys.exit(1)
    return settings

def compute_initial_guess(time, f0, name):
    time_mid_index = int(len(time)/2)

    if "FUNcube" in name:
        return [4000, 0.01, time[time_mid_index], f0]
    elif "NOAA" in name:
        return [2500, 0.005, time[time_mid_index], f0-1200]

# Retrives the initial guess based on DopTrack measurement/processing
def retrieve_initial_guess(file, name, year, t_init):
    yml_filename = file.rsplit('_', 1)[0]
    found_yml = False

    if os.path.exists(DopTrackLoc + f"products/L1C/{name}/{year}/{yml_filename}.yml"):
        yml_path = DopTrackLoc + f"products/L1C/{name}/{year}/{yml_filename}.yml"
        found_yml = True
    else:
        parts = file.split('_')
        date_part = parts[2]
        date_time = datetime.strptime(date_part, "%Y%m%d%H%M")

        for m in range(-10,10):
            candidate_datetime = date_time + timedelta(minutes=m)
            yml_filename = f"{parts[0]}_{parts[1]}_"+candidate_datetime.strftime("%Y%m%d%H%M")
            if os.path.exists(DopTrackLoc + f"products/L1C/{name}/{year}/{yml_filename}.yml"):
                yml_path = DopTrackLoc + f"products/L1C/{name}/{year}/{yml_filename}.yml"
                found_yml = True
                break

    if found_yml:
        with open(yml_path, 'r') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
        DT_tca = settings['product']['tca']
        DT_fca = settings['product']['fca']
        DT_rr_path = yml_path.rsplit('.',1)[0] + ".dat"
        
        DT_rr = np.loadtxt(DT_rr_path, skiprows=1)
        f_shifted = compute_doppler(DT_rr[0,1], DT_fca)
        df_est = abs(f_shifted - DT_fca)

        diff = DT_tca - t_init
        seconds_diff = diff.total_seconds()
        return [1*df_est, 0.01, seconds_diff, DT_fca]
    
    else:
        return None

def db_to_ratio(db):
    return 10**(db/10)

def ratio_to_db(ratio):
    return 10*np.log10(ratio)

def load_data(file):
    #with open(EXTRACTION_LOC + "ExtractedSignal/" + file + ".txt", 'r') as f:
    #    header = f.readline().strip()
    #parts = dict(item.split(":") for item in header.split("|")[1:])
    #mask_width = float(parts["MW"])

    #freq_data = np.loadtxt(EXTRACTION_LOC + "RawSignalsFreq/" + file + ".txt", skiprows=1)
    #SNR_data = np.loadtxt(EXTRACTION_LOC + "RawSignalsPower/" + file + ".txt", skiprows=1)

    # shift SNR data because of halve timebin offset
    #median_SNR = np.median(SNR_data[:,1])
    #multi = 1.15
    #multi = 0
    #treshold = db_to_ratio(7.1)

    # First retrieve noise to compute SNR
    noise_data = np.loadtxt(EXTRACTION_LOC + "ExtractedNoise/" + file + ".txt", skiprows=1)
    
    noise_times = noise_data[:,0]
    noise_powers = noise_data[:,1]

    with open(EXTRACTION_LOC + "RawSignalsFreq/" + file + ".txt") as f:
        header = f.readline().strip()
        parts = dict(item.split(":") for item in header.split("|")[1:])
        mask_width = float(parts["MW"])
        freq_data = []
        for line in f:
            ts, arr_str = line.strip().split("\t")
            arr = list(map(float, arr_str.split(",")))
            freq_data.append((ts, arr))

    with open(EXTRACTION_LOC + "RawSignalsPower/" + file + ".txt") as f:
        header = f.readline().strip()
        power_data = []
        for line in f:
            ts, arr_str = line.strip().split("\t")
            arr = list(map(float, arr_str.split(",")))
            power_data.append((ts, arr))

    sig_time = [float(ts) for ts, arr in freq_data]
    signal_freq = [arr for ts, arr in freq_data]
    signal_power = [arr for ts, arr in power_data]

    idx = np.searchsorted(noise_times, sig_time)
    noise_at_signal_entries = noise_powers[idx]
    signal_snr = [np.array(power)/ noise for power, noise in zip(signal_power, noise_at_signal_entries)]

    SNR_treshold = db_to_ratio(6.0)

    filtered_timestamps = []
    filtered_freqs = []
    filtered_powers = []
    filtered_noises = []
    filtered_snrs = []

    # Apply a weak SNR cut
    for t, freqs, powers, noise, snrs in zip(sig_time, signal_freq, signal_power, noise_at_signal_entries, signal_snr):
        f_kept = []
        p_kept = []
        snr_kept = []
        for f, p, snr in zip(freqs, powers, snrs):
            if snr >= SNR_treshold:
                f_kept.append(f)
                p_kept.append(p)
                snr_kept.append(snr)

        if f_kept:  # keep only if something remained
            filtered_timestamps.append(t)
            filtered_freqs.append(f_kept)
            filtered_powers.append(p_kept)
            filtered_snrs.append(snr_kept)
            filtered_noises.append(noise)

    return np.array(filtered_timestamps), filtered_freqs, filtered_powers, filtered_snrs, filtered_noises, mask_width 

def fit_curve(time, freq, ai, bi, ci, di, limits):
    params, cov = curve_fit(tanh, time, freq, p0=[ai,bi,ci,di], bounds=limits)
    return params

def plot_fit(file, time, freq, sig_snr, a,b,c,d, ai,bi,ci,di, tune, freq_offset, mask_width):
    signal_freq = freq - tune
    fig_1, ax_1 = plt.subplots(figsize=(8,5)) 
    full_time = np.arange(time[0], time[-1])

    ax_1.plot(full_time, tanh(full_time, ai,bi,ci,di)-tune, color="red", linestyle = "dotted", alpha = 0.8, label = f"Initial Guess (a={ai:.0f}, b={bi:.5f}, c={ci:.2f}, d={di:.3f})", zorder=1)  
    ax_1.plot(full_time, tanh(full_time, a,b,c,d)-tune, color="red", linestyle = "dashed", label = f"Best Fit (a={a:.0f}, b={b:.5f}, c={c:.2f}, d={d:.3f})", zorder=3)   
    scat = ax_1.scatter(time, signal_freq, s=10, label = "Extracted Signal", c=ratio_to_db(sig_snr),zorder=2)
    cbar = plt.colorbar(scat)
    cbar.set_label('Signal-to-Noise Ratio [dB]') 
    ax_1.set_xlabel('Time [s]')
    ax_1.set_ylabel(f'Frequency w.r.t. Tuning Frequency [Hz]')
    ax_1.set_title(f'Extracted Signal Frequency of pass {file}')

    #ax_1.set_ylim(0, 40)
    ax_1.set_xlim(time[0], time[-1])
    #ax_1.xaxis.set_inverted(True)
    #if mask_width != 0:
        #ax_1.set_ylim(freq_offset-(mask_width/2), freq_offset+(mask_width/2))
    ax_1.grid()
    #fig_1.set_size_inches(19, 10)
    #ax_1.legend()
    ax_1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig_1.tight_layout()
    os.makedirs(PLOT_LOC+"FittedSignal", exist_ok=True)
    fig_1.savefig(os.path.join(PLOT_LOC, f"FittedSignal/fitted_signal_{file}.png"))
    plt.close(fig_1)
    gc.collect()

def plot_range_rate(time, extracted_rr, fitted_rr, file):
    fig_1, ax_1 = plt.subplots(figsize=(10,5)) 

    ax_1.plot(time, fitted_rr, color="red", label = "Range-rate of Best Fit")   
    ax_1.scatter(time, extracted_rr, s=10, label = "Extracted Range-rate")
    ax_1.set_xlabel('Time [s]')
    ax_1.set_ylabel('Range-Rate [m/s]')
    ax_1.set_title(f'Extracted Range-Rate and Best Fit')

    ax_1.set_xlim(time[0], time[-1])
    ax_1.grid()
    fig_1.tight_layout()
    os.makedirs(PLOT_LOC+"FittedRangeRate", exist_ok=True)
    fig_1.savefig(os.path.join(PLOT_LOC, f"FittedRangeRate/rangerate_{file}.png"))
    plt.close(fig_1)
    gc.collect()

def plot_residuals(time, rr_residuals, file):
    fig_1, ax_1 = plt.subplots(figsize=(10,5)) 

    ax_1.hist(rr_residuals, range=[-100,100])
    ax_1.set_xlabel('Range-rate Residual [m/s]')
    ax_1.set_ylabel('Occurance [-]')
    ax_1.set_title(f'Range-Rate Residuals Distribution')

    ax_1.grid()
    #fig_1.set_size_inches(19, 10)
    fig_1.tight_layout()
    os.makedirs(PLOT_LOC+"FitResidualsHist", exist_ok=True)
    fig_1.savefig(os.path.join(PLOT_LOC, f"FitResidualsHist/residuals_hist_{file}.png"))
    plt.close(fig_1)

    fig_2, ax_2 = plt.subplots(figsize=(10,5)) 

    ax_2.plot(time, rr_residuals)
    ax_2.set_ylim(-100,100)
    ax_2.set_xlabel('Time [s]')
    ax_2.set_ylabel('Residual [m/s]]')
    ax_2.set_title(f'Range-Rate Residuals as Function of Time')

    ax_2.grid()
    #fig_1.set_size_inches(19, 10)
    fig_2.tight_layout()
    os.makedirs(PLOT_LOC+"FitResiduals", exist_ok=True)
    fig_2.savefig(os.path.join(PLOT_LOC, f"FitResiduals/residuals_{file}.png"))
    plt.close(fig_2)

    gc.collect()

def update_processed_log(processed_file):
    with open(LOG_FILE, 'a') as f:
        f.write(processed_file + '\n')    

def selection_around_fit(times, freqs, powers, noises, snrs, a,b,c,d, freq_limit):
    model_output = tanh(times, a,b,c,d)

    filtered_timestamps = []
    filtered_freqs = []
    filtered_powers = []
    filtered_noises = []
    filtered_snrs = []

    for t, freqs, powers, noise, snrs, expected in zip(times, freqs, powers, noises, snrs, model_output):
        f_kept = []
        p_kept = []
        snr_kept = []
        for f, p, snr in zip(freqs, powers, snrs):
            if abs(f - expected) <= freq_limit:
                f_kept.append(f)
                p_kept.append(p)
                snr_kept.append(snr)

        if f_kept:  # keep only if something remained
            filtered_timestamps.append(t)
            filtered_freqs.append(f_kept)
            filtered_powers.append(p_kept)
            filtered_snrs.append(snr_kept)
            filtered_noises.append(noise)

    return np.array(filtered_timestamps), filtered_freqs, filtered_powers, filtered_noises, filtered_snrs

def get_max_snr_frequencies(freq_data, snr_data, power_data):
    freq = []
    snr = []
    power = []

    for i in range(len(freq_data)):
        freqs = freq_data[i]
        snrs = snr_data[i]
        powers = power_data[i]

        max_snr_index = np.argmax(snrs)
        freq.append(freqs[max_snr_index])
        snr.append(snrs[max_snr_index])
        power.append(powers[max_snr_index])

    return np.array(freq), np.array(snr), np.array(power)

def store_extracted_signal(time, freq, power, noise, snr, label, mask_width): 
    data = np.column_stack((time, freq))
    data = np.column_stack((data, power))
    data = np.column_stack((data, noise))
    data = np.column_stack((data, snr))
    np.savetxt(os.path.join(EXTRACTION_LOC, f"ExtractedSignal/{label}.txt"), data, fmt='%f', delimiter='\t', header=f'Time[s]\tFrequency[Hz]|MW:{mask_width}|TUNE:{TUNE}\tPower [-]\tNoise [-],\tSNR [-]', comments='')

def store_fit_result(a,b,c,d, label):
    pars = {'a': float(a),
            'b': float(b),
            'c': float(c),
            'd': float(d)
            }
    with open(os.path.join(EXTRACTION_LOC, f"ExtractedFit/{label}.yml"), "w") as f:
        yaml.dump(pars, f, default_flow_style=False)


def frequency_filter(times, freqs, powers, noises, snrs):
    binwidth = 25 # in Hz
    k = 20 # number of bins around bin under test

    flattened_freqs = [item for sublist in freqs for item in sublist]
    n_samp = len(flattened_freqs)
    freq_bins = np.arange(min(flattened_freqs), max(flattened_freqs) + binwidth, binwidth)
    freq_hist, bin_edges = np.histogram(flattened_freqs, bins = freq_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #mask = freq_hist > occurance_treshold*n_samp
    #dominant_freqs = bin_centers[mask]
    #freq_mask.extend(dominant_freqs)
    #print(dominant_freqs)

    dominant_freqs = []
    for i in range(len(freq_hist)):
        # Define neighborhood (clip edges)
        left = max(0, i - k)
        right = min(len(freq_hist), i + k + 1)

        # Exclude the bin itself
        neighbors = np.delete(freq_hist[left:right], np.where(np.arange(left, right) == i))

        # Compute local threshold
        local_mean = np.mean(neighbors)
        local_std = np.std(neighbors)
        threshold = local_mean + local_std

        # Check if current bin exceeds threshold
        if freq_hist[i] > threshold:
            dominant_freqs.append(bin_centers[i])

    freq_mask = np.array(dominant_freqs)

    filtered_timestamps = []
    filtered_freqs = []
    filtered_powers = []
    filtered_noises = []
    filtered_snrs = []

    for t, freqs, powers, noise, snrs in zip(times, freqs, powers, noises, snrs):
        f_kept = []
        p_kept = []
        snr_kept = []
        for f, p, snr in zip(freqs, powers, snrs):
            not_masked = True
            for mask_freq in freq_mask:
                if abs(f - mask_freq) < binwidth:
                    not_masked = False

            if not_masked:
                f_kept.append(f)
                p_kept.append(p)
                snr_kept.append(snr)

        if f_kept:  # keep only if something remained
            filtered_timestamps.append(t)
            filtered_freqs.append(f_kept)
            filtered_powers.append(p_kept)
            filtered_snrs.append(snr_kept)
            filtered_noises.append(noise)

    return np.array(filtered_timestamps), filtered_freqs, filtered_powers, filtered_noises, filtered_snrs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: curvefit.py <base_filepath>")
        sys.exit(1)

    FILE = sys.argv[1]
    try:    
        # First extract relevant info from the filename
        parts = FILE.split('_')
        name = parts[0]  
        date_part = parts[2]
        year = date_part[:4]
        center_frequency = 1000*int(parts[-1].replace("kHz", ""))
        
        # Then find and load the yml file to retrieve tuning freq and supposed rec start time
        yml = find_yml(FILE, name, year)
        TUNE = yml['Sat']['State']['Tuning Frequency']
        REC_START = yml['Sat']['Record']['time1 UTC'].replace(tzinfo=timezone.utc)
        freq_offset = center_frequency - TUNE

        # Retrieve data and perform fit
        full_time_data, full_freq_data, full_power_data, full_snr_data, full_noise_data, mask_width = load_data(FILE)
        full_time_data, full_freq_data, full_power_data, full_noise_data, full_snr_data = frequency_filter(full_time_data, full_freq_data, full_power_data, full_noise_data, full_snr_data)

        freq_data, snr_data, power_data = get_max_snr_frequencies(full_freq_data, full_snr_data, full_power_data)

        initial_guess = compute_initial_guess(full_time_data, center_frequency, name)
        DT_initial_guess = None

        if "FUNcube" in name:
            DT_initial_guess = retrieve_initial_guess(FILE, name, year, REC_START)
        
        if DT_initial_guess is not None:
            print("Using DopTrack data initial guess...")
            ai, bi, ci, di = DT_initial_guess[0], DT_initial_guess[1], DT_initial_guess[2], DT_initial_guess[3]
            limits = [[1000, 0.002, DT_initial_guess[2]-60, DT_initial_guess[3]-(mask_width/4)], [mask_width/2, 0.05, DT_initial_guess[2]+60, DT_initial_guess[3]+(mask_width/4)]]

        else:
            ai, bi, ci, di = initial_guess[0], initial_guess[1], initial_guess[2], initial_guess[3]
            limits = [[1000, 0.002,initial_guess[2]-60, initial_guess[3]-(mask_width/4)], [mask_width/2, 0.05, initial_guess[2]+60, initial_guess[3]+(mask_width/4)]]
        
        
        # only select data 300 seconds before and after estimated TCA
        #treshold = 300
        #mask = (data[:, 0] > ci-treshold) & (data[:, 0] < ci+treshold)
        #data = data[mask]
        
        # Select limted datapoints close to fit, and for each selection take  

        #filtered_time, filtered_freq, filtered_power, filtered_noise, filtered_snr = selection_around_fit(full_time_data, full_freq_data, full_power_data, full_noise_data, full_snr_data, ai,bi,ci,di, 2500)
        #freq_data, snr_data, power_data = get_max_snr_frequencies(filtered_freq, filtered_snr, filtered_power)
        #a,b,c,d = fit_curve(filtered_time, freq_data, ai, bi, ci, di, limits)

        if DT_initial_guess:
            selection_width = 1000
        else:
            selection_width = 2500
        filtered_time, filtered_freq, filtered_power, filtered_noise, filtered_snr = selection_around_fit(full_time_data, full_freq_data, full_power_data, full_noise_data, full_snr_data, ai,bi,ci,di, selection_width)
        freq_data, snr_data, power_data = get_max_snr_frequencies(filtered_freq, filtered_snr, filtered_power)
        a,b,c,d = fit_curve(filtered_time, freq_data, ai, bi, ci, di, limits)
        changes = np.abs(np.array([(a-ai), (b-bi), (c-ci), (d-di)]))

        #size_decay = 0.95
        size_decay = 0.9
        #max_change = np.max(changes)
        #while changes[0] > 10 or changes[1] > 0.0005 or changes[2] > 0.01 or changes [3] > 10:
        min_width = 500
        while selection_width >= min_width:
            filtered_time, filtered_freq, filtered_power, filtered_noise, filtered_snr = selection_around_fit(full_time_data, full_freq_data, full_power_data, full_noise_data, full_snr_data, a,b,c,d, selection_width)
            freq_data, snr_data, power_data = get_max_snr_frequencies(filtered_freq, filtered_snr, filtered_power)
            a_new,b_new,c_new,d_new = fit_curve(filtered_time, freq_data, a, b, c, d, limits)

            changes = np.abs(np.array([(a_new-a), (b_new-b), (c_new-c), (d_new-d)]))
            #print(changes[0] > 50, changes[1] > 0.0005, changes[2] > 0.01, changes [3] > 10)
            #print(changes)
            max_change = np.max(changes)
            a = a_new
            b = b_new
            c = c_new
            d = d_new
            selection_width *= size_decay

        plot_fit(FILE, filtered_time, freq_data, snr_data, a,b,c,d, ai, bi, ci, di,  TUNE, freq_offset, mask_width)
        store_fit_result(a,b,c,d, FILE)
        store_extracted_signal(filtered_time, freq_data, power_data, filtered_noise, snr_data, FILE, mask_width)

        # Compute range-rate
        extracted_rr = compute_range_rate(freq_data, d)
        fitted_rr = compute_range_rate(tanh(filtered_time,a,b,c,d), d)
        range_rate_residuals = extracted_rr - fitted_rr
        plot_range_rate(filtered_time, extracted_rr, fitted_rr, FILE)
        plot_residuals(filtered_time, range_rate_residuals, FILE)
        
        update_processed_log(FILE)

    except Exception as e:
        print(f"[ERROR] Failed to process {FILE}: {e}")
        sys.exit(1)
